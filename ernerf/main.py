import os
import argparse
import logging
import torch
import numpy as np

from .nerf_triplane.provider import NeRFDataset, NeRFDataset_Test
from .nerf_triplane.utils import seed_everything, PSNRMeter, LPIPSMeter, LMDMeter
from .nerf_triplane.network import NeRFNetwork
from torch import optim

# Disable tf32 features to fix low numerical accuracy on RTX 30xx GPUs.
try:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
except AttributeError:
    logging.info('This PyTorch version does not support tf32.')


def load_and_freeze_head_checkpoint(model, head_ckpt):
    """Load the head checkpoint and freeze the loaded parameters."""
    model_dict = torch.load(head_ckpt, map_location='cpu')['model']
    missing_keys, unexpected_keys = model.load_state_dict(model_dict, strict=False)

    if missing_keys:
        logging.warning(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        logging.warning(f"Unexpected keys: {unexpected_keys}")

    for k, v in model.named_parameters():
        if k in model_dict:
            v.requires_grad = False


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()

    # Basic options
    parser.add_argument('path', type=str)
    parser.add_argument('-O', action='store_true', help="Equals --fp16 --cuda_ray --exp_eye")
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--seed', type=int, default=0)

    # Modes
    mode_group = parser.add_argument_group('Modes')
    mode_group.add_argument('--test', action='store_true', help="Test mode (load model and test dataset)")
    mode_group.add_argument('--test_train', action='store_true', help="Test mode (load model and train dataset)")

    # Data options
    data_group = parser.add_argument_group('Data options')
    data_group.add_argument('--data_range', type=int, nargs='*', default=[0, -1], help="Data range to use")
    data_group.add_argument('--pose', type=str, default="data/data_kf.json", help="Pose source (transforms.json)")
    data_group.add_argument('--au', type=str, default="data/au.csv", help="Eye blink area")

    # Training options
    train_group = parser.add_argument_group('Training options')
    train_group.add_argument('--iters', type=int, default=200000, help="Training iterations")
    train_group.add_argument('--lr', type=float, default=1e-2, help="Initial learning rate")
    train_group.add_argument('--lr_net', type=float, default=1e-3, help="Initial learning rate for network")
    train_group.add_argument('--ckpt', type=str, default='latest')
    train_group.add_argument('--num_rays', type=int, default=4096 * 16,
                             help="Number of rays sampled per image for each training step")
    train_group.add_argument('--cuda_ray', action='store_true', help="Use CUDA raymarching instead of PyTorch")
    train_group.add_argument('--max_steps', type=int, default=16,
                             help="Max number of steps sampled per ray (only valid when using --cuda_ray)")
    train_group.add_argument('--num_steps', type=int, default=16,
                             help="Number of steps sampled per ray (only valid when NOT using --cuda_ray)")
    train_group.add_argument('--upsample_steps', type=int, default=0,
                             help="Number of steps up-sampled per ray (only valid when NOT using --cuda_ray)")
    train_group.add_argument('--update_extra_interval', type=int, default=16,
                             help="Iteration interval to update extra status (only valid when using --cuda_ray)")
    train_group.add_argument('--max_ray_batch', type=int, default=4096,
                             help="Batch size of rays at inference to avoid OOM (only valid when NOT using --cuda_ray)")

    # Loss options
    loss_group = parser.add_argument_group('Loss options')
    loss_group.add_argument('--warmup_step', type=int, default=10000, help="Warm-up steps")
    loss_group.add_argument('--amb_aud_loss', type=int, default=1, help="Use ambient audio loss")
    loss_group.add_argument('--amb_eye_loss', type=int, default=1, help="Use ambient eye loss")
    loss_group.add_argument('--unc_loss', type=int, default=1, help="Use uncertainty loss")
    loss_group.add_argument('--lambda_amb', type=float, default=1e-4, help="Lambda for ambient loss")

    # Network options
    network_group = parser.add_argument_group('Network options')
    network_group.add_argument('--fp16', action='store_true', help="Use AMP mixed precision training")
    network_group.add_argument('--bg_img', type=str, default='white', help="Background image")
    network_group.add_argument('--fbg', action='store_true', help="Frame-wise background")
    network_group.add_argument('--exp_eye', action='store_true', help="Explicitly control the eyes")
    network_group.add_argument('--fix_eye', type=float, default=-1,
                               help="Fixed eye area, negative to disable, set to 0-0.3 for a reasonable eye")
    network_group.add_argument('--smooth_eye', action='store_true', help="Smooth the eye area sequence")
    network_group.add_argument('--torso_shrink', type=float, default=0.8,
                               help="Shrink background coordinates to allow more flexibility in deformation")

    # Dataset options
    dataset_group = parser.add_argument_group('Dataset options')
    dataset_group.add_argument('--color_space', type=str, default='srgb', help="Color space, supports (linear, srgb)")
    dataset_group.add_argument('--preload', type=int, default=0,
                               help="0 means load data from disk on-the-fly, 1 means preload to CPU, 2 means preload to GPU.")
    dataset_group.add_argument('--bound', type=float, default=1,
                               help="Assume the scene is bounded in box[-bound, bound]^3, if > 1, will invoke adaptive ray marching.")
    dataset_group.add_argument('--scale', type=float, default=4, help="Scale camera location into box[-bound, bound]^3")
    dataset_group.add_argument('--offset', type=float, nargs='*', default=[0, 0, 0], help="Offset of camera location")
    dataset_group.add_argument('--dt_gamma', type=float, default=1 / 256,
                               help="dt_gamma (>=0) for adaptive ray marching. Set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
    dataset_group.add_argument('--min_near', type=float, default=0.05, help="Minimum near distance for camera")
    dataset_group.add_argument('--density_thresh', type=float, default=10,
                               help="Threshold for density grid to be occupied (sigma)")
    dataset_group.add_argument('--density_thresh_torso', type=float, default=0.01,
                               help="Threshold for density grid to be occupied (alpha)")
    dataset_group.add_argument('--patch_size', type=int, default=1,
                               help="[Experimental] Render patches in training to apply LPIPS loss. 1 means disabled, use [64, 32, 16] to enable")

    # Lips options
    lips_group = parser.add_argument_group('Lips options')
    lips_group.add_argument('--init_lips', action='store_true', help="Initialize lips region")
    lips_group.add_argument('--finetune_lips', action='store_true',
                            help="Use LPIPS and landmarks to fine-tune lips region")
    lips_group.add_argument('--smooth_lips', action='store_true',
                            help="Smooth the encoding in an exponential decay way")

    # Torso options
    torso_group = parser.add_argument_group('Torso options')
    torso_group.add_argument('--torso', action='store_true', help="Fix head and train torso")
    torso_group.add_argument('--head_ckpt', type=str, default='', help="Head model checkpoint")

    # GUI options
    gui_group = parser.add_argument_group('GUI options')
    gui_group.add_argument('--gui', action='store_true', help="Start a GUI")
    gui_group.add_argument('--W', type=int, default=450, help="GUI width")
    gui_group.add_argument('--H', type=int, default=450, help="GUI height")
    gui_group.add_argument('--radius', type=float, default=3.35, help="Default GUI camera radius from center")
    gui_group.add_argument('--fovy', type=float, default=21.24, help="Default GUI camera field of view in y-axis")
    gui_group.add_argument('--max_spp', type=int, default=1, help="GUI rendering max samples per pixel")

    # Additional options
    extra_group = parser.add_argument_group('Additional options')
    extra_group.add_argument('--att', type=int, default=2,
                             help="Audio attention mode (0 = turn off, 1 = left-direction, 2 = bi-direction)")
    extra_group.add_argument('--aud', type=str, default='',
                             help="Audio source (empty will load the default, else should be a path to a npy file)")
    extra_group.add_argument('--emb', action='store_true', help="Use audio class + embedding instead of logits")
    extra_group.add_argument('--ind_dim', type=int, default=4, help="Individual code dimension, 0 to turn off")
    extra_group.add_argument('--ind_num', type=int, default=10000,
                             help="Number of individual codes, should be larger than training dataset size")
    extra_group.add_argument('--ind_dim_torso', type=int, default=8,
                             help="Individual code dimension for torso, 0 to turn off")
    extra_group.add_argument('--amb_dim', type=int, default=2, help="Ambient dimension")
    extra_group.add_argument('--part', action='store_true', help="Use partial training data (1/10)")
    extra_group.add_argument('--part2', action='store_true', help="Use partial training data (first 15s)")
    extra_group.add_argument('--train_camera', action='store_true', help="Optimize camera pose")
    extra_group.add_argument('--smooth_path', action='store_true',
                             help="Smooth camera pose trajectory with a window size")
    extra_group.add_argument('--smooth_path_window', type=int, default=7, help="Smoothing window size")

    # ASR options
    asr_group = parser.add_argument_group('ASR options')
    asr_group.add_argument('--asr', action='store_true', help="Load ASR for real-time app")
    asr_group.add_argument('--asr_wav', type=str, default='', help="Load the wav and use as input")
    asr_group.add_argument('--asr_play', action='store_true', help="Play out the audio")
    asr_group.add_argument('--asr_model', type=str, default='deepspeech')
    asr_group.add_argument('--asr_save_feats', action='store_true', help="Save ASR features")
    asr_group.add_argument('--fps', type=int, default=50, help="Audio FPS")
    asr_group.add_argument('-l', type=int, default=10, help="Sliding window left length (unit: 20ms)")
    asr_group.add_argument('-m', type=int, default=50, help="Sliding window middle length (unit: 20ms)")
    asr_group.add_argument('-r', type=int, default=10, help="Sliding window right length (unit: 20ms)")

    opt = parser.parse_args()

    if opt.O:
        opt.fp16 = True
        opt.exp_eye = True

    opt.cuda_ray = True

    if opt.patch_size > 1:
        assert opt.num_rays % (opt.patch_size ** 2) == 0, "patch_size ** 2 should be divisible by num_rays."

    logging.info(opt)
    seed_everything(opt.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeRFNetwork(opt)

    if opt.torso and opt.head_ckpt != '':
        load_and_freeze_head_checkpoint(model, opt.head_ckpt)

    criterion = torch.nn.MSELoss(reduction='none')

    if opt.test:
        if opt.gui:
            metrics = []
        else:
            metrics = [PSNRMeter(), LPIPSMeter(device=device), LMDMeter(backend='fan')]

        trainer = Trainer('ngp', opt, model, device=device, workspace=opt.workspace, criterion=criterion,
                          fp16=opt.fp16, metrics=metrics, use_checkpoint=opt.ckpt)

        if opt.test_train:
            test_set = NeRFDataset(opt, device=device, type='train')
            test_set.training = False
            test_set.num_rays = -1
            test_loader = test_set.dataloader()
        else:
            test_loader = NeRFDataset(opt, device=device, type='test').dataloader()

        # Temporary fix for update_extra_states
        model.aud_features = test_loader._data.auds
        model.eye_areas = test_loader._data.eye_area

        if opt.gui:
            from nerf_triplane.gui import NeRFGUI

            with NeRFGUI(opt, trainer, test_loader) as gui:
                gui.render()
        else:
            trainer.test(test_loader)
            if test_loader.has_gt:
                trainer.evaluate(test_loader)
    else:
        optimizer_fn = lambda model: optim.AdamW(model.get_params(opt.lr, opt.lr_net), betas=(0, 0.99), eps=1e-8)
        train_loader = NeRFDataset(opt, device=device, type='train').dataloader()

        assert len(train_loader) < opt.ind_num, f"Dataset too large: {len(train_loader)} frames. Increase --ind_num."

        # Temporary fix for update_extra_states
        model.aud_features = train_loader._data.auds
        model.eye_area = train_loader._data.eye_area
        model.poses = train_loader._data.poses

        # Decay to 0.1 * init_lr at last iteration step
        if opt.finetune_lips:
            scheduler_fn = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer,
                                                                         lambda iter: 0.05 ** (iter / opt.iters))
        else:
            scheduler_fn = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer,
                                                                         lambda iter: 0.5 ** (iter / opt.iters))

        metrics = [PSNRMeter(), LPIPSMeter(device=device)]
        eval_interval = max(1, int(5000 / len(train_loader)))

        trainer = Trainer('ngp', opt, model, device=device, workspace=opt.workspace, optimizer=optimizer_fn,
                          criterion=criterion, ema_decay=0.95, fp16=opt.fp16, lr_scheduler=scheduler_fn,
                          scheduler_update_every_step=True, metrics=metrics, use_checkpoint=opt.ckpt,
                          eval_interval=eval_interval)

        with open(os.path.join(opt.workspace, 'opt.txt'), 'a') as f:
            f.write(str(opt))

        if opt.gui:
            from nerf_triplane.gui import NeRFGUI

            with NeRFGUI(opt, trainer, train_loader) as gui:
                gui.render()
        else:
            valid_loader = NeRFDataset(opt, device=device, type='val', downscale=1).dataloader()
            max_epochs = np.ceil(opt.iters / len(train_loader)).astype(np.int32)
            logging.info(f"Max epochs = {max_epochs}")
            trainer.train(train_loader, valid_loader, max_epochs)

            # Free memory
            del train_loader, valid_loader
            torch.cuda.empty_cache()

            # Test and evaluate
            test_loader = NeRFDataset(opt, device=device, type='test').dataloader()

            if test_loader.has_gt:
                trainer.evaluate(test_loader)

            trainer.test(test_loader)
