import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import tqdm
from scipy.ndimage import binary_dilation
from sklearn.neighbors import NearestNeighbors

# Configure logging to replace print statements
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("video_processing.log", mode='w')
    ]
)
logger = logging.getLogger(__name__)

# Constants for audio and video processing
AUDIO_PTIME = 0.020  # 20ms audio packetization
VIDEO_CLOCK_RATE = 90000  # Clock rate for video
VIDEO_FPS = 25  # Frames per second for video
VIDEO_PTIME = 1 / VIDEO_FPS  # Packetization time for video
VIDEO_TIME_BASE = 1 / VIDEO_CLOCK_RATE  # Time base for video
SAMPLE_RATE = 16000  # Sample rate for audio
AUDIO_TIME_BASE = 1 / SAMPLE_RATE  # Time base for audio


def extract_audio(video_path: Path, output_path: Path, sample_rate: int = 16000) -> None:
    """
    Extracts audio from a video file using ffmpeg and saves it as a WAV file.

    Args:
        video_path (Path): Path to the input video file.
        output_path (Path): Path where the extracted audio will be saved.
        sample_rate (int, optional): Desired audio sample rate. Defaults to 16000.

    Raises:
        FileNotFoundError: If the input video file does not exist.
        subprocess.CalledProcessError: If ffmpeg command fails.
    """
    logger.info(f"===== Extracting audio from {video_path} to {output_path} =====")

    if not video_path.is_file():
        logger.error(f"Video file {video_path} does not exist.")
        raise FileNotFoundError(f"Video file {video_path} does not exist.")

    cmd = [
        'ffmpeg',
        '-i', str(video_path),
        '-f', 'wav',
        '-ar', str(sample_rate),
        '-y',  # Overwrite output files without asking
        str(output_path)
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.info("===== Audio extraction completed =====")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to extract audio: {e.stderr.decode().strip()}")
        raise


def extract_audio_features(audio_path: Path, mode: str = 'wav2vec') -> None:
    """
    Extracts audio features using either wav2vec or DeepSpeech.

    Args:
        audio_path (Path): Path to the input audio file.
        mode (str, optional): Feature extraction mode ('wav2vec' or 'deepspeech'). Defaults to 'wav2vec'.

    Raises:
        ValueError: If an unsupported mode is provided.
        subprocess.CalledProcessError: If the feature extraction command fails.
    """
    logger.info(f"===== Extracting audio features for {audio_path} using {mode} =====")

    if mode == 'wav2vec':
        cmd = ['python', 'nerf/asr.py', '--wav', str(audio_path), '--save_feats']
    elif mode == 'deepspeech':
        cmd = ['python', 'data_utils/deepspeech_features/extract_ds_features.py', '--input', str(audio_path)]
    else:
        logger.error(f"Unsupported mode '{mode}'. Choose 'wav2vec' or 'deepspeech'.")
        raise ValueError(f"Unsupported mode '{mode}'. Choose 'wav2vec' or 'deepspeech'.")

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.info("===== Audio features extraction completed =====")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to extract audio features: {e.stderr.decode().strip()}")
        raise


def extract_images(video_path: Path, output_dir: Path, fps: int = 25) -> None:
    """
    Extracts frames from a video file and saves them as JPEG images at a specified frame rate.

    Args:
        video_path (Path): Path to the input video file.
        output_dir (Path): Directory where the extracted images will be saved.
        fps (int, optional): Frames per second to extract. Defaults to 25.

    Raises:
        FileNotFoundError: If the input video file does not exist.
        subprocess.CalledProcessError: If ffmpeg command fails.
    """
    logger.info(f"===== Extracting images from {video_path} to {output_dir} at {fps} FPS =====")

    if not video_path.is_file():
        logger.error(f"Video file {video_path} does not exist.")
        raise FileNotFoundError(f"Video file {video_path} does not exist.")

    output_pattern = output_dir / "%d.jpg"
    cmd = [
        'ffmpeg',
        '-i', str(video_path),
        '-vf', f'fps={fps}',
        '-qmin', '1',
        '-q:v', '1',
        '-start_number', '0',
        str(output_pattern)
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.info("===== Image extraction completed =====")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to extract images: {e.stderr.decode().strip()}")
        raise


def extract_semantics(ori_imgs_dir: Path, parsing_dir: Path) -> None:
    """
    Extracts semantic segmentation data from original images using a face parsing script.

    Args:
        ori_imgs_dir (Path): Directory containing original images.
        parsing_dir (Path): Directory where parsed semantic images will be saved.

    Raises:
        subprocess.CalledProcessError: If the parsing command fails.
    """
    logger.info(f"===== Extracting semantics from {ori_imgs_dir} to {parsing_dir} =====")

    cmd = [
        'python',
        'data_utils/face_parsing/test.py',
        '--respath', str(parsing_dir),
        '--imgpath', str(ori_imgs_dir)
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.info("===== Semantic extraction completed =====")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to extract semantics: {e.stderr.decode().strip()}")
        raise


def extract_landmarks(ori_imgs_dir: Path) -> None:
    """
    Extracts facial landmarks from images using the face_alignment library.

    Args:
        ori_imgs_dir (Path): Directory containing original images.

    Raises:
        FileNotFoundError: If no JPEG images are found in the directory.
    """
    logger.info(f"===== Extracting face landmarks from {ori_imgs_dir} =====")

    import face_alignment
    from tqdm import tqdm

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
    image_paths = list(ori_imgs_dir.glob('*.jpg'))

    if not image_paths:
        logger.error(f"No JPEG images found in {ori_imgs_dir}.")
        raise FileNotFoundError(f"No JPEG images found in {ori_imgs_dir}.")

    for image_path in tqdm(image_paths, desc="Extracting landmarks"):
        input_image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)  # [H, W, 3]
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        preds = fa.get_landmarks(input_image)
        if preds:
            landmarks = preds[0].reshape(-1, 2)[:, :2]
            landmarks_path = image_path.with_suffix('.lms')
            np.savetxt(str(landmarks_path), landmarks, fmt='%f')
    del fa
    logger.info("===== Face landmarks extraction completed =====")


def extract_background(base_dir: Path, ori_imgs_dir: Path) -> None:
    """
    Extracts a background image by analyzing semantic segmentation and selecting background pixels.

    Args:
        base_dir (Path): Base directory for saving the background image.
        ori_imgs_dir (Path): Directory containing original images.

    Raises:
        FileNotFoundError: If no JPEG images are found in the directory.
    """
    logger.info(f"===== Extracting background image from {ori_imgs_dir} =====")

    image_paths = list(ori_imgs_dir.glob('*.jpg'))

    if not image_paths:
        logger.error(f"No JPEG images found in {ori_imgs_dir}.")
        raise FileNotFoundError(f"No JPEG images found in {ori_imgs_dir}.")

    # Sample 1/20 of the images for processing
    sampled_image_paths = image_paths[::20]
    sample_image = cv2.imread(str(sampled_image_paths[0]), cv2.IMREAD_UNCHANGED)  # [H, W, 3]
    h, w = sample_image.shape[:2]

    all_xys = np.mgrid[0:h, 0:w].reshape(2, -1).T  # [H*W, 2]
    distss = []

    for image_path in tqdm(sampled_image_paths, desc="Extracting background"):
        parsing_path = image_path.with_name(image_path.stem.replace('ori_imgs', 'parsing') + '.png')
        parse_img = cv2.imread(str(parsing_path))
        if parse_img is None:
            logger.warning(f"Parsing image {parsing_path} not found. Skipping.")
            continue
        bg = (parse_img[..., 0] == 255) & (parse_img[..., 1] == 255) & (parse_img[..., 2] == 255)
        fg_xys = np.argwhere(~bg)
        if fg_xys.size == 0:
            logger.warning(f"No foreground pixels found in {parsing_path}. Skipping.")
            continue
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(fg_xys)
        dists, _ = nbrs.kneighbors(all_xys)
        distss.append(dists)

    if not distss:
        logger.error("No distance data collected for background extraction.")
        raise ValueError("No distance data collected for background extraction.")

    distss = np.stack(distss)
    max_dist = np.max(distss, axis=0)
    max_id = np.argmax(distss, axis=0)

    bc_pixs = max_dist > 5
    bc_pixs_id = np.nonzero(bc_pixs)
    bc_ids = max_id[bc_pixs]

    imgs = [cv2.imread(str(image_path)) for image_path in sampled_image_paths]
    imgs = np.stack(imgs).reshape(-1, h * w, 3)  # [num_samples, H*W, 3]

    bc_img = np.zeros((h * w, 3), dtype=np.uint8)
    bc_img[bc_pixs_id] = imgs[bc_ids, bc_pixs_id, :]
    bc_img = bc_img.reshape(h, w, 3)

    max_dist = max_dist.reshape(h, w)
    bc_pixs = max_dist > 5
    bg_xys = np.argwhere(~bc_pixs)
    fg_xys = np.argwhere(bc_pixs)
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(fg_xys)
    distances, indices = nbrs.kneighbors(bg_xys)
    bg_fg_xys = fg_xys[indices[:, 0]]
    bc_img[bg_xys[:, 0], bg_xys[:, 1], :] = bc_img[bg_fg_xys[:, 0], bg_xys[:, 1], :]

    # Save the background image
    bc_image_path = base_dir / 'bc.jpg'
    cv2.imwrite(str(bc_image_path), bc_img)
    logger.info("===== Background image extraction completed =====")


def extract_torso_and_gt(base_dir: Path, ori_imgs_dir: Path) -> None:
    """
    Extracts torso and ground truth (gt) images by inpainting head and neck regions.

    Args:
        base_dir (Path): Base directory for saving torso and gt images.
        ori_imgs_dir (Path): Directory containing original images.

    Raises:
        FileNotFoundError: If required files are missing.
    """
    logger.info(f"===== Extracting torso and gt images for {base_dir} =====")

    from scipy.ndimage import binary_dilation, binary_erosion

    # Load background image
    bg_image_path = base_dir / 'bc.jpg'
    bg_image = cv2.imread(str(bg_image_path), cv2.IMREAD_UNCHANGED)
    if bg_image is None:
        logger.error(f"Background image {bg_image_path} not found.")
        raise FileNotFoundError(f"Background image {bg_image_path} not found.")

    image_paths = list(ori_imgs_dir.glob('*.jpg'))

    for image_path in tqdm.tqdm(image_paths, desc="Extracting torso and gt"):
        # Read original image
        ori_image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)  # [H, W, 3]

        # Read semantic segmentation image
        seg_path = image_path.with_name(image_path.stem.replace('ori_imgs', 'parsing') + '.png')
        seg = cv2.imread(str(seg_path))
        if seg is None:
            logger.warning(f"Semantic segmentation image {seg_path} not found. Skipping.")
            continue

        # Define semantic parts based on color coding
        head_part = (seg[..., 0] == 255) & (seg[..., 1] == 0) & (seg[..., 2] == 0)
        neck_part = (seg[..., 0] == 0) & (seg[..., 1] == 255) & (seg[..., 2] == 0)
        torso_part = (seg[..., 0] == 0) & (seg[..., 1] == 0) & (seg[..., 2] == 255)
        bg_part = (seg[..., 0] == 255) & (seg[..., 1] == 255) & (seg[..., 2] == 255)

        # Create ground truth image by replacing background
        gt_image = ori_image.copy()
        gt_image[bg_part] = bg_image[bg_part]
        gt_imgs_dir = image_path.parent.parent / 'gt_imgs'
        gt_imgs_dir.mkdir(parents=True, exist_ok=True)
        gt_image_path = image_path.with_name(image_path.stem.replace('ori_imgs', 'gt_imgs') + '.jpg')
        cv2.imwrite(str(gt_image_path), gt_image)

        # Create torso image by inpainting head and neck regions
        torso_image = gt_image.copy()  # RGB
        torso_image[head_part] = bg_image[head_part]
        torso_alpha = 255 * np.ones((gt_image.shape[0], gt_image.shape[1], 1), dtype=np.uint8)  # Alpha channel

        # Vertical inpainting for torso parts
        L = 9  # 8 + 1
        torso_coords = np.argwhere(torso_part)  # [M, 2]
        inds = np.lexsort((torso_coords[:, 0], torso_coords[:, 1]))
        torso_coords_sorted = torso_coords[inds]

        # Select top pixel for each column
        unique_cols, unique_indices = np.unique(torso_coords_sorted[:, 1], return_index=True)
        top_torso_coords = torso_coords_sorted[unique_indices]  # [m, 2]

        # Filter top torso pixels that are above the head
        top_torso_coords_up = top_torso_coords.copy()
        top_torso_coords_up[:, 0] -= 1  # Move up by one pixel
        mask = head_part[tuple(top_torso_coords_up.T)]
        top_torso_coords = top_torso_coords[mask]

        # Get colors for inpainting
        top_torso_colors = gt_image[tuple(top_torso_coords.T)]  # [m, 3]

        # Construct inpaint coordinates (vertically up)
        inpaint_torso_coords = top_torso_coords[:, None, :] - np.arange(L).reshape(L, 1, 1)
        inpaint_torso_coords = inpaint_torso_coords.reshape(-1, 2)  # [Lm, 2]
        inpaint_torso_colors = (top_torso_colors[:, None, :] * (0.98 ** np.arange(L))).reshape(-1, 3)

        # Set inpaint colors
        torso_image[tuple(inpaint_torso_coords.T)] = inpaint_torso_colors

        # Create mask for inpainted regions
        inpaint_torso_mask = np.zeros_like(torso_image[..., 0], dtype=bool)
        inpaint_torso_mask[tuple(inpaint_torso_coords.T)] = True

        # Repeat similar process for neck parts
        push_down = 4
        L_neck = 49  # 48 + push_down + 1

        # Dilate neck part for better inpainting
        neck_part_dilated = binary_dilation(neck_part, structure=np.array([[0, 1, 0],
                                                                            [0, 1, 0],
                                                                            [0, 1, 0]], dtype=bool), iterations=3)
        neck_coords = np.argwhere(neck_part_dilated)  # [M, 2]
        inds = np.lexsort((neck_coords[:, 0], neck_coords[:, 1]))
        neck_coords_sorted = neck_coords[inds]

        # Select top pixel for each column
        unique_cols_neck, unique_indices_neck = np.unique(neck_coords_sorted[:, 1], return_index=True)
        top_neck_coords = neck_coords_sorted[unique_indices_neck]  # [m, 2]

        # Filter top neck pixels that are above the head
        top_neck_coords_up = top_neck_coords.copy()
        top_neck_coords_up[:, 0] -= 1  # Move up by one pixel
        mask_neck = head_part[tuple(top_neck_coords_up.T)]
        top_neck_coords = top_neck_coords[mask_neck]

        # Push neck pixels down to make inpainting more natural
        offset_down = np.minimum(np.full(top_neck_coords.shape[0], push_down), top_neck_coords.shape[0] - 1)
        top_neck_coords[:, 0] += offset_down

        # Get colors for inpainting neck
        top_neck_colors = gt_image[tuple(top_neck_coords.T)]  # [m, 3]

        # Construct inpaint coordinates (vertically up)
        inpaint_neck_coords = top_neck_coords[:, None, :] - np.arange(L_neck).reshape(L_neck, 1, 1)
        inpaint_neck_coords = inpaint_neck_coords.reshape(-1, 2)  # [Lm, 2]
        inpaint_neck_colors = (top_neck_colors[:, None, :] * (0.98 ** np.arange(L_neck))).reshape(-1, 3)

        # Set inpaint colors
        torso_image[tuple(inpaint_neck_coords.T)] = inpaint_neck_colors

        # Apply Gaussian blur to inpainted areas to avoid artifacts
        inpaint_mask = np.zeros_like(torso_image[..., 0], dtype=bool)
        inpaint_mask[tuple(inpaint_neck_coords.T)] = True
        blur_img = torso_image.copy()
        blur_img = cv2.GaussianBlur(blur_img, (5, 5), cv2.BORDER_DEFAULT)
        torso_image[inpaint_mask] = blur_img[inpaint_mask]

        # Create final mask and apply alpha channel
        mask_final = neck_part_dilated | torso_part | inpaint_torso_mask
        if hasattr(inpaint_torso_mask, '__len__') and inpaint_torso_mask is not None:
            mask_final = mask_final | inpaint_torso_mask
        torso_image[~mask_final] = 0
        torso_alpha[~mask_final] = 0

        # Save torso image with alpha channel
        torso_imgs_dir = base_dir / 'torso_imgs'
        torso_imgs_dir.mkdir(parents=True, exist_ok=True)
        torso_image_path = image_path.with_name(image_path.stem.replace('ori_imgs', 'torso_imgs') + '.png')
        torso_image_rgba = np.concatenate([torso_image, torso_alpha], axis=-1)
        cv2.imwrite(str(torso_image_path), torso_image_rgba)

    logger.info("===== Torso and GT images extraction completed =====")


def face_tracking(ori_imgs_dir: Path) -> None:
    """
    Performs face tracking on original images using a face tracking script.

    Args:
        ori_imgs_dir (Path): Directory containing original images.

    Raises:
        FileNotFoundError: If no JPEG images are found in the directory.
        subprocess.CalledProcessError: If the face tracking command fails.
    """
    logger.info(f"===== Performing face tracking on {ori_imgs_dir} =====")

    image_paths = list(ori_imgs_dir.glob('*.jpg'))

    if not image_paths:
        logger.error(f"No JPEG images found in {ori_imgs_dir}.")
        raise FileNotFoundError(f"No JPEG images found in {ori_imgs_dir}.")

    # Read one image to get height and width
    tmp_image = cv2.imread(str(image_paths[0]), cv2.IMREAD_UNCHANGED)  # [H, W, 3]
    if tmp_image is None:
        logger.error(f"Failed to read image {image_paths[0]}.")
        raise FileNotFoundError(f"Failed to read image {image_paths[0]}.")

    h, w = tmp_image.shape[:2]

    cmd = [
        'python',
        'data_utils/face_tracking/face_tracker.py',
        '--path', str(ori_imgs_dir),
        '--img_h', str(h),
        '--img_w', str(w),
        '--frame_num', str(len(image_paths))
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.info("===== Face tracking completed =====")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to perform face tracking: {e.stderr.decode().strip()}")
        raise


def save_transforms(base_dir: Path, ori_imgs_dir: Path) -> None:
    """
    Saves transformation matrices and other metadata into JSON files for training and validation sets.

    Args:
        base_dir (Path): Base directory containing tracking parameters.
        ori_imgs_dir (Path): Directory containing original images.

    Raises:
        FileNotFoundError: If the tracking parameters file does not exist.
    """
    logger.info(f"===== Saving transforms for {base_dir} =====")

    import torch

    track_params_path = base_dir / 'track_params.pt'
    if not track_params_path.is_file():
        logger.error(f"Tracking parameters file {track_params_path} not found.")
        raise FileNotFoundError(f"Tracking parameters file {track_params_path} not found.")

    params_dict = torch.load(str(track_params_path))
    focal_len = params_dict.get('focal', [1.0])[0]
    euler_angle = params_dict.get('euler', torch.zeros((0, 3)))
    trans = params_dict.get('trans', torch.zeros((0, 3))) / 10.0
    valid_num = euler_angle.shape[0]

    def euler2rot(euler_angles: torch.Tensor) -> torch.Tensor:
        """
        Converts Euler angles to rotation matrices.

        Args:
            euler_angles (torch.Tensor): Tensor of shape [batch_size, 3] containing Euler angles.

        Returns:
            torch.Tensor: Tensor of shape [batch_size, 3, 3] containing rotation matrices.
        """
        batch_size = euler_angles.shape[0]
        theta = euler_angles[:, 0].reshape(-1, 1, 1)
        phi = euler_angles[:, 1].reshape(-1, 1, 1)
        psi = euler_angles[:, 2].reshape(-1, 1, 1)
        one = torch.ones((batch_size, 1, 1), dtype=torch.float32, device=euler_angles.device)
        zero = torch.zeros((batch_size, 1, 1), dtype=torch.float32, device=euler_angles.device)

        rot_x = torch.cat((
            torch.cat((one, zero, zero), dim=2),
            torch.cat((zero, theta.cos(), theta.sin()), dim=2),
            torch.cat((zero, -theta.sin(), theta.cos()), dim=2),
        ), dim=1)

        rot_y = torch.cat((
            torch.cat((phi.cos(), zero, -phi.sin()), dim=2),
            torch.cat((zero, one, zero), dim=2),
            torch.cat((phi.sin(), zero, phi.cos()), dim=2),
        ), dim=1)

        rot_z = torch.cat((
            torch.cat((psi.cos(), -psi.sin(), zero), dim=2),
            torch.cat((psi.sin(), psi.cos(), zero), dim=2),
            torch.cat((zero, zero, one), dim=2)
        ), dim=1)

        return torch.bmm(rot_x, torch.bmm(rot_y, rot_z))

    # Split data into training and validation sets
    train_val_split = int(valid_num * 10 / 11)
    train_ids = torch.arange(0, train_val_split)
    val_ids = torch.arange(train_val_split, valid_num)

    rot = euler2rot(euler_angle)
    rot_inv = rot.permute(0, 2, 1)
    trans_inv = -torch.bmm(rot_inv, trans.unsqueeze(2)).squeeze(2)

    pose = torch.eye(4, dtype=torch.float32)
    save_ids = ['train', 'val']
    train_val_ids = [train_ids, val_ids]
    mean_z = -float(torch.mean(trans[:, 2]).item())

    for split in range(2):
        transform_dict = {
            'focal_len': float(focal_len),
            'cx': float(w / 2.0),
            'cy': float(h / 2.0),
            'frames': []
        }
        ids = train_val_ids[split]
        save_id = save_ids[split]

        for i in ids:
            i = i.item()
            frame_dict = {
                'img_id': i,
                'aud_id': i,
                'transform_matrix': pose.numpy().tolist()
            }

            pose[:3, :3] = rot_inv[i]
            pose[:3, 3] = trans_inv[i]

            frame_dict['transform_matrix'] = pose.numpy().tolist()
            transform_dict['frames'].append(frame_dict)

        transform_file = base_dir / f'transforms_{save_id}.json


        with open(os.path.join(transform_file), 'w') as fp:
            json.dump(transform_dict, fp, indent=2, separators=(',', ': '))

    logger(f'[INFO] ===== finished saving transforms =====')


def main():
    parser = argparse.ArgumentParser(description="Video and Audio Processing Pipeline")
    parser.add_argument(
        '--video',
        type=Path,
        required=True,
        help='Path to the input video file.'
    )
    parser.add_argument(
        '--output_dir',
        type=Path,
        required=True,
        help='Directory where all outputs will be saved.'
    )
    parser.add_argument(
        '--feature_mode',
        type=str,
        choices=['wav2vec', 'deepspeech'],
        default='wav2vec',
        help="Mode for audio feature extraction ('wav2vec' or 'deepspeech')."
    )
    parser.add_argument(
        '--sample_rate',
        type=int,
        default=16000,
        help='Sample rate for audio extraction. Default is 16000 Hz.'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=25,
        help='Frames per second for image extraction. Default is 25 FPS.'
    )

    args = parser.parse_args()

    # Create base output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Define output paths
    audio_output_path = args.output_dir / "audio.wav"
    images_output_dir = args.output_dir / "ori_imgs"
    parsing_output_dir = args.output_dir / "parsing"
    landmarks_output_dir = args.output_dir / "landmarks"
    background_output_dir = args.output_dir / "background"
    torso_gt_output_dir = args.output_dir / "torso_gt"
    transforms_output_dir = args.output_dir / "transforms"

    # Create necessary subdirectories
    images_output_dir.mkdir(parents=True, exist_ok=True)
    parsing_output_dir.mkdir(parents=True, exist_ok=True)
    landmarks_output_dir.mkdir(parents=True, exist_ok=True)
    background_output_dir.mkdir(parents=True, exist_ok=True)
    torso_gt_output_dir.mkdir(parents=True, exist_ok=True)
    transforms_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Step 1: Extract audio from video
        extract_audio(
            video_path=args.video,
            output_path=audio_output_path,
            sample_rate=args.sample_rate
        )

        # Step 2: Extract audio features
        extract_audio_features(
            audio_path=audio_output_path,
            mode=args.feature_mode
        )

        # Step 3: Extract frames as images from video
        extract_images(
            video_path=args.video,
            output_dir=images_output_dir,
            fps=args.fps
        )

        # Step 4: Extract semantic segmentation data
        extract_semantics(
            ori_imgs_dir=images_output_dir,
            parsing_dir=parsing_output_dir
        )

        # Step 5: Extract facial landmarks
        extract_landmarks(
            ori_imgs_dir=images_output_dir
        )

        # Step 6: Extract background image
        extract_background(
            base_dir=args.output_dir,
            ori_imgs_dir=images_output_dir
        )

        # Step 7: Extract torso and ground truth images
        extract_torso_and_gt(
            base_dir=args.output_dir,
            ori_imgs_dir=images_output_dir
        )

        # Step 8: Perform face tracking
        face_tracking(
            ori_imgs_dir=images_output_dir
        )

        # Step 9: Save transformation matrices and metadata
        save_transforms(
            base_dir=args.output_dir,
            ori_imgs_dir=images_output_dir
        )

        logger.info("===== Video and Audio Processing Pipeline completed successfully =====")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()


