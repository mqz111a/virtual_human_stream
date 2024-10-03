import os
import time
import copy
import glob
import pickle
import multiprocessing as mp
import asyncio
import cv2
import torch
import numpy as np
from threading import Thread
from av import AudioFrame, VideoFrame
from tqdm import tqdm

from musetalk.utils.blending import get_image_blending
from musetalk.utils.utils import load_diffusion_model, load_audio_model
from ttsreal import EdgeTTS, VoitsTTS, XTTS
from museasr import MuseASR


def read_images(image_list):
    """
    Read images from a list of file paths.

    Args:
        image_list (list of str): List of image file paths.

    Returns:
        list of ndarray: List of images read from the file paths.
    """
    frames = []
    print('Reading images...')
    for img_path in tqdm(image_list, desc='Loading images'):
        frame = cv2.imread(img_path)
        if frame is not None:
            frames.append(frame)
        else:
            print(f"Warning: Could not read image {img_path}")
    return frames


def mirror_index(size, index):
    """
    Calculate a mirrored index for looping sequences in a forward-backward manner.

    Args:
        size (int): Size of the sequence.
        index (int): Current index.

    Returns:
        int: Mirrored index.
    """
    if size == 0:
        return 0
    turn = index // size
    res = index % size
    if turn % 2 == 0:
        return res
    else:
        return size - res - 1


def inference(render_event, batch_size, latents_out_path, audio_feat_queue, audio_out_queue, res_frame_queue):
    """
    Inference function running in a separate process to generate frames based on audio features.

    Args:
        render_event (mp.Event): Event to control rendering.
        batch_size (int): Batch size for processing.
        latents_out_path (str): Path to the latent representations.
        audio_feat_queue (mp.Queue): Queue to receive audio features.
        audio_out_queue (mp.Queue): Queue to receive audio frames.
        res_frame_queue (mp.Queue): Queue to send resulting frames.
    """
    # Load models
    vae, unet, pe = load_diffusion_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timesteps = torch.tensor([0], device=device)
    pe = pe.half()
    vae.vae = vae.vae.half()
    unet.model = unet.model.half()

    # Load latent representations
    input_latent_list_cycle = torch.load(latents_out_path)
    length = len(input_latent_list_cycle)
    index = 0
    count = 0
    counttime = 0
    print('Start inference process')

    while True:
        if render_event.is_set():
            try:
                whisper_chunks = audio_feat_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue

            is_all_silence = True
            audio_frames = []
            for _ in range(batch_size * 2):
                frame, frame_type = audio_out_queue.get()
                audio_frames.append((frame, frame_type))
                if frame_type == 0:
                    is_all_silence = False

            if is_all_silence:
                for i in range(batch_size):
                    res_frame_queue.put((None, mirror_index(length, index), audio_frames[i * 2:i * 2 + 2]))
                    index += 1
            else:
                # Perform inference
                t = time.perf_counter()
                whisper_batch = np.stack(whisper_chunks)
                latent_batch = []
                for i in range(batch_size):
                    idx = mirror_index(length, index + i)
                    latent = input_latent_list_cycle[idx]
                    latent_batch.append(latent)
                latent_batch = torch.cat(latent_batch, dim=0)

                audio_feature_batch = torch.from_numpy(whisper_batch).to(device=device, dtype=unet.model.dtype)
                audio_feature_batch = pe(audio_feature_batch)
                latent_batch = latent_batch.to(dtype=unet.model.dtype)

                pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample
                recon = vae.decode_latents(pred_latents)

                counttime += (time.perf_counter() - t)
                count += batch_size
                if count >= 100:
                    print(f"Actual average inference FPS: {count / counttime:.4f}")
                    count = 0
                    counttime = 0
                for i, res_frame in enumerate(recon):
                    res_frame_queue.put((res_frame, mirror_index(length, index), audio_frames[i * 2:i * 2 + 2]))
                    index += 1
        else:
            time.sleep(1)
    print('Inference process stopped')


@torch.no_grad()
class MuseReal:
    """
    MuseReal class handles the real-time rendering of talking avatars using MuseTalk.

    Attributes:
        opt: Options for configuration.
        W: Width of the video frames.
        H: Height of the video frames.
        fps: Frames per second.
        avatar_id: Identifier for the avatar.
        batch_size: Batch size for processing.
        res_frame_queue: Queue to receive resulting frames.
    """

    def __init__(self, opt):
        """
        Initialize the MuseReal object.

        Args:
            opt: Configuration options.
        """
        self.opt = opt
        self.W = opt.W
        self.H = opt.H
        self.fps = opt.fps

        # MuseTalk settings
        self.avatar_id = opt.avatar_id
        self.bbox_shift = opt.bbox_shift
        self.avatar_path = os.path.join("./data/avatars", self.avatar_id)
        self.full_imgs_path = os.path.join(self.avatar_path, "full_imgs")
        self.coords_path = os.path.join(self.avatar_path, "coords.pkl")
        self.latents_out_path = os.path.join(self.avatar_path, "latents.pt")
        self.mask_out_path = os.path.join(self.avatar_path, "mask")
        self.mask_coords_path = os.path.join(self.avatar_path, "mask_coords.pkl")
        self.avatar_info_path = os.path.join(self.avatar_path, "avatar_info.json")

        self.batch_size = opt.batch_size
        self.idx = 0
        self.res_frame_queue = mp.Queue(self.batch_size * 2)

        # Load models and avatar data
        self._load_models()
        self._load_avatar()

        # Initialize ASR and TTS
        self.asr = MuseASR(opt, self.audio_processor)
        self.asr.warm_up()
        if opt.tts == "edgetts":
            self.tts = EdgeTTS(opt, self)
        elif opt.tts == "gpt-sovits":
            self.tts = VoitsTTS(opt, self)
        elif opt.tts == "xtts":
            self.tts = XTTS(opt, self)
        else:
            raise ValueError(f"Unknown TTS option: {opt.tts}")

        # Start inference process
        self.render_event = mp.Event()
        self.inference_process = mp.Process(target=inference, args=(
            self.render_event,
            self.batch_size,
            self.latents_out_path,
            self.asr.feat_queue,
            self.asr.output_queue,
            self.res_frame_queue,
        ))
        self.inference_process.start()

    def _load_models(self):
        """
        Load necessary models for processing.
        """
        self.audio_processor = load_audio_model()

    def _load_avatar(self):
        """
        Load avatar images and related data.
        """
        # Load coordinate data
        with open(self.coords_path, 'rb') as f:
            self.coord_list_cycle = pickle.load(f)

        # Load full images
        input_img_list = glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]'))
        input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        self.frame_list_cycle = read_images(input_img_list)

        # Load mask coordinates
        with open(self.mask_coords_path, 'rb') as f:
            self.mask_coords_list_cycle = pickle.load(f)

        # Load mask images
        input_mask_list = glob.glob(os.path.join(self.mask_out_path, '*.[jpJP][pnPN]*[gG]'))
        input_mask_list = sorted(input_mask_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        self.mask_list_cycle = read_images(input_mask_list)

    def put_msg_txt(self, msg):
        """
        Put a message text into the TTS system.

        Args:
            msg (str): The message text.
        """
        self.tts.put_msg_txt(msg)

    def put_audio_frame(self, audio_chunk):
        """
        Put an audio frame into the ASR system.

        Args:
            audio_chunk (ndarray): Audio chunk data (16kHz, 20ms PCM).
        """
        self.asr.put_audio_frame(audio_chunk)

    def pause_talk(self):
        """
        Pause the TTS and ASR systems.
        """
        self.tts.pause_talk()
        self.asr.pause_talk()

    def _mirror_index(self, index):
        """
        Calculate the mirrored index for the avatar sequence.

        Args:
            index (int): Current index.

        Returns:
            int: Mirrored index.
        """
        size = len(self.coord_list_cycle)
        return mirror_index(size, index)

    def process_frames(self, quit_event, loop=None, audio_track=None, video_track=None):
        """
        Process frames and audio, and send them to the respective tracks.

        Args:
            quit_event (Event): Event to signal quitting.
            loop (asyncio.AbstractEventLoop): Event loop.
            audio_track: Audio track to send audio frames.
            video_track: Video track to send video frames.
        """
        while not quit_event.is_set():
            try:
                res_frame, idx, audio_frames = self.res_frame_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue

            if audio_frames[0][1] == 1 and audio_frames[1][1] == 1:
                # All frames are silent, use full image
                combined_frame = self.frame_list_cycle[idx]
            else:
                # Get bounding box and original frame
                bbox = self.coord_list_cycle[idx]
                ori_frame = copy.deepcopy(self.frame_list_cycle[idx])
                x1, y1, x2, y2 = bbox
                try:
                    res_frame = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
                except Exception as e:
                    print(f"Error resizing frame: {e}")
                    continue
                mask = self.mask_list_cycle[idx]
                mask_crop_box = self.mask_coords_list_cycle[idx]
                # Combine frames using blending
                combined_frame = get_image_blending(ori_frame, res_frame, bbox, mask, mask_crop_box)

            image = combined_frame
            new_frame = VideoFrame.from_ndarray(image, format="bgr24")
            asyncio.run_coroutine_threadsafe(video_track._queue.put(new_frame), loop)

            for audio_frame in audio_frames:
                frame_data, frame_type = audio_frame
                frame_data = (frame_data * 32767).astype(np.int16)
                new_audio_frame = AudioFrame(format='s16', layout='mono', samples=frame_data.shape[0])
                new_audio_frame.planes[0].update(frame_data.tobytes())
                new_audio_frame.sample_rate = 16000
                asyncio.run_coroutine_threadsafe(audio_track._queue.put(new_audio_frame), loop)
        print('MuseReal process_frames thread stopped')

    def render(self, quit_event, loop=None, audio_track=None, video_track=None):
        """
        Start rendering by processing ASR, TTS, and handling frame processing.

        Args:
            quit_event (Event): Event to signal quitting.
            loop (asyncio.AbstractEventLoop): Event loop.
            audio_track: Audio track to send audio frames.
            video_track: Video track to send video frames.
        """
        self.tts.render(quit_event)
        process_thread = Thread(target=self.process_frames, args=(quit_event, loop, audio_track, video_track))
        process_thread.start()

        self.render_event.set()  # Start inference process render
        print('MuseReal rendering started')
        while not quit_event.is_set():
            # Run ASR step
            self.asr.run_step()
            # Control the video track queue size to prevent overflow
            if video_track._queue.qsize() >= 1.5 * self.batch_size:
                sleep_time = 0.04 * video_track._queue.qsize() * 0.8
                print(f'Sleeping to control queue size: {video_track._queue.qsize()}')
                time.sleep(sleep_time)
        self.render_event.clear()  # End inference process render
        print('MuseReal render loop stopped')
