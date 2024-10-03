import os
import time
import math
import copy
import glob
import pickle
import asyncio
import multiprocessing as mp

import numpy as np
import cv2
import torch
from tqdm import tqdm
from av import AudioFrame, VideoFrame

from wav2lip.models import Wav2Lip
from basereal import BaseReal
from ttsreal import EdgeTTS, VoitsTTS, XTTS
from lipasr import LipASR

# Set the device for inference
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} for inference.')


def _load(checkpoint_path):
    """
    Load a checkpoint from the given path, handling CPU and CUDA devices.

    Args:
        checkpoint_path (str): Path to the checkpoint file.

    Returns:
        dict: Loaded checkpoint.
    """
    if device == 'cuda':
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    return checkpoint


def load_model(checkpoint_path):
    """
    Load the Wav2Lip model from a checkpoint.

    Args:
        checkpoint_path (str): Path to the Wav2Lip checkpoint file.

    Returns:
        torch.nn.Module: The Wav2Lip model in evaluation mode.
    """
    model = Wav2Lip()
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = _load(checkpoint_path)
    state_dict = checkpoint["state_dict"]
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('module.', '')  # Remove 'module.' prefix if present
        new_state_dict[new_key] = v
    model.load_state_dict(new_state_dict)
    model = model.to(device)
    return model.eval()


def read_imgs(img_list):
    """
    Read images from a list of file paths.

    Args:
        img_list (list): List of image file paths.

    Returns:
        list: List of image arrays.
    """
    frames = []
    print('Reading images...')
    for img_path in tqdm(img_list, desc='Loading images'):
        frame = cv2.imread(img_path)
        if frame is not None:
            frames.append(frame)
        else:
            print(f"Warning: Could not read image {img_path}")
    return frames


def _mirror_index(size, index):
    """
    Calculate a mirrored index for looping back and forth through a sequence.

    Args:
        size (int): The size of the sequence.
        index (int): The current index.

    Returns:
        int: The mirrored index.
    """
    if size == 0:
        return 0
    turn = index // size
    res = index % size
    if turn % 2 == 0:
        return res
    else:
        return size - res - 1


class InferenceWorker:
    """
    InferenceWorker runs the Wav2Lip model inference in a separate process.

    It loads the model, receives audio features from a queue, processes them,
    and outputs the synthesized frames to another queue.
    """

    def __init__(self, render_event, batch_size, face_imgs_path, audio_feat_queue, audio_out_queue):
        """
        Initialize the InferenceWorker.

        Args:
            render_event (multiprocessing.Event): Event to control rendering.
            batch_size (int): Batch size for processing.
            face_imgs_path (str): Path to the directory containing face images.
            audio_feat_queue (multiprocessing.Queue): Queue to receive audio features.
            audio_out_queue (multiprocessing.Queue): Queue to receive audio frames.
        """
        self.render_event = render_event
        self.batch_size = batch_size
        self.face_imgs_path = face_imgs_path
        self.audio_feat_queue = audio_feat_queue
        self.audio_out_queue = audio_out_queue
        self.res_frame_queue = mp.Queue(self.batch_size * 2)
        self.process = mp.Process(target=self.run)
        self.process.start()

    def run(self):
        """
        The main loop that runs in a separate process.

        It loads the Wav2Lip model, processes audio features, and generates frames.
        """
        model = load_model("./models/wav2lip.pth")
        input_face_list = sorted(
            glob.glob(os.path.join(self.face_imgs_path, '*.[jpJP][pnPN]*[gG]')),
            key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
        )
        face_list_cycle = read_imgs(input_face_list)
        length = len(face_list_cycle)
        index = 0
        count = 0
        counttime = 0
        print('Inference worker started')
        while True:
            if self.render_event.is_set():
                start_time = time.perf_counter()
                try:
                    mel_batch = self.audio_feat_queue.get(timeout=1)
                except queue.Empty:
                    continue

                is_all_silence = True
                audio_frames = []
                for _ in range(self.batch_size * 2):
                    frame, frame_type = self.audio_out_queue.get()
                    audio_frames.append((frame, frame_type))
                    if frame_type == 0:
                        is_all_silence = False

                if is_all_silence:
                    # If all frames are silent, just output the frames without processing
                    for i in range(self.batch_size):
                        self.res_frame_queue.put((None, _mirror_index(length, index), audio_frames[i * 2:i * 2 + 2]))
                        index += 1
                else:
                    t = time.perf_counter()
                    img_batch = []
                    for i in range(self.batch_size):
                        idx = _mirror_index(length, index + i)
                        face = face_list_cycle[idx]
                        img_batch.append(face)
                    img_batch = np.asarray(img_batch)
                    mel_batch = np.asarray(mel_batch)

                    img_masked = img_batch.copy()
                    # Mask the lower half of the face
                    img_masked[:, face.shape[0] // 2:, :] = 0

                    # Prepare input for the model
                    img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.0
                    mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

                    img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
                    mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

                    with torch.no_grad():
                        pred = model(mel_batch, img_batch)
                    pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.0

                    counttime += (time.perf_counter() - t)
                    count += self.batch_size
                    if count >= 100:
                        avg_fps = count / counttime
                        print(f"Actual average inference FPS: {avg_fps:.4f}")
                        count = 0
                        counttime = 0
                    for i, res_frame in enumerate(pred):
                        self.res_frame_queue.put((res_frame, _mirror_index(length, index), audio_frames[i * 2:i * 2 + 2]))
                        index += 1
            else:
                time.sleep(1)
        print('Inference worker stopped')

    def get_result(self):
        """
        Get the result frame queue.

        Returns:
            multiprocessing.Queue: Queue containing the result frames.
        """
        return self.res_frame_queue

    def stop(self):
        """
        Stop the inference worker process.
        """
        self.process.terminate()
        self.process.join()


class LipReal(BaseReal):
    """
    LipReal handles the real-time lip-sync rendering using Wav2Lip and audio processing.

    It initializes the ASR, TTS, and runs the rendering loop.
    """

    def __init__(self, opt):
        """
        Initialize the LipReal object.

        Args:
            opt: Configuration options.
        """
        super().__init__(opt)
        self.W = opt.W
        self.H = opt.H
        self.fps = opt.fps  # Frames per second

        self.avatar_id = opt.avatar_id
        self.avatar_path = os.path.join("./data/avatars", self.avatar_id)
        self.full_imgs_path = os.path.join(self.avatar_path, "full_imgs")
        self.face_imgs_path = os.path.join(self.avatar_path, "face_imgs")
        self.coords_path = os.path.join(self.avatar_path, "coords.pkl")
        self.batch_size = opt.batch_size
        self.idx = 0

        self.res_frame_queue = mp.Queue(self.batch_size * 2)
        self._load_avatar()

        self.asr = LipASR(opt, self)
        self.asr.warm_up()

        # Initialize Text-to-Speech system
        if opt.tts == "edgetts":
            self.tts = EdgeTTS(opt, self)
        elif opt.tts == "gpt-sovits":
            self.tts = VoitsTTS(opt, self)
        elif opt.tts == "xtts":
            self.tts = XTTS(opt, self)
        else:
            raise ValueError(f"Unknown TTS option: {opt.tts}")

        self.render_event = mp.Event()
        self.inference_worker = InferenceWorker(
            self.render_event,
            self.batch_size,
            self.face_imgs_path,
            self.asr.feat_queue,
            self.asr.output_queue
        )

    def _load_avatar(self):
        """
        Load the avatar images and coordinate data.
        """
        with open(self.coords_path, 'rb') as f:
            self.coord_list_cycle = pickle.load(f)
        input_img_list = sorted(
            glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]')),
            key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
        )
        self.frame_list_cycle = read_imgs(input_img_list)

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
            audio_chunk (ndarray): The audio data chunk.
        """
        self.asr.put_audio_frame(audio_chunk)

    def pause_talk(self):
        """
        Pause the TTS and ASR systems.
        """
        self.tts.pause_talk()
        self.asr.pause_talk()

    async def process_frames(self, quit_event, loop=None, audio_track=None, video_track=None):
        """
        Coroutine to process frames asynchronously.

        Args:
            quit_event (asyncio.Event): Event to signal quitting.
            loop (asyncio.AbstractEventLoop): The event loop.
            audio_track: The audio track to send frames to.
            video_track: The video track to send frames to.
        """
        while not quit_event.is_set():
            try:
                res_frame, idx, audio_frames = await asyncio.get_event_loop().run_in_executor(
                    None, self.inference_worker.get_result().get, True, 1
                )
            except queue.Empty:
                continue

            if all(af[1] != 0 for af in audio_frames):  # All frames are silent
                combine_frame = self.frame_list_cycle[idx]
            else:
                bbox = self.coord_list_cycle[idx]
                combine_frame = copy.deepcopy(self.frame_list_cycle[idx])
                y1, y2, x1, x2 = bbox
                try:
                    res_frame = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
                except Exception as e:
                    print(f"Error resizing frame: {e}")
                    continue
                combine_frame[y1:y2, x1:x2] = res_frame

            image = combine_frame
            new_frame = VideoFrame.from_ndarray(image, format="bgr24")
            await video_track._queue.put(new_frame)

            for audio_frame in audio_frames:
                frame_data, frame_type = audio_frame
                frame_data = (frame_data * 32767).astype(np.int16)
                new_audio_frame = AudioFrame(format='s16', layout='mono', samples=frame_data.shape[0])
                new_audio_frame.planes[0].update(frame_data.tobytes())
                new_audio_frame.sample_rate = 16000
                await audio_track._queue.put(new_audio_frame)
        print('LipReal process_frames coroutine stopped')

    def render(self, quit_event, loop=None, audio_track=None, video_track=None):
        """
        Start the rendering loop.

        Args:
            quit_event (asyncio.Event): Event to signal quitting.
            loop (asyncio.AbstractEventLoop): The event loop.
            audio_track: The audio track to send frames to.
            video_track: The video track to send frames to.
        """
        self.tts.render(quit_event)
        asyncio.ensure_future(self.process_frames(quit_event, loop, audio_track, video_track))

        self.render_event.set()  # Start inference worker rendering
        print('LipReal rendering started')
        while not quit_event.is_set():
            self.asr.run_step()
            # Control the rate to prevent queue overflow
            if video_track._queue.qsize() >= 5:
                sleep_time = 0.04 * video_track._queue.qsize() * 0.8
                time.sleep(sleep_time)
        self.render_event.clear()  # Stop inference worker rendering
        print('LipReal render loop stopped')
