import os
import time
import asyncio
import cv2
import numpy as np
import torch
from av import AudioFrame, VideoFrame

from asrreal import ASR
from ttsreal import EdgeTTS, VoitsTTS, XTTS


class NeRFReal:
    """
    NeRFReal handles real-time rendering using NeRF models,
    integrating ASR and TTS functionalities.
    """

    def __init__(self, opt, trainer, data_loader, debug=True):
        """
        Initialize the NeRFReal object.

        Args:
            opt: Configuration options.
            trainer: Trainer object containing the NeRF model.
            data_loader: DataLoader providing input data.
            debug (bool): Whether to enable debug mode.
        """
        self.opt = opt
        self.W = opt.W
        self.H = opt.H
        self.trainer = trainer
        self.data_loader = data_loader
        self.loader = iter(data_loader)
        self.customimg_index = 0

        # Initialize ASR
        self.asr = ASR(opt)
        self.asr.warm_up()

        # Initialize TTS
        if opt.tts == "edgetts":
            self.tts = EdgeTTS(opt, self)
        elif opt.tts == "gpt-sovits":
            self.tts = VoitsTTS(opt, self)
        elif opt.tts == "xtts":
            self.tts = XTTS(opt, self)
        else:
            raise ValueError(f"Unsupported TTS option: {opt.tts}")

        # Initialize RTMP streaming if needed
        if self.opt.transport == 'rtmp':
            from rtmp_streaming import StreamerConfig, Streamer
            sc = StreamerConfig()
            sc.source_width = self.W
            sc.source_height = self.H
            sc.stream_width = self.W
            sc.stream_height = self.H
            if self.opt.fullbody:
                sc.source_width = self.opt.fullbody_width
                sc.source_height = self.opt.fullbody_height
                sc.stream_width = self.opt.fullbody_width
                sc.stream_height = self.opt.fullbody_height
            sc.stream_fps = 25
            sc.stream_bitrate = 1000000
            sc.stream_profile = 'baseline'
            sc.audio_channel = 1
            sc.sample_rate = 16000
            sc.stream_server = self.opt.push_url
            self.streamer = Streamer()
            self.streamer.init(sc)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.opt.asr:
            self.asr.stop()

    def put_msg_txt(self, msg):
        """
        Send a text message to the TTS system.

        Args:
            msg (str): The message text.
        """
        self.tts.put_msg_txt(msg)

    def put_audio_frame(self, audio_chunk):
        """
        Provide an audio frame to the ASR system.

        Args:
            audio_chunk (np.ndarray): Audio data chunk (16kHz, 20ms PCM).
        """
        self.asr.put_audio_frame(audio_chunk)

    def pause_talk(self):
        """
        Pause both TTS and ASR systems.
        """
        self.tts.pause_talk()
        self.asr.pause_talk()

    def mirror_index(self, index):
        """
        Calculate the mirrored index for looping sequences in a forward-backward manner.

        Args:
            index (int): Current index.

        Returns:
            int: Mirrored index.
        """
        size = self.opt.customvideo_imgnum
        if size == 0:
            return 0
        turn = index // size
        res = index % size
        if turn % 2 == 0:
            return res
        else:
            return size - res - 1

    def test_step(self, loop=None, audio_track=None, video_track=None):
        """
        Perform a test step, processing data and outputting frames.

        Args:
            loop (asyncio.AbstractEventLoop): The event loop.
            audio_track: Audio track to send audio frames.
            video_track: Video track to send video frames.
        """
        try:
            data = next(self.loader)
        except StopIteration:
            self.loader = iter(self.data_loader)
            data = next(self.loader)

        if self.opt.asr:
            # Use the live audio stream
            data['auds'] = self.asr.get_next_feat()

        audiotype = 0
        for _ in range(2):
            frame, frame_type = self.asr.get_audio_out()
            audiotype += frame_type
            if self.opt.transport == 'rtmp':
                self.streamer.stream_frame_audio(frame)
            else:
                frame = (frame * 32767).astype(np.int16)
                new_frame = AudioFrame(format='s16', layout='mono', samples=frame.shape[0])
                new_frame.planes[0].update(frame.tobytes())
                new_frame.sample_rate = 16000
                asyncio.run_coroutine_threadsafe(audio_track._queue.put(new_frame), loop)

        if self.opt.customvideo and audiotype != 0:
            # Use custom video frames
            self.loader = iter(self.data_loader)  # Reset loader
            img_index = self.mirror_index(self.customimg_index)
            img_path = os.path.join(self.opt.customvideo_img, f"{int(img_index)}.png")
            image = cv2.imread(img_path)
            if image is None:
                print(f"Warning: Could not read image {img_path}")
                return
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if self.opt.transport == 'rtmp':
                self.streamer.stream_frame(image)
            else:
                new_frame = VideoFrame.from_ndarray(image, format="rgb24")
                asyncio.run_coroutine_threadsafe(video_track._queue.put(new_frame), loop)
            self.customimg_index += 1
        else:
            self.customimg_index = 0
            outputs = self.trainer.test_gui_with_data(data, self.W, self.H)
            image = (outputs['image'] * 255).astype(np.uint8)
            if not self.opt.fullbody:
                if self.opt.transport == 'rtmp':
                    self.streamer.stream_frame(image)
                else:
                    new_frame = VideoFrame.from_ndarray(image, format="rgb24")
                    asyncio.run_coroutine_threadsafe(video_track._queue.put(new_frame), loop)
            else:
                # Handle full body rendering
                frame_index = data['index'][0]
                fullbody_img_path = os.path.join(self.opt.fullbody_img, f"{frame_index}.jpg")
                image_fullbody = cv2.imread(fullbody_img_path)
                if image_fullbody is None:
                    print(f"Warning: Could not read full body image {fullbody_img_path}")
                    return
                image_fullbody = cv2.cvtColor(image_fullbody, cv2.COLOR_BGR2RGB)
                start_x = self.opt.fullbody_offset_x
                start_y = self.opt.fullbody_offset_y
                image_fullbody[start_y:start_y+image.shape[0], start_x:start_x+image.shape[1]] = image
                if self.opt.transport == 'rtmp':
                    self.streamer.stream_frame(image_fullbody)
                else:
                    new_frame = VideoFrame.from_ndarray(image_fullbody, format="rgb24")
                    asyncio.run_coroutine_threadsafe(video_track._queue.put(new_frame), loop)

    def render(self, quit_event, loop=None, audio_track=None, video_track=None):
        """
        Start the rendering loop.

        Args:
            quit_event (threading.Event): Event to signal quitting.
            loop (asyncio.AbstractEventLoop): The event loop.
            audio_track: Audio track to send audio frames.
            video_track: Video track to send video frames.
        """
        self.tts.render(quit_event)
        count = 0
        total_time = 0
        start_time = time.perf_counter()
        total_frames = 0

        while not quit_event.is_set():
            t_start = time.perf_counter()
            for _ in range(2):
                self.asr.run_step()
            self.test_step(loop, audio_track, video_track)
            elapsed = time.perf_counter() - t_start
            total_time += elapsed
            count += 1
            total_frames += 1

            if count == 100:
                avg_fps = count / total_time
                print(f"Actual average inference FPS: {avg_fps:.4f}")
                count = 0
                total_time = 0

            if self.opt.transport == 'rtmp':
                delay = start_time + total_frames * 0.04 - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
            else:
                if video_track._queue.qsize() >= 5:
                    time.sleep(0.04 * video_track._queue.qsize() * 0.8)

        print('NeRFReal render loop stopped')
