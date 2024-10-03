import argparse
import logging
import os
import sys
import time
from pathlib import Path
from queue import Queue
from threading import Thread, Event
from typing import Optional

import numpy as np
import pyaudio
import resampy
import soundfile as sf
import torch
import torch.nn.functional as F
from transformers import AutoModelForCTC, AutoProcessor


def _read_frame(stream: pyaudio.Stream, exit_event: Event, queue: Queue, chunk: int) -> None:
    """Continuously read audio frames from the input stream and enqueue them."""
    while not exit_event.is_set():
        try:
            frame = stream.read(chunk, exception_on_overflow=False)
            frame = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32767  # Normalize to [-1, 1]
            queue.put(frame)
        except Exception as e:
            logging.error(f"Error reading frame: {e}")
            break
    logging.info('[INFO] Read frame thread ends')


def _play_frame(stream: pyaudio.Stream, exit_event: Event, queue: Queue, chunk: int) -> None:
    """Continuously play audio frames from the output queue."""
    while not exit_event.is_set():
        try:
            frame = queue.get(timeout=0.1)
            frame = (frame * 32767).astype(np.int16).tobytes()
            stream.write(frame, chunk)
        except Queue.Empty:
            continue
        except Exception as e:
            logging.error(f"Error playing frame: {e}")
            break
    logging.info('[INFO] Play frame thread ends')


class ASR:
    """Automatic Speech Recognition class using HuggingFace's Transformers."""

    def __init__(self, opt: argparse.Namespace):
        self.opt = opt
        self.play_audio = opt.asr_play
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.fps = opt.fps  # Frames per second (e.g., 50 for 20ms per frame)
        self.sample_rate = 16000
        self.chunk = self.sample_rate // self.fps  # Samples per chunk (e.g., 320 for 20ms)
        self.mode = 'live' if not opt.wav else 'file'

        # Determine audio dimension based on the model
        if 'esperanto' in self.opt.asr_model:
            self.audio_dim = 44
        elif 'deepspeech' in self.opt.asr_model:
            self.audio_dim = 29
        else:
            self.audio_dim = 32

        # Context and stride sizes for feature extraction
        self.context_size = opt.m
        self.stride_left_size = opt.l
        self.stride_right_size = opt.r
        self.text = '[START]\n'
        self.terminated = False
        self.frames: list = []

        # Initialize frame buffer with padding if necessary
        if self.stride_left_size > 0:
            self.frames.extend([np.zeros(self.chunk, dtype=np.float32)] * self.stride_left_size)

        self.exit_event = Event()
        self.audio_instance = pyaudio.PyAudio()

        # Initialize audio streams and threading
        if self.mode == 'file':
            self.file_stream, self.file_sample_rate = self.create_file_stream()
            if self.file_sample_rate != self.sample_rate:
                logging.warning(f"Audio sample rate {self.file_sample_rate} does not match target {self.sample_rate}. Resampling.")
                self.file_stream = resampy.resample(self.file_stream, self.file_sample_rate, self.sample_rate)
        else:
            self.input_stream = self.audio_instance.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk
            )
            self.queue = Queue()
            self.read_thread = Thread(target=_read_frame, args=(self.input_stream, self.exit_event, self.queue, self.chunk))
            self.read_thread.start()

        if self.play_audio:
            self.output_stream = self.audio_instance.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                output=True,
                frames_per_buffer=self.chunk
            )
            self.output_queue = Queue()
            self.play_thread = Thread(target=_play_frame, args=(self.output_stream, self.exit_event, self.output_queue, self.chunk))
            self.play_thread.start()

        self.idx = 0  # Current audio frame index

        # Load ASR model and processor
        logging.info(f'[INFO] Loading ASR model {self.opt.asr_model}...')
        self.processor = AutoProcessor.from_pretrained(self.opt.asr_model)
        self.model = AutoModelForCTC.from_pretrained(self.opt.asr_model).to(self.device)
        self.model.eval()

        # Initialize feature saving
        if self.opt.asr_save_feats:
            self.all_feats: list = []

        # Initialize feature buffer for efficient processing
        self.feat_buffer_size = 4
        self.feat_buffer_idx = 0
        self.feat_queue = torch.zeros(self.feat_buffer_size * self.context_size, self.audio_dim, dtype=torch.float32, device=self.device)

        # Parameters for attention window
        self.front = self.feat_buffer_size * self.context_size - 8  # Fake padding
        self.tail = 8
        self.att_feats = [torch.zeros(self.audio_dim, 16, dtype=torch.float32, device=self.device) for _ in range(4)]  # Initial padding

        # Warm-up steps
        self.warm_up_steps = self.context_size + self.stride_right_size + 8 + 6  # Adjusted for clarity

        self.listening = False
        self.playing = False

    def listen(self) -> None:
        """Start listening to audio input and playing audio output if enabled."""
        if self.mode == 'live' and not self.listening:
            logging.info('[INFO] Starting read frame thread...')
            self.listening = True

        if self.play_audio and not self.playing:
            logging.info('[INFO] Starting play frame thread...')
            self.playing = True

    def stop(self) -> None:
        """Stop all audio streams and threads."""
        self.exit_event.set()

        if self.play_audio and hasattr(self, 'output_stream'):
            self.output_stream.stop_stream()
            self.output_stream.close()
            if self.playing:
                self.play_thread.join()
                self.playing = False

        if self.mode == 'live' and hasattr(self, 'input_stream'):
            self.input_stream.stop_stream()
            self.input_stream.close()
            if self.listening:
                self.read_thread.join()
                self.listening = False

        self.audio_instance.terminate()
        logging.info('[INFO] ASR stopped.')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()
        if self.mode == 'file':
            self.text += '\n[END]'
            logging.info(self.text)

    def get_next_feat(self) -> torch.Tensor:
        """Retrieve the next set of features for model input."""
        while len(self.att_feats) < 8:
            if self.front < self.tail:
                feat = self.feat_queue[self.front:self.tail]
            else:
                feat = torch.cat((self.feat_queue[self.front:], self.feat_queue[:self.tail]), dim=0)

            self.front = (self.front + 2) % self.feat_queue.shape[0]
            self.tail = (self.tail + 2) % self.feat_queue.shape[0]

            self.att_feats.append(feat.permute(1, 0))

        att_feat = torch.stack(self.att_feats, dim=0)  # [8, audio_dim, 16]
        self.att_feats.pop(0)  # Remove the oldest feature

        return att_feat

    def run_step(self) -> None:
        """Process a single step of audio frame extraction and transcription."""
        if self.terminated:
            return

        frame = self.get_audio_frame()

        if frame is None:
            self.terminated = True
        else:
            self.frames.append(frame)
            if self.play_audio:
                self.output_queue.put(frame)

            if len(self.frames) < (self.stride_left_size + self.context_size + self.stride_right_size):
                return

        inputs = np.concatenate(self.frames)

        if not self.terminated:
            self.frames = self.frames[-(self.stride_left_size + self.stride_right_size):]

        logits, _, text = self.frame_to_text(inputs)
        feats = logits

        if self.opt.asr_save_feats:
            self.all_feats.append(feats.cpu())

        start = self.feat_buffer_idx * self.context_size
        end = start + feats.shape[0]
        self.feat_queue[start:end] = feats
        self.feat_buffer_idx = (self.feat_buffer_idx + 1) % self.feat_buffer_size

        if text:
            self.text += f' {text}'

        if self.terminated:
            self.text += '\n[END]'
            logging.info(self.text)
            if self.opt.asr_save_feats:
                self.save_features()

    def create_file_stream(self) -> (np.ndarray, int):
        """Load audio from a file and return the audio stream and sample rate."""
        try:
            stream, sample_rate = sf.read(self.opt.wav)
            stream = stream.astype(np.float32)

            if stream.ndim > 1:
                logging.warning(f'[WARN] Audio has {stream.shape[1]} channels; using the first channel.')
                stream = stream[:, 0]

            if sample_rate != self.sample_rate:
                logging.warning(f'[WARN] Resampling from {sample_rate} to {self.sample_rate} Hz.')
                stream = resampy.resample(stream, sample_rate, self.sample_rate)

            logging.info(f'[INFO] Loaded audio stream from {self.opt.wav}: {stream.shape}')
            return stream, self.sample_rate
        except Exception as e:
            logging.error(f"Error loading audio file: {e}")
            sys.exit(1)

    def get_audio_frame(self) -> Optional[np.ndarray]:
        """Retrieve the next audio frame from the queue or file."""
        if self.mode == 'file':
            if self.idx < len(self.file_stream):
                frame = self.file_stream[self.idx: self.idx + self.chunk]
                self.idx += self.chunk
                return frame
            else:
                return None
        else:
            try:
                frame = self.queue.get(timeout=1)
                self.idx += self.chunk
                return frame
            except Queue.Empty:
                return None

    def frame_to_text(self, frame: np.ndarray) -> (torch.Tensor, torch.Tensor, str):
        """Convert an audio frame to text using the ASR model."""
        inputs = self.processor(frame, sampling_rate=self.sample_rate, return_tensors="pt", padding=True).to(self.device)

        with torch.no_grad():
            result = self.model(inputs.input_values)
            logits = result.logits  # [batch_size, sequence_length, num_labels]

        # Adjust logits based on stride
        left = max(0, self.stride_left_size)
        right = logits.shape[1] - self.stride_right_size if not self.terminated else logits.shape[1]
        logits = logits[:, left:right, :]

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0].lower()

        return logits.squeeze(0), predicted_ids.squeeze(0), transcription

    def run(self) -> None:
        """Run the ASR pipeline."""
        self.listen()
        try:
            while not self.terminated:
                self.run_step()
        except KeyboardInterrupt:
            logging.info('[INFO] Interrupted by user.')
        finally:
            self.stop()

    def save_features(self) -> None:
        """Save the extracted features to a NumPy file."""
        try:
            feats = torch.cat(self.all_feats, dim=0)  # [N, C]
            window_size = 16
            padding = window_size // 2
            feats = feats.view(-1, self.audio_dim).permute(1, 0).contiguous()  # [C, M]
            feats = feats.unsqueeze(0).unsqueeze(-1)  # [1, C, M, 1]
            unfold_feats = F.unfold(feats, kernel_size=(window_size, 1), padding=(padding, 0), stride=(2, 1))  # [1, C * window_size, L]
            unfold_feats = unfold_feats.view(self.audio_dim, window_size, -1).permute(2, 1, 0).contiguous()  # [L, window_size, C]

            output_path = Path(self.opt.wav).with_suffix('.npy')
            if 'esperanto' in self.opt.asr_model:
                output_path = Path(self.opt.wav).with_suffix('_eo.npy')

            np.save(output_path, unfold_feats.cpu().numpy())
            logging.info(f"[INFO] Saved logits to {output_path}")
        except Exception as e:
            logging.error(f"Error saving features: {e}")

    def clear_queue(self) -> None:
        """Clear the input and output queues to reduce latency."""
        logging.info('[INFO] Clearing queues...')
        if self.mode == 'live' and hasattr(self, 'queue'):
            with self.queue.mutex:
                self.queue.queue.clear()
        if self.play_audio and hasattr(self, 'output_queue'):
            with self.output_queue.mutex:
                self.output_queue.queue.clear()

    def warm_up(self) -> None:
        """Warm up the ASR model to stabilize latency."""
        self.listen()
        logging.info(f'[INFO] Warming up ASR model, expected latency = {self.warm_up_steps / self.fps:.6f}s')
        start_time = time.time()
        for _ in range(self.warm_up_steps):
            self.run_step()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_time = time.time() - start_time
        logging.info(f'[INFO] Warm-up done, actual latency = {elapsed_time:.6f}s')
        self.clear_queue()


def setup_logging(log_level: str = 'INFO') -> None:
    """Configure logging for the ASR application."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("asr.log", mode='w')
        ]
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="ASR (Automatic Speech Recognition) Pipeline")
    parser.add_argument('--wav', type=str, default='', help='Path to the input WAV file. Leave empty for live mode.')
    parser.add_argument('--play', action='store_true', help="Play out the audio.")
    parser.add_argument('--model', type=str, default='facebook/wav2vec2-large-960h', help='ASR model name or path.')
    parser.add_argument('--save_feats', action='store_true', help='Save extracted features to a file.')
    parser.add_argument('--fps', type=int, default=50, help='Audio frames per second (e.g., 50 for 20ms frames).')
    parser.add_argument('-l', type=int, default=10, help='Left stride size for context.')
    parser.add_argument('-m', type=int, default=50, help='Context size.')
    parser.add_argument('-r', type=int, default=10, help='Right stride size for context.')
    parser.add_argument('--log_level', type=str, default='INFO', help='Logging level (DEBUG, INFO, WARNING, ERROR).')
    return parser.parse_args()


def main() -> None:
    """Main function to execute the ASR pipeline."""
    opt = parse_arguments()
    setup_logging(opt.log_level)

    # Validate ASR model
    if 'deepspeech' in opt.model:
        logging.error("DeepSpeech features should not use this code to extract.")
        sys.exit(1)

    if opt.wav and not Path(opt.wav).is_file():
        logging.error(f"Audio file {opt.wav} does not exist.")
        sys.exit(1)

    with ASR(opt) as asr:
        if opt.save_feats:
            asr.warm_up()
        asr.run()


if __name__ == '__main__':
    main()
