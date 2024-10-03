import time
import queue
from threading import Thread, Event

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCTC, AutoProcessor, Wav2Vec2Processor, HubertModel

from baseasr import BaseASR


class ASR(BaseASR):
    """
    Automatic Speech Recognition (ASR) class that processes audio frames
    and extracts features using a specified ASR model.

    Inherits from BaseASR.
    """

    def __init__(self, opt):
        """
        Initialize the ASR model and set up the processing parameters.

        Args:
            opt: An object containing configuration options.
        """
        super().__init__(opt)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Determine audio dimension based on the model
        if 'esperanto' in self.opt.asr_model:
            self.audio_dim = 44
        elif 'deepspeech' in self.opt.asr_model:
            self.audio_dim = 29
        elif 'hubert' in self.opt.asr_model:
            self.audio_dim = 1024
        else:
            self.audio_dim = 32

        # Prepare context cache sizes
        self.context_size = opt.m  # Middle context size
        self.stride_left_size = opt.l  # Left stride size
        self.stride_right_size = opt.r  # Right stride size

        # Initialize frames buffer with zeros for left padding
        self.frames = []
        if self.stride_left_size > 0:
            zero_frame = np.zeros(self.chunk, dtype=np.float32)
            self.frames.extend([zero_frame] * self.stride_left_size)

        # Load the ASR model and processor
        print(f'[INFO] Loading ASR model {self.opt.asr_model}...')
        if 'hubert' in self.opt.asr_model:
            self.processor = Wav2Vec2Processor.from_pretrained(opt.asr_model)
            self.model = HubertModel.from_pretrained(opt.asr_model).to(self.device)
        else:
            self.processor = AutoProcessor.from_pretrained(opt.asr_model)
            self.model = AutoModelForCTC.from_pretrained(opt.asr_model).to(self.device)

        # Feature buffer for efficient recording of features
        self.feat_buffer_size = 4
        self.feat_buffer_idx = 0
        self.feat_queue = torch.zeros(
            self.feat_buffer_size * self.context_size,
            self.audio_dim,
            dtype=torch.float32,
            device=self.device
        )

        # Initialize pointers for feature buffer
        self.front = self.feat_buffer_size * self.context_size - 8  # Fake padding
        self.tail = 8

        # Attention features window
        self.att_feats = [torch.zeros(self.audio_dim, 16, device=self.device)] * 4  # Zero padding

        # Warm-up steps needed based on context and strides
        self.warm_up_steps = (
            self.context_size + self.stride_left_size + self.stride_right_size
        )

    def get_audio_frame(self):
        """
        Get the next audio frame from the queue.

        Returns:
            tuple: (frame, frame_type), where 'frame' is the audio frame array,
            and 'frame_type' indicates if it's a real frame (0) or a placeholder (1).
        """
        try:
            frame = self.queue.get(block=False)
            frame_type = 0  # Real frame
        except queue.Empty:
            frame = np.zeros(self.chunk, dtype=np.float32)
            frame_type = 1  # Placeholder frame

        return frame, frame_type

    def get_next_feat(self):
        """
        Get the next feature window for the ASR model.

        Returns:
            Tensor: A tensor representing the attention feature window.
        """
        if self.opt.att > 0:
            while len(self.att_feats) < 8:
                feat = self._extract_feat_segment()
                self.att_feats.append(feat.permute(1, 0))

            att_feat = torch.stack(self.att_feats, dim=0)  # Shape: [8, audio_dim, 16]
            self.att_feats = self.att_feats[1:]  # Discard the oldest feature
        else:
            feat = self._extract_feat_segment()
            att_feat = feat.permute(1, 0).unsqueeze(0)  # Shape: [1, audio_dim, time_steps]

        return att_feat

    def _extract_feat_segment(self):
        """
        Extract a segment of features from the feature queue.

        Returns:
            Tensor: A tensor containing the feature segment.
        """
        if self.front < self.tail:
            feat = self.feat_queue[self.front:self.tail]
        else:
            feat = torch.cat(
                [self.feat_queue[self.front:], self.feat_queue[:self.tail]],
                dim=0
            )

        # Update pointers
        self.front = (self.front + 2) % self.feat_queue.shape[0]
        self.tail = (self.tail + 2) % self.feat_queue.shape[0]

        return feat

    def run_step(self):
        """
        Process a single step by getting an audio frame, running ASR,
        and updating the feature queue.
        """
        frame, frame_type = self.get_audio_frame()
        self.frames.append(frame)
        self.output_queue.put((frame, frame_type))

        total_frames_needed = (
            self.stride_left_size + self.context_size + self.stride_right_size
        )

        # If not enough frames, return early
        if len(self.frames) < total_frames_needed:
            return

        # Concatenate frames to create input
        inputs = np.concatenate(self.frames)
        # Keep only the necessary frames
        self.frames = self.frames[-(self.stride_left_size + self.stride_right_size):]

        # Process the input frame through the ASR model
        logits = self._frame_to_logits(inputs)

        # Update feature queue efficiently
        start = self.feat_buffer_idx * self.context_size
        end = start + logits.shape[0]
        self.feat_queue[start:end] = logits
        self.feat_buffer_idx = (self.feat_buffer_idx + 1) % self.feat_buffer_size

    def _frame_to_logits(self, frame):
        """
        Convert an audio frame to model logits.

        Args:
            frame (ndarray): The audio frame data.

        Returns:
            Tensor: The model logits.
        """
        # Process the frame with the ASR processor
        inputs = self.processor(
            frame,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True
        )

        with torch.no_grad():
            # Run the model to get logits
            result = self.model(inputs.input_values.to(self.device))
            if 'hubert' in self.opt.asr_model:
                logits = result.last_hidden_state  # [Batch, Time, Hidden]
            else:
                logits = result.logits  # [Batch, Time, Hidden]

        # Extract the relevant part of logits, excluding strides
        left = max(0, self.stride_left_size)
        right = logits.shape[1] - self.stride_right_size + 1
        logits = logits[:, left:right]

        return logits[0]  # Return logits for the first (only) batch

    def warm_up(self):
        """
        Warm up the ASR model by running several initial steps.
        """
        print(
            f'[INFO] Warming up ASR model, expected latency = '
            f'{self.warm_up_steps / self.fps:.6f}s'
        )
        start_time = time.time()
        for _ in range(self.warm_up_steps):
            self.run_step()
        elapsed_time = time.time() - start_time
        print(f'[INFO] Warm-up done, actual latency = {elapsed_time:.6f}s')

    def run(self):
        """
        Run the ASR processing loop.
        """
        self.warm_up()
        while not self.terminated:
            self.run_step()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="ASR Feature Extraction")
    parser.add_argument('--wav', type=str, default='', help='Path to input WAV file')
    parser.add_argument('--play', action='store_true', help='Play out the audio')
    parser.add_argument('--model', type=str, default='facebook/hubert-large-ls960-ft', help='ASR model to use')
    parser.add_argument('--save_feats', action='store_true', help='Save features to file')
    parser.add_argument('--fps', type=int, default=50, help='Frames per second for audio processing')
    parser.add_argument('-l', type=int, default=10, help='Left context size (in units of 20ms)')
    parser.add_argument('-m', type=int, default=50, help='Middle context size (in units of 20ms)')
    parser.add_argument('-r', type=int, default=10, help='Right context size (in units of 20ms)')
    opt = parser.parse_args()

    # Adjust options for ASR
    opt.asr_wav = opt.wav
    opt.asr_play = opt.play
    opt.asr_model = opt.model
    opt.asr_save_feats = opt.save_feats

    if 'deepspeech' in opt.asr_model:
        raise ValueError("DeepSpeech features should not use this code to extract.")

    # Initialize and run ASR
    asr = ASR(opt)
    asr.run()
