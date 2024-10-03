import time
import numpy as np
import queue
from queue import Queue
import multiprocessing as mp
from baseasr import BaseASR
from musetalk.whisper.audio2feature import Audio2Feature

class MuseASR(BaseASR):
    def __init__(self, opt, audio_processor:Audio2Feature):
        super().__init__(opt)
        self.audio_processor = audio_processor

    def extract_audio_features(self):
        inputs = np.concatenate(self.frames) # [N * chunk]
        whisper_feature = self.audio_processor.audio2feat(inputs)
        whisper_chunks = self.audio_processor.feature2chunks(feature_array=whisper_feature, fps=self.fps/2, batch_size=self.batch_size, start=self.stride_left_size/2)
        self.feat_queue.put(whisper_chunks)

    def run_step(self):
        for _ in range(self.batch_size*2):
            audio_frame, type = self.get_audio_frame()
            self.frames.append(audio_frame)
            self.output_queue.put((audio_frame, type))

        if len(self.frames) > self.stride_left_size + self.stride_right_size:
            self.extract_audio_features()
            # discard the old part to save memory
            self.frames = self.frames[-(self.stride_left_size + self.stride_right_size):]