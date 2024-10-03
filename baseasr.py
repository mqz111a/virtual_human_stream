import time
import numpy as np
import queue
from multiprocessing import Queue
import multiprocessing as mp


class BaseASR:
    """
    Base class for Automatic Speech Recognition (ASR) processing.

    This class handles audio frame queuing, feature extraction, and buffering
    for ASR models.

    Attributes:
        opt: Configuration options.
        parent: Optional parent object.
        fps (int): Frames per second for audio processing.
        sample_rate (int): Audio sample rate in Hz.
        chunk (int): Number of samples per audio chunk.
        queue (Queue): Input audio frame queue.
        output_queue (Queue): Output audio frame queue.
        batch_size (int): Batch size for processing.
        frames (list): List of audio frames.
        stride_left_size (int): Size of the left stride (in frames).
        stride_right_size (int): Size of the right stride (in frames).
        feat_queue (Queue): Queue for features.
    """

    def __init__(self, opt, parent=None):
        """
        Initialize the BaseASR object.

        Args:
            opt: An object containing configuration options.
            parent: Optional parent object.
        """
        self.opt = opt
        self.parent = parent

        self.fps = opt.fps  # Frames per second
        self.sample_rate = 16000  # Audio sample rate in Hz
        self.chunk = self.sample_rate // self.fps  # Samples per chunk (e.g., 320 samples for 20ms at 16kHz)
        self.queue = Queue()
        self.output_queue = mp.Queue()

        self.batch_size = opt.batch_size

        self.frames = []
        self.stride_left_size = opt.l
        self.stride_right_size = opt.r
        self.feat_queue = mp.Queue(maxsize=2)

    def pause_talk(self):
        """
        Clear the input audio queue to pause processing.
        """
        self.queue.queue.clear()

    def put_audio_frame(self, audio_chunk):
        """
        Put an audio chunk into the input queue.

        Args:
            audio_chunk (np.ndarray): Audio data chunk (e.g., 16kHz, 20ms PCM).
        """
        self.queue.put(audio_chunk)

    def get_audio_frame(self):
        """
        Retrieve the next audio frame from the input queue.

        Returns:
            tuple: (frame, frame_type), where 'frame' is the audio data array,
            and 'frame_type' indicates the source/type of the frame.
        """
        try:
            frame = self.queue.get(block=True, timeout=0.01)
            frame_type = 0  # Real-time audio frame
        except queue.Empty:
            if self.parent and self.parent.curr_state > 1:
                # Get custom audio stream from parent
                frame = self.parent.get_audio_stream(self.parent.curr_state)
                frame_type = self.parent.curr_state
            else:
                # Return silent frame if no audio is available
                frame = np.zeros(self.chunk, dtype=np.float32)
                frame_type = 1  # Silent frame

        return frame, frame_type

    def get_audio_out(self):
        """
        Retrieve the next output audio frame.

        Returns:
            tuple: (frame, frame_type), the original audio PCM data to be used elsewhere.
        """
        return self.output_queue.get()

    def warm_up(self):
        """
        Warm up the ASR processing by pre-filling buffers.
        """
        total_frames = self.stride_left_size + self.stride_right_size
        for _ in range(total_frames):
            audio_frame, frame_type = self.get_audio_frame()
            self.frames.append(audio_frame)
            self.output_queue.put((audio_frame, frame_type))
        # Discard initial frames to align processing
        for _ in range(self.stride_left_size):
            self.output_queue.get()

    def run_step(self):
        """
        Placeholder method for running a processing step.
        Should be implemented in subclasses.
        """
        pass

    def get_next_feat(self, block=True, timeout=None):
        """
        Get the next set of features from the feature queue.

        Args:
            block (bool): Whether to block if no features are available.
            timeout (float): Timeout in seconds for blocking.

        Returns:
            The next set of features from the feature queue.
        """
        return self.feat_queue.get(block=block, timeout=timeout)
