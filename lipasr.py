import numpy as np
from baseasr import BaseASR
from wav2lip import audio

class LipASR(BaseASR):
    """
    LipASR class extends BaseASR to extract audio features suitable for lip-sync applications.
    It processes audio frames to generate mel-spectrogram chunks that can be used for lip synchronization.
    """

    def run_step(self):
        """
        Process audio frames to extract mel-spectrogram features and place them into the feature queue.
        This method reads audio frames, computes the mel-spectrogram, segments it into chunks,
        and queues these chunks for further processing.
        """
        # Collect audio frames and place them into the output queue
        self._collect_audio_frames()

        # Ensure there are enough frames to proceed with feature extraction
        if len(self.frames) <= self.stride_left_size + self.stride_right_size:
            return

        # Extract mel-spectrogram and segment it into chunks
        mel_chunks = self._extract_mel_chunks()

        # Place the mel chunks into the feature queue for downstream processing
        self.feat_queue.put(mel_chunks)

        # Discard old frames to manage memory and maintain the correct frame window
        self._discard_old_frames()

    def _collect_audio_frames(self):
        """Collect audio frames and place them into the output queue."""
        for _ in range(self.batch_size * 2):
            frame, frame_type = self.get_audio_frame()
            self.frames.append(frame)
            self.output_queue.put((frame, frame_type))

    def _extract_mel_chunks(self):
        """Extract mel-spectrogram chunks from the collected audio frames."""
        # Concatenate frames to create a continuous audio signal
        inputs = np.concatenate(self.frames)  # Shape: [N * chunk]
        mel = audio.melspectrogram(inputs)  # Compute mel-spectrogram of the input signal

        # Constants for mel-spectrogram indexing
        mel_frames_per_second = 80  # Number of mel-spectrogram frames per second (for 16kHz audio with hop length 200)
        audio_fps = self.fps  # Frames per second of the audio input

        # Calculate left and right indices to exclude stride frames from processing
        left = int(max(0, self.stride_left_size * mel_frames_per_second / audio_fps))
        right = int(min(mel.shape[1], mel.shape[1] - self.stride_right_size * mel_frames_per_second / audio_fps))

        # Parameters for segmenting the mel-spectrogram into chunks
        mel_idx_multiplier = mel_frames_per_second * 2 / audio_fps  # Multiplier to align audio frames with mel frames
        mel_step_size = 16  # Number of mel frames per chunk
        num_frames = int((len(self.frames) - self.stride_left_size - self.stride_right_size) / 2)

        # Segment mel-spectrogram into chunks
        mel_chunks = [self._get_mel_chunk(mel, left, i, mel_idx_multiplier, mel_step_size) for i in range(num_frames)]
        return mel_chunks

    def _get_mel_chunk(self, mel, left, i, mel_idx_multiplier, mel_step_size):
        """Get a single mel-spectrogram chunk given the index."""
        start_idx = int(left + i * mel_idx_multiplier)
        end_idx = start_idx + mel_step_size

        # Ensure the indices are within the valid range
        if end_idx > mel.shape[1]:
            # If the end index exceeds the mel frames, take the last mel_step_size frames
            return mel[:, -mel_step_size:]
        return mel[:, start_idx:end_idx]

    def _discard_old_frames(self):
        """Discard old frames to manage memory and maintain the correct frame window."""
        self.frames = self.frames[-(self.stride_left_size + self.stride_right_size):]
