import os
import glob
import cv2
import numpy as np
import soundfile as sf
from tqdm import tqdm


def read_images(image_paths):
    """
    Read images from a list of file paths.

    Parameters:
    - image_paths (list of str): List of image file paths.

    Returns:
    - frames (list of ndarray): List of image arrays.
    """
    frames = []
    print('Reading images...')
    for img_path in tqdm(image_paths, desc='Loading images'):
        frame = cv2.imread(img_path)
        if frame is not None:
            frames.append(frame)
        else:
            print(f"Warning: Could not read image {img_path}")
    return frames


class BaseReal:
    """
    Base class for handling real-time audio and image processing for custom avatars.

    Attributes:
    - opt: Configuration options.
    - sample_rate (int): Audio sample rate in Hz.
    - chunk (int): Number of samples per audio chunk.
    - curr_state (int): Current state identifier (e.g., 0 for normal, 1 for silent).
    - custom_img_cycle (dict): Maps audio types to lists of images.
    - custom_audio_cycle (dict): Maps audio types to audio data arrays.
    - custom_audio_index (dict): Current audio index per audio type.
    - custom_index (dict): Current image index per audio type.
    - custom_opt (dict): Custom options per audio type.
    """

    def __init__(self, opt):
        """
        Initialize the BaseReal object.

        Args:
        - opt: Configuration options containing 'fps' and 'customopt'.
        """
        self.opt = opt
        self.sample_rate = 16000
        self.chunk = self.sample_rate // opt.fps  # Samples per chunk (e.g., 320 samples for 20ms at 16kHz)

        self.curr_state = 0  # Current state (e.g., 0 for normal, 1 for silent)
        self.custom_img_cycle = {}
        self.custom_audio_cycle = {}
        self.custom_audio_index = {}
        self.custom_index = {}
        self.custom_opt = {}
        self._load_custom()

    def _load_custom(self):
        """
        Load custom images and audio for each custom audio type.
        """
        for item in self.opt.customopt:
            print(f"Loading custom options for audiotype {item['audiotype']}")
            img_path = item['imgpath']
            audio_path = item['audiopath']
            audiotype = item['audiotype']

            # Get list of image files
            image_pattern = os.path.join(img_path, '*.[jpJP][pnPN]*[gG]')
            image_files = glob.glob(image_pattern)
            # Sort images by numeric filename
            image_files = sorted(
                image_files,
                key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
            )

            # Read images
            images = read_images(image_files)
            self.custom_img_cycle[audiotype] = images

            # Read audio file
            audio_data, sample_rate = sf.read(audio_path, dtype='float32')
            if sample_rate != self.sample_rate:
                print(f"Warning: Sample rate mismatch for {audio_path}. "
                      f"Expected {self.sample_rate}, got {sample_rate}.")
                # Optionally handle resampling here if needed

            self.custom_audio_cycle[audiotype] = audio_data

            # Initialize indices and options
            self.custom_audio_index[audiotype] = 0
            self.custom_index[audiotype] = 0
            self.custom_opt[audiotype] = item

    def mirror_index(self, size, index):
        """
        Calculate the mirrored index for looping sequences in a forward-backward fashion.

        Args:
        - size (int): Total number of items in the sequence.
        - index (int): Current index.

        Returns:
        - int: Mirrored index.
        """
        if size == 0:
            print("Error: Size cannot be zero in mirror_index")
            return 0

        turn = index // size
        res = index % size
        if turn % 2 == 0:
            return res
        else:
            return size - res - 1

    def get_audio_stream(self, audiotype):
        """
        Get the next audio chunk for the specified audio type.

        Args:
        - audiotype: Identifier for the audio type.

        Returns:
        - stream (ndarray): Audio data chunk of size self.chunk.
        """
        idx = self.custom_audio_index.get(audiotype, 0)
        audio_data = self.custom_audio_cycle.get(audiotype)

        if audio_data is None:
            print(f"Error: No audio data found for audiotype {audiotype}")
            return np.zeros(self.chunk, dtype=np.float32)

        end_idx = idx + self.chunk
        stream = audio_data[idx:end_idx]
        self.custom_audio_index[audiotype] = end_idx

        # If the stream is shorter than the chunk, pad with zeros
        if len(stream) < self.chunk:
            padding = np.zeros(self.chunk - len(stream), dtype=stream.dtype)
            stream = np.concatenate((stream, padding))
            self.curr_state = 1  # Switch to silent state or reset

        # Check if we've reached the end of the audio data
        if self.custom_audio_index[audiotype] >= len(audio_data):
            self.curr_state = 1  # Switch to silent state or reset
            # Optionally reset for looping
            # self.custom_audio_index[audiotype] = 0

        return stream

    def get_image_frame(self, audiotype):
        """
        Get the next image frame for the specified audio type.

        Args:
        - audiotype: Identifier for the audio type.

        Returns:
        - image (ndarray): Image array corresponding to the current frame.
        """
        images = self.custom_img_cycle.get(audiotype)
        if images is None:
            print(f"Error: No image data found for audiotype {audiotype}")
            return None

        idx = self.custom_index.get(audiotype, 0)
        size = len(images)
        if size == 0:
            print(f"Error: No images available for audiotype {audiotype}")
            return None

        # Get mirrored index for looping
        mirror_idx = self.mirror_index(size, idx)
        image = images[mirror_idx]

        # Update index
        self.custom_index[audiotype] = idx + 1

        return image

    def set_curr_state(self, audiotype, reinit=False):
        """
        Set the current state to the specified audio type.

        Args:
        - audiotype: Identifier for the audio type.
        - reinit (bool): Whether to reinitialize indices for the audio type.
        """
        if audiotype not in self.custom_audio_cycle:
            print(f"Error: Invalid audiotype {audiotype}")
            return

        self.curr_state = audiotype
        if reinit:
            self.custom_audio_index[audiotype] = 0
            self.custom_index[audiotype] = 0
