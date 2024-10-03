import os
from glob import glob
from dataclasses import dataclass, field

def get_image_list(data_root, split):
    """
    Get a list of image file paths from a given data root and dataset split.

    Parameters:
    - data_root (str): Root directory of the data.
    - split (str): Dataset split name (e.g., 'train', 'val', 'test').

    Returns:
    - filelist (list of str): List of full paths to image files.
    """
    filelist = []

    filelist_path = os.path.join('filelists', f'{split}.txt')
    with open(filelist_path, 'r') as f:
        for line in f:
            line = line.strip()
            if ' ' in line:
                line = line.split()[0]
            filelist.append(os.path.join(data_root, line))

    return filelist

@dataclass
class HParams:
    """
    Hyperparameters configuration class.

    Attributes:
    - num_mels (int): Number of mel-spectrogram channels.
    - rescale (bool): Whether to rescale audio prior to preprocessing.
    - rescaling_max (float): Maximum value for rescaling.
    - use_lws (bool): Use LWS for STFT and phase reconstruction.
    - n_fft (int): Number of FFT components.
    - hop_size (int): Hop size in samples.
    - win_size (int): Window size in samples.
    - sample_rate (int): Audio sampling rate.
    - frame_shift_ms (float): Frame shift in milliseconds.
    - signal_normalization (bool): Normalize spectrograms.
    - allow_clipping_in_normalization (bool): Allow clipping during normalization.
    - symmetric_mels (bool): Symmetric scaling of mel spectrograms.
    - max_abs_value (float): Maximum absolute value for normalization.
    - preemphasize (bool): Apply pre-emphasis filter.
    - preemphasis (float): Pre-emphasis coefficient.
    - min_level_db (int): Minimum level in decibels.
    - ref_level_db (int): Reference level in decibels.
    - fmin (int): Minimum frequency.
    - fmax (int): Maximum frequency.
    - img_size (int): Image size for processing.
    - fps (int): Frames per second.
    - batch_size (int): Training batch size.
    - initial_learning_rate (float): Initial learning rate.
    - nepochs (int): Number of training epochs.
    - num_workers (int): Number of data loader workers.
    - checkpoint_interval (int): Interval for saving checkpoints.
    - eval_interval (int): Interval for model evaluation.
    - save_optimizer_state (bool): Save optimizer state in checkpoints.
    - syncnet_wt (float): Weight for SyncNet loss.
    - syncnet_batch_size (int): Batch size for SyncNet training.
    - syncnet_lr (float): Learning rate for SyncNet.
    - syncnet_eval_interval (int): Evaluation interval for SyncNet.
    - syncnet_checkpoint_interval (int): Checkpoint interval for SyncNet.
    - disc_wt (float): Weight for discriminator loss.
    - disc_initial_learning_rate (float): Initial learning rate for discriminator.
    - sentences (list): Placeholder for sentences (if any).
    """
    # Audio processing parameters
    num_mels: int = 80
    rescale: bool = True
    rescaling_max: float = 0.9

    # STFT parameters
    use_lws: bool = False
    n_fft: int = 800
    hop_size: int = 200
    win_size: int = 800
    sample_rate: int = 16000
    frame_shift_ms: float = None

    # Spectrogram normalization parameters
    signal_normalization: bool = True
    allow_clipping_in_normalization: bool = True
    symmetric_mels: bool = True
    max_abs_value: float = 4.0

    # Pre-emphasis filter
    preemphasize: bool = True
    preemphasis: float = 0.97

    # Spectrogram limits
    min_level_db: int = -100
    ref_level_db: int = 20
    fmin: int = 55
    fmax: int = 7600

    # Training parameters
    img_size: int = 96
    fps: int = 25
    batch_size: int = 16
    initial_learning_rate: float = 1e-4
    nepochs: int = 1_000_000_000  # Large number; training can be stopped manually
    num_workers: int = 16
    checkpoint_interval: int = 3000
    eval_interval: int = 3000
    save_optimizer_state: bool = True

    # SyncNet parameters
    syncnet_wt: float = 0.0  # Will be set to 0.03 later for faster convergence
    syncnet_batch_size: int = 64
    syncnet_lr: float = 1e-4
    syncnet_eval_interval: int = 10000
    syncnet_checkpoint_interval: int = 10000

    # Discriminator parameters
    disc_wt: float = 0.07
    disc_initial_learning_rate: float = 1e-4

    # Additional fields
    sentences: list = field(default_factory=list)

def hparams_debug_string(hparams):
    """
    Generate a formatted string of hyperparameters for debugging.

    Parameters:
    - hparams (HParams): An instance of the HParams class.

    Returns:
    - debug_str (str): Formatted string of hyperparameters.
    """
    hp = [
        f"  {name}: {getattr(hparams, name)}"
        for name in sorted(hparams.__dataclass_fields__)
        if name != "sentences"
    ]
    return "Hyperparameters:\n" + "\n".join(hp)

# Instantiate default hyperparameters
hparams = HParams()

if __name__ == "__main__":
    print(hparams_debug_string(hparams))
