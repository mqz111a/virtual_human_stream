import os
import json
import logging
from pathlib import Path
from typing import Optional, Tuple, List

import cv2
import glob
import tqdm
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as F

from .utils import get_audio_features, get_rays, get_bg_coords, convert_poses
from ..encoding import get_encoder
from .renderer import NeRFRenderer


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("nerf_dataset.log", mode='w')
    ]
)
logger = logging.getLogger(__name__)


def nerf_matrix_to_ngp(pose: np.ndarray, scale: float = 0.33, offset: List[float] = [0, 0, 0]) -> np.ndarray:
    """
    Converts a NeRF pose matrix to the format expected by instant-ngp.

    Args:
        pose (np.ndarray): Original pose matrix of shape [4, 4].
        scale (float, optional): Scaling factor. Defaults to 0.33.
        offset (List[float], optional): Translation offset. Defaults to [0, 0, 0].

    Returns:
        np.ndarray: Converted pose matrix of shape [4, 4].
    """
    new_pose = np.array([
        [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale + offset[0]],
        [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale + offset[1]],
        [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale + offset[2]],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    return new_pose


def smooth_camera_path(poses: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Smooths the camera trajectory by averaging poses within a sliding window.

    Args:
        poses (np.ndarray): Array of pose matrices of shape [N, 4, 4].
        kernel_size (int, optional): Size of the smoothing kernel. Defaults to 5.

    Returns:
        np.ndarray: Smoothed pose matrices of shape [N, 4, 4].
    """
    N = poses.shape[0]
    K = kernel_size // 2

    trans = poses[:, :3, 3].copy()  # [N, 3]
    rots = poses[:, :3, :3].copy()  # [N, 3, 3]

    for i in range(N):
        start = max(0, i - K)
        end = min(N, i + K + 1)
        poses[i, :3, 3] = trans[start:end].mean(axis=0)
        avg_rotation = Rotation.from_matrix(rots[start:end]).mean()
        poses[i, :3, :3] = avg_rotation.as_matrix()

    return poses


def polygon_area(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculates the area of a polygon given its vertices.

    Args:
        x (np.ndarray): X-coordinates of the vertices.
        y (np.ndarray): Y-coordinates of the vertices.

    Returns:
        float: Absolute area of the polygon.
    """
    x_ = x - x.mean()
    y_ = y - y.mean()
    correction = x_[-1] * y_[0] - y_[-1] * x_[0]
    main_area = np.dot(x_[:-1], y_[1:]) - np.dot(y_[:-1], x_[1:])
    return 0.5 * np.abs(main_area + correction)


def visualize_poses(poses: np.ndarray, size: float = 0.1) -> None:
    """
    Visualizes camera poses using trimesh.

    Args:
        poses (np.ndarray): Array of pose matrices of shape [B, 4, 4].
        size (float, optional): Size of the camera axes. Defaults to 0.1.
    """
    logger.info(f'[INFO] Visualizing poses: {poses.shape}')

    axes = trimesh.creation.axis(axis_length=4)
    box = trimesh.primitives.Box(extents=(2, 2, 2)).as_outline()
    box.visual.vertex_colors = [128, 128, 128, 255] * len(box.vertices)
    objects = [axes, box]

    for pose in poses:
        pos = pose[:3, 3]
        directions = [pose[:3, 0], pose[:3, 1], pose[:3, 2]]
        a = pos + size * directions[0] + size * directions[1] + size * directions[2]
        b = pos - size * directions[0] + size * directions[1] + size * directions[2]
        c = pos - size * directions[0] - size * directions[1] + size * directions[2]
        d = pos + size * directions[0] - size * directions[1] + size * directions[2]

        dir_vector = (a + b + c + d) / 4 - pos
        dir_vector /= np.linalg.norm(dir_vector) + 1e-8
        o = pos + dir_vector * 3

        segs = np.array([
            [pos, a],
            [pos, b],
            [pos, c],
            [pos, d],
            [a, b],
            [b, c],
            [c, d],
            [d, a],
            [pos, o]
        ])
        segs = trimesh.load_path(segs)
        objects.append(segs)

    scene = trimesh.Scene(objects)
    scene.show()


class NeRFDatasetTest(Dataset):
    """
    NeRF Dataset for Testing.

    This dataset class handles loading of poses, audio features, eye areas, torso images,
    and background images. It provides a DataLoader for batching and efficient data access.

    Args:
        opt: Configuration options.
        device (torch.device): Device to load data onto.
        downscale (int, optional): Downscale factor for image resolution. Defaults to 1.
    """

    def __init__(self, opt, device: torch.device, downscale: int = 1):
        super().__init__()

        self.opt = opt
        self.device = device
        self.downscale = downscale
        self.scale = opt.scale  # Camera radius scale
        self.offset = opt.offset  # Camera offset
        self.bound = opt.bound  # Bounding box half-length
        self.fp16 = opt.fp16

        self.start_index = opt.data_range[0]
        self.end_index = opt.data_range[1]

        self.training = False
        self.num_rays = -1
        self.preload = opt.preload  # 0 = disk, 1 = CPU, 2 = GPU

        # Load NeRF-compatible format data
        self._load_transform(opt.pose)

        # Load audio features if not in live-streaming mode
        if not self.opt.asr:
            self._load_audio_features()
        else:
            self.auds = None

        # Load action units
        if self.opt.exp_eye:
            self._load_action_units()

        # Load torso images
        if self.opt.torso_imgs:
            self._load_torso_images()

        # Load background image
        self._load_background_image()

        # Stack poses
        self.poses = np.stack(self.poses, axis=0)
        self.poses = torch.from_numpy(self.poses).to(self.device)

        # Smooth camera path if required
        if self.opt.smooth_path:
            self.poses = smooth_camera_path(self.poses.cpu().numpy(), self.opt.smooth_path_window)
            self.poses = torch.from_numpy(self.poses).to(self.device)

        # Move audio features to device if available
        if self.auds is not None:
            self.auds = self.auds.to(self.device)

        if self.opt.exp_eye:
            self.eye_area = torch.from_numpy(self.eye_area).view(-1, 1).to(self.device)

        # Load intrinsics
        self._load_intrinsics(transform)

        # Build background coordinates
        self.bg_coords = get_bg_coords(self.H, self.W, self.device)  # [1, H*W, 2]

    def _load_transform(self, pose_path: str):
        """
        Loads transformation matrices from a JSON file.

        Args:
            pose_path (str): Path to the JSON file containing poses.
        """
        with open(pose_path, 'r') as f:
            transform = json.load(f)

        # Load image size
        self.H = int(transform['cy']) * 2 // self.downscale
        self.W = int(transform['cx']) * 2 // self.downscale

        # Read frames
        frames = transform["frames"]

        # Slice dataset
        if self.end_index == -1:
            self.end_index = len(frames)
        frames = frames[self.start_index:self.end_index]

        logger.info(f'[INFO] Loaded {len(frames)} frames.')

        # Initialize pose list
        self.poses = []
        self.auds = []
        self.eye_area = []
        self.torso_img = []

        # Load audio features if applicable
        if not self.opt.asr and not self.opt.aud:
            raise ValueError("Audio features must be provided if not in live-streaming mode.")

        for f in tqdm.tqdm(frames, desc='Loading data'):
            # Convert pose matrix
            pose = np.array(f['transform_matrix'], dtype=np.float32)  # [4, 4]
            pose = nerf_matrix_to_ngp(pose, scale=self.scale, offset=self.offset)
            self.poses.append(pose)

            # Load corresponding audio
            if not self.opt.asr and self.opt.aud:
                aud = self.auds[min(f['aud_id'], self.auds.shape[0] - 1)]
                self.auds.append(aud)

            # Load eye area
            if self.opt.exp_eye:
                area = self._get_eye_area(f)
                self.eye_area.append(area)

    def _load_audio_features(self):
        """
        Loads pre-extracted audio features from a .npy file.
        """
        aud_features_path = self.opt.aud
        aud_features = np.load(aud_features_path)

        aud_features = torch.from_numpy(aud_features).float()

        # Support both [N, 16] labels and [N, 16, K] logits
        if aud_features.ndim == 3:
            aud_features = aud_features.permute(0, 2, 1)  # [N, 29, 16]

            if self.opt.emb:
                logger.info('[INFO] Applying argmax to audio features for embedding mode.')
                aud_features = aud_features.argmax(dim=1)  # [N, 16]
        else:
            assert self.opt.emb, "Audio features must be labels if not logits."
            aud_features = aud_features.long()

        logger.info(f'[INFO] Loaded audio features from {aud_features_path}: {aud_features.shape}')
        self.auds = aud_features

    def _load_action_units(self):
        """
        Loads action units from a CSV file.
        """
        au_path = self.opt.au
        au_blink_info = pd.read_csv(au_path)
        au_blink = au_blink_info[' AU45_r'].values

        for f in tqdm.tqdm(self.opt.data_range, desc='Loading action units'):
            area = au_blink[f['img_id']]
            area = np.clip(area, 0, 2) / 2
            self.eye_area.append(area)

        if self.opt.smooth_eye:
            logger.info('[INFO] Smoothing eye areas with a 5-window average.')
            self.eye_area = self._smooth_eye_areas(self.eye_area)

        self.eye_area = torch.from_numpy(np.array(self.eye_area, dtype=np.float32)).view(-1, 1)

    def _smooth_eye_areas(self, eye_areas: List[float], window_size: int = 5) -> np.ndarray:
        """
        Applies a naive sliding window average to smooth eye areas.

        Args:
            eye_areas (List[float]): List of eye area values.
            window_size (int, optional): Size of the sliding window. Defaults to 5.

        Returns:
            np.ndarray: Smoothed eye area values.
        """
        smoothed = np.copy(eye_areas)
        half_window = window_size // 2
        for i in range(len(smoothed)):
            start = max(0, i - half_window)
            end = min(len(smoothed), i + half_window + 1)
            smoothed[i] = np.mean(eye_areas[start:end])
        return smoothed

    def _get_eye_area(self, frame: dict) -> float:
        """
        Retrieves and processes the eye area from a frame.

        Args:
            frame (dict): Frame data containing 'img_id'.

        Returns:
            float: Processed eye area value.
        """
        au_blink = self.eye_area[frame['img_id']]
        area = np.clip(au_blink, 0, 2) / 2
        return area

    def _load_torso_images(self):
        """
        Loads torso images either by preloading them into memory or storing their file paths.
        """
        for f in tqdm.tqdm(self.poses, desc='Loading torso images'):
            torso_img_path = os.path.join(self.opt.torso_imgs, f'{f["img_id"]}.png')

            if self.preload > 0:
                torso_img = cv2.imread(torso_img_path, cv2.IMREAD_UNCHANGED)  # [H, W, 4]
                torso_img = cv2.cvtColor(torso_img, cv2.COLOR_BGRA2RGBA)
                torso_img = torso_img.astype(np.float32) / 255  # Normalize to [0, 1]
                self.torso_img.append(torso_img)
            else:
                self.torso_img.append(torso_img_path)

        if self.opt.torso_imgs:
            if self.preload > 0:
                self.torso_img = torch.from_numpy(np.stack(self.torso_img, axis=0))  # [N, H, W, C]
                if self.preload > 1:  # GPU
                    self.torso_img = self.torso_img.half().to(self.device)
            else:
                self.torso_img = np.array(self.torso_img)
            logger.info('[INFO] Torso images loaded.')

    def _load_background_image(self):
        """
        Loads the pre-extracted background image or creates a default background.
        """
        if self.opt.bg_img.lower() == 'white':
            bg_img = np.ones((self.H, self.W, 3), dtype=np.float32)
        elif self.opt.bg_img.lower() == 'black':
            bg_img = np.zeros((self.H, self.W, 3), dtype=np.float32)
        else:
            bg_img_path = self.opt.bg_img
            bg_img = cv2.imread(bg_img_path, cv2.IMREAD_UNCHANGED)  # [H, W, 3]
            if bg_img is None:
                raise FileNotFoundError(f"Background image {bg_img_path} not found.")
            if bg_img.shape[0] != self.H or bg_img.shape[1] != self.W:
                logger.warning('[WARN] Resizing background image to match dataset resolution.')
                bg_img = cv2.resize(bg_img, (self.W, self.H), interpolation=cv2.INTER_AREA)
            bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
            bg_img = bg_img.astype(np.float32) / 255  # Normalize to [0, 1]

        self.bg_img = torch.from_numpy(bg_img).float()
        if self.preload > 1 or not self.opt.torso_imgs:
            self.bg_img = self.bg_img.half().to(self.device)

        logger.info('[INFO] Background image loaded.')

    def _load_intrinsics(self, transform: dict):
        """
        Loads camera intrinsics from the transform dictionary.

        Args:
            transform (dict): Transform dictionary containing intrinsics.
        """
        fl_x = fl_y = transform['focal_len']
        cx = transform['cx'] / self.downscale
        cy = transform['cy'] / self.downscale

        self.intrinsics = np.array([fl_x, fl_y, cx, cy], dtype=np.float32)

    def mirror_index(self, index: int) -> int:
        """
        Mirrors the index for alternating camera paths.

        Args:
            index (int): Original index.

        Returns:
            int: Mirrored index.
        """
        size = self.poses.shape[0]
        turn = index // size
        res = index % size
        if turn % 2 == 0:
            return res
        else:
            return size - res - 1

    def collate_fn(self, index: List[int]) -> dict:
        """
        Collate function for DataLoader.

        Args:
            index (List[int]): List containing a single index.

        Returns:
            dict: Dictionary containing batched data.
        """
        B = len(index)  # Batch size, expected to be 1
        results = {}

        # Audio features
        if self.auds is not None:
            auds = get_audio_features(self.auds, self.opt.att, index[0]).to(self.device)
            results['auds'] = auds

        # Mirror index for alternating camera paths
        index[0] = self.mirror_index(index[0])

        # Retrieve poses
        poses = self.poses[index].to(self.device)  # [B, 4, 4]
        rays = get_rays(poses, self.intrinsics, self.H, self.W, self.num_rays, self.opt.patch_size)

        results['index'] = index  # For individual code
        results['H'] = self.H
        results['W'] = self.W
        results['rays_o'] = rays['rays_o']
        results['rays_d'] = rays['rays_d']

        # Eye area
        if self.opt.exp_eye:
            results['eye'] = self.eye_area[index].to(self.device)  # [B, 1]
        else:
            results['eye'] = None

        # Background image with torso if applicable
        if self.opt.torso_imgs:
            bg_torso_img = self.torso_img[index]
            if self.preload == 0:  # On-the-fly loading
                bg_torso_img = cv2.imread(bg_torso_img[0], cv2.IMREAD_UNCHANGED)  # [H, W, 4]
                if bg_torso_img is None:
                    raise FileNotFoundError(f"Torso image {bg_torso_img[0]} not found.")
                bg_torso_img = cv2.cvtColor(bg_torso_img, cv2.COLOR_BGRA2RGBA)
                bg_torso_img = bg_torso_img.astype(np.float32) / 255  # Normalize
                bg_torso_img = torch.from_numpy(bg_torso_img).unsqueeze(0).to(self.device)
            else:
                bg_torso_img = bg_torso_img[..., :3] * bg_torso_img[..., 3:] + self.bg_img * (1 - bg_torso_img[..., 3:])
                bg_torso_img = bg_torso_img.view(B, -1, 3).to(self.device)

            if not self.opt.torso:
                bg_img = bg_torso_img
            else:
                bg_img = self.bg_img.view(1, -1, 3).repeat(B, 1, 1).to(self.device)
        else:
            bg_img = self.bg_img.view(1, -1, 3).repeat(B, 1, 1).to(self.device)

        results['bg_color'] = bg_img
        results['bg_coords'] = self.bg_coords  # [1, H*W, 2]
        results['poses'] = poses  # [B, 4, 4]

        return results

    def dataloader(self) -> DataLoader:
        """
        Creates a DataLoader for the dataset.

        Returns:
            DataLoader: DataLoader instance.
        """
        # Determine dataset size
        if self.auds is not None:
            size = self.auds.shape[0]
        else:
            size = 2 * self.poses.shape[0]  # Live stream test

        loader = DataLoader(
            dataset=list(range(size)),
            batch_size=1,
            collate_fn=self.collate_fn,
            shuffle=False,
            num_workers=0
        )
        loader.has_gt = False  # Placeholder for evaluation flags

        logger.info('[INFO] DataLoader created.')
        return loader



class NeRFDataset:
    def __init__(self, opt, device, type='train', downscale=1):
        super().__init__()
        
        self.opt = opt
        self.device = device
        self.type = type # train, val, test
        self.downscale = downscale
        self.root_path = opt.path
        self.preload = opt.preload # 0 = disk, 1 = cpu, 2 = gpu
        self.scale = opt.scale # camera radius scale to make sure camera are inside the bounding box.
        self.offset = opt.offset # camera offset
        self.bound = opt.bound # bounding box half length, also used as the radius to random sample poses.
        self.fp16 = opt.fp16

        self.start_index = opt.data_range[0]
        self.end_index = opt.data_range[1]

        self.training = self.type in ['train', 'all', 'trainval']
        self.num_rays = self.opt.num_rays if self.training else -1

        # load nerf-compatible format data.
      
         # load all splits (train/valid/test)
        if type == 'all':
            transform_paths = glob.glob(os.path.join(self.root_path, '*.json'))
            transform = None
            for transform_path in transform_paths:
                with open(transform_path, 'r') as f:
                    tmp_transform = json.load(f)
                    if transform is None:
                        transform = tmp_transform
                    else:
                        transform['frames'].extend(tmp_transform['frames'])
        # load train and val split
        elif type == 'trainval':
            with open(os.path.join(self.root_path, f'transforms_train.json'), 'r') as f:
                transform = json.load(f)
            with open(os.path.join(self.root_path, f'transforms_val.json'), 'r') as f:
                transform_val = json.load(f)
            transform['frames'].extend(transform_val['frames'])
        # only load one specified split
        else:
            # no test, use val as test
            _split = 'val' if type == 'test' else type
            with open(os.path.join(self.root_path, f'transforms_{_split}.json'), 'r') as f:
                transform = json.load(f)

        # load image size
        if 'h' in transform and 'w' in transform:
            self.H = int(transform['h']) // downscale
            self.W = int(transform['w']) // downscale
        else:
            self.H = int(transform['cy']) * 2 // downscale
            self.W = int(transform['cx']) * 2 // downscale
        
        # read images
        frames = transform["frames"]

        # use a slice of the dataset
        if self.end_index == -1: # abuse...
            self.end_index = len(frames)

        frames = frames[self.start_index:self.end_index]
        print(f'[INFO] load {len(frames)} {type} frames.')

        # only load pre-calculated aud features when not live-streaming
        if not self.opt.asr:

            # empty means the default self-driven extracted features.
            if self.opt.aud == '':
                if 'esperanto' in self.opt.asr_model:
                    aud_features = np.load(os.path.join(self.root_path, 'aud_eo.npy'))
                elif 'deepspeech' in self.opt.asr_model:
                    aud_features = np.load(os.path.join(self.root_path, 'aud_ds.npy'))
                # elif 'hubert_cn' in self.opt.asr_model:
                #     aud_features = np.load(os.path.join(self.root_path, 'aud_hu_cn.npy'))
                elif 'hubert' in self.opt.asr_model:
                    aud_features = np.load(os.path.join(self.root_path, 'aud_hu.npy'))
                else:
                    aud_features = np.load(os.path.join(self.root_path, 'aud.npy'))
            # cross-driven extracted features. 
            else:
                aud_features = np.load(self.opt.aud)

            aud_features = torch.from_numpy(aud_features)

            # support both [N, 16] labels and [N, 16, K] logits
            if len(aud_features.shape) == 3:
                aud_features = aud_features.float().permute(0, 2, 1) # [N, 16, 29] --> [N, 29, 16]    

                if self.opt.emb:
                    print(f'[INFO] argmax to aud features {aud_features.shape} for --emb mode')
                    aud_features = aud_features.argmax(1) # [N, 16]
            
            else:
                assert self.opt.emb, "aud only provide labels, must use --emb"
                aud_features = aud_features.long()

            print(f'[INFO] load {self.opt.aud} aud_features: {aud_features.shape}')

        # load action units
        import pandas as pd
        au_blink_info=pd.read_csv(os.path.join(self.root_path, 'au.csv'))
        au_blink = au_blink_info[' AU45_r'].values

        self.torso_img = []
        self.images = []

        self.poses = []
        self.exps = []

        self.auds = []
        self.face_rect = []
        self.lhalf_rect = []
        self.lips_rect = []
        self.eye_area = []
        self.eye_rect = []

        for f in tqdm.tqdm(frames, desc=f'Loading {type} data'):

            f_path = os.path.join(self.root_path, 'gt_imgs', str(f['img_id']) + '.jpg')

            if not os.path.exists(f_path):
                print('[WARN]', f_path, 'NOT FOUND!')
                continue
            
            pose = np.array(f['transform_matrix'], dtype=np.float32) # [4, 4]
            pose = nerf_matrix_to_ngp(pose, scale=self.scale, offset=self.offset)
            self.poses.append(pose)

            if self.preload > 0:
                image = cv2.imread(f_path, cv2.IMREAD_UNCHANGED) # [H, W, 3] o [H, W, 4]
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = image.astype(np.float32) / 255 # [H, W, 3/4]

                self.images.append(image)
            else:
                self.images.append(f_path)

            # load frame-wise bg
        
            torso_img_path = os.path.join(self.root_path, 'torso_imgs', str(f['img_id']) + '.png')

            if self.preload > 0:
                torso_img = cv2.imread(torso_img_path, cv2.IMREAD_UNCHANGED) # [H, W, 4]
                torso_img = cv2.cvtColor(torso_img, cv2.COLOR_BGRA2RGBA)
                torso_img = torso_img.astype(np.float32) / 255 # [H, W, 3/4]

                self.torso_img.append(torso_img)
            else:
                self.torso_img.append(torso_img_path)

            # find the corresponding audio to the image frame
            if not self.opt.asr and self.opt.aud == '':
                aud = aud_features[min(f['aud_id'], aud_features.shape[0] - 1)] # careful for the last frame...
                self.auds.append(aud)

            # load lms and extract face
            lms = np.loadtxt(os.path.join(self.root_path, 'ori_imgs', str(f['img_id']) + '.lms')) # [68, 2]

            lh_xmin, lh_xmax = int(lms[31:36, 1].min()), int(lms[:, 1].max()) # actually lower half area
            xmin, xmax = int(lms[:, 1].min()), int(lms[:, 1].max())
            ymin, ymax = int(lms[:, 0].min()), int(lms[:, 0].max())
            self.face_rect.append([xmin, xmax, ymin, ymax])
            self.lhalf_rect.append([lh_xmin, lh_xmax, ymin, ymax])

            if self.opt.exp_eye:
                # eyes_left = slice(36, 42)
                # eyes_right = slice(42, 48)

                # area_left = polygon_area(lms[eyes_left, 0], lms[eyes_left, 1])
                # area_right = polygon_area(lms[eyes_right, 0], lms[eyes_right, 1])

                # # area percentage of two eyes of the whole image...
                # area = (area_left + area_right) / (self.H * self.W) * 100

                # action units blink AU45
                area = au_blink[f['img_id']]
                area = np.clip(area, 0, 2) / 2
                # area = area + np.random.rand() / 10
                self.eye_area.append(area)

                xmin, xmax = int(lms[36:48, 1].min()), int(lms[36:48, 1].max())
                ymin, ymax = int(lms[36:48, 0].min()), int(lms[36:48, 0].max())
                self.eye_rect.append([xmin, xmax, ymin, ymax])

            if self.opt.finetune_lips:
                lips = slice(48, 60)
                xmin, xmax = int(lms[lips, 1].min()), int(lms[lips, 1].max())
                ymin, ymax = int(lms[lips, 0].min()), int(lms[lips, 0].max())

                # padding to H == W
                cx = (xmin + xmax) // 2
                cy = (ymin + ymax) // 2

                l = max(xmax - xmin, ymax - ymin) // 2
                xmin = max(0, cx - l)
                xmax = min(self.H, cx + l)
                ymin = max(0, cy - l)
                ymax = min(self.W, cy + l)

                self.lips_rect.append([xmin, xmax, ymin, ymax])
        
        # load pre-extracted background image (should be the same size as training image...)

        if self.opt.bg_img == 'white': # special
            bg_img = np.ones((self.H, self.W, 3), dtype=np.float32)
        elif self.opt.bg_img == 'black': # special
            bg_img = np.zeros((self.H, self.W, 3), dtype=np.float32)
        else: # load from file
            # default bg
            if self.opt.bg_img == '':
                self.opt.bg_img = os.path.join(self.root_path, 'bc.jpg')
            bg_img = cv2.imread(self.opt.bg_img, cv2.IMREAD_UNCHANGED) # [H, W, 3]
            if bg_img.shape[0] != self.H or bg_img.shape[1] != self.W:
                bg_img = cv2.resize(bg_img, (self.W, self.H), interpolation=cv2.INTER_AREA)
            bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
            bg_img = bg_img.astype(np.float32) / 255 # [H, W, 3/4]

        self.bg_img = bg_img

        self.poses = np.stack(self.poses, axis=0)

        # smooth camera path...
        if self.opt.smooth_path:
            self.poses = smooth_camera_path(self.poses, self.opt.smooth_path_window)
            
        self.poses = torch.from_numpy(self.poses) # [N, 4, 4]

        if self.preload > 0:
            self.images = torch.from_numpy(np.stack(self.images, axis=0)) # [N, H, W, C]
            self.torso_img = torch.from_numpy(np.stack(self.torso_img, axis=0)) # [N, H, W, C]
        else:
            self.images = np.array(self.images)
            self.torso_img = np.array(self.torso_img)

        if self.opt.asr:
            # live streaming, no pre-calculated auds
            self.auds = None
        else:
            # auds corresponding to images
            if self.opt.aud == '':
                self.auds = torch.stack(self.auds, dim=0) # [N, 32, 16]
            # auds is novel, may have a different length with images
            else:
                self.auds = aud_features
        
        self.bg_img = torch.from_numpy(self.bg_img)

        if self.opt.exp_eye:
            self.eye_area = np.array(self.eye_area, dtype=np.float32) # [N]
            print(f'[INFO] eye_area: {self.eye_area.min()} - {self.eye_area.max()}')

            if self.opt.smooth_eye:

                # naive 5 window average
                ori_eye = self.eye_area.copy()
                for i in range(ori_eye.shape[0]):
                    start = max(0, i - 1)
                    end = min(ori_eye.shape[0], i + 2)
                    self.eye_area[i] = ori_eye[start:end].mean()

            self.eye_area = torch.from_numpy(self.eye_area).view(-1, 1) # [N, 1]

        
        # calculate mean radius of all camera poses
        self.radius = self.poses[:, :3, 3].norm(dim=-1).mean(0).item()
        #print(f'[INFO] dataset camera poses: radius = {self.radius:.4f}, bound = {self.bound}')

        
        # [debug] uncomment to view all training poses.
        # visualize_poses(self.poses.numpy())

        # [debug] uncomment to view examples of randomly generated poses.
        # visualize_poses(rand_poses(100, self.device, radius=self.radius).cpu().numpy())

        if self.preload > 1:
            self.poses = self.poses.to(self.device)

            if self.auds is not None:
                self.auds = self.auds.to(self.device)

            self.bg_img = self.bg_img.to(torch.half).to(self.device)

            self.torso_img = self.torso_img.to(torch.half).to(self.device)
            self.images = self.images.to(torch.half).to(self.device)
            
            if self.opt.exp_eye:
                self.eye_area = self.eye_area.to(self.device)

        # load intrinsics
        if 'focal_len' in transform:
            fl_x = fl_y = transform['focal_len']
        elif 'fl_x' in transform or 'fl_y' in transform:
            fl_x = (transform['fl_x'] if 'fl_x' in transform else transform['fl_y']) / downscale
            fl_y = (transform['fl_y'] if 'fl_y' in transform else transform['fl_x']) / downscale
        elif 'camera_angle_x' in transform or 'camera_angle_y' in transform:
            # blender, assert in radians. already downscaled since we use H/W
            fl_x = self.W / (2 * np.tan(transform['camera_angle_x'] / 2)) if 'camera_angle_x' in transform else None
            fl_y = self.H / (2 * np.tan(transform['camera_angle_y'] / 2)) if 'camera_angle_y' in transform else None
            if fl_x is None: fl_x = fl_y
            if fl_y is None: fl_y = fl_x
        else:
            raise RuntimeError('Failed to load focal length, please check the transforms.json!')

        cx = (transform['cx'] / downscale) if 'cx' in transform else (self.W / 2)
        cy = (transform['cy'] / downscale) if 'cy' in transform else (self.H / 2)
    
        self.intrinsics = np.array([fl_x, fl_y, cx, cy])

        # directly build the coordinate meshgrid in [-1, 1]^2
        self.bg_coords = get_bg_coords(self.H, self.W, self.device) # [1, H*W, 2] in [-1, 1]


    def mirror_index(self, index):
        size = self.poses.shape[0]
        turn = index // size
        res = index % size
        if turn % 2 == 0:
            return res
        else:
            return size - res - 1


    def collate(self, index):

        B = len(index) # a list of length 1
        # assert B == 1

        results = {}

        # audio use the original index
        if self.auds is not None:
            auds = get_audio_features(self.auds, self.opt.att, index[0]).to(self.device)
            results['auds'] = auds

        # head pose and bg image may mirror (replay --> <-- --> <--).
        index[0] = self.mirror_index(index[0])

        poses = self.poses[index].to(self.device) # [B, 4, 4]
        
        if self.training and self.opt.finetune_lips:
            rect = self.lips_rect[index[0]]
            results['rect'] = rect
            rays = get_rays(poses, self.intrinsics, self.H, self.W, -1, rect=rect)
        else:
            rays = get_rays(poses, self.intrinsics, self.H, self.W, self.num_rays, self.opt.patch_size)

        results['index'] = index # for ind. code
        results['H'] = self.H
        results['W'] = self.W
        results['rays_o'] = rays['rays_o']
        results['rays_d'] = rays['rays_d']

        # get a mask for rays inside rect_face
        if self.training:
            xmin, xmax, ymin, ymax = self.face_rect[index[0]]
            face_mask = (rays['j'] >= xmin) & (rays['j'] < xmax) & (rays['i'] >= ymin) & (rays['i'] < ymax) # [B, N]
            results['face_mask'] = face_mask
            
            xmin, xmax, ymin, ymax = self.lhalf_rect[index[0]]
            lhalf_mask = (rays['j'] >= xmin) & (rays['j'] < xmax) & (rays['i'] >= ymin) & (rays['i'] < ymax) # [B, N]
            results['lhalf_mask'] = lhalf_mask

        if self.opt.exp_eye:
            results['eye'] = self.eye_area[index].to(self.device) # [1]
            if self.training:
                results['eye'] += (np.random.rand()-0.5) / 10
                xmin, xmax, ymin, ymax = self.eye_rect[index[0]]
                eye_mask = (rays['j'] >= xmin) & (rays['j'] < xmax) & (rays['i'] >= ymin) & (rays['i'] < ymax) # [B, N]
                results['eye_mask'] = eye_mask

        else:
            results['eye'] = None

        # load bg
        bg_torso_img = self.torso_img[index]
        if self.preload == 0: # on the fly loading
            bg_torso_img = cv2.imread(bg_torso_img[0], cv2.IMREAD_UNCHANGED) # [H, W, 4]
            bg_torso_img = cv2.cvtColor(bg_torso_img, cv2.COLOR_BGRA2RGBA)
            bg_torso_img = bg_torso_img.astype(np.float32) / 255 # [H, W, 3/4]
            bg_torso_img = torch.from_numpy(bg_torso_img).unsqueeze(0)
        bg_torso_img = bg_torso_img[..., :3] * bg_torso_img[..., 3:] + self.bg_img * (1 - bg_torso_img[..., 3:])
        bg_torso_img = bg_torso_img.view(B, -1, 3).to(self.device)

        if not self.opt.torso:
            bg_img = bg_torso_img
        else:
            bg_img = self.bg_img.view(1, -1, 3).repeat(B, 1, 1).to(self.device)

        if self.training:
            bg_img = torch.gather(bg_img, 1, torch.stack(3 * [rays['inds']], -1)) # [B, N, 3]

        results['bg_color'] = bg_img

        if self.opt.torso and self.training:
            bg_torso_img = torch.gather(bg_torso_img, 1, torch.stack(3 * [rays['inds']], -1)) # [B, N, 3]
            results['bg_torso_color'] = bg_torso_img

        images = self.images[index] # [B, H, W, 3/4]
        if self.preload == 0:
            images = cv2.imread(images[0], cv2.IMREAD_UNCHANGED) # [H, W, 3]
            images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
            images = images.astype(np.float32) / 255 # [H, W, 3]
            images = torch.from_numpy(images).unsqueeze(0)
        images = images.to(self.device)

        if self.training:
            C = images.shape[-1]
            images = torch.gather(images.view(B, -1, C), 1, torch.stack(C * [rays['inds']], -1)) # [B, N, 3/4]
            
        results['images'] = images

        if self.training:
            bg_coords = torch.gather(self.bg_coords, 1, torch.stack(2 * [rays['inds']], -1)) # [1, N, 2]
        else:
            bg_coords = self.bg_coords # [1, N, 2]

        results['bg_coords'] = bg_coords

        # results['poses'] = convert_poses(poses) # [B, 6]
        # results['poses_matrix'] = poses # [B, 4, 4]
        results['poses'] = poses # [B, 4, 4]
            
        return results

    def dataloader(self):

        if self.training:
            # training len(poses) == len(auds)
            size = self.poses.shape[0]
        else:
            # test with novel auds, then use its length
            if self.auds is not None:
                size = self.auds.shape[0]
            # live stream test, use 2 * len(poses), so it naturally mirrors.
            else:
                size = 2 * self.poses.shape[0]

        loader = DataLoader(list(range(size)), batch_size=1, collate_fn=self.collate, shuffle=self.training, num_workers=0)
        loader._data = self # an ugly fix... we need poses in trainer.

        # do evaluate if has gt images and use self-driven setting
        loader.has_gt = (self.opt.aud == '')

        return loader        