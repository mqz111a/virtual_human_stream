import argparse
import glob
import json
import os
import pickle
import shutil

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from diffusers import AutoencoderKL
from face_alignment import NetworkSize
from mmpose.apis import inference_topdown, init_model
from mmpose.structures import merge_data_samples
from tqdm import tqdm

try:
    from utils.face_parsing import FaceParsing
except ModuleNotFoundError:
    from musetalk.utils.face_parsing import FaceParsing


def video_to_images(video_path, output_dir, ext='.png', max_frames=10000000):
    """
    Extract frames from a video file and save them as images.

    Parameters:
    - video_path (str): Path to the input video file.
    - output_dir (str): Directory to save the extracted images.
    - ext (str): Image file extension (default: '.png').
    - max_frames (int): Maximum number of frames to extract (default: 10000000).
    """
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames = min(frame_count, max_frames)
    for count in tqdm(range(total_frames), desc="Extracting frames"):
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(os.path.join(output_dir, f"{count:08d}{ext}"), frame)
        else:
            break
    cap.release()


def read_images(image_paths):
    """
    Read a list of image file paths into memory.

    Parameters:
    - image_paths (list of str): List of image file paths.

    Returns:
    - frames (list of ndarray): List of image arrays.
    """
    print('Reading images...')
    frames = [cv2.imread(img_path) for img_path in tqdm(image_paths, desc="Loading images")]
    return frames


def get_landmarks_and_bboxes(image_paths, model, face_aligner, bbox_shift=0):
    """
    Get landmarks and bounding boxes for a list of images.

    Parameters:
    - image_paths (list of str): List of image file paths.
    - model: MMPose model for human pose estimation.
    - face_aligner: FaceAlignment object for face detection.
    - bbox_shift (int): Shift value for the upper boundary of the face bounding box (default: 0).

    Returns:
    - coords_list (list of tuple): List of bounding box coordinates (x1, y1, x2, y2).
    - frames (list of ndarray): List of image arrays.
    """
    frames = read_images(image_paths)
    coords_list = []
    coord_placeholder = (0.0, 0.0, 0.0, 0.0)
    print(f'Getting key landmarks and face bounding boxes with bbox_shift: {bbox_shift}')
    for frame in tqdm(frames, desc="Processing frames"):
        # Perform human pose estimation
        result = inference_topdown(model, frame)
        result = merge_data_samples(result)
        keypoints = result.pred_instances.keypoints

        if keypoints.shape[0] == 0:
            coords_list.append(coord_placeholder)
            continue

        # Extract face landmarks
        face_landmarks = keypoints[0][23:91].astype(np.int32)

        # Get bounding boxes using face detection
        bbox = face_aligner.get_detections_for_batch(np.expand_dims(frame, axis=0))

        if bbox[0] is None:
            coords_list.append(coord_placeholder)
            continue

        # Adjust upper boundary of the face bounding box
        half_face_coord = face_landmarks[29].copy()
        if bbox_shift != 0:
            half_face_coord[1] += bbox_shift

        half_face_dist = np.max(face_landmarks[:, 1]) - half_face_coord[1]
        upper_bound = half_face_coord[1] - half_face_dist

        # Construct face landmark bounding box
        f_landmark = (
            np.min(face_landmarks[:, 0]),
            int(upper_bound),
            np.max(face_landmarks[:, 0]),
            np.max(face_landmarks[:, 1])
        )
        x1, y1, x2, y2 = f_landmark

        if y2 - y1 <= 0 or x2 - x1 <= 0 or x1 < 0:
            coords_list.append(bbox[0])
            print("Error in bounding box:", bbox[0])
        else:
            coords_list.append(f_landmark)
    return coords_list, frames


class FaceAlignment:
    """
    Custom FaceAlignment class for face detection in batches.

    Parameters:
    - landmarks_type (int): Type of landmarks to detect.
    - network_size (int): Size of the network model (default: NetworkSize.LARGE).
    - device (str): Device to run the model on ('cuda' or 'cpu').
    - flip_input (bool): Whether to flip input images (default: False).
    - face_detector (str): Face detector type to use (default: 'sfd').
    - verbose (bool): Verbosity flag (default: False).
    """
    def __init__(self, landmarks_type, network_size=NetworkSize.LARGE,
                 device='cuda', flip_input=False, face_detector='sfd', verbose=False):
        self.device = device
        self.flip_input = flip_input
        self.landmarks_type = landmarks_type
        self.verbose = verbose

        network_size = int(network_size)
        if 'cuda' in device:
            torch.backends.cudnn.benchmark = True
            print('CUDA acceleration is enabled.')

        # Dynamically import the specified face detector module
        face_detector_module = __import__('face_detection.detection.' + face_detector,
                                          fromlist=[face_detector])

        self.face_detector = face_detector_module.FaceDetector(device=device, verbose=verbose)

    def get_detections_for_batch(self, images):
        """
        Detect faces in a batch of images.

        Parameters:
        - images (ndarray): Batch of images in BGR format (batch_size, H, W, 3).

        Returns:
        - results (list of tuple): List of bounding boxes (x1, y1, x2, y2) for each image.
        """
        images_rgb = images[..., ::-1]  # Convert BGR to RGB
        detected_faces = self.face_detector.detect_from_batch(images_rgb.copy())
        results = [
            (int(d[0][0]), int(d[0][1]), int(d[0][2]), int(d[0][3])) if len(d) > 0 else None
            for d in detected_faces
        ]
        return results


def get_mask_tensor():
    """
    Creates a mask tensor for image processing.

    Returns:
    - mask_tensor (Tensor): A mask tensor of shape (256, 256).
    """
    mask_tensor = torch.zeros((256, 256))
    mask_tensor[:128, :] = 1
    return mask_tensor


def preprocess_image(img_input, device, half_mask=False):
    """
    Preprocess an image for model input.

    Parameters:
    - img_input (str or ndarray): Image file path or image array.
    - device (str or torch.device): Device to place the tensor on.
    - half_mask (bool): Whether to apply a half mask to the image (default: False).

    Returns:
    - img_tensor (Tensor): Preprocessed image tensor ready for model input.
    """
    if isinstance(img_input, str):
        img = cv2.imread(img_input)
    else:
        img = img_input
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LANCZOS4)
    img = img / 255.0
    img_tensor = torch.FloatTensor(img).permute(2, 0, 1)
    if half_mask:
        mask = get_mask_tensor()
        img_tensor *= mask
    normalize = transforms.Normalize(mean=[0.5], std=[0.5])
    img_tensor = normalize(img_tensor)
    img_tensor = img_tensor.unsqueeze(0).to(device)
    return img_tensor


def encode_latents(image, vae):
    """
    Encode an image tensor into latent space using the VAE.

    Parameters:
    - image (Tensor): Preprocessed image tensor.
    - vae (AutoencoderKL): VAE model for encoding.

    Returns:
    - init_latents (Tensor): Latent representation of the image.
    """
    with torch.no_grad():
        init_latent_dist = vae.encode(image.to(vae.dtype)).latent_dist
    init_latents = vae.config.scaling_factor * init_latent_dist.sample()
    return init_latents


def get_latents_for_unet(img, device, vae):
    """
    Get the latent inputs for the U-Net model.

    Parameters:
    - img (ndarray): Image array in BGR format.
    - device (str or torch.device): Device to place tensors on.
    - vae (AutoencoderKL): VAE model for encoding.

    Returns:
    - latent_model_input (Tensor): Concatenated latent inputs for U-Net.
    """
    # Preprocess images with and without half mask
    ref_image_masked = preprocess_image(img, device, half_mask=True)
    masked_latents = encode_latents(ref_image_masked, vae)
    ref_image_full = preprocess_image(img, device, half_mask=False)
    ref_latents = encode_latents(ref_image_full, vae)
    latent_model_input = torch.cat([masked_latents, ref_latents], dim=1)
    return latent_model_input


def get_crop_box(box, expand):
    """
    Calculate a crop box expanded around a given bounding box.

    Parameters:
    - box (tuple): Original bounding box (x1, y1, x2, y2).
    - expand (float): Expansion factor.

    Returns:
    - crop_box (list): Expanded crop box coordinates [x_start, y_start, x_end, y_end].
    - s (int): Half size of the expanded bounding box.
    """
    x1, y1, x2, y2 = box
    x_center = (x1 + x2) // 2
    y_center = (y1 + y2) // 2
    s = int(max(x2 - x1, y2 - y1) * expand / 2)
    crop_box = [x_center - s, y_center - s, x_center + s, y_center + s]
    return crop_box, s


def face_segmentation(image, face_parser):
    """
    Perform face segmentation on an image.

    Parameters:
    - image (PIL.Image): Image to segment.
    - face_parser (FaceParsing): Face parsing model.

    Returns:
    - seg_image (PIL.Image or None): Segmented image or None if no face is found.
    """
    seg_image = face_parser(image)
    if seg_image is None:
        print("Error: No person segment found.")
        return None
    return seg_image.resize(image.size)


def prepare_image_material(image, face_box, face_parser, upper_boundary_ratio=0.5, expand=1.2):
    """
    Prepare materials for image processing, including masks and crop boxes.

    Parameters:
    - image (ndarray): Image array in BGR format.
    - face_box (tuple): Face bounding box (x1, y1, x2, y2).
    - face_parser (FaceParsing): Face parsing model.
    - upper_boundary_ratio (float): Ratio to keep upper boundary of the talking area (default: 0.5).
    - expand (float): Expansion factor for the crop box (default: 1.2).

    Returns:
    - mask_array (ndarray): Mask array for the image.
    - crop_box (list): Coordinates of the crop box.
    """
    # Convert image to PIL Image
    body = Image.fromarray(image[:, :, ::-1])  # Convert BGR to RGB

    # Calculate crop box
    crop_box, _ = get_crop_box(face_box, expand)
    x_start, y_start, x_end, y_end = crop_box
    face_large = body.crop(crop_box)

    # Perform face segmentation
    mask_image = face_segmentation(face_large, face_parser)
    if mask_image is None:
        mask_array = np.zeros((face_large.size[1], face_large.size[0]), dtype=np.uint8)
        return mask_array, crop_box

    x1, y1, x2, y2 = face_box
    # Adjust coordinates relative to the cropped image
    x1_rel, y1_rel = x1 - x_start, y1 - y_start
    x2_rel, y2_rel = x2 - x_start, y2 - y_start

    # Crop mask to face bounding box
    mask_small = mask_image.crop((x1_rel, y1_rel, x2_rel, y2_rel))
    mask_image_full = Image.new('L', face_large.size, 0)
    mask_image_full.paste(mask_small, (x1_rel, y1_rel, x2_rel, y2_rel))

    width, height = mask_image_full.size
    top_boundary = int(height * upper_boundary_ratio)
    modified_mask_image = Image.new('L', face_large.size, 0)
    modified_mask_image.paste(mask_image_full.crop((0, top_boundary, width, height)), (0, top_boundary))

    # Apply Gaussian blur to smooth the mask edges
    blur_kernel_size = int(0.1 * face_large.size[0] // 2 * 2) + 1
    mask_array = cv2.GaussianBlur(np.array(modified_mask_image), (blur_kernel_size, blur_kernel_size), 0)
    return mask_array, crop_box


def is_video_file(file_path):
    """
    Check if a file is a video file based on its extension.

    Parameters:
    - file_path (str): Path to the file.

    Returns:
    - is_video (bool): True if the file is a video, False otherwise.
    """
    video_exts = ['.mp4', '.mkv', '.flv', '.avi', '.mov']
    file_ext = os.path.splitext(file_path)[1].lower()
    return file_ext in video_exts


def create_directory(dir_path):
    """
    Create a directory if it doesn't exist.

    Parameters:
    - dir_path (str): Path to the directory.
    """
    os.makedirs(dir_path, exist_ok=True)


def create_musetalk_human(file_path, avatar_id, model, face_aligner, vae, face_parser, device):
    """
    Main function to create MuseTalk avatar materials.

    Parameters:
    - file_path (str): Path to the input video file or directory containing images.
    - avatar_id (str): Identifier for the avatar.
    - model: MMPose model for inference.
    - face_aligner: FaceAlignment object for face detection.
    - vae: VAE model for encoding.
    - face_parser: FaceParsing model for segmentation.
    - device (str or torch.device): Device to run models on.
    """
    # Set up paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(current_dir, f'../data/avatars/avatar_{avatar_id}')
    save_full_path = os.path.join(save_path, 'full_imgs')
    create_directory(save_full_path)
    mask_out_path = os.path.join(save_path, 'mask')
    create_directory(mask_out_path)
    mask_coords_path = os.path.join(save_path, 'mask_coords.pkl')
    coords_path = os.path.join(save_path, 'coords.pkl')
    latents_out_path = os.path.join(save_path, 'latents.pt')

    # Save avatar info
    avatar_info = {
        "avatar_id": avatar_id,
        "video_path": file_path,
        "bbox_shift": 5
    }
    with open(os.path.join(save_path, 'avatar_info.json'), "w") as f:
        json.dump(avatar_info, f)

    # Process input file
    if os.path.isfile(file_path):
        if is_video_file(file_path):
            # Extract frames from video
            video_to_images(file_path, save_full_path, ext='.png')
        else:
            # Single image file
            shutil.copy(file_path, os.path.join(save_full_path, os.path.basename(file_path)))
    else:
        # Directory containing image files
        image_files = sorted(glob.glob(os.path.join(file_path, '*.[jpJP][pnPN]*[gG]')))
        for src in image_files:
            dst = os.path.join(save_full_path, os.path.basename(src))
            shutil.copy(src, dst)

    # Get list of input images
    input_img_list = sorted(glob.glob(os.path.join(save_full_path, '*.[jpJP][pnPN]*[gG]')))
    print("Extracting landmarks...")
    # Get landmarks and bounding boxes
    coord_list, frame_list = get_landmarks_and_bboxes(input_img_list, model, face_aligner, bbox_shift=5)
    input_latent_list = []
    coord_placeholder = (0.0, 0.0, 0.0, 0.0)

    # Encode images to latent space
    for bbox, frame in zip(coord_list, frame_list):
        if bbox == coord_placeholder:
            continue
        x1, y1, x2, y2 = map(int, bbox)
        crop_frame = frame[y1:y2, x1:x2]
        resized_crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
        latents = get_latents_for_unet(resized_crop_frame, device, vae)
        input_latent_list.append(latents)

    # Create cyclic frames for continuity
    frame_list_cycle = frame_list + frame_list[::-1]
    coord_list_cycle = coord_list + coord_list[::-1]
    mask_coords_list_cycle = []

    # Generate masks and save images
    for i, (frame, face_box) in enumerate(tqdm(zip(frame_list_cycle, coord_list_cycle),
                                               total=len(frame_list_cycle),
                                               desc="Processing masks")):
        # Save full image
        cv2.imwrite(os.path.join(save_full_path, f"{str(i).zfill(8)}.png"), frame)
        # Generate mask and crop box
        mask, crop_box = prepare_image_material(frame, face_box, face_parser)
        # Save mask image
        cv2.imwrite(os.path.join(mask_out_path, f"{str(i).zfill(8)}.png"), mask)
        mask_coords_list_cycle.append(crop_box)

    # Save processing results
    with open(mask_coords_path, 'wb') as f:
        pickle.dump(mask_coords_list_cycle, f)

    with open(coords_path, 'wb') as f:
        pickle.dump(coord_list_cycle, f)

    # Save latent representations
    torch.save(input_latent_list, latents_out_path)


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description="Create MuseTalk avatar materials.")
    parser.add_argument("--file",
                        type=str,
                        required=True,
                        help="Path to the input video file or directory containing images.")
    parser.add_argument("--avatar_id",
                        type=str,
                        default='3',
                        help="Identifier for the avatar.")
    args = parser.parse_args()

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize models
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Initialize FaceAlignment
    face_aligner = FaceAlignment(1, flip_input=False, device=device)

    # Initialize MMPose model
    config_file = os.path.join(current_dir, 'utils/dwpose/rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py')
    checkpoint_file = os.path.abspath(os.path.join(current_dir, '../models/dwpose/dw-ll_ucoco_384.pth'))
    model = init_model(config_file, checkpoint_file, device=device)

    # Initialize VAE model
    vae = AutoencoderKL.from_pretrained(os.path.abspath(os.path.join(current_dir, '../models/sd-vae-ft-mse')))
    vae.to(device)

    # Initialize FaceParsing model
    face_parser = FaceParsing(
        os.path.abspath(os.path.join(current_dir, '../models/face-parse-bisent/resnet18-5c106cde.pth')),
        os.path.abspath(os.path.join(current_dir, '../models/face-parse-bisent/79999_iter.pth'))
    )

    # Create MuseTalk human avatar materials
    create_musetalk_human(args.file, args.avatar_id, model, face_aligner, vae, face_parser, device)
