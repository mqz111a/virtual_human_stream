import argparse
import os
import pickle
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
import torch
import face_detection


def create_directories(paths):
    """
    Create directories if they don't exist.

    Parameters:
    - paths (list of str): List of directory paths to create.
    """
    for path in paths:
        os.makedirs(path, exist_ok=True)


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
    count = 0
    while True:
        if count >= max_frames:
            break
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(os.path.join(output_dir, f"{count:08d}{ext}"), frame)
            count += 1
        else:
            break
    cap.release()


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
    for img_path in tqdm(image_paths, desc="Loading images"):
        frame = cv2.imread(img_path)
        if frame is not None:
            frames.append(frame)
        else:
            print(f"Warning: Could not read image {img_path}")
    return frames


def smooth_boxes(boxes, window_size):
    """
    Smooth bounding boxes over a temporal window.

    Parameters:
    - boxes (list of ndarray): List of bounding boxes.
    - window_size (int): Size of the smoothing window.

    Returns:
    - smoothed_boxes (list of ndarray): Smoothed bounding boxes.
    """
    smoothed_boxes = []
    for i in range(len(boxes)):
        if i + window_size > len(boxes):
            window = boxes[len(boxes) - window_size:]
        else:
            window = boxes[i: i + window_size]
        smoothed_box = np.mean(window, axis=0)
        smoothed_boxes.append(smoothed_box)
    return smoothed_boxes


def detect_faces(images, device, batch_size=16, pads=(0, 10, 0, 0), smooth=True, smooth_window=5):
    """
    Detect faces in a list of images.

    Parameters:
    - images (list of ndarray): List of images.
    - device (str): Device to run face detection ('cpu' or 'cuda').
    - batch_size (int): Batch size for face detection (default: 16).
    - pads (tuple): Padding (top, bottom, left, right) (default: (0, 10, 0, 0)).
    - smooth (bool): Whether to smooth the bounding boxes over time (default: True).
    - smooth_window (int): Size of the smoothing window (default: 5).

    Returns:
    - results (list): List of tuples containing cropped face images and their coordinates.
    """
    # Initialize face detector
    detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D,
                                            flip_input=False, device=device)

    while True:
        predictions = []
        try:
            for i in tqdm(range(0, len(images), batch_size), desc="Detecting faces"):
                batch = images[i:i + batch_size]
                batch_predictions = detector.get_detections_for_batch(np.array(batch))
                predictions.extend(batch_predictions)
        except RuntimeError as e:
            if batch_size == 1:
                raise RuntimeError('Image too big to run face detection on GPU. Consider reducing the image size.') from e
            batch_size = max(1, batch_size // 2)
            print(f'Recovering from OOM error; New batch size: {batch_size}')
            continue
        break

    results = []
    pady1, pady2, padx1, padx2 = pads
    for idx, (rect, image) in enumerate(zip(predictions, images)):
        if rect is None:
            faulty_frame_path = os.path.join('temp', 'faulty_frame.jpg')
            os.makedirs('temp', exist_ok=True)
            cv2.imwrite(faulty_frame_path, image)
            raise ValueError(f'Face not detected in frame {idx}! Saved to {faulty_frame_path}. Ensure the video contains a face in all frames.')

        x1 = max(0, rect[0] - padx1)
        y1 = max(0, rect[1] - pady1)
        x2 = min(image.shape[1], rect[2] + padx2)
        y2 = min(image.shape[0], rect[3] + pady2)

        results.append([x1, y1, x2, y2])

    boxes = np.array(results)
    if smooth:
        boxes = smooth_boxes(boxes, window_size=smooth_window)

    cropped_faces = []
    for image, box in zip(images, boxes):
        x1, y1, x2, y2 = map(int, box)
        cropped_face = image[y1:y2, x1:x2]
        cropped_faces.append((cropped_face, (x1, y1, x2, y2)))

    del detector
    return cropped_faces


def main(args):
    """
    Main function to process video and extract face images and coordinates.

    Parameters:
    - args: Parsed command-line arguments.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} for inference.')

    # Set up paths
    avatar_path = os.path.join("./results/avatars", args.avatar_id)
    full_imgs_path = os.path.join(avatar_path, "full_imgs")
    face_imgs_path = os.path.join(avatar_path, "face_imgs")
    coords_path = os.path.join(avatar_path, "coords.pkl")
    create_directories([avatar_path, full_imgs_path, face_imgs_path])

    print(f'Processing video: {args.video_path}')

    # Extract frames from video
    video_to_images(args.video_path, full_imgs_path, ext='.png')

    # Get list of extracted images
    input_img_list = sorted(glob(os.path.join(full_imgs_path, '*.png')))
    if not input_img_list:
        input_img_list = sorted(glob(os.path.join(full_imgs_path, '*.[jpJP][pnPN]*[gG]')))

    if not input_img_list:
        raise FileNotFoundError('No images found in the specified directory.')

    # Read images
    frames = read_images(input_img_list)

    if not frames:
        raise ValueError('No frames read from images.')

    # Face detection and cropping
    face_det_results = detect_faces(frames, device=device, batch_size=args.face_det_batch_size,
                                    pads=tuple(args.pads), smooth=not args.nosmooth)

    coord_list = []
    for idx, (cropped_face, coords) in enumerate(face_det_results):
        resized_crop_face = cv2.resize(cropped_face, (args.img_size, args.img_size))
        cv2.imwrite(os.path.join(face_imgs_path, f"{idx:08d}.png"), resized_crop_face)
        coord_list.append(coords)

    # Save coordinates
    with open(coords_path, 'wb') as f:
        pickle.dump(coord_list, f)

    print(f'Processing complete. Results saved in {avatar_path}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a video to extract face images for lip-syncing using Wav2Lip models.')
    parser.add_argument('--img_size', default=96, type=int, help='Size of the cropped face images (default: 96)')
    parser.add_argument('--avatar_id', default='wav2lip_avatar1', type=str, help='ID for the avatar (default: wav2lip_avatar1)')
    parser.add_argument('--video_path', required=True, type=str, help='Path to the input video')
    parser.add_argument('--nosmooth', action='store_true', help='Disable smoothing of face detections over time')
    parser.add_argument('--pads', nargs=4, type=int, default=[0, 10, 0, 0],
                        help='Padding (top, bottom, left, right). Adjust to include chin at least (default: [0, 10, 0, 0])')
    parser.add_argument('--face_det_batch_size', type=int, default=16, help='Batch size for face detection (default: 16)')
    args = parser.parse_args()

    main(args)
