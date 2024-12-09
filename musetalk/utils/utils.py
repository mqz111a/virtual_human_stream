import os
import cv2
import numpy as np
import torch
from musetalk.whisper.audio2feature import Audio2Feature
from musetalk.models.vae import VAE
from musetalk.models.unet import UNet,PositionalEncoding

def check_ffmpeg_path():
    ffmpeg_path = os.getenv('FFMPEG_PATH')
    if ffmpeg_path is None:
        print("please download ffmpeg-static and export to FFMPEG_PATH. \nFor example: export FFMPEG_PATH=/musetalk/ffmpeg-4.4-amd64-static")
    elif ffmpeg_path not in os.getenv('PATH'):
        print("add ffmpeg to path")
        os.environ["PATH"] = f"{ffmpeg_path}:{os.environ['PATH']}"

def load_model(model_class, model_path, **kwargs):
    return model_class(model_path=model_path, **kwargs)

def load_all_model():
    audio_processor = load_model(Audio2Feature, "./models/whisper/tiny.pt")
    vae = load_model(VAE, "./models/sd-vae-ft-mse/")
    unet = load_model(UNet, "./models/musetalk/musetalk.json", model_path ="./models/musetalk/pytorch_model.bin")
    pe = PositionalEncoding(d_model=384)
    return audio_processor, vae, unet, pe

def get_file_type(video_path):
    _, ext = os.path.splitext(video_path)
    file_types = {
        '.jpg': 'image',
        '.jpeg': 'image',
        '.png': 'image',
        '.bmp': 'image',
        '.tif': 'image',
        '.tiff': 'image',
        '.avi': 'video',
        '.mp4': 'video',
        '.mov': 'video',
        '.flv': 'video',
        '.mkv': 'video'
    }
    return file_types.get(ext.lower(), 'unsupported')

def get_video_fps(video_path):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    video.release()
    return fps

def datagen(whisper_chunks, vae_encode_latents, batch_size=8, delay_frame=0):
    whisper_batch, latent_batch = [], []
    for i, w in enumerate(whisper_chunks):
        idx = (i+delay_frame)%len(vae_encode_latents)
        latent = vae_encode_latents[idx]
        whisper_batch.append(w)
        latent_batch.append(latent)

        if len(latent_batch) >= batch_size:
            whisper_batch = np.stack(whisper_batch)
            latent_batch = torch.cat(latent_batch, dim=0)
            yield whisper_batch, latent_batch
            whisper_batch, latent_batch = [], []

    # the last batch may smaller than batch size
    if len(latent_batch) > 0:
        whisper_batch = np.stack(whisper_batch)
        latent_batch = torch.cat(latent_batch, dim=0)
        yield whisper_batch, latent_batch

def load_audio_model():
    return load_model(Audio2Feature, "./models/whisper/tiny.pt")

def load_diffusion_model():
    vae = load_model(VAE, "./models/sd-vae-ft-mse/")
    unet = load_model(UNet, "./models/musetalk/musetalk.json", model_path ="./models/musetalk/pytorch_model.bin")
    pe = PositionalEncoding(d_model=384)
    return vae, unet, pe