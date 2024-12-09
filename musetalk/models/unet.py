import torch
import torch.nn as nn
import math
import json

from diffusers import UNet2DConditionModel
import sys
import time
import numpy as np
import os


class PositionalEncoding(nn.Module):
    def __init__(self, d_model=384, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.pe = self.calculate_pe(d_model, max_len)

    def calculate_pe(self, d_model, max_len):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        return pe

    def forward(self, x):
        b, seq_len, d_model = x.size()
        pe = self.pe[:, :seq_len, :]
        x = x + pe.to(x.device)
        return x


class UNet():
    def __init__(self,
                 unet_config,
                 model_path,
                 use_float16=False,
                 ):
        self.unet_config = unet_config
        self.model_path = model_path
        self.use_float16 = use_float16
        self.load_model_and_device()

    def load_model_and_device(self):
        with open(self.unet_config, 'r') as f:
            unet_config = json.load(f)
        self.model = UNet2DConditionModel(**unet_config)
        self.pe = PositionalEncoding(d_model=384)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weights = torch.load(self.model_path) if torch.cuda.is_available() else torch.load(self.model_path,
                                                                                           map_location=self.device)
        self.model.load_state_dict(weights)
        if self.use_float16:
            self.model = self.model.half()
        self.model.to(self.device)


if __name__ == "__main__":
    unet = UNet()