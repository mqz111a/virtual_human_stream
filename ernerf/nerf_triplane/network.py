import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..encoding import get_encoder
from .renderer import NeRFRenderer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("nerf_network.log", mode='w')
    ]
)
logger = logging.getLogger(__name__)


class AudioAttNet(nn.Module):
    """
    Audio Attention Network using convolutional layers followed by a linear softmax layer.
    Applies attention over the temporal dimension of audio features.
    """

    def __init__(self, dim_aud: int = 64, seq_len: int = 8):
        super(AudioAttNet, self).__init__()
        self.seq_len = seq_len
        self.dim_aud = dim_aud

        self.attention_conv_net = nn.Sequential(
            nn.Conv1d(in_channels=self.dim_aud, out_channels=16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Conv1d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Conv1d(in_channels=8, out_channels=4, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Conv1d(in_channels=4, out_channels=2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Conv1d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, inplace=True)
        )

        self.attention_net = nn.Sequential(
            nn.Linear(in_features=self.seq_len, out_features=self.seq_len, bias=True),
            nn.Softmax(dim=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the AudioAttNet.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, dim_aud].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, dim_aud].
        """
        # Permute to [batch_size, dim_aud, seq_len] for Conv1d
        y = x.permute(0, 2, 1)
        y = self.attention_conv_net(y)
        # Reshape for attention_net: [batch_size, seq_len]
        y = self.attention_net(y.view(y.size(0), self.seq_len)).view(y.size(0), self.seq_len, 1)
        # Weighted sum over the sequence length dimension
        return torch.sum(y * x, dim=1)  # [batch_size, dim_aud]


class AudioNet(nn.Module):
    """
    Audio Feature Extractor Network using convolutional layers followed by fully connected layers.
    Extracts meaningful features from raw audio inputs.
    """

    def __init__(self, dim_in: int = 29, dim_aud: int = 64, win_size: int = 16):
        super(AudioNet, self).__init__()
        self.win_size = win_size
        self.dim_aud = dim_aud

        self.encoder_conv = nn.Sequential(
            nn.Conv1d(in_channels=dim_in, out_channels=32, kernel_size=3, stride=2, padding=1, bias=True),
            # [batch, 32, 8]
            nn.LeakyReLU(0.02, inplace=True),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1, bias=True),  # [batch, 32, 4]
            nn.LeakyReLU(0.02, inplace=True),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1, bias=True),  # [batch, 64, 2]
            nn.LeakyReLU(0.02, inplace=True),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=True),  # [batch, 64, 1]
            nn.LeakyReLU(0.02, inplace=True),
        )

        self.encoder_fc1 = nn.Sequential(
            nn.Linear(in_features=64, out_features=64),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Linear(in_features=64, out_features=dim_aud),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the AudioNet.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, dim_in, win_size].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, dim_aud].
        """
        half_w = self.win_size // 2
        # Center crop the input
        x = x[:, :, 8 - half_w:8 + half_w]
        x = self.encoder_conv(x).squeeze(-1)
        x = self.encoder_fc1(x)
        return x


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) with configurable layers and dimensions.
    """

    def __init__(self, dim_in: int, dim_out: int, dim_hidden: int, num_layers: int):
        super(MLP, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        layers = []
        for layer_idx in range(num_layers):
            in_dim = dim_in if layer_idx == 0 else dim_hidden
            out_dim = dim_out if layer_idx == num_layers - 1 else dim_hidden
            layers.append(nn.Linear(in_features=in_dim, out_features=out_dim, bias=False))
            if layer_idx < num_layers - 1:
                layers.append(nn.ReLU(inplace=True))
                # Optional dropout can be added here if needed
                # layers.append(nn.Dropout(p=0.1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MLP.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.net(x)


class NeRFNetwork(NeRFRenderer):
    """
    Neural Radiance Fields (NeRF) Network incorporating audio features and optional torso deformation.
    Extends the NeRFRenderer with additional audio and torso processing capabilities.
    """

    def __init__(self, opt):
        super(NeRFNetwork, self).__init__(opt)

        # Audio embedding configuration
        self.emb = opt.emb
        self.audio_in_dim = self._determine_audio_input_dim(opt.asr_model)
        self.embedding = nn.Embedding(num_embeddings=self.audio_in_dim,
                                      embedding_dim=self.audio_in_dim) if self.emb else None

        # Audio networks
        self.audio_dim = 32  # Can be made configurable
        self.audio_net = AudioNet(dim_in=self.audio_in_dim, dim_aud=self.audio_dim)

        # Optional Audio Attention Network
        self.att = opt.att
        self.audio_att_net = AudioAttNet(dim_aud=self.audio_dim, seq_len=opt.seq_len) if self.att > 0 else None

        # Encoders for spatial coordinates
        self.num_levels = 12
        self.level_dim = 1
        self.bound = opt.bound  # Assuming 'bound' is defined in opt
        self.encoder_xy, self.in_dim_xy = get_encoder(
            encoder_type='hashgrid',
            input_dim=2,
            num_levels=self.num_levels,
            level_dim=self.level_dim,
            base_resolution=64,
            log2_hashmap_size=14,
            desired_resolution=512 * self.bound
        )
        self.encoder_yz, self.in_dim_yz = get_encoder(
            encoder_type='hashgrid',
            input_dim=2,
            num_levels=self.num_levels,
            level_dim=self.level_dim,
            base_resolution=64,
            log2_hashmap_size=14,
            desired_resolution=512 * self.bound
        )
        self.encoder_xz, self.in_dim_xz = get_encoder(
            encoder_type='hashgrid',
            input_dim=2,
            num_levels=self.num_levels,
            level_dim=self.level_dim,
            base_resolution=64,
            log2_hashmap_size=14,
            desired_resolution=512 * self.bound
        )

        self.in_dim = self.in_dim_xy + self.in_dim_yz + self.in_dim_xz

        # Sigma (density) network
        self.num_layers_sigma = 3
        self.hidden_dim_sigma = 64
        self.geo_feat_dim = 64
        self.eye_att_net = MLP(dim_in=self.in_dim, dim_out=1, dim_hidden=16, num_layers=2)
        self.eye_dim = 1 if getattr(opt, 'exp_eye', False) else 0
        self.sigma_net = MLP(dim_in=self.in_dim + self.audio_dim + self.eye_dim, dim_out=1 + self.geo_feat_dim,
                             dim_hidden=self.hidden_dim_sigma, num_layers=self.num_layers_sigma)

        # Color network
        self.num_layers_color = 2
        self.hidden_dim_color = 64
        self.encoder_dir, self.in_dim_dir = get_encoder(
            encoder_type='spherical_harmonics',
            input_dim=3  # Assuming direction is 3D
        )
        self.color_net = MLP(dim_in=self.in_dim_dir + self.geo_feat_dim + getattr(opt, 'individual_dim', 0),
                             dim_out=3,
                             dim_hidden=self.hidden_dim_color,
                             num_layers=self.num_layers_color)

        # Uncertainty network
        self.unc_net = MLP(dim_in=self.in_dim, dim_out=1, dim_hidden=32, num_layers=2)

        # Audio channel attention network
        self.aud_ch_att_net = MLP(dim_in=self.in_dim, dim_out=self.audio_dim, dim_hidden=64, num_layers=2)

        # Optional torso deformation networks
        self.torso = getattr(opt, 'torso', False)
        if self.torso:
            self._initialize_torso_network(opt)

        self.testing = False

    def _determine_audio_input_dim(self, asr_model: str) -> int:
        """
        Determines the audio input dimension based on the ASR model name.

        Args:
            asr_model (str): Name of the ASR model.

        Returns:
            int: Audio input dimension.
        """
        if 'esperanto' in asr_model:
            return 44
        elif 'deepspeech' in asr_model:
            return 29
        elif 'hubert' in asr_model:
            return 1024
        else:
            return 32

    def _initialize_torso_network(self, opt):
        """
        Initializes the torso deformation and color networks.

        Args:
            opt: Configuration options.
        """
        # Torso deformation network
        self.register_parameter(
            'anchor_points',
            nn.Parameter(torch.tensor([[0.01, 0.01, 0.1, 1.0],
                                       [-0.1, -0.1, 0.1, 1.0],
                                       [0.1, -0.1, 0.1, 1.0]]))
        )
        self.torso_deform_encoder, self.torso_deform_in_dim = get_encoder(
            encoder_type='frequency',
            input_dim=2,
            multires=8
        )
        self.anchor_encoder, self.anchor_in_dim = get_encoder(
            encoder_type='frequency',
            input_dim=6,
            multires=3
        )
        self.torso_deform_net = MLP(
            dim_in=self.torso_deform_in_dim + self.anchor_in_dim + getattr(opt, 'individual_dim_torso', 0),
            dim_out=2,
            dim_hidden=32,
            num_layers=3)

        # Torso color network
        self.torso_encoder, self.torso_in_dim = get_encoder(
            encoder_type='tiledgrid',
            input_dim=2,
            num_levels=16,
            level_dim=2,
            base_resolution=16,
            log2_hashmap_size=16,
            desired_resolution=2048
        )
        self.torso_net = MLP(dim_in=self.torso_in_dim + self.torso_deform_in_dim + self.anchor_in_dim + getattr(opt,
                                                                                                                'individual_dim_torso',
                                                                                                                0),
                             dim_out=4,
                             dim_hidden=32,
                             num_layers=3)

    def forward_torso(self, x: torch.Tensor, poses: torch.Tensor, c: Optional[torch.Tensor] = None) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for torso deformation and color prediction.

        Args:
            x (torch.Tensor): Spatial coordinates tensor of shape [N, 2], normalized to [-1, 1].
            poses (torch.Tensor): Pose matrices tensor of shape [1, 4, 4].
            c (Optional[torch.Tensor]): Individual code tensor of shape [1, ind_dim].

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Alpha, color, and deformation tensors.
        """
        # Shrink coordinates
        x = x * self.opt.torso_shrink

        # Deformation based on pose
        wrapped_anchor = self.anchor_points.unsqueeze(0) @ torch.inverse(poses).permute(0, 2, 1)
        wrapped_anchor = (wrapped_anchor[:, :, :2] / wrapped_anchor[:, :, 3].unsqueeze(-1) / wrapped_anchor[:, :,
                                                                                             2].unsqueeze(-1)).view(
            x.size(0), -1)

        enc_anchor = self.anchor_encoder(wrapped_anchor)
        enc_x = self.torso_deform_encoder(x)

        if c is not None:
            h = torch.cat([enc_x, enc_anchor.repeat(x.size(0), 1), c.repeat(x.size(0), 1)], dim=-1)
        else:
            h = torch.cat([enc_x, enc_anchor.repeat(x.size(0), 1)], dim=-1)

        dx = self.torso_deform_net(h)
        x = (x + dx).clamp(-1, 1)

        x = self.torso_encoder(x, bound=1)

        h = torch.cat([x, h], dim=-1)
        h = self.torso_net(h)

        alpha = torch.sigmoid(h[..., :1]) * (1 + 2 * 0.001) - 0.001
        color = torch.sigmoid(h[..., 1:]) * (1 + 2 * 0.001) - 0.001

        return alpha, color, dx

    @staticmethod
    @torch.jit.script
    def split_xyz(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Splits the input tensor into xy, yz, and xz components.

        Args:
            x (torch.Tensor): Input tensor of shape [N, 3].

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Split tensors.
        """
        xy = x[:, :-1]
        yz = x[:, 1:]
        xz = torch.cat([x[:, :1], x[:, -1:]], dim=-1)
        return xy, yz, xz

    def encode_x(self, xyz: torch.Tensor, bound: float) -> torch.Tensor:
        """
        Encodes spatial coordinates using hashgrid encoders.

        Args:
            xyz (torch.Tensor): Spatial coordinates tensor of shape [N, 3], normalized to [-bound, bound].
            bound (float): Boundary for normalization.

        Returns:
            torch.Tensor: Encoded features tensor.
        """
        xy, yz, xz = self.split_xyz(xyz)
        feat_xy = self.encoder_xy(xy, bound=bound)
        feat_yz = self.encoder_yz(yz, bound=bound)
        feat_xz = self.encoder_xz(xz, bound=bound)

        return torch.cat([feat_xy, feat_yz, feat_xz], dim=-1)

    def encode_audio(self, a: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """
        Encodes audio features.

        Args:
            a (Optional[torch.Tensor]): Audio features tensor of shape [batch_size, dim_aud].

        Returns:
            Optional[torch.Tensor]: Encoded audio features tensor.
        """
        if a is None:
            return None

        if self.emb and self.embedding is not None:
            a = self.embedding(a).transpose(-1, -2).contiguous()  # [batch_size, dim_aud, ...]

        enc_a = self.audio_net(a)  # [batch_size, audio_dim]

        if self.att and self.audio_att_net is not None:
            enc_a = self.audio_att_net(enc_a.unsqueeze(0))  # [batch_size, audio_dim]

        return enc_a

    def predict_uncertainty(self, unc_inp: torch.Tensor) -> torch.Tensor:
        """
        Predicts uncertainty based on the input.

        Args:
            unc_inp (torch.Tensor): Input tensor for uncertainty prediction.

        Returns:
            torch.Tensor: Uncertainty tensor.
        """
        if self.testing or not getattr(self.opt, 'unc_loss', False):
            return torch.zeros_like(unc_inp)
        else:
            return torch.log(1 + torch.exp(self.unc_net(unc_inp.detach())))

    def forward(self,
                x: torch.Tensor,
                d: torch.Tensor,
                enc_a: torch.Tensor,
                c: Optional[torch.Tensor] = None,
                e: Optional[torch.Tensor] = None) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for the NeRFNetwork.

        Args:
            x (torch.Tensor): Spatial coordinates tensor of shape [N, 3], normalized to [-bound, bound].
            d (torch.Tensor): Direction tensor of shape [N, 3], normalized to [-1, 1].
            enc_a (torch.Tensor): Encoded audio features tensor of shape [batch_size, audio_dim].
            c (Optional[torch.Tensor]): Individual code tensor of shape [batch_size, ind_dim].
            e (Optional[torch.Tensor]): Eye feature tensor of shape [batch_size, 1].

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                sigma, color, aud_ch_att, eye_att, uncertainty.
        """
        enc_x = self.encode_x(x, bound=self.bound)

        sigma_result = self.density(x, enc_a, e, enc_x)
        sigma = sigma_result['sigma']
        geo_feat = sigma_result['geo_feat']
        aud_ch_att = sigma_result['ambient_aud']
        eye_att = sigma_result['ambient_eye']

        # Encode direction
        enc_d = self.encoder_dir(d)

        if c is not None:
            h = torch.cat([enc_d, geo_feat, c.repeat(x.size(0), 1)], dim=-1)
        else:
            h = torch.cat([enc_d, geo_feat], dim=-1)

        # Predict color
        h_color = self.color_net(h)
        color = torch.sigmoid(h_color) * (1 + 2 * 0.001) - 0.001

        # Predict uncertainty
        uncertainty = self.predict_uncertainty(enc_x)
        uncertainty = uncertainty.unsqueeze(-1)  # [N, 1]

        return sigma, color, aud_ch_att, eye_att, uncertainty

    def density(self,
                x: torch.Tensor,
                enc_a: torch.Tensor,
                e: Optional[torch.Tensor],
                enc_x: Optional[torch.Tensor] = None) -> dict:
        """
        Computes density and geometric features.

        Args:
            x (torch.Tensor): Spatial coordinates tensor of shape [N, 3], normalized to [-bound, bound].
            enc_a (torch.Tensor): Encoded audio features tensor of shape [batch_size, audio_dim].
            e (Optional[torch.Tensor]): Eye feature tensor of shape [batch_size, 1].
            enc_x (Optional[torch.Tensor]): Encoded spatial features tensor of shape [N, in_dim].

        Returns:
            dict: Dictionary containing 'sigma', 'geo_feat', 'ambient_aud', and 'ambient_eye'.
        """
        if enc_x is None:
            enc_x = self.encode_x(x, bound=self.bound)

        enc_a_expanded = enc_a.repeat(x.size(0), 1)  # [N, audio_dim]
        aud_ch_att = self.aud_ch_att_net(enc_x)  # [N, audio_dim]
        enc_w = enc_a_expanded * aud_ch_att  # [N, audio_dim]

        if e is not None:
            eye_att = torch.sigmoid(self.eye_att_net(enc_x))  # [N, 1]
            e = e * eye_att  # [N, 1]
            h = torch.cat([enc_x, enc_w, e], dim=-1)  # [N, in_dim + audio_dim + eye_dim]
        else:
            h = torch.cat([enc_x, enc_w], dim=-1)  # [N, in_dim + audio_dim]

        h = self.sigma_net(h)  # [N, 1 + geo_feat_dim]

        sigma = torch.exp(h[..., 0])  # [N, 1]
        geo_feat = h[..., 1:]  # [N, geo_feat_dim]

        ambient_aud = aud_ch_att.norm(dim=-1, keepdim=True)  # [N, 1]
        ambient_eye = torch.sigmoid(self.eye_att_net(enc_x)) if e is not None else torch.zeros(x.size(0), 1,
                                                                                               device=x.device)

        return {
            'sigma': sigma,
            'geo_feat': geo_feat,
            'ambient_aud': ambient_aud,
            'ambient_eye': ambient_eye,
        }

    def get_params(self, lr: float, lr_net: float, wd: float = 0.0) -> list:
        """
        Retrieves parameters for optimization, separating them based on different learning rates and weight decay.

        Args:
            lr (float): Learning rate for specific parameter groups.
            lr_net (float): Learning rate for neural network parameters.
            wd (float, optional): Weight decay. Defaults to 0.0.

        Returns:
            list: List of parameter dictionaries for the optimizer.
        """
        params = []

        # Torso parameters
        if self.torso:
            params.extend([
                {'params': self.torso_encoder.parameters(), 'lr': lr},
                {'params': self.torso_deform_encoder.parameters(), 'lr': lr, 'weight_decay': wd},
                {'params': self.torso_net.parameters(), 'lr': lr_net, 'weight_decay': wd},
                {'params': self.torso_deform_net.parameters(), 'lr': lr_net, 'weight_decay': wd},
                {'params': self.anchor_points, 'lr': lr_net, 'weight_decay': wd}
            ])
            if hasattr(self, 'individual_codes_torso') and self.individual_dim_torso > 0:
                params.append({'params': self.individual_codes_torso, 'lr': lr_net, 'weight_decay': wd})

            return params

        # General parameters
        params.extend([
            {'params': self.audio_net.parameters(), 'lr': lr_net, 'weight_decay': wd},
            {'params': self.encoder_xy.parameters(), 'lr': lr},
            {'params': self.encoder_yz.parameters(), 'lr': lr},
            {'params': self.encoder_xz.parameters(), 'lr': lr},
            {'params': self.sigma_net.parameters(), 'lr': lr_net, 'weight_decay': wd},
            {'params': self.color_net.parameters(), 'lr': lr_net, 'weight_decay': wd},
        ])

        if self.att and self.audio_att_net is not None:
            params.append({'params': self.audio_att_net.parameters(), 'lr': lr_net * 5, 'weight_decay': 0.0001})

        if self.emb and self.embedding is not None:
            params.append({'params': self.embedding.parameters(), 'lr': lr})

        if hasattr(self, 'individual_dim') and self.individual_dim > 0:
            params.append({'params': self.individual_codes, 'lr': lr_net, 'weight_decay': wd})

        if getattr(self, 'train_camera', False):
            params.extend([
                {'params': self.camera_dT, 'lr': 1e-5, 'weight_decay': 0.0},
                {'params': self.camera_dR, 'lr': 1e-5, 'weight_decay': 0.0}
            ])

        params.extend([
            {'params': self.aud_ch_att_net.parameters(), 'lr': lr_net, 'weight_decay': wd},
            {'params': self.unc_net.parameters(), 'lr': lr_net, 'weight_decay': wd},
            {'params': self.eye_att_net.parameters(), 'lr': lr_net, 'weight_decay': wd},
        ])

        return params
