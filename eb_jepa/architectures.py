# eb_jepa/architectures.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.models import mobilenet_v2
from torchvision.models.mobilenetv2 import InvertedResidual
from torchvision.ops import Conv2dNormActivation
import timm



class Projector(nn.Module):
    """MLP projector built from a spec string like '256-512-128'."""

    def __init__(self, mlp_spec):
        super().__init__()
        layers = []
        f = list(map(int, mlp_spec.split("-")))
        for i in range(len(f) - 2):
            layers.append(nn.Linear(f[i], f[i + 1]))
            layers.append(nn.BatchNorm1d(f[i + 1]))
            layers.append(nn.ReLU(True))
        layers.append(nn.Linear(f[-2], f[-1], bias=False))
        self.net = nn.Sequential(*layers)
        self.out_dim = f[-1]  # Store output dimension as attribute

    def forward(self, x):
        return self.net(x)


class CustomInvertedResidual(nn.Module):
    """InvertedResidual block supporting custom kernel size and padding."""

    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int = 1,
        expand_ratio: float = 6,
        kernel_size: int = 2,
        padding: int = 1,
    ):
        super().__init__()
        self.stride = stride
        self.use_res_connect = self.stride == 1 and inp == oup

        hidden_dim = int(round(inp * expand_ratio))
        layers = []

        # 1. Expansion phase (1x1 Conv)
        if expand_ratio != 1:
            layers.append(
                Conv2dNormActivation(
                    inp,
                    hidden_dim,
                    kernel_size=1,
                    norm_layer=nn.BatchNorm2d,
                    activation_layer=nn.ReLU6,
                )
            )

        # 2. Depthwise phase (custom kernel_size x kernel_size Conv)
        layers.append(
            Conv2dNormActivation(
                hidden_dim,
                hidden_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=hidden_dim,
                norm_layer=nn.BatchNorm2d,
                activation_layer=nn.ReLU6,
            )
        )

        # 3. Projection phase (Linear 1x1 Conv)
        layers.extend([
            nn.Conv2d(hidden_dim, oup, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(oup),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV2Encoder(nn.Module):
    """MobileNetV2 image encoder with custom layer 18 replacement."""

    def __init__(self, width_mult: float = 1.0):
        super().__init__()

        backbone = mobilenet_v2(weights=None, width_mult=width_mult)

        # Channel output from block 16 is 160 (at width_mult=1.0)
        in_channels = int(160 * width_mult)
        out_channels = in_channels * 2

        # Replace layer (17) with InvertedResidual stride=2 to downsample
        backbone.features[17] = InvertedResidual(
            inp=in_channels,
            oup=out_channels,
            stride=2,
            expand_ratio=6,
        )

        in_channels = out_channels
        out_channels = in_channels * 2

        # Replace layer (18) with custom InvertedResidual where kernel_size=2, padding=0 to have a 1x1 output
        backbone.features[18] = CustomInvertedResidual(
            inp=in_channels,
            oup=out_channels,
            stride=1,
            expand_ratio=6,
            kernel_size=2,
            padding=0,
        )

        in_channels = out_channels
        out_channels = in_channels * 2
        
        backbone.features.append(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )

        self.features = backbone.features

        self.feature_dim = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x).flatten(1)


class EfficientSpatialProjection(nn.Module):
    """
    Collapses spatial map (H x W -> 1 x 1) using a learned Depthwise Conv,
    followed by a Pointwise 1x1 Conv to project channels (in_channels -> out_channels).
    """
    def __init__(self, in_channels: int = 960, out_channels: int = 1280, spatial_size: tuple = (4, 4)):
        super().__init__()
        # 1. Depthwise Conv: Learns spatial weighting to collapse H x W -> 1 x 1
        self.dw_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=spatial_size,
            stride=spatial_size,
            groups=in_channels,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.act1 = nn.SiLU(inplace=True)
        
        # 2. Pointwise Conv: Efficiently expands 960 channels -> 1280
        self.pw_conv = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=1, 
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.bn1(self.dw_conv(x)))
        x = self.act2(self.bn2(self.pw_conv(x)))
        return x  # Output Shape: [B, 1280, 1, 1]


class MobileNetV4Encoder(nn.Module):
    """MobileNetV4 image encoder using timm feature extraction."""
    
    def __init__(self, width_mult: float = 1.0, img_size: int = 128):
        super().__init__()
        
        # 1. Store timm backbone as an internal attribute
        self.backbone = timm.create_model(
            'mobilenetv4_conv_medium', 
            pretrained=False, 
            features_only=True,
            out_indices=(4,)
        )
        
        # Get the feature channel count for stage 4 dynamically
        in_channels = self.backbone.feature_info.channels()[-1]  # 960 for mnv4 small
        
        # Determine spatial feature size (Stage 5 downsamples input by 32x)
        spatial_dim = img_size // 32  # For 128x128 -> 4x4
        
        # 2. Add EfficientSpatialProjection to collapse spatial dimensions and project channels
        self.projection = EfficientSpatialProjection(
            in_channels=in_channels,
            out_channels=1280,
            spatial_size=(spatial_dim, spatial_dim)  # Assuming final feature map is 4x4
        )

        self.feature_dim = 1280  # Final embedding dimension after projection

        # Verify parameters
        num_params = sum(p.numel() for p in self.backbone.parameters())
        print(f"Backbone Parameters: {num_params / 1e6:.2f}M")
        num_params = sum(p.numel() for p in self.projection.parameters())
        print(f"Projection Parameters: {num_params / 1e6:.2f}M")

    def forward(self, x: torch.Tensor):
        # Extract 960-channel feature map: [B, 960, H/32, W/32]
        feat_map = self.backbone(x)[0]  # Shape: (B, 960, H/32, W/32) -> (B, 960, 4, 4) for 128x128
        
        # Project spatially & channel-wise: [B, 1280, 1, 1]
        embedding = self.projection(feat_map)
        return embedding.flatten(1)  # Return shape: [B, 1280]


class LinearProbeHead(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)
    def forward(self, feat_map):          # [B, C, H, W]
        pooled = F.adaptive_avg_pool2d(feat_map, (1, 1))  # [B, C, 1, 1]
        flattened = pooled.flatten(1)
        return self.fc(flattened)


class GatedPredictor(nn.Module):
    """
    GatedPredictor with independent reset gates (2D) for state mean (mu) and variance (log_var),
    and a single update gate (1D) strictly for state transition.
    
    Dimension Breakdown:
    - z (update gate): D    -> Used exclusively for mu residual transition
    - r (reset gates): 2D   -> Independent reset masks for (r_mu, r_var)
    - c (candidates):  2D   -> Raw proposal features for (c_mu, c_var)
    """
    def __init__(self, state_size: int, action_size: int, bias: bool = True):
        super().__init__()
        self.state_size = state_size
        
        # Parallel action projections: z (D) + r (2D) + c (2D) = 5 * state_size
        self.W_a = nn.Linear(action_size, 5 * state_size, bias=bias)
        
        # State gate projections: z (D) + r (2D) = 3 * state_size
        self.U_zr = nn.Linear(state_size, 3 * state_size, bias=bias)
        
        # Candidate transformation operating on the masked 2D space
        self.U_h = nn.Linear(2 * state_size, 2 * state_size, bias=bias)
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_a.weight)
        nn.init.xavier_uniform_(self.U_zr.weight)
        nn.init.xavier_uniform_(self.U_h.weight)
        
        if self.W_a.bias is not None:
            nn.init.zeros_(self.W_a.bias)
            nn.init.zeros_(self.U_zr.bias)
            nn.init.zeros_(self.U_h.bias)

    def forward(self, ref_state: torch.Tensor, action: torch.Tensor):
        # 1. Action projections: split into z (D), r (2D), c (2D)
        W_a_z, W_a_r, W_a_c = self.W_a(action).split(
            [self.state_size, 2 * self.state_size, 2 * self.state_size], dim=-1
        )

        # 2. State gate projections: split into z (D) and r (2D)
        U_z, U_r = self.U_zr(ref_state).split(
            [self.state_size, 2 * self.state_size], dim=-1
        )

        # 3. Compute gate activations
        z = torch.sigmoid(W_a_z + U_z)       # (batch, D)  - Update gate for mu only
        r = torch.sigmoid(W_a_r + U_r)       # (batch, 2D) - Independent reset gates (r_mu, r_var)

        # 4. Apply 2D reset gate to concatenated reference state [ref_state, ref_state]
        ref_state_2d = torch.cat([ref_state, ref_state], dim=-1)
        candidate_raw = W_a_c + self.U_h(r * ref_state_2d) # (batch, 2D)

        # 5. Separate candidate outputs into mu and log_var pathways
        c_mu, c_var = candidate_raw.chunk(2, dim=-1)

        # --- Pathway A: Mean State (mu) ---
        # Gated residual state transition with tanh activation
        mu = (1 - z) * ref_state + z * torch.tanh(c_mu)

        # --- Pathway B: Log Variance (log_var) ---
        # Direct linear feature output, clamped for NLL loss numerical stability
        log_var = torch.clamp(c_var, min=-10.0, max=5.0)

        return mu, log_var


class ConvProbeHead(nn.Module):
    def __init__(self, in_channels, num_classes, out_channels=None, expand_ratio=6):
        super().__init__()
        out_channels = out_channels or in_channels
        self.block = InvertedResidual(in_channels, out_channels, stride=2, expand_ratio=expand_ratio)
        self.fc = nn.Linear(out_channels, num_classes)
    def forward(self, feat_map):
        new_feat_map = self.block(feat_map)
        pooled = F.adaptive_avg_pool2d(new_feat_map, (1, 1))
        flattened = pooled.flatten(1)
        return self.fc(flattened)


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.drop = nn.Dropout(dropout)
        self.shortcut = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        res = self.shortcut(x)
        x = self.act(self.norm(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return self.act(x + res)


class InverseDynamicsModel(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int):
        super().__init__()
        # Input dim is 2 * emb_dim due to pairwise fusion [diff, prod]
        in_dim = state_dim * 2
        
        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        self.res1 = ResidualBlock(hidden_dim, hidden_dim)
        self.res2 = ResidualBlock(hidden_dim, hidden_dim // 2)
        
        self.head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, action_dim)
        )

    def _forward_single(self, z1, z2):
        diff = z1 - z2
        prod = z1 * z2
        h = torch.cat([diff, prod], dim=-1)
        
        feat = self.in_proj(h)
        feat = self.res1(feat)
        feat = self.res2(feat)
        return self.head(feat)

    def forward(self, z1, z2):
        # Enforce anti-symmetry: f(z1, z2) == -f(z2, z1)
        pos = self._forward_single(z1, z2)
        neg = self._forward_single(z2, z1)
        return 0.5 * (pos - neg)


class FourierPositionalEncoding(nn.Module):
    """Maps low-dim continuous coordinates to high-dim spatial frequencies."""
    def __init__(self, input_dim: int = 2, num_bands: int = 16, max_freq: float = 10.0):
        super().__init__()
        scales = torch.logspace(0, np.log10(max_freq), num_bands)
        self.register_buffer("scales", scales)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 2] -> [B, 2, num_bands]
        x_proj = x.unsqueeze(-1) * self.scales * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1).flatten(start_dim=1)


class ForwardDynamicsModel(nn.Module):
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        num_bands: int = 16,
        min_log_var: float = -6.0,
        max_log_var: float = 2.0
    ):
        super().__init__()
        self.min_log_var = min_log_var
        self.max_log_var = max_log_var
        
        self.pe = FourierPositionalEncoding(input_dim=2, num_bands=num_bands)
        pos_dim = 2 * num_bands * 2
        
        self.pos_proj = nn.Sequential(
            nn.Linear(pos_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )
        
        in_dim = state_dim + (hidden_dim // 2)
        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        self.res1 = ResidualBlock(hidden_dim, hidden_dim)
        self.res2 = ResidualBlock(hidden_dim, hidden_dim)
        
        # Dual output heads: Mean offset (Delta z) and Log-Variance
        self.mean_head = nn.Linear(hidden_dim, state_dim)
        self.log_var_head = nn.Linear(hidden_dim, state_dim)

    def forward(self, z1: torch.Tensor, delta_xy: torch.Tensor):
        pos_feat = self.pos_proj(self.pe(delta_xy))
        h = torch.cat([z1, pos_feat], dim=-1)
        h = self.in_proj(h)
        h = self.res1(h)
        h = self.res2(h)
        
        # Mean prediction (Residual shift)
        mu = z1 + self.mean_head(h)
        
        # Bounded log-variance for stability
        log_var = self.log_var_head(h)
        log_var = torch.clamp(log_var, self.min_log_var, self.max_log_var)
        var = torch.exp(log_var)
        
        return mu, var


if __name__ == "__main__":
    encoder = MobileNetV4Encoder()
    
    dummy_input = torch.rand(2, 3, 128, 128)
    dummy_outputs = encoder(dummy_input)
    
    print(f"Dummy Input Shape: {dummy_input.shape}")
    print(f"Dummy Output Shape: {dummy_outputs.shape}")  # Expected: [2, 1280, 1, 1]