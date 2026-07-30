# eb_jepa/architectures.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2
from torchvision.models.mobilenetv2 import InvertedResidual
from torchvision.ops import Conv2dNormActivation
    
from eb_jepa.nn_utils import init_module_weights


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
        batch_norm: bool = True
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
                    norm_layer=nn.BatchNorm2d if batch_norm else nn.Identity,
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
                norm_layer=nn.BatchNorm2d if batch_norm else nn.Identity,
                activation_layer=nn.ReLU6,
            )
        )

        # 3. Projection phase (Linear 1x1 Conv)
        layers.extend([
            nn.Conv2d(hidden_dim, oup, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(oup) if batch_norm else nn.Identity(oup),
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

        # Replace layer (17) with custom InvertedResidual
        backbone.features[17] = CustomInvertedResidual(
            inp=in_channels,
            oup=out_channels,
            stride=2,
            expand_ratio=6,
            kernel_size=3,
            padding=1,
            batch_norm=False,
        )

        in_channels = out_channels
        out_channels = in_channels * 2

        # Replace layer (18) with custom InvertedResidual
        backbone.features[18] = CustomInvertedResidual(
            inp=in_channels,
            oup=out_channels,
            stride=1,
            expand_ratio=6,
            kernel_size=2,
            padding=0,
            batch_norm=False,
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


class LinearProbeHead(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)
    def forward(self, feat_map):          # [B, C, H, W]
        pooled = F.adaptive_avg_pool2d(feat_map, (1, 1))  # [B, C, 1, 1]
        flattened = pooled.flatten(1)
        return self.fc(flattened)


class StatePredictor(nn.Module):
    """Single-step predictor: (state, action) -> next_state, via GRUCell."""

    def __init__(self, hidden_size: int = 512, action_dim: int = 2, final_ln=None):
        super().__init__()
        self.cell = nn.GRUCell(input_size=action_dim, hidden_size=hidden_size)
        self.final_ln = final_ln if final_ln is not None else nn.Identity()

    def forward(self, ref_state, action):
        """
        Args:
            ref_state: [B, D]
            action: [B, A]
        Returns:
            pred_state: [B, D]
        """
        pred_state = self.cell(action, ref_state)
        return self.final_ln(pred_state)


class InverseDynamicsModel(nn.Module):
    """
    Predicts the action that caused a transition from ref_state to goal_state.
    Used as auxiliary task for representation learning.
    """

    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.apply(init_module_weights)

    def _single_forward(self, ref_state, goal_state):
        """
        Args:
            ref_state: Reference state, shape [B, D]
            goal_state: Goal state, shape [B, D]
        Returns:
            predicted_action: Action predicted to transform ref_state to goal_state, shape [B, A]
        """
        combined_states = torch.cat([ref_state, goal_state], dim=1)
        return self.model(combined_states)
    
    def forward(self, ref_state, goal_state):
        pos = self._single_forward(ref_state, goal_state)
        neg = self._single_forward(goal_state, ref_state)
        return 0.5 * (pos - neg)


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

class RelativeDistancePredictor(nn.Module):
    def __init__(self, emb_dim=1024, out_dim=2):
        super().__init__()
        # Input dim is 3 * emb_dim due to pairwise fusion [diff, abs_diff, prod]
        in_dim = emb_dim * 3
        
        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        self.res1 = ResidualBlock(512, 512)
        self.res2 = ResidualBlock(512, 256)
        
        self.head = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, out_dim)
        )

    def _forward_single(self, z1, z2):
        diff = z1 - z2
        abs_diff = torch.abs(diff)
        prod = z1 * z2
        h = torch.cat([diff, abs_diff, prod], dim=-1)
        
        feat = self.in_proj(h)
        feat = self.res1(feat)
        feat = self.res2(feat)
        return self.head(feat)

    def forward(self, z1, z2):
        # Enforce anti-symmetry: f(z1, z2) == -f(z2, z1)
        pos = self._forward_single(z1, z2)
        neg = self._forward_single(z2, z1)
        return 0.5 * (pos - neg)
    


if __name__ == "__main__":
    input = torch.rand((1, 3, 96, 96))
    print(f"{input.shape = }")
    backbone = mobilenet_v2(weights=None, width_mult=1.0)
    output = backbone.features(input)
    print(f"{output.shape = }")


