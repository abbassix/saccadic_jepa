# eb_jeoa/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class HingeStdLoss(torch.nn.Module):
    def __init__(
        self,
        std_margin: float = 1.0,
    ):
        """
        Encourages each feature to maintain at least a minimum standard deviation.
        Features with std below the margin incur a penalty of (std_margin - std).
        Args:
            std_margin (float, default=1.0):
                Minimum desired standard deviation per feature.
        """
        super().__init__()
        self.std_margin = std_margin

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [N, D] where N is number of samples, D is feature dimension
        Returns:
            std_loss: Scalar tensor with the hinge loss on standard deviations
        """
        x = x - x.mean(dim=0, keepdim=True)
        std = torch.sqrt(x.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(self.std_margin - std))
        return std_loss


class CovarianceLoss(torch.nn.Module):
    def __init__(self):
        """
        Penalizes off-diagonal elements of the covariance matrix to encourage
        feature decorrelation.

        Normalizes by D * (D - 1) where D is feature dimensionality.
        """
        super().__init__()

    def off_diagonal(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [N, D] where N is number of samples, D is feature dimension
        """
        batch_size = x.shape[0]
        num_features = x.shape[-1]
        x = x - x.mean(dim=0, keepdim=True)
        cov = (x.T @ x) / (batch_size - 1)  # [D, D]
        # Calculate off-diagonal loss
        cov_loss = self.off_diagonal(cov).pow(2).mean()

        return cov_loss


class InverseDynamicsLoss(nn.Module):
    def __init__(self, idm: nn.Module):
        super().__init__()
        self.idm = idm

    def forward(self, ref_state, goal_state, action):
        """
        Args:
            ref_state, goal_state: [B, D]
            action: [B, A]
        """
        if action is None:
            return torch.tensor(0.0, device=ref_state.device)
        pred_actions = self.idm(ref_state, goal_state)
        return F.mse_loss(pred_actions, action)


class VC_IDM_Regularizer(nn.Module):
    """VicReg-style (std + cov) regularizer plus optional inverse-dynamics loss."""

    def __init__(
        self,
        idm: nn.Module = None,
        std_margin: float = 1.0,
        projector: nn.Module = None,
        idm_after_proj: bool = False,
    ):
        super().__init__()
        self.projector = nn.Identity() if projector is None else projector
        self.idm_after_proj = idm_after_proj

        self.std_loss_fn = HingeStdLoss(std_margin=std_margin)
        self.cov_loss_fn = CovarianceLoss()
        self.idm_loss_fn = InverseDynamicsLoss(idm) if idm is not None else None

    def forward(self, ref_state, goal_state, actions=None):
        """
        Args:
            ref_state, goal_state: [B, D] encoder outputs
            actions: [B, A]
        """
        ref_proj = self.projector(ref_state)

        if self.idm_loss_fn is not None:
            if self.idm_after_proj:
                idm_loss = self.idm_loss_fn(ref_proj, self.projector(goal_state), actions)
            else:
                idm_loss = self.idm_loss_fn(ref_state, goal_state, actions)
        else:
            idm_loss = torch.zeros_like(goal_state.sum())

        std_loss = self.std_loss_fn(ref_proj)
        cov_loss = self.cov_loss_fn(ref_proj)

        loss_dict = {
            "cov_loss": cov_loss,
            "std_loss": std_loss,
            "idm_loss": idm_loss,
        }

        return loss_dict


def all_reduce(x, op):
    """All-reduce operation for distributed training."""
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        op = dist.ReduceOp.__dict__[op]
        dist.all_reduce(x, op=op)
        return x
    else:
        return x


def epps_pulley(x, t_min=-3, t_max=3, n_points=10):
    """Epps-Pulley test statistic for Gaussianity."""
    # integration points
    t = torch.linspace(t_min, t_max, n_points, device=x.device)
    # theoretical CF for N(0, 1)
    exp_f = torch.exp(-0.5 * t**2)
    # ECF
    x_t = x.unsqueeze(2) * t  # (N, M, T)
    ecf = (1j * x_t).exp().mean(0)
    ecf = all_reduce(ecf, op="AVG")
    # weighted L2 distance
    err = exp_f * (ecf - exp_f).abs() ** 2
    T = torch.trapz(err, t, dim=1)
    return T


class BCS(nn.Module):
    """BCS (Batched Characteristic Slicing) loss for SIGReg."""

    def __init__(self, num_slices=256, lmbd=10.0):
        super().__init__()
        self.num_slices = num_slices
        self.step = 0
        self.lmbd = lmbd

    def forward(self, z1, z2):
        with torch.no_grad():
            dev = z1.device
            g = torch.Generator(device=dev)
            g.manual_seed(self.step)
            proj_shape = (z1.size(1), self.num_slices)
            A = torch.randn(proj_shape, device=dev, generator=g)
            A /= A.norm(p=2, dim=0)
        view1 = z1 @ A
        view2 = z2 @ A

        self.step += 1
        bcs = (epps_pulley(view1).mean() + epps_pulley(view2).mean()) / 2
        return bcs


class SIG_IDM_Regularizer(nn.Module):
    """
    SIG_IDM Regularizer using BCS (Batched Characteristic Slicing) 
    instead of VC (Variance-Covariance) regularization.
    """

    def __init__(
        self,
        idm: nn.Module = None,
        num_slices: int = 256,
        projector: nn.Module = None,
        idm_after_proj: bool = False,
    ):
        super().__init__()
        self.projector = nn.Identity() if projector is None else projector
        self.idm_after_proj = idm_after_proj

        self.bcs_loss_fn = BCS(num_slices=num_slices)
        self.idm_loss_fn = InverseDynamicsLoss(idm) if idm is not None else None

    def forward(self, ref_state, goal_state, actions=None):
        """
        Args:
            ref_state, goal_state: [B, D] encoder outputs
            actions: [B, A]
        """
        ref_proj = self.projector(ref_state)
        # Note: BCS typically requires two views/inputs to compare distribution alignment
        # In a standard IDM context, we treat ref_state and goal_state as the views
        goal_proj = self.projector(goal_state)

        # 1. Inverse Dynamics Loss
        if self.idm_after_proj:
            idm_loss = self.idm_loss_fn(ref_proj, goal_proj, actions)
        else:
            idm_loss = self.idm_loss_fn(ref_state, goal_state, actions)

        # 2. BCS Regularization
        # Encourages the distribution of projections to match Gaussianity
        bcs_loss = self.bcs_loss_fn(ref_proj, goal_proj)

        loss_dict = {
            "bcs_loss": bcs_loss,
            "idm_loss": idm_loss,
        }

        return loss_dict
