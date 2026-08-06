# eb_jepa/jepa.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from eb_jepa.logging import get_logger

logging = get_logger(__name__)


class JEPAbase(nn.Module):
    """Base JEPA class for planning and inference only. Use JEPA subclass for training."""

    def __init__(self, encoder, predictor):
        """Initialize JEPAbase with encoder, action encoder, and predictor."""
        super().__init__()
        # Observation Encoder
        self.encoder = encoder
        # Predictor
        self.predictor = predictor

    def save(self, file):
        torch.save(self.state_dict(), file)

    def load(self, file):
        self.load_state_dict(torch.load(file), weights_only=False)

    @torch.no_grad()
    def encode(self, observation):
        """Encode the encoder output."""
        return self.encoder(observation)

    @torch.no_grad()
    def feats(self, observation):
        """Return the encoder's last feature map before the final projection layer."""
        return self.encoder.features(observation)


class JEPA(JEPAbase):
    """Trainable JEPA: predicts `encoder(goal_crop)` from `encoder(ref_crop)` and `action`."""

    def __init__(self, encoder, predictor, regularizer, predcost):
        super().__init__(encoder, predictor)
        self.regularizer = regularizer
        self.predcost = predcost

    @torch.no_grad()
    def infer(self, ref_crop, action):
        """Predict the next state's embedding (mu, log_var) given current state and an action."""
        ref_state = self.encoder(ref_crop)
        return self.predictor(ref_state, action)

    def forward(self, ref_crop, action, goal_crop):
        """
        Args:
            ref_crop: [B, C, H, W]
            action: [B, A]
            goal_crop: [B, C, H, W]
        Returns:
            reg_loss_dict
            pred_loss
        """
        ref_state = self.encoder(ref_crop)
        goal_state = self.encoder(goal_crop)
        
        pred_goal_state, pred_log_var = self.predictor(ref_state, action)

        reg_loss_dict = self.regularizer(ref_state, goal_state, action)

        # Convert log_variance to variance for GaussianNLLLoss
        pred_var = torch.exp(pred_log_var)

        # Call the instantiated loss function (input, target, var)
        goal_state = goal_state.detach()  # Detach goal_state to prevent gradients from flowing into the encoder
        pred_loss = self.predcost(pred_goal_state, goal_state, pred_var)
        
        return reg_loss_dict, pred_loss


class JEPAProbe(nn.Module):
    """JEPA with a trainable prediction head. The JEPA encoder is kept fixed.
    Head could be a linear layer or a conv layer followed by a linear layer."""

    def __init__(self, jepa, head, hcost, n_layers_to_remove=2):
        """Initialize with a frozen JEPA, prediction head, and head loss function."""
        super().__init__()
        self.jepa = jepa
        self.head = head
        self.hcost = hcost

        if n_layers_to_remove == 1:
            # Remove the 19th layer from the encoder to get the feature map before the final projection layer
            self.jepa.encoder.features = self.jepa.encoder.features[:-1]
        elif n_layers_to_remove == 2:
            # Remove the 18th and 19th layers from the encoder to get the feature map before the final projection layer
            self.jepa.encoder.features = self.jepa.encoder.features[:-2]

    @torch.no_grad()
    def infer(self, observation):
        """Encode observations through JEPA and apply the prediction head."""
        state = self.jepa.feats(observation)
        return self.head(state)

    def forward(self, observation, target):
        """Forward pass for training the head (JEPA encoder gradients are detached)."""
        with torch.no_grad():
            state = self.jepa.feats(observation)
        output = self.head(state.detach())
        return self.hcost(output, target)
