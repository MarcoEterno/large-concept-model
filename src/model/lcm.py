import torch
from torch import nn

from src.model.core_lcm import CoreLCM
from src.model.encoder import Encoder


class LCM(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.encoder = Encoder(config.n_tokens_per_concept)
        self.core = CoreLCM(config)

    def forward(self, idx: torch.Tensor, targets=None):
        input_concepts = self.encoder(idx)
        target_concepts = self.encoder(targets) if targets is not None else None
        return self.core(input_concepts, target_concepts)

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        return self.core.configure_optimizers(weight_decay, learning_rate, device_type)
