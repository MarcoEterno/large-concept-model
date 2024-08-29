import torch
from torch import nn

from src.model.core_lcm import CoreLCM
from src.model.decoder import Decoder
from src.model.encoder import Encoder


class LCM(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.encoder = Encoder(config.n_tokens_per_concept)
        self.core = CoreLCM(config)
        self.decoder = Decoder(config)

    def forward(self, idx: torch.Tensor, targets=None): # idx is of shape (B, T)
        # B: batch size, T: number tokens, C: number concepts, D: embedding dimension
        input_concepts = self.encoder(idx)  # input_concepts is of shape (B, C, D)
        target_concepts = self.encoder(targets) if targets is not None else None  # target_concepts is of shape (B, C, D) or None

        # TODO predict auto-regressively concepts until satified
        predicted_concepts = self.core(input_concepts, target_concepts)  # predicted_concepts is of shape (B, C, D)

        # TODO decode predicted_concepts to words
        return self.decoder(predicted_concepts)  # return shape (B, T*, vocab_size)

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        return self.core.configure_optimizers(weight_decay, learning_rate, device_type)
