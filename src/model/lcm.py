import tiktoken
import torch
from torch import nn

from src.model.config import LCMConfig
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
        self.enc = tiktoken.get_encoding("gpt2")

    def forward(self, idx: torch.Tensor, targets=None):  # tokens is of shape (B, T)
        # B: batch size, T: number tokens, C: number concepts, D: embedding dimension
        input_concepts = self.encoder(idx)  # input_concepts is of shape (B, C, D)
        target_concepts = self.encoder(
            targets) if targets is not None else None  # target_concepts is of shape (B, C, D) or None

        # TODO predict auto-regressively concepts until satified
        predicted_concepts, loss = self.core(input_concepts,
                                             target_concepts)  # predicted_concepts is of shape (B, C, D)

        # TODO add auto-regressivity
        tokens = torch.Tensor(input_concepts.shape[0], 0).to(input_concepts.device).long()  # (B, 1)
        logits, loss = self.decoder(predicted_concepts, tokens)  # return shape (B, T*, vocab_size)

        # take the logits at the last position
        logits = logits[:, -1, :]  # (B, vocab_size)
        # get the probabilities
        idx_g = logits.argmax(dim=-1)
        # append to the sequence
        return self.enc.decode(idx_g.tolist())

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        return self.core.configure_optimizers(weight_decay, learning_rate, device_type)


if __name__ == '__main__':
    config = LCMConfig()
    model = LCM(config)
    text = "The quick brown fox jumps over the lazy dog."
    generated_text = model(text)
    print(generated_text)
