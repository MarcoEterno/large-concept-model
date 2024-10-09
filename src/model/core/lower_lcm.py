import logging

import torch
from torch import nn

from src.model.config import CoreLCMConfig, DATA_ROOT_PATH
from src.model.core.core_lcm import CoreLCM
from src.model.encoder.encoder import Encoder

logger = logging.getLogger(__name__)


class Lower_LCM(nn.Module):
    def __init__(self, config_core,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config_core = config_core
        self.encoder = Encoder(n_tokens_per_concept=config_core.n_tokens_per_concept)
        self.core = CoreLCM(config_core)

        self.enc = self.encoder.tokenizer# tiktoken.get_encoding("gpt2")
        self.rng = torch.Generator()
        self.rng.manual_seed(42)
        # self.to(DEVICE)  # TODO: exercise

    def forward(self, x, target=None):
        """
        Forward pass of the model

        Args:
            x: input tensor of shape (B, C, D)
            target: target tensor of shape (B, T)

        Returns:
            concept: tensor of shape (B, T, vocab_size)
            loss: tensor of shape (B, T)
        """
        # ATTENTION: the forward target is token embeddings, so we need to encode the target
        # TODO: check that input tokens are always tokenized with BERT
        x = self.encoder(x)
        target = self.encoder(target) if target is not None else None
        x, loss_core = self.core(x, target)  # x is of shape (B, C, D), loss is of shape (B, C)
        return x, loss_core

    def infer(
            self,
            inputs: str | torch.Tensor | list[str],
            num_concepts_to_generate: int,

    ):
        raise NotImplementedError
        # tokens is of shape (B, T)
        assert num_concepts_to_generate <= self.config_core.block_size, "Cannot generate more tokens than decoder block size"

        # B: batch size, T: number tokens, C: number concepts, D: embedding dimension
        
        input_concepts = self.encoder(inputs)  # input_concepts is of shape (B, C, D)
        logger.info(f"Input encoded in {input_concepts.shape[1]} concepts")

        decoded_concepts = self.decode_concepts(input_concepts)
        logger.info(f"Input concepts: {decoded_concepts[0]}")  # get first element in batch

        # predict auto-regressively concepts TODO : stop generation when satified or max length
        logger.info(f"Generating {num_concepts_to_generate} concepts")
        concepts = input_concepts
        while concepts.size(1) < num_concepts:
            print('\u271D', end='')
            predicted_concepts, loss_core = self.core(input_concepts)  # predicted_concepts is of shape (B, C, D)
            last_predicted_concepts = predicted_concepts[:, -1:, :]  # take the last concept
            concepts = torch.cat([concepts, last_predicted_concepts], dim=-2)

        decoded_concepts = self.decode_concepts(concepts)
        logger.info(f"Concepts: {decoded_concepts[0]}")  # get first element in batch

        # decode concepts auto-regressively  TODO : stop generation when satified or max length
        logger.info(f"Generating {num_tokens_to_generate} tokens")
        tokens = torch.empty(concepts.shape[0], 0, device=concepts.device, dtype=torch.long)  # (B, 0)
        while tokens.size(1) < num_tokens_to_generate:
            print('.', end='')
            logits, loss_decoder = self.decoder(concepts, tokens)  # return shape (B, T*, vocab_size)
            last_logits = logits[:, -1, :]  # look at the last token  # logits is of shape (B, vocab_size)
            last_token = self.choose_token(last_logits, mode="top_k")  # (B, 1)
            tokens = torch.cat([tokens, last_token], dim=-1)  # (B, T*)

        # decode tokens to text
        return [self.enc.decode(sentence.tolist()) for sentence in tokens]

    def load(self, checkpoint_path, device):
        checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
        self.load_state_dict(checkpoint['model'], strict=False)# TODO change to strict=True
        self.eval()
        return self

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        return self.core.configure_optimizers(weight_decay, learning_rate, device_type)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    model = Lower_LCM(config_core=CoreLCMConfig())
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_built() else 'cpu'
    model.load(checkpoint_path=f"{DATA_ROOT_PATH}/checkpoints/lower_lcm_ntc-8_nlayer-12_nhead-8_n_embd-1024_step-04100.pt", device=device)
    print(model)
