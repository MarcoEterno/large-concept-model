import logging

import tiktoken
import torch
from torch import nn
from torch.nn import functional as F

from src.model.config import CoreLCMConfig, DATA_ROOT_PATH
from src.model.config import DecoderConfig
from src.model.core_lcm import CoreLCM
from src.model.decoder import Decoder
from src.model.encoder import Encoder

logger = logging.getLogger(__name__)


class LCM(nn.Module):
    def __init__(self, config_core, config_decoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config_core = config_core
        self.config_decoder = config_decoder
        self.encoder = Encoder(config_core.n_tokens_per_concept)
        self.core = CoreLCM(config_core)
        self.decoder = Decoder(config_decoder)

        self.enc = tiktoken.get_encoding("gpt2")
        self.rng = torch.Generator()
        self.rng.manual_seed(42)
        # self.to(DEVICE)  # TODO: exercise

    def infer(
            self,
            inputs: str | torch.Tensor | list[str],
            num_concepts: int,
            num_tokens_to_generate: int,
    ):  # tokens is of shape (B, T)
        assert num_concepts <= self.config_core.block_size, "Cannot generate more concepts than core block size"
        assert num_tokens_to_generate <= self.config_decoder.block_size, "Cannot generate more tokens than decoder block size"

        # B: batch size, T: number tokens, C: number concepts, D: embedding dimension
        input_concepts = self.encoder(inputs)  # input_concepts is of shape (B, C, D)
        logger.info(f"Input encoded in {input_concepts.shape[1]} concepts")

        decoded_concepts = self.decode_concepts(input_concepts)
        logger.info(f"Input concepts: {decoded_concepts[0]}")  # get first element in batch

        # predict auto-regressively concepts TODO : stop generation when satified or max length
        logger.info(f"Generating {num_concepts - input_concepts.shape[1]} concepts")
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

    def decode_concepts(self, concepts) -> list[str]:
        # concepts is of shape (B, C, D)
        logits = self.decoder.lm_head(concepts)  # (B, C, vocal_size)
        tokens = logits.argmax(dim=-1)  # (B, C)

        return [self.enc.decode(sentence.tolist()) for sentence in tokens]

    def choose_token(self, logits, mode='greedy', top_k=50):
        assert mode in ['greedy', 'top_k'], "Mode must be either 'greedy' or 'top_k'"

        if mode == 'greedy':
            # greedy decoding to the most probable token
            return logits.argmax(dim=-1).unsqueeze(-1)  # shape (B, 1)
        elif mode == 'top_k':
            # get the probabilities
            probs = F.softmax(logits, dim=-1)  # shape (B, vocab_size)
            # do top-k sampling of 50 (huggingface pipeline default)
            # topk_probs here becomes (5, 50), topk_indices is (5, 50)
            topk_probs, topk_indices = torch.topk(probs, top_k, dim=-1)
            # select a token from the top-k probabilities
            # note: multinomial does not demand the input to sum to 1
            ix = torch.multinomial(topk_probs, 1, generator=self.rng)  # (B, 1)
            # gather the corresponding indices
            return torch.gather(topk_indices, -1, ix)  # (B, 1)

    def load(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model'], strict=False)  # TODO change to strict=True when decoder is trained
        model.decoder = Decoder.from_pretrained('gpt2')  # TODO decoder will be trained
        self.eval()
        return self

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        return self.core.configure_optimizers(weight_decay, learning_rate, device_type)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    model = LCM(config_core=CoreLCMConfig(), config_decoder=DecoderConfig())
    model.load(checkpoint_path=f"{DATA_ROOT_PATH}/checkpoint/core_lcm_04500.pt")
    generated_text = model.infer(
        "Clever, astute, insighful, smart, sagation",  # input text (can be already tokenized)
        num_concepts=3,  # number of total concepts: input + generated
        num_tokens_to_generate=32  # number of tokens to generate
    )
    print(generated_text[0])  # first sentence in batch
