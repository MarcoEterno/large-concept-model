import os
from dataclasses import dataclass
from pathlib import Path

import torch
from dotenv import load_dotenv

load_dotenv()

DATA_ROOT_PATH = os.getenv("DATA_ROOT_PATH", str(Path(__file__).parents[2] / "data"))
N_TOKENS_PER_CONCEPT = 8

DEVICE = 'mps' if torch.backends.mps.is_built() else 'cuda' if torch.cuda.is_available() else 'cpu'
TOP_K = 5
HIGHER_THAN_P = 0.05  # tokens with probability less than this will be discarded. # if no token is left, the token with the highest probability will be chosen


@dataclass
class CoreLCMConfig:
    n_tokens_per_concept: int = N_TOKENS_PER_CONCEPT  # number of tokens per concept
    block_size: int = 1024 // N_TOKENS_PER_CONCEPT  # max sequence length in concept space
    vocab_size: int = 50257  # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12  # number of layers
    n_head: int = 8  # number of heads (was 12 in GPT-2)
    n_embd: int = 1024  # embedding dimension (was 768 in GPT-2)


@dataclass
class DecoderConfig:
    block_size: int = 1024  # max sequence length
    vocab_size: int = 50257  # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12  # number of layers
    n_head: int = 12  # number of heads
    n_embd: int = 768  # embedding dimension


@dataclass
class GPTConfig:
    block_size: int = 1024  # max sequence length
    vocab_size: int = 50257  # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12  # number of layers
    n_head: int = 12  # number of heads
    n_embd: int = 768  # embedding dimension
