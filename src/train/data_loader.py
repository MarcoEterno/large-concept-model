import os

import numpy as np
import torch

from transformers import GPT2Tokenizer

from src.model.config import DATA_ROOT_PATH, N_TOKENS_PER_CONCEPT

from src.model.encoder import Encoder



def load_tokens(filename, device=None):
    if device is None:
        print("WARNING: device not specified during data loading. loading on cpu")
        device = torch.device('cpu')
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    ptt = torch.tensor(npt, dtype=torch.long, device=device)
    return ptt

# TODO: load the tokens on device when converting to pytorch tensor

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split, master_process, device=None):
        if device is None:
            print("WARNING: device not specified during data loading. loading on cpu")
            self.device = torch.device('cpu')
        else:
            self.device = device

        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        # get the shard filenames
        data_root = os.path.join(DATA_ROOT_PATH, "edu_fineweb10B")
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard], device = self.device)
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position: self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T)  # inputs
        y = (buf[1:]).view(B, T)  # targets

        # advance the position in the tensor
        self.current_position += B * T * self.num_processes

        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank

        return x, y


class DataLoaderWithConcepts:
    def __init__(self, B, T, process_rank, num_processes, split, master_process, device=None):
        if device is None:
            print("WARNING: device not specified during data loading. loading on cpu")
            self.device = torch.device('cpu')
        else:
            self.device = device

        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        self.encoder = Encoder(n_tokens_per_concept=N_TOKENS_PER_CONCEPT).to(device)
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

        # get the shard filenames
        data_root = os.path.join(DATA_ROOT_PATH, "edu_fineweb10B-gpt2")
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard], device=self.device)
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position: self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T)  # inputs
        y = (buf[1:]).view(B, T)  # targets

        # calculating the input concepts. ATTENTION: tokens must be detokenized and reencoded with bert tokenizer!!
        # batch decoding
        # x_text = self.gpt2_tokenizer.batch_decode(x, skip_special_tokens=True)
        x_text = [
            self.gpt2_tokenizer.decode([token for token in token_ids if token != self.gpt2_tokenizer.pad_token_id], skip_special_tokens=True)
            for token_ids in x
        ]
        x_bert = self.encoder.tokenizer.encode(x_text, add_special_tokens=True, return_tensors='pt')
        concepts = self.encoder.encode_text(x_text).view(B, T // N_TOKENS_PER_CONCEPT, -1)

        # advance the position in the tensor
        self.current_position += B * T * self.num_processes

        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank

        return x, y, concepts

