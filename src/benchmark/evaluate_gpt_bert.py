import os

import numpy as np
import tiktoken
import torch
import torch.nn.functional as F

from src.model.config import DATA_ROOT_PATH, N_TOKENS_PER_CONCEPT, DEVICE
from src.model.encoder.encoder import Encoder
from src.model.gpt import GPT


def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt


class DataLoaderLite:
    def __init__(self, B, T, n_tokens_per_concept, split):
        self.B = B
        self.T = T
        self.n_tokens_per_concept = n_tokens_per_concept
        assert T % n_tokens_per_concept == 0, "T must be divisible by n_tokens_per_concept"
        assert split in {'train', 'val'}

        # get the shard filenames
        data_root = os.path.join(DATA_ROOT_PATH, "edu_fineweb10B-gpt2")
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        print(f"found {len(shards)} shards for split {split}")
        self.reset()
        print(f"{len(self.tokens) // (B * T + self.n_tokens_per_concept)} batch per shard")

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position: self.current_position + B * T + self.n_tokens_per_concept]
        inputs = (buf[:-self.n_tokens_per_concept]).view(B, T)  # inputs
        targets = (buf[self.n_tokens_per_concept:]).view(B, T)  # targets
        # advance the position in the tensor
        self.current_position += B * T

        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T + self.n_tokens_per_concept) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = 0

        return inputs, targets


def generate_n_tokens(model, inputs: torch.Tensor, num_tokens: int):
    xgen = inputs.to(DEVICE)
    for _ in range(num_tokens):
        # forward the model to get the logits
        with torch.no_grad():
            # with torch.autocast(device_type="cpu", dtype=torch.bfloat16): # use bfloat16 for inference, not on mps
            logits, loss = model(xgen)  # (B, T, vocab_size)

            # take the logits at the last position
            logits = logits[:, -1, :]  # (B, vocab_size)

            # get the probabilities
            probs = F.softmax(logits, dim=-1)

            next_tokens = probs.argmax(dim=-1).unsqueeze(-1)

            # append to the sequence
            xgen = torch.cat((xgen, next_tokens), dim=-1)

    return xgen


# Generate text from GPT2 then encode concepts with BERT
# Compare to original concept encoded with BERT

# define parameters
B = 1  # micro batch size
T = 128 # 1024  # sequence length
device = "mps" if torch.backends.mps.is_built() else "cuda" if torch.cuda.is_available() else "cpu"

# define models
val_loader = DataLoaderLite(B=B, T=T, n_tokens_per_concept=N_TOKENS_PER_CONCEPT, split="val")
enc = tiktoken.get_encoding("gpt2")

model = GPT.from_pretrained("gpt2")
model.to(device)
model.eval()

encoder = Encoder(n_tokens_per_concept=N_TOKENS_PER_CONCEPT)
encoder.to(device)

with torch.no_grad():
    val_loss_accum = 0.0
    val_loss_steps = 20
    for _ in range(val_loss_steps):
        # 1. Get tokenized text
        inputs, targets = val_loader.next_batch()
        inputs.to(device), targets.to(device)

        # 2. De-tokenize to get text
        target_text = [enc.decode(tokens.tolist()) for tokens in targets]
        # for sentence in text:
        #     print()
        #     print(sentence)

        # 3. Generate tokens with GPT2
        outputs = []
        for i in range(0, T, N_TOKENS_PER_CONCEPT):
            x = inputs[:, :i+N_TOKENS_PER_CONCEPT]

            output = generate_n_tokens(model, inputs=x, num_tokens=N_TOKENS_PER_CONCEPT)
            output = output[:, -N_TOKENS_PER_CONCEPT:]
            outputs.append(output)

        generated = torch.cat(outputs, dim=-1)

        # 4. Decode generated tokens to text
        generated_text = [enc.decode(tokens.tolist()) for tokens in generated]
        # for sentence in generated_text:
        #     print()
        #     print(sentence)

        # TODO tokens need to be reencoded in groups of eight, to prevent tokens mismatchs from happening
        # 4. Encode text with BERT
        concepts = encoder(generated_text)
        # print(concepts.shape)

        # 5. Encode original text with BERT
        target_concepts = encoder(target_text)
        # print(target_concepts.shape)

        # 6. Compute similarity
        dim = min(concepts.shape[-2], target_concepts.shape[-2])  # different encoders can have different dimensions
        sim = F.cosine_similarity(concepts[:dim, -1], target_concepts[:dim, -1], dim=-1).mean()
        loss = 1 - sim
        # print(sim)

        val_loss_accum += loss.detach()
        print(f"Step {_}: {val_loss_accum / (_ + 1)}")

print(val_loss_accum / val_loss_steps)
