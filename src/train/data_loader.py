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

    def old_next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position: self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T)  # inputs
        y = (buf[1:]).view(B, T)  # targets

        # calculating the input concepts. ATTENTION: tokens must be detokenized and reencoded with bert tokenizer!!
        # batch decoding
        # x_text = self.gpt2_tokenizer.batch_decode(x, skip_special_tokens=True)
        x_text = [
            self.gpt2_tokenizer.decode(
                [token for token in token_ids if token != self.gpt2_tokenizer.pad_token_id],
                skip_special_tokens=True
            )
            for token_ids in x
        ]

        # Encode texts and ensure fixed-length encodings
        encoded_concepts = []
        max_length = max(self.encoder.encode_text(x_text[i]).size(1) for i in range(len(x_text))) # a little suboptimal to loop through the texts twice, but doable

        pad_token = torch.tensor([0]*self.encoder.encode_text(x_text[0]).size(-1), device=self.device, dtype = torch.long)
        for i in range(len(x_text)):
            encoded = self.encoder.encode_text(x_text[i])
            # Pad or truncate to the fixed length
            if len(encoded) < max_length:
                # Pad with zeros if the encoding is shorter than max_length
                encoded = torch.cat([encoded, pad_token.repeat(max_length - len(encoded))])

            encoded_concepts.append(encoded)

        # Convert to tensor
        concepts = torch.tensor(encoded_concepts, device=self.device, dtype=torch.long).view(B, -1)
        # TODO: attention: to avoid mismatch between tokens and concepts, concepts should be encoded from groups of eight gpt2 tokens, that are transformed to text and retokenized.

        # advance the position in the tensor
        self.current_position += B * T * self.num_processes

        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank

        return x, y, concepts

    def less_old_next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position: self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T)  # inputs
        y = (buf[1:]).view(B, T)  # targets

        # Chunk the targets into future concepts
        num_chunks = T // N_TOKENS_PER_CONCEPT
        y_chunks = y[:, :num_chunks * N_TOKENS_PER_CONCEPT].view(B, num_chunks, N_TOKENS_PER_CONCEPT)

        # Flatten y_chunks for decoding
        y_chunks_flat = y_chunks.view(-1, N_TOKENS_PER_CONCEPT)  # Shape: (B * num_chunks, N)

        y_text = [self.gpt2_tokenizer.batch_decode(y_chunks[:,i,:], skip_special_tokens=True) for i in range(y_chunks.size(1))] # list of lists, each sublist has B elements

        # Re-encode the text chunks with the BERT tokenizer
        y_encoded = [[self.encoder.tokenizer.encode(y_text[i][j], padding=False, truncation=True, return_tensors='pt', device=self.device)for j in range(len(y_text[0]))] for i in range(len(y_text))]

        # Move tensors to the correct device
        #y_encoded = [torch.cat(y_encoded[i], dim=0).to(self.device) for i in range(len(y_encoded))]

        # Now we encode text to concepts with the self.encoder
        # Pass the encoded inputs through the encoder to get concepts
        concepts = torch.cat([self.encoder(y_encoded[i]) for i in range(len(y_encoded))], dim=0).view(B, num_chunks, -1)

        # Update current position for the next batch
        self.current_position += B * T

        return x, y, concepts  # Return inputs, targets, and concepts
        encoded_concepts = []
        max_length = max(self.encoder.encode_text(x_text[i]).size(1) for i in
                         range(len(x_text)))  # a little suboptimal to loop through the texts twice, but doable

        pad_token = torch.tensor([0] * self.encoder.encode_text(x_text[0]).size(-1), device=self.device,
                                 dtype=torch.long)
        for i in range(len(x_text)):
            encoded = self.encoder.encode_text(x_text[i])
            # Pad or truncate to the fixed length
            if len(encoded) < max_length:
                # Pad with zeros if the encoding is shorter than max_length
                encoded = torch.cat([encoded, pad_token.repeat(max_length - len(encoded))])

            encoded_concepts.append(encoded)

        # Convert to tensor
        concepts = torch.tensor(encoded_concepts, device=self.device, dtype=torch.long).view(B, -1)
        # TODO: attention: to avoid mismatch between tokens and concepts, concepts should be encoded from groups of eight gpt2 tokens, that are transformed to text and retokenized.

        # advance the position in the tensor
        self.current_position += B * T * self.num_processes

        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank

        return x, y, concepts

    def next_batch(self):
        B, T = self.B, self.T
        N = N_TOKENS_PER_CONCEPT  # Number of tokens per concept

        # Get the buffer of tokens
        buf = self.tokens[self.current_position: self.current_position + B * T + 1]
        x = buf[:-1].view(B, T)  # Inputs
        y = buf[1:].view(B, T)  # Targets

        # Chunk the targets into future concepts
        num_chunks = T // N
        y_chunks = y[:, :num_chunks * N].view(B, num_chunks, N)  # Shape: (B, num_chunks, N)

        # Flatten y_chunks for processing
        y_chunks_flat = y_chunks.view(-1, N)  # Shape: (B * num_chunks, N)
        y_chunks_list = y_chunks_flat.tolist()  # Convert to list of lists

        concepts_list = []

        # Process each chunk individually
        for chunk_tokens in y_chunks_list:
            # Decode the chunk into text using GPT-2 tokenizer
            text = self.gpt2_tokenizer.decode(chunk_tokens, skip_special_tokens=True)

            # Re-tokenize the text with GPT-2 tokenizer
            # Since GPT-2 tokenizer doesn't support padding and we have variable-length sequences,
            # we process each sequence individually
            tokens = self.encoder.tokenizer.encode(text, padding=True, truncation=True, return_tensors='pt')

            # Now, encode the tokens into a concept using the Encoder
            # Since the Encoder uses the BERT tokenizer and model, we'll pass the text directly
            # The Encoder will handle variable-length sequences internally

            # Encode the text into a concept
            concept = self.encoder.encode_text(text, encode_in_single_concept=True)  # Shape: (1, hidden_size)

            # Append the concept to the list
            concepts_list.append(concept)

        # Stack concepts into a tensor
        concepts = torch.cat(concepts_list, dim=0)  # Shape: (B * num_chunks, hidden_size)

        # Reshape concepts to (B, num_chunks, hidden_size)
        hidden_size = concepts.size(-1)
        concepts = concepts.view(B, num_chunks, hidden_size)

        # Update current position for the next batch
        self.current_position += B * T * self.num_processes

        # If loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard], device=self.device)
            self.current_position = B * T * self.process_rank

        return x, y, concepts  # Return inputs, targets, and concepts

if __name__ == '__main__':

    def test_data_loader():
        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_built() else 'cpu'
        B = 2
        T = 1024
        process_rank = 0
        num_processes = 1
        split = 'train'
        master_process = True

        dl = DataLoaderWithConcepts(B, T, process_rank, num_processes, split, master_process, device=device)
        for i in range(3):
            x, y, concepts = dl.next_batch()
            print(x, y, concepts)

    test_data_loader()