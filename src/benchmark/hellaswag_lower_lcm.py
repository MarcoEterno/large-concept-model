"""
Downloads and evaluates HellaSwag in Python.
https://github.com/rowanz/hellaswag

Example HellaSwag json item:

{"ind": 24, "activity_label": "Roof shingle removal", "ctx_a": "A man is sitting on a roof.", "ctx_b": "he", "ctx": "A man is sitting on a roof. he", "split": "val", "split_type": "indomain", "label": 3, "endings": ["is using wrap to wrap a pair of skis.", "is ripping level tiles off.", "is holding a rubik's cube.", "starts pulling up roofing on a roof."], "source_id": "activitynet~v_-JhWjGDPHMY"}

ind: dataset ID
activity_label: The ActivityNet or WikiHow label for this example
context: There are two formats. The full context is in ctx. When the context ends in an (incomplete) noun phrase, like for ActivityNet, this incomplete noun phrase is in ctx_b, and the context up until then is in ctx_a. This can be useful for models such as BERT that need the last sentence to be complete. However, it's never required. If ctx_b is nonempty, then ctx is the same thing as ctx_a, followed by a space, then ctx_b.
endings: a list of 4 endings. The correct index is given by label (0,1,2, or 3)
split: train, val, or test.
split_type: indomain if the activity label is seen during training, else zeroshot
source_id: Which video or WikiHow article this example came from

gpt2 (124M)
- eleuther harness reports acc 28.92%, acc_norm 31.14% (multiple choice style)
- this script: 10042 acc: 0.2859 acc_norm: 0.2955 (completion style)

gpt2-xl (1558M)
- eleuther harness reports acc 40.04%, acc_norm 50.89% (multiple choice style)
- this script: 10042 acc: 0.3842 acc_norm: 0.4893 (completion style)

The validation set of HellaSwag has a total of 10,042 examples.
"""

import os
import json
import requests
from tqdm import tqdm
import torch
from torch.nn import functional as F
from transformers import GPT2LMHeadModel, BertTokenizer


from src.model.config import DATA_ROOT_PATH, CoreLCMConfig
from src.model.encoder.encoder import Encoder
from src.model.core.lower_lcm import Lower_LCM

# -----------------------------------------------------------------------------
DATA_CACHE_DIR = os.path.join(DATA_ROOT_PATH, "hellaswag")

def download_file(url: str, fname: str, chunk_size=1024):
    """Helper function to download a file from a given url"""
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm(
        desc=fname,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)

hellaswags = {
    "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
    "val": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
    "test": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
}

enc = BertTokenizer.from_pretrained("bert-base-uncased")

def download(split):
    """Downloads HellaSwag DATA_CACHE_DIR"""
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    data_url = hellaswags[split]
    data_filename = os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl")
    if not os.path.exists(data_filename):
        print(f"Downloading {data_url} to {data_filename}...")
        download_file(data_url, data_filename)

def render_example(example):
    """
    Given the example as a dictionary, render it as three torch tensors:
    - tokens (the tokens of context + completion, of size 4xN, as there are always 4 candidates)
    - mask (is 1 in the region of the candidate completion, where we evaluate likelihoods)
    - label (the index of the correct completion, which we hope has the highest likelihood)
    """
    ctx = example["ctx"]
    label = example["label"]
    endings = example["endings"]

    # data needed to reproduce this eval on the C size
    data = {
        "label": label,
        "ctx_tokens": None,
        "ending_tokens": [],
    }

    # gather up all the tokens
    ctx_tokens = enc.encode(ctx)
    data["ctx_tokens"] = ctx_tokens
    tok_rows = []
    mask_rows = []
    for end in endings:
        end_tokens = enc.encode(end) # note: in gpt2 tokenizer we needed to prepend " " to the text
        tok_rows.append(ctx_tokens + end_tokens)
        mask_rows.append([0]*len(ctx_tokens) + [1]*len(end_tokens))
        data["ending_tokens"].append(end_tokens)

    # have to be careful during the collation because the number of tokens in each row can differ
    max_len = max(len(row) for row in tok_rows)
    tokens = torch.zeros((4, max_len), dtype=torch.long)
    mask = torch.zeros((4, max_len), dtype=torch.long)
    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, :len(tok_row)] = torch.tensor(tok_row)
        mask[i, :len(mask_row)] = torch.tensor(mask_row)

    return data, tokens, mask, label

def iterate_examples(split):
    # there are 10,042 examples in total in val
    download(split)
    with open(os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl"), "r") as f:
        for line in f:
            example = json.loads(line)
            yield example

@torch.no_grad()
def evaluate_gpt(model_type, device, one_example_every_n = 1):

    torch.set_float32_matmul_precision('high') # use tf32
    model = GPT2LMHeadModel.from_pretrained(model_type)
    model.to(device)
    # model = torch.compile(model) # optionally torch compile the model

    num_correct_norm = 0
    num_correct = 0
    iterations = 0
    num_examples_evaluated_so_far = 0

    for example in iterate_examples("val"):
        # skip examples
        if iterations % one_example_every_n != 0:
            iterations += 1
            continue

        data, tokens, mask, label = render_example(example)
        tokens = tokens.to(device)
        mask = mask.to(device)

        # get the logits
        logits = model(tokens).logits
        # evaluate the autoregressive loss at all positions
        shift_logits = (logits[..., :-1, :]).contiguous()
        shift_tokens = (tokens[..., 1:]).contiguous()
        flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_shift_tokens = shift_tokens.view(-1)
        shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
        shift_losses = shift_losses.view(tokens.size(0), -1)
        # now get the average loss just for the completion region (where mask == 1), in each row
        shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
        masked_shift_losses = shift_losses * shift_mask
        # sum and divide by the number of 1s in the mask
        sum_loss = masked_shift_losses.sum(dim=1)
        avg_loss = sum_loss / shift_mask.sum(dim=1)
        # now we have a loss for each of the 4 completions
        # the one with the lowest loss should be the most likely
        pred = sum_loss.argmin().item()
        pred_norm = avg_loss.argmin().item()

        # accumulate stats
        iterations += 1
        num_examples_evaluated_so_far += 1
        num_correct += int(pred == label)
        num_correct_norm += int(pred_norm == label)
        print(f"{iterations} acc_norm: {num_correct_norm}/{num_examples_evaluated_so_far}={num_correct_norm/num_examples_evaluated_so_far:.4f}")

        # debug: pretty print a few examples, and the losses in each case
        if iterations < 10:
            print("---")
            print(f"Context:\n {example['ctx']}")
            print(f"Endings:")
            for i, end in enumerate(example["endings"]):
                print(f"{i} (loss: {avg_loss[i].item():.4f}) {end}")
            print(f"predicted: {pred_norm}, actual: {label}")

@torch.no_grad()
def evaluate_lcm_checkpoint(model_checkpoint, n_tokens_per_concept, device, one_example_every_n = 1):
    torch.set_float32_matmul_precision('high') # use tf32

    checkpoint = torch.load(model_checkpoint, map_location=torch.device(device), weights_only=False)
    state_dict = checkpoint["model"]
    model_lower = Lower_LCM(config_core=CoreLCMConfig())
    model_lower.load_state_dict(state_dict, strict=False)
    model = model_lower.core.to(device)
    model.eval()
    # model = torch.compile(model) # optionally torch compile the model and the encoder, helpful for bigger models

    encoder = Encoder(n_tokens_per_concept=n_tokens_per_concept).to(device) #TODO: change to model.encoder
    encoder.eval()

    num_correct_norm = 0
    num_correct = 0
    iterations = 0
    num_examples_evaluated_so_far = 0

    for example in iterate_examples("val"):
        # skip examples
        if iterations % one_example_every_n != 0:
            iterations += 1
            continue

        data, tokens, token_mask, label = render_example(example)
        tokens = tokens.to(device)
        token_mask = token_mask.to(device)

        context_concepts = encoder.encode_tokens(torch.tensor(data["ctx_tokens"], device=device))

        # Pad ending tokens to the maximum length
        max_len = max(len(ending) for ending in data["ending_tokens"])
        padded_endings = [ending + [encoder.tokenizer.pad_token_id] * (max_len - len(ending)) for ending in data["ending_tokens"]]
        ending_concepts = encoder.encode_tokens(torch.tensor(padded_endings).to(device))

        # concatenate the context and the ending concepts by copying the context concepts 4 times and pasting them before ending concepts
        concepts_real = torch.cat([context_concepts.repeat(4, 1, 1), ending_concepts], dim=1)

        # create the mask for the concepts
        max_ending_concept_len = ending_concepts.size(1)
        ending_concept_mask = torch.zeros(4, max_ending_concept_len, dtype=torch.float32, device=device)
        for i in range(4):
            concept_position = -1
            for position, ending_token in enumerate(data["ending_tokens"][i]):
                if position % n_tokens_per_concept != 0:
                    continue
                concept_position += 1
                if ending_token == encoder.tokenizer.pad_token_id:
                    break
                ending_concept_mask[i, concept_position] = 1
        concept_mask = torch.cat([torch.zeros(context_concepts.shape[:-1],dtype=torch.float32, device=device).repeat(4, 1), ending_concept_mask], dim=1)

        # get the forecasted concepts
        concept_forecast, loss = model(concepts=concepts_real)

        # evaluate the autoregressive loss at all positions
        shift_concepts_forecast = (concept_forecast[..., :-1, :]).contiguous()
        shift_concepts_real = (concepts_real[...,1:,:]).contiguous()

        #flatten the forecasts and the concepts
        flat_shift_concepts_forecasts = shift_concepts_forecast.view(-1, shift_concepts_forecast.size(-1))
        flat_shift_concepts_real = shift_concepts_real.view(-1, shift_concepts_real.size(-1))

        # calculate loss
        #shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_concepts, reduction='none')
        shift_losses = 1 - F.cosine_similarity(shift_concepts_forecast, shift_concepts_real, dim=-1)# check what is the mean for, should be mean on the concepts of the single sentence
        # shift_losses = shift_losses.view(concepts_real.size(0), -1)

        # now get the average loss just for the completion region (where mask == 1), in each row
        # we need to create a new mask for it, that carefully separates the input from the continuation
        shift_concept_mask = concept_mask[:, 1:].contiguous() # we must shift mask, so we start at the last prompt token
        masked_shift_losses = shift_losses * shift_concept_mask

        # sum and divide by the number of 1s in the mask
        sum_loss = masked_shift_losses.sum(dim=1)
        avg_loss = sum_loss / shift_concept_mask.sum(dim=1)

        # now we have a loss for each of the 4 completions
        # the one with the lowest loss should be the most likely
        pred = avg_loss.argmin().item()
        pred_norm = avg_loss.argmin().item()

        # accumulate stats
        iterations += 1
        num_examples_evaluated_so_far += 1
        num_correct += int(pred == label)
        num_correct_norm += int(pred_norm == label)
        print(f"{iterations} acc_norm: {num_correct_norm}/{num_examples_evaluated_so_far}={num_correct_norm/num_examples_evaluated_so_far:.4f}")

        # debug: pretty print a few examples, and the losses in each case
        if iterations < 202:
            print("---")
            print(f"Context:\n {example['ctx']}")
            print(f"Endings:")
            for i, end in enumerate(example["endings"]):
                print(f"{i} (loss: {avg_loss[i].item():.4f}) {end}")
            print(f"predicted: {pred_norm}, actual: {label}")

def evaluate_lower_lcm(model, n_tokens_per_concept, device, one_example_every_n=1, print_to_video=True):
    # TODO add batch size >1 to speed up
    torch.set_float32_matmul_precision('high')  # use tf32
    model.to(device)
    model.eval()
    # model = torch.compile(model) # optionally torch compile the model and the encoder, helpful for bigger models

    encoder = Encoder(n_tokens_per_concept=n_tokens_per_concept)
    encoder.eval()

    num_correct_norm = 0
    num_correct = 0
    iterations = 0
    num_examples_evaluated_so_far = 0

    for example in iterate_examples("val"):
        # skip examples
        if iterations % one_example_every_n != 0:
            iterations += 1
            continue

        data, tokens, token_mask, label = render_example(example)
        tokens = tokens.to(device)
        token_mask = token_mask.to(device)

        context_concepts = encoder.encode_tokens(torch.tensor(data["ctx_tokens"], dtype=torch.long, device=device)).to(device)

        # Pad ending tokens to the maximum length
        max_len = max(len(ending) for ending in data["ending_tokens"])
        padded_endings = [ending + [encoder.tokenizer.pad_token_id] * (max_len - len(ending)) for ending in
                          data["ending_tokens"]]
        ending_concepts = encoder.encode_tokens(torch.tensor(padded_endings, dtype=torch.long).to(device))

        # concatenate the context and the ending concepts by copying the context concepts 4 times and pasting them before ending concepts
        concepts_real = torch.cat([context_concepts.repeat(4, 1, 1), ending_concepts], dim=1)

        # create the mask for the concepts
        max_ending_concept_len = ending_concepts.size(1)
        ending_concept_mask = torch.zeros(4, max_ending_concept_len, dtype=torch.long, device=device)
        for i in range(4):
            concept_position = -1
            for position, ending_token in enumerate(data["ending_tokens"][i]):
                if position % n_tokens_per_concept != 0:
                    continue
                concept_position += 1
                if ending_token == encoder.tokenizer.pad_token_id:
                    break
                ending_concept_mask[i, concept_position] = 1
        concept_mask = torch.cat(
            [torch.zeros(context_concepts.shape[:-1], dtype=torch.long, device=device).repeat(4, 1),
             ending_concept_mask], dim=1)

        # get the forecasted concepts
        concept_forecast, loss = model.core(concepts_real)

        # evaluate the autoregressive loss at all positions
        shift_concepts_forecast = (concept_forecast[..., :-1, :]).contiguous()
        shift_concepts_real = (concepts_real[..., 1:, :]).contiguous()

        # flatten the forecasts and the concepts
        flat_shift_concepts_forecasts = shift_concepts_forecast.view(-1, shift_concepts_forecast.size(-1))
        flat_shift_concepts_real = shift_concepts_real.view(-1, shift_concepts_real.size(-1))

        # calculate loss
        # shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_concepts, reduction='none')
        shift_losses = 1 - F.cosine_similarity(shift_concepts_forecast, shift_concepts_real,
                                               dim=-1)  # check what is the mean for, should be mean on the concepts of the single sentence
        # shift_losses = shift_losses.view(concepts_real.size(0), -1)

        # now get the average loss just for the completion region (where mask == 1), in each row
        # we need to create a new mask for it, that carefully separates the input from the continuation
        shift_concept_mask = (
        concept_mask[..., :, 1:]).contiguous()  # we must shift mask, so we start at the last prompt token
        masked_shift_losses = shift_losses * shift_concept_mask

        # sum and divide by the number of 1s in the mask
        sum_loss = masked_shift_losses.sum(dim=1)
        avg_loss = sum_loss / shift_concept_mask.sum(dim=1)

        # now we have a loss for each of the 4 completions
        # the one with the lowest loss should be the most likely
        pred = avg_loss.argmin().item()
        pred_norm = avg_loss.argmin().item()

        # accumulate stats
        iterations += 1
        num_examples_evaluated_so_far += 1
        num_correct += int(pred == label)
        num_correct_norm += int(pred_norm == label)

        if print_to_video:
            print(
                f"{iterations} acc_norm: {num_correct_norm}/{num_examples_evaluated_so_far}={num_correct_norm / num_examples_evaluated_so_far:.4f}")

            # debug: pretty print a few examples, and the losses in each case
            if num_examples_evaluated_so_far < 3 :
                print("---")
                print(f"Context:\n {example['ctx']}")
                print(f"Endings:")
                for i, end in enumerate(example["endings"]):
                    print(f"{i} (loss: {avg_loss[i].item():.4f}) {end}")
                print(f"predicted: {pred_norm}, actual: {label}")

    return num_correct_norm, num_examples_evaluated_so_far

def load_checkpoint(model_checkpoint, device):
    checkpoint = torch.load(model_checkpoint, map_location=torch.device(device), weights_only=False)
    state_dict = checkpoint["model"]
    model_lower = Lower_LCM(config_core=CoreLCMConfig())
    model_lower.load_state_dict(state_dict, strict=False)
    model_lower = model_lower.to(device)
    model_lower.eval()
    # model = torch.compile(model) # optionally torch compile the model and the encoder, helpful for bigger models
    return model_lower


if __name__ == "__main__":
    import argparse
    from src.model.config import DEVICE
    parser = argparse.ArgumentParser()
    checkpoint_file = os.path.join(DATA_ROOT_PATH,"checkpoints",  "lower_lcm_ntc-8_nlayer-12_nhead-8_n_embd-1024_step-04100.pt")

    model=load_checkpoint(model_checkpoint=checkpoint_file, device=DEVICE)

    parser.add_argument("-mod", "--model", type=Lower_LCM, default=model)
    parser.add_argument("-m", "--model_checkpoint", type=str, default= checkpoint_file, help="the checkpoint file to use")
    parser.add_argument("-ntc", "--n_tokens_per_concept", type=int, default=8, help="the number of tokens per concept")
    parser.add_argument("-d", "--device", type=str, default=DEVICE, help="the device to use")
    parser.add_argument("-n", "--one_example_every_n", type=int, default=10, help="evaluate one example every n")
    parser.add_argument("-p", "--print_to_video", type=bool, default=False, help="prints some examples to stdout")
    args = parser.parse_args()
    evaluate_lcm_checkpoint(args.model_checkpoint, args.n_tokens_per_concept, args.device, args.one_example_every_n)
    
    #evaluate_lower_lcm(args.model, args.n_tokens_per_concept, args.device, args.one_example_every_n, args.print_to_video)
    # scores for lower lcm with compression = 10:
    # ntc=8 => acc_norm: 0.2716
    # ntc=4 => acc_norm: 0.2667
    # ntc=2 => acc_norm: 0.2577
    # ntc=1 => acc_norm: 0.2711
    # ntc=16 => acc_norm: 0.2219