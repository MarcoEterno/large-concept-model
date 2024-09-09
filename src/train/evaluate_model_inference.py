import torch
import torch.functional as F
import torch.distributed as dist

from src.benchmark.hellaswag_gpt import get_most_likely_row
from src.train.train_lcm import Trainer
from src.train.hellaswag import render_example, iterate_examples


def sample_model_inference(trainer: Trainer, step):
    trainer.model.eval()
    num_return_sequences = 4
    max_length = 32
    tokens = trainer.model.encoder.encode("Hello, I'm a language model,")
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    xgen = tokens.to(trainer.device)
    sample_rng = torch.Generator(device=trainer.device)
    sample_rng.manual_seed(42 + trainer.ddp_rank)
    while xgen.size(1) < max_length:
        # forward the model to get the logits
        with torch.no_grad():
            with torch.autocast(device_type=trainer.device_type, dtype=torch.bfloat16):
                logits, loss = trainer.model(xgen)  # (B, T, vocab_size)
            # take the logits at the last position
            logits = logits[:, -1, :]  # (B, vocab_size)
            # get the probabilities
            probs = F.softmax(logits, dim=-1)
            # do top-k sampling of 50 (huggingface pipeline default)
            # topk_probs here becomes (5, 50), topk_indices is (5, 50)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            # select a token from the top-k probabilities
            # note: multinomial does not demand the input to sum to 1
            ix = torch.multinomial(topk_probs, 1, generator=sample_rng)  # (B, 1)
            # gather the corresponding indices
            xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
            # append to the sequence
            xgen = torch.cat((xgen, xcol), dim=1)

    # print the generated text
    for i in range(num_return_sequences):
        tokens = xgen[i, :max_length].tolist()
        decoded = trainer.model.encoder.decode(tokens)
        print(f"rank {trainer.ddp_rank} sample {i}: {decoded}")

def eval_hellaswag(trainer: Trainer, step):
    num_correct_norm = 0
    num_total = 0
    for i, example in enumerate(iterate_examples("val")):
        # only process examples where i % ddp_world_size == ddp_rank
        if i % trainer.ddp_world_size != trainer.ddp_rank:
            continue
        # render the example into tokens and labels
        _, tokens, mask, label = render_example(example)
        tokens = tokens.to(trainer.device)
        mask = mask.to(trainer.device)
        # get the logits
        with torch.no_grad():
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                logits, loss = trainer.model(tokens)
            pred_norm = get_most_likely_row(tokens, mask, logits)
        num_total += 1
        num_correct_norm += int(pred_norm == label)
    # reduce the stats across all processes
    if trainer.ddp:
        num_total = torch.tensor(num_total, dtype=torch.long, device=trainer.device)
        num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=trainer.device)
        dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
        num_total = num_total.item()
        num_correct_norm = num_correct_norm.item()
    acc_norm = num_correct_norm / num_total
    if trainer.master_process:
        print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
        with open(trainer.log_file, "a") as f:
            f.write(f"{step} hella {acc_norm:.4f}\n")