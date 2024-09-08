import os
from dataclasses import dataclass

import torch
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from src.model.config import N_TOKENS_PER_CONCEPT

@dataclass
class TrainerConfig:
    total_batch_size: int = 524288  # 2**19, ~0.5M, in number of tokens
    B: int = 1 * N_TOKENS_PER_CONCEPT  # micro batch size
    T: int = 1024  # sequence length, was 1024 in GPT-2

    eval_freq: int = 10
    eval_hellaswag_freq: int = 10
    eval_model_inference_freq: int = 10
    checkpoint_freq: int= 10

    max_lr: float = 1e-3 # 6e-4 is the default for GPT-2
    min_lr: float = max_lr * 0.1
    warmup_steps: int = 715
    max_steps: int = 19073  # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens

    weight_decay: float = 0.1
    learning_rate: float = 1e-3 # 6e-4 is the default for GPT-2
    seed: int = 1337


def setup_ddp(self):
    # set up DDP (distributed data parallel).
    # torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
    ddp = int(os.environ.get('RANK', -1)) != -1  # is this a ddp run?
    if ddp:
        # use of DDP atm demands CUDA, we set the device appropriately according to rank
        assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
    else:
        # vanilla, non-DDP run
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        # attempt to autodetect device
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        print(f"using device: {device}")

    return ddp, ddp_rank, ddp_local_rank, ddp_world_size, master_process, device