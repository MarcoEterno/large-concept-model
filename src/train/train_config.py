import os
from dataclasses import dataclass
from datetime import datetime

import torch
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from src.model.config import N_TOKENS_PER_CONCEPT, DATA_ROOT_PATH


@dataclass
class TrainerConfig:
    total_batch_size: int = 524288  # 2**19, ~0.5M, in number of tokens
    B: int = 2 * N_TOKENS_PER_CONCEPT  # micro batch size
    T: int = 1024  # sequence length, was 1024 in GPT-2

    # TODO: change for cloud run!
    eval_freq: int = 10
    eval_n_examples:int = 20
    eval_hellaswag_freq: int = 1
    eval_hellaswag_compression: int = 1 # TODO set to 1 for final run
    eval_model_inference_freq: int = 500


    checkpoint_freq: int= 2

    max_lr: float = 1e-3 # 6e-4 is the default for GPT-2
    min_lr: float = max_lr * 0.1
    warmup_steps: int = 715
    max_steps: int = 19073  # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens

    weight_decay: float = 0.1
    learning_rate: float = 1e-3 # 6e-4 is the default for GPT-2
    seed: int = 1337

    use_compile: bool = False


def setup_ddp():
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

def create_log_file_and_dir(self):
    log_dir = os.path.join(DATA_ROOT_PATH, "log")
    os.makedirs(log_dir, exist_ok=True)
    # call the log file as the current time
    log_file = os.path.join(log_dir, f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")
    with open(log_file, "w") as f:  # open for writing to clear the file
        pass
    return log_file, log_dir