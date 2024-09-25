import math
import os
import time

import torch

# distributed computing imports
from torch.distributed import destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from src.model.config import N_TOKENS_PER_CONCEPT, DecoderConfig
from src.model.decoder import Decoder
from src.train.data_loader import DataLoaderWithConcepts
from src.train.train_config import TrainerConfig, setup_ddp, create_log_file_and_dir


# -----------------------------------------------------------------------------
# simple launch:
# python train_lcm.py
# DDP launch for e.g. 4 GPUs:
# torchrun --standalone --nproc_per_node=4 train_decoder.py

# importing this class seems to take a lot of time
class Trainer:
    def __init__(self, model, config):
        self.ddp, self.ddp_rank, self.ddp_local_rank, self.ddp_world_size, self.master_process, self.device = setup_ddp()
        self.device_type = "cuda" if self.device.startswith(
            "cuda") else "mps" if torch.backends.mps.is_built() else "cpu"

        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.seed)

        assert config.total_batch_size % (config.B * config.T * self.ddp_world_size) == 0, \
            "make sure total_batch_size is divisible by B * T * ddp_world_size"
        self.grad_accum_steps = config.total_batch_size // (config.B * config.T * self.ddp_world_size)
        if self.master_process:
            print(f"total desired batch size: {config.total_batch_size}")
            print(f"=> calculated gradient accumulation steps: {self.grad_accum_steps}")

        # data loaders
        self.train_loader = DataLoaderWithConcepts(
            B=config.B,
            T=config.T,
            process_rank=self.ddp_rank,
            num_processes=self.ddp_world_size,
            split="train",
            master_process=self.master_process,
            device=self.device
        )

        self.val_loader = DataLoaderWithConcepts(
            B=config.B,
            T=config.T,
            process_rank=self.ddp_rank,
            num_processes=self.ddp_world_size,
            split="val",
            master_process=self.master_process,
            device=self.device
        )

        # create model
        torch.set_float32_matmul_precision('high')
        self.model = model
        model.to(self.device)
        self.use_compile = config.use_compile  # torch.compile interferes with HellaSwag eval and Generation. TODO fix
        if self.use_compile:
            model = torch.compile(model)
        if self.ddp:
            model = DDP(model, device_ids=[self.ddp_local_rank])

        self.raw_model = model.module if self.ddp else model  # always contains the "raw" unwrapped model
        self.optimizer = self.raw_model.configure_optimizers(
            weight_decay=config.weight_decay,
            learning_rate=config.learning_rate,
            device_type=self.device_type
        )

        self.log_file, self.log_dir = create_log_file_and_dir(self)

        # TODO: chage to self.config.---
        self.max_lr = config.max_lr
        self.min_lr = config.min_lr
        self.warmup_steps = config.warmup_steps
        self.max_steps = config.max_steps

        self.eval_freq = config.eval_freq
        self.eval_n_examples = config.eval_n_examples
        self.eval_hellaswag_freq = config.eval_hellaswag_freq
        self.eval_hellaswag_compression = config.eval_hellaswag_compression
        self.eval_model_inference_freq = config.eval_model_inference_freq
        self.checkpoint_freq = config.checkpoint_freq

    def get_lr(self, it):
        # 1) linear warmup for warmup_iters steps
        if it < self.warmup_steps:
            return self.max_lr * (it + 1) / self.warmup_steps
        # 2) if it > lr_decay_iters, return min learning rate
        if it > self.max_steps:
            return self.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff starts at 1 and goes to 0
        return self.min_lr + coeff * (self.max_lr - self.min_lr)

    def run(self):
        for step in range(self.max_steps):
            t0 = time.time()
            last_step = (step == self.max_steps - 1)

            # once in a while evaluate our validation loss
            if (step % self.eval_freq == 0 and step != 0) or last_step:
                self.eval(step, last_step)

            # once in a while evaluate hellaswag # TODO: write hellaswag eval for LCM
            """
            if (step % self.eval_hellaswag_freq == 0 or last_step) and (not self.use_compile):
                num_correct, num_evaluated =evaluate_lower_lcm(self.model, n_tokens_per_concept = N_TOKENS_PER_CONCEPT, device=self.device, one_example_every_n=self.eval_hellaswag_compression, print_to_video=self.master_process)
                if self.master_process:
                    with open(self.log_file, "a") as f:
                        f.write(f"{step} hellaswag {num_correct / num_evaluated:.4f}\n")"""
            #
            # # once in a while generate from the model (except step 0, which is noise)
            # if ((step > 0 and step % self.eval_model_inference_freq == 0) or last_step) and (not self.use_compile):
            #     eval_model_inference(self,step)
            #
            # do one step of the optimization
            loss_accum, lr, norm = self.optimize_one_step(step)  # return just to print

            # print stats
            t1 = time.time()
            dt = t1 - t0  # time difference in seconds
            tokens_processed = self.train_loader.B * self.train_loader.T * self.grad_accum_steps * self.ddp_world_size
            tokens_per_sec = tokens_processed / dt
            if self.master_process or not self.ddp:
                print(
                    f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt * 1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
                with open(self.log_file, "a") as f:
                    f.write(f"{step} train {loss_accum.item():.6f}\n")

        if self.ddp:
            destroy_process_group()

    def save_checkpoint(self, step, val_loss_accum):
        # optionally write model checkpoints
        from src.model.config import CoreLCMConfig as conf
        checkpoint_path = os.path.join(
            self.log_dir,
            f"lower_lcm_ntc-{conf.n_tokens_per_concept}_nlayer-{conf.n_layer}_nhead-{conf.n_head}_n_embd-{conf.n_embd}_step-{step:05d}.pt")

        checkpoint = {
            'model': self.raw_model.state_dict(),
            'config': self.raw_model.config,
            'step': step,
            'val_loss': val_loss_accum.item(),
            'optimizer': self.optimizer.state_dict(),
            'rng_state': torch.get_rng_state(),
        }
        print("saving_checkpoint")
        torch.save(checkpoint, checkpoint_path)

    def eval(self, step, last_step):
        model.eval()
        self.val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            for _ in range(self.eval_n_examples):
                x, y, concepts = self.val_loader.next_batch()
                x, y, concepts = x.to(self.device), y.to(self.device), concepts.to(
                    self.device)  # TODO: this should already be on device after changing the data loader
                if self.device_type == "cuda" or self.device_type == "cpu":
                    with torch.autocast(device_type=self.device_type,
                                        dtype=torch.bfloat16):  # bfloat16 is faster for evaluation
                        logits, loss = model(x, concepts, target_tokens=y)
                elif self.device_type == "mps":
                    logits, loss = model(x, concepts, target_tokens=y)  # MPS does not support bfloat16

                else:
                    raise ValueError(f"device_type {self.device_type} not supported")

                average_loss = loss / self.eval_n_examples
                val_loss_accum += average_loss.detach()

        if self.ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if self.master_process:
            print(f"validation loss: {val_loss_accum.item():.4f}")
            with open(self.log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")
            if step > 0 and (step % self.checkpoint_freq == 0 or last_step):
                self.save_checkpoint(step, val_loss_accum)

    def optimize_one_step(self, step):
        model.train()
        self.optimizer.zero_grad()
        loss_accum = 0.0
        for micro_step in range(self.grad_accum_steps):
            print('.', end='', flush=True)
            x, y = self.train_loader.next_batch()
            x, y = x.to(self.device), y.to(self.device)
            concepts = x[:, :N_TOKENS_PER_CONCEPT, :]

            # this field is also used by the forward pass.
            if self.ddp:
                model.require_backward_grad_sync = (micro_step == self.grad_accum_steps - 1)

            if self.device_type == "cuda" or self.device_type == "cpu":
                with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
            elif self.device_type == "mps":
                logits, loss = model(x, y)

            # we have to scale the loss to account for gradient accumulation,
            # because the gradients just add on each successive backward().
            # addition of gradients corresponds to a SUM in the objective, but
            # instead of a SUM we want MEAN. Scale the loss here so it comes out right
            loss = loss / self.grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()

        if self.ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # determine and set the learning rate for this iteration
        lr = self.get_lr(step)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.optimizer.step()
        if self.device_type == "cuda":
            torch.cuda.synchronize()  # wait for the GPU to finish work

        return loss_accum, lr, norm


if __name__ == "__main__":
    model = Decoder(DecoderConfig())
    trainer = Trainer(model, TrainerConfig())
    trainer.run()
