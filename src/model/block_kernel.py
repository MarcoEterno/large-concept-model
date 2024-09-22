import math

from torch import nn
from torch.nn import functional as F
import torch

class GeneralCausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0 # I would call the n_embed the single head dimensionality, and then multiply by heads to get the total dimensionality.
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.compressed_attention = nn.Linear(config.n_embd, 2*config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
    """
    FUNCTIONING IMPLEMENTATION WITH 3 MATRICES
    def forward(self, xt):
        B, T, C = xt.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(xt)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y
    

    FUNCTIONING ALL PYTORCH IMPLEMENTATION WITH 3 MATRICES OF ATTENTION, ON CUDA IS 15/12 TIMES SLOWER.
    def new_scaled_dot_product_attention(self, query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False,
                                     scale=None) -> torch.Tensor:
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
        attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
        if is_causal:
            assert attn_mask is None
            temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(query.dtype)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias += attn_mask
        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
        return attn_weight @ value
    
    """

    def compressed_dot_product_attention(self, xt, qk_t, value, attn_mask=None, dropout_p=0.0, is_causal=False,
                                         scale=None) -> torch.Tensor:
        """
        Shape legend:
        - :math:`N: \text{Batch size} ... : \text{Any number of other batch dimensions (optional)}`
        - :math:`S: \text{Source sequence length}`
        - :math:`L: \text{Target sequence length}`
        - :math:`E: \text{Embedding dimension of the query and key}`
        - :math:`Ev: \text{Embedding dimension of the value}`
        """
        B, nh, T, hs = value.shape   # (B, nh, T, hs)
        scale_factor = 1 / math.sqrt(hs) if scale is None else scale
        attn_bias = torch.zeros(T, T, dtype=value.dtype, device=value.device) # will broadcast over the batch dimension later
        if is_causal:
            assert attn_mask is None
            temp_mask = torch.ones(T, T, dtype=torch.bool, device=value.device).tril(diagonal=0)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(value.dtype)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias += attn_mask
        attn_weight = xt @ qk_t.transpose(-2, -1)  * scale_factor # penultimo della prima si scontra con l'ultimo della seconda
        attn_weight += attn_bias # broadcasted over the batch dimension
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True) # wtf, why is dropout True? # TODO: check if this is correct
        return attn_weight @ value

    def forward(self, xt):
        B, T, C = xt.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qk_tv = self.compressed_attention(xt)
        qk_t, v = qk_tv.split(self.n_embd, dim=2)
        qk_t = qk_t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        xt = xt.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        y = self.compressed_dot_product_attention(xt, qk_t, v, is_causal=True)  # compressed attention
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = GeneralCausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


if __name__ == '__main__':
    def test_new_block_equals_gpt_block():
        from src.model.config import GPTConfig
        from src.model.gpt_block import Block as GPTBlock
        import torch
        config = GPTConfig()

        device = "mps" if torch.backends.mps.is_built() else "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        # set a seed for reproducibility
        torch.manual_seed(42)
        block = Block(config).to(device)
        torch.manual_seed(42)
        gpt_block = GPTBlock(config).to(device)
        x = torch.randn(1, 10, config.n_embd).to(device)
        # time the block function
        with torch.autograd.profiler.profile(use_cuda=False) as profiler_concepts:
            for i in range(10):
                yc = block(x)
        with torch.autograd.profiler.profile(use_cuda=False) as profiler_gpt:
            for i in range(10):
                y = block(x)
        print(profiler_concepts.key_averages().table(sort_by="cpu_time_total"))
        print(profiler_gpt.key_averages().table(sort_by="cpu_time_total"))
        assert (torch.allclose(yc, y))

    def explore_block_speed():
        from src.model.config import GPTConfig
        import torch
        config = GPTConfig()

        device = "mps" if torch.backends.mps.is_built() else "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        # set a seed for reproducibility
        torch.manual_seed(42)
        block = Block(config).to(device)
        x = torch.randn(1, 10, config.n_embd, device=device)
        x = block(x)
        # time the block function
        with torch.autograd.profiler.profile(use_cuda=False) as profiler:
            for i in range(10):
                x = block(x)
        print(profiler.key_averages().table(sort_by="self_cpu_time_total"))
        print(x.shape)

    explore_block_speed()
