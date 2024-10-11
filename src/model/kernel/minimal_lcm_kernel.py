import math
import time

from torch import nn
from torch.nn import functional as F
import torch

from src.model.config import DecoderConfig, N_TOKENS_PER_CONCEPT
decoderconfig = DecoderConfig()


class MinimalCausalConceptAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0  # I would call the n_embed the single head dimensionality, and then multiply by heads to get the total dimensionality.
        # matrices that will create all the Q,K,V matrices. they are made from a single contiguous matrix in memory to speed up access
        self.general_token_attention = nn.Linear(config.n_embd, config.n_embd + 2 * config.concept_embedding_dim)
        self.general_concept_attention = nn.Linear(config.concept_embedding_dim,
                                                   config.n_embd + config.concept_embedding_dim)

        # dimensions
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.concept_embedding_dim = config.concept_embedding_dim

        # output projection
        self.c_proj = nn.Linear(config.concept_embedding_dim, config.concept_embedding_dim)
        self.c_proj.NANOGPT_SCALE_INIT = 1

        self.t_proj = nn.Linear(config.n_embd, config.n_embd)
        self.t_proj.NANOGPT_SCALE_INIT = 1

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

    # now writing the attention kernel for concepts
    def forward(self, xt, xc):
        """
        General forward of the attention block for both concepts and tokens
        """

        B, T, D = xt.size()  # batch size, tokens length, embedding dimensionality (n_embd)
        Bc, C, Dc = xc.size()  # batch size, concepts length, embedding dimensionality (n_embd)
        assert B == Bc, f"Batch size for tokens and concepts must be the same, got {B} and {Bc}"

        # in our general implementation Vt, Qt_tkt and Qc_tkt will be multiplied by the tokens while the others will be multiplied by the concepts
        all_token_matrices = self.general_token_attention(xt)
        all_concept_matrices = self.general_concept_attention(xc)

        # MATRICES FOR THE FLASH ATTENTION IMPLEMENTATION
        Qct, Kct, Vtt = all_token_matrices.split(
            [self.concept_embedding_dim, self.concept_embedding_dim, self.n_embd], dim=2)
        Kcc, Vtc = all_concept_matrices.split(
            [self.concept_embedding_dim, self.n_embd], dim=2)

        # reshaping the matrices
        Qct = Qct.view(B, T, self.n_head, Dc // self.n_head).transpose(1, 2)
        Kct = Kct.view(B, T, self.n_head, Dc // self.n_head).transpose(1, 2)
        Vtt = Vtt.view(B, T, self.n_head, D // self.n_head).transpose(1, 2)
        Kcc = Kcc.view(B, C, self.n_head, Dc // self.n_head).transpose(1, 2)
        Vtc = Vtc.view(B, C, self.n_head, D // self.n_head).transpose(1, 2)

        # Create masks using the updated functions
        mask_tc = create_token_to_concept_mask(N_TOKENS_PER_CONCEPT, T, C, device=xt.device)  # shape: (T,C)
        # mask_ct = create_concept_to_token_mask(N_TOKENS_PER_CONCEPT, C, T, device=xt.device) # shape: (C,T)

        # attention for tokens and concepts. BE CAREFUL: CODE IS STILL NOT OPTIMIZED, LOGICAL_NOT TAKES 6% OF GPU TIME. EVERYTHING ELSE IS IN ORDER
        xtt = F.scaled_dot_product_attention(Qct, Kct, Vtt, is_causal=True)  # Shape: (B, nh, T, hs_t)
        xtc = F.scaled_dot_product_attention(Qct, Kcc, Vtc,
                                             attn_mask=mask_tc)  # F.scaled_dot_product_attention(Qct, Kcc, Vtc, is_causal=False)  # Shape: (B, nh, T, hs_t)
        # xct = self.new_scaled_dot_product_attention(Qcc, Kct, Vct, attn_mask=mask_ct) #F.scaled_dot_product_attention(Qcc, Kct, Vct, is_causal=False)  # Shape: (B, nh, C, hs_c)
        # xcc = F.scaled_dot_product_attention(Qcc, Kcc, Vcc, is_causal=True)  # Shape: (B, nh, C, hs_c)

        xt_embed = xtt + xtc
        # xc_embed = xct + xcc

        # output heads reassemble
        xt_new = xt_embed.transpose(1, 2).contiguous().view(B, T, D)
        # xc_new = xc_embed.transpose(1, 2).contiguous().view(B, C, Dc)
        # xc_new = xc
        # output projection
        xt = self.t_proj(xt_new)
        # xc = self.c_proj(xc_new)

        return xt, xc


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.t_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        # self.c_fc = nn.Linear(config.concept_embedding_dim, 4 * config.concept_embedding_dim)
        self.gelu = nn.GELU(approximate='tanh')
        self.t_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.t_proj.NANOGPT_SCALE_INIT = 1
        # self.c_proj = nn.Linear(4 * config.concept_embedding_dim, config.concept_embedding_dim)
        # self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, xt, xc):
        xt = self.t_fc(xt)
        # xc = self.c_fc(xc)

        xt = self.gelu(xt)
        # xc = self.gelu(xc)

        xt = self.t_proj(xt)
        # xc = self.c_proj(xc)
        return xt, xc


class MinimalBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1t = nn.LayerNorm(config.n_embd)
        # self.ln_1c = nn.LayerNorm(config.concept_embedding_dim)
        self.attn = MinimalCausalConceptAttention(config)
        self.ln_2t = nn.LayerNorm(config.n_embd)
        # self.ln_2c = nn.LayerNorm(config.concept_embedding_dim)
        self.mlp = MLP(config)

    def forward(self, xt, xc):
        attn_xt, attn_xc = self.attn(self.ln_1t(xt), xc)  # self.ln_1c(xc) in general
        xt = xt + attn_xt
        # xc = xc + attn_xc
        mlp_xt, mlp_xc = self.mlp(self.ln_2t(xt), xc)  # self.ln_1c(xc) in general
        xt = xt + mlp_xt
        # xc = xc + mlp_xc
        return xt, xc

def get_random_value(change_interval_minutes=1.0, start_epoch=0, last_value = decoderconfig.n_future_concepts_seen):
    # Total number of possible values
    total_values = start_epoch + last_value

    # Current Unix time in seconds
    current_time = int(time.time())

    # Calculate the duration of each interval in seconds
    change_interval_seconds = int(change_interval_minutes * 60)

    # Calculate how many complete intervals have passed since start_epoch
    elapsed_intervals = (current_time - start_epoch) // change_interval_seconds

    # Determine the current value by cycling through 0 to 16
    current_value = elapsed_intervals % total_values

    return current_value


def create_token_to_concept_mask(n_tokens_per_concept, n_tokens, n_concepts, device):
    """
    Create a causal attention mask for token-to-concept attention.
    A token at index i can attend to concepts with indices less than i // n_tokens_per_concept + 1.

    Args:
        n_tokens_per_concept: Number of tokens per concept
        n_tokens: Total number of tokens (T)
        n_concepts: Total number of concepts (C)
        device: Device to use

    Returns:
        mask: torch.tensor of shape (T, C)
    """

    # we need to

    i_indices = torch.arange(n_tokens, device=device).unsqueeze(1)  # Shape: (T, 1)
    j_indices = torch.arange(n_concepts, device=device).unsqueeze(0)  # Shape: (1, C)
    mask = (j_indices <= i_indices // n_tokens_per_concept + get_random_value(change_interval_minutes=1, start_epoch=0, last_value = decoderconfig.n_future_concepts_seen)).bool()  # Shape: (T, C) # randomizing n_concepts seen by the model makes it more robust
    return mask


def create_concept_to_token_mask(n_tokens_per_concept, n_concepts, n_tokens, device):
    """
    Create a causal attention mask for concept-to-token attention.
    A concept at index j can only attend to tokens with indices less than (j + 1) * n_tokens_per_concept.

    Args:
        n_tokens_per_concept: Number of tokens per concept
        n_concepts: Total number of concepts (C)
        n_tokens: Total number of tokens (T)
        device: Device to use

    Returns:
        mask: torch.tensor of shape (C, T)
    """
    i_indices = torch.arange(n_concepts, device=device).unsqueeze(1)  # Shape: (C, 1)
    j_indices = torch.arange(n_tokens, device=device).unsqueeze(0)  # Shape: (1, T)
    mask = (j_indices < n_tokens_per_concept * (i_indices + 1)).bool()  # Shape: (C, T)
    return mask


if __name__ == '__main__':
    def test_create_causal_attention_mask():
        B = 1
        num_tokens = 12
        num_concepts = 10
        n_tokens_per_concept = 2
        device = 'mps' if torch.backends.mps.is_built() else 'cuda' if torch.cuda.is_available() else 'cpu'
        mask_tc = create_token_to_concept_mask(n_tokens_per_concept, num_tokens, num_concepts, device)
        mask_ct = create_token_to_concept_mask(n_tokens_per_concept, n_tokens=num_tokens, n_concepts=num_concepts,
                                               device=device)
        print("mask_tc")
        print(mask_tc)
        print("mask_ct")
        print(mask_ct)

        config = DecoderConfig()

        device = "mps" if torch.backends.mps.is_built() else "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        # set a seed for reproducibility
        torch.manual_seed(42)

        block = MinimalBlock(config).to(device)
        xt = torch.randn(1, 20, config.n_embd, device=device)
        xc = torch.randn(1, 10, config.concept_embedding_dim, device=device)
        xt, xc = block(xt, xc)
        # time the block function
        with torch.autograd.profiler.profile(use_cuda=False) as profiler:
            for i in range(10):
                xt, xc = block(xt, xc)
        print(profiler.key_averages().table(sort_by="self_cpu_time_total"))
        print(xt.shape, xc.shape)


    def test_new_block_equals_gpt_block():
        from src.model.config import DecoderConfig
        from src.model.kernel.gpt_block import Block as GPTBlock
        import torch
        config = DecoderConfig()

        device = "mps" if torch.backends.mps.is_built() else "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        # set a seed for reproducibility
        torch.manual_seed(42)
        block = MinimalBlock(config).to(device)
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
        from src.model.config import DecoderConfig
        import torch
        config = DecoderConfig

        device = "mps" if torch.backends.mps.is_built() else "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        # set a seed for reproducibility
        torch.manual_seed(42)
        block = MinimalBlock(config).to(device)
        x = torch.randn(1, 10, config.n_embd, device=device)
        x = block(x)
        # time the block function
        with torch.autograd.profiler.profile(use_cuda=False) as profiler:
            for i in range(10):
                x = block(x)
        print(profiler.key_averages().table(sort_by="self_cpu_time_total"))
        print(x.shape)


    def test_attention():
        from src.model.config import DecoderConfig
        import torch
        config = DecoderConfig()

        device = "mps" if torch.backends.mps.is_built() else "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        # set a seed for reproducibility
        torch.manual_seed(42)

        block = MinimalBlock(config).to(device)
        xt = torch.randn(1, 10, config.n_embd, device=device)
        xc = torch.randn(1, 10, config.concept_embedding_dim, device=device)
        xt, xc = block(xt, xc)
        # time the block function
        with torch.autograd.profiler.profile(use_cuda=False) as profiler:
            for i in range(10):
                xt, xc = block(xt, xc)
        print(profiler.key_averages().table(sort_by="self_cpu_time_total"))
        print(xt.shape, xc.shape)

    def test_get_random_value():
        print(get_random_value(change_interval_minutes=1/60, start_epoch=0, last_value = decoderconfig.n_future_concepts_seen))
        #make the system wait two seconds:
        time.sleep(2)
        print(get_random_value(change_interval_minutes=1/60, start_epoch=0, last_value=decoderconfig.n_future_concepts_seen))
        time.sleep(10)
        print(get_random_value(change_interval_minutes=1 / 60, start_epoch=0,
                               last_value=decoderconfig.n_future_concepts_seen))



    test_get_random_value()

