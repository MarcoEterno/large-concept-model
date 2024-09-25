import inspect
from logging import raiseExceptions

import torch
import torch.nn as nn
from torch.nn import functional as F

from src.model.config import DecoderConfig
from src.model.block_kernel import GeneralBlock


class Decoder(nn.Module):
    """
    The decoder for now is a simple transformer that takes a sequence of concepts and a sequence of tokens and
    predicts the next token in the sequence.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd), # only 8/9 of the block will be made of tokens
            cpe=nn.Embedding(config.block_size, config.concept_embedding_dim), # only 1/9 of the block will be made of concepts
            h=nn.ModuleList([GeneralBlock(config) for _ in range(config.n_layer)]),
            ln_t = nn.LayerNorm(config.n_embd)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens, concepts, target_tokens=None):
        """
        Predict the next token in the sequence given the current sequence of tokens and concepts.

        For now, the forward method allows for block sizes that are double of the token block size.
        in practice, given that concepts are far less than tokens, the maximum block size will be Block_Size * (N_tc + 1)/N_tc
        """
        # tokens (B, T) are to embed
        B, T = tokens.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=tokens.device)  # shape (T)
        token_pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(tokens)  # token embeddings of shape (B, T, n_embd)
        xt = tok_emb + token_pos_emb

        # concepts (B, C, D) are vectors
        B, C, Dc = concepts.size()
        assert C <= self.config.block_size, f"Cannot forward sequence of concepts of length {C}, block size is only {self.config.block_size}"
        # forward the token and position embeddings
        pos = torch.arange(0, C, dtype=torch.long, device=tokens.device)  # shape (C)
        concept_pos_emb = self.transformer.cpe(pos)  # position embeddings of shape (C, n_embd)
        xc = concepts + concept_pos_emb

        # forward the blocks of the transformer
        for block in self.transformer.h:
            xt, xc = block(xt, xc)

        # forward the final layer norm and the classifier
        xt = self.transformer.ln_t(xt)
        logits = self.lm_head(xt)  # (B, T, vocab_size)
        loss = None
        if target_tokens is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_tokens.view(-1))
        return logits, loss

    # TODO: substitute with load from checkpoint
    @classmethod
    def from_pretrained(cls, model_type):
        # raise NotImplementedError

        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained core: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257  # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024  # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = DecoderConfig(**config_args)
        model = Decoder(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]  # discard this mask / buffer, not a param
        sd_keys = [k for k in sd_keys if not k.endswith('transformer.cpe.weight')]  # cpe is not available

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]  # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]  # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        # copy the embeddings for concept positional encoding
        sd['transformer.cpe.weight'].copy_(sd['transformer.wpe.weight']) # slight abuse since there is no reason to share the weights

        return model

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        master_process = True

        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if master_process:
            print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

if __name__ == "__main__":

    def test_forward():
        model = Decoder(DecoderConfig())
        # create the tokens ground truth of shape (25)
        text = torch.randint(0, 50257, (25,)).long()

        tokens = text[:-1].view(6,4)
        target_tokens = text[1:].view(6,4)
        concepts = torch.randn(6, 4, 1024)

        logits, loss = model(tokens, concepts, target_tokens)
        print(logits.shape, loss)


    model = Decoder(DecoderConfig())
    optimizer = model.configure_optimizers(0.1, 0.1, 'cuda')
    print("done initializing")

    test_forward()
