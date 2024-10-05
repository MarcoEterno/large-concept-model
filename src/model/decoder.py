import inspect

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import GPT2Tokenizer, BertTokenizer

from src.model.block_kernel import GeneralBlock
from src.model.config import DecoderConfig
from src.model.encoder import Encoder


def top_k_top_p_filtering(logits, top_k: int, top_p: float = 0.0):
    """
    Filter a distribution of logits using top-k and top-p (nucleus) filtering. top k is much more efficient!
    """
    assert logits.shape[0] == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = -float('Inf')

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the last token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = -float('Inf')
    return logits


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
            wpe=nn.Embedding(config.block_size, config.n_embd),  # only 8/9 of the block will be made of tokens
            cpe=nn.Embedding(config.block_size, config.concept_embedding_dim),
            # only 1/9 of the block will be made of concepts
            h=nn.ModuleList([GeneralBlock(config) for _ in range(config.n_layer)]),
            ln_t=nn.LayerNorm(config.n_embd)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

        # give the model the gpt2 tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

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

    def generate(self, xt, xc, max_len=128, temperature=1.0, top_k=0, top_p=0.9, device='cpu', print_to_video=False):
        """
        Generate a sequence of tokens given a sequence of concepts.

        This method generates a sequence of tokens given a sequence of concepts. The method is autoregressive, meaning
        that the model generates one token at a time. The method stops generating tokens when the model generates the
        token
        """
        assert xt.size(0) == 1, "only batch size 1 is supported for generation"
        assert xc.size(0) == 1, "only batch size 1 is supported for generation"
        B, T = xt.size()
        B, C, Dc = xc.size()

        # use the forward method to generate in a loop untill max_len is reached of eos token is generated
        for _ in range(max_len):
            logits, _ = self.forward(xt, xc)  # (B, T, vocab_size)
            logits = logits[:, -1, :] / temperature
            logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            xt = torch.cat([xt, next_token], dim=-1)
            if print_to_video:
                print(self.tokenizer.decode(next_token.squeeze(0, 1)))  # decide whether to print all sentence or not
            if next_token.squeeze(0, 1) == self.tokenizer.eos_token_id:
                break
        return xt

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
        sd['transformer.cpe.weight'].copy_(
            sd['transformer.wpe.weight'])  # slight abuse since there is no reason to share the weights

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

    def load_checkpoint(self, checkpoint_path, device):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.load_state_dict(checkpoint['model'])
        self.eval()
        return self


if __name__ == "__main__":
    def sample_model_inference():
        # load the model from checkpoint
        model = Decoder(DecoderConfig())
        checkpoint_path = "/Users/marcoeterno/Desktop/Coding/large-concept-model/data/checkpoints/decoder_ntc-8_nlayer-12_nhead-16_n_embd-768-concept_dim1024step-11400.pt"
        print(checkpoint_path)
        model.load_checkpoint(checkpoint_path, device='mps')
        model.eval()
        print(model)

        # generate a sequence of tokens
        text = "I am a language model"
        xt = model.tokenizer.encode(text, return_tensors='pt', device='mps')
        xt = model.generate(xt, xc=torch.empty(1, 0, 1024), max_len=128, temperature=1.0, top_k=5, top_p=0.0, device='mps',
                       print_to_video=True)
        print(model.tokenizer.decode(xt.squeeze(0, 1)))

    def test_forward():
        model = Decoder(DecoderConfig())
        # create the tokens ground truth of shape (25)
        text = torch.randint(0, 50257, (25,)).long()

        tokens = text[:-1].view(6, 4)
        target_tokens = text[1:].view(6, 4)
        concepts = torch.randn(6, 4, 1024)

        logits, loss = model(tokens, concepts, target_tokens)
        print(logits.shape, loss)


    def test_generate():
        device = 'mps' if torch.backends.mps.is_built() else 'cuda' if torch.cuda.is_available() else 'cpu'
        model = Decoder(DecoderConfig()).to(device)
        # create the tokens ground truth of shape (25) and batch size = 1
        text = torch.randint(0, 50257, (25,), device=device).long()

        tokens = text[:-1].unsqueeze(0)
        target_tokens = text[1:].unsqueeze(0)
        concepts = torch.randn(3, 1024, device=device).unsqueeze(0)

        model.generate(tokens, concepts, max_len=128, temperature=1.0, top_k=5, top_p=0.0, device='mps',
                       print_to_video=True)

    def test_model_inference_with_given_concepts():
        # load the decoder from checkpoint

        device = 'cpu' if torch.backends.mps.is_built() else 'cuda' if torch.cuda.is_available() else 'cpu'

        model = Decoder(DecoderConfig())
        checkpoint_path = "/Users/marcoeterno/Desktop/Coding/large-concept-model/data/checkpoints/decoder_ntc-8_nlayer-12_nhead-16_n_embd-768-concept_dim1024step-19072.pt"
        print(checkpoint_path)
        model.load_checkpoint(checkpoint_path, device=device)
        model.eval()
        print(model)

        encoder = Encoder(n_tokens_per_concept=8).to(device)

        # generate a sequence of tokens
        text = """Non-steroidal anti-inflammatory drugs are members of a therapeutic drug class which reduces pain, decreases inflammation, decreases fever, and prevents blood clots. Side effects depend on the specific drug, its dose and duration of use, but largely include an increased risk of gastrointestinal ulcers and bleeds, heart attack, and kidney disease.
The term non-steroidal, common from around 1960, distinguishes these drugs from corticosteroids, another class of anti-inflammatory drugs, which during the 1950s had acquired a bad reputation due to overuse and side-effect problems after their introduction in 1948
"""
        beginning_text = "Non-steroidal anti-inflammatory drugs are members of a therapeutic drug class which reduces pain,"

        xt = model.tokenizer.encode(beginning_text, return_tensors='pt').to(device)

        xc = encoder.encode_text(text , encode_in_single_concept=False) # size (1, 8, 1024)
        xt = model.generate(xt, xc=xc, max_len=100, temperature=0.1, top_k=1, top_p=0.0, device=device,
                       print_to_video=True)
        print(model.tokenizer.decode(xt.squeeze(0, 1)))

    test_model_inference_with_given_concepts()

