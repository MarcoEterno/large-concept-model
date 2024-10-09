import torch
from torch import nn
from transformers import BertTokenizer, BertModel


class Encoder(nn.Module):
    def __init__(self, n_tokens_per_concept: int, tokenizer=None, model=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_tokens_per_concept = n_tokens_per_concept
        self.tokenizer = tokenizer if tokenizer else BertTokenizer.from_pretrained(
            'bert-large-uncased', clean_up_tokenization_spaces=True)
        self.model = model if model else BertModel.from_pretrained('bert-large-uncased')

    def forward(self, *args, **kwargs):
        # had to take out the kwargs from the forward method because python thought it was an init method instead
        """
        Accepts input in the form of a string, a list of strings, or a tensor of tokens (batched or not).

        - If a string is passed, it will be tokenized and encoded into concepts.
        - If a list of strings is passed, each string will be tokenized separately and encoded into concepts,
            with the results concatenated into a single tensor.
        - If a tensor of tokens is passed, it will be encoded into concepts.
        """
        if len(args) == 1 and isinstance(args[0], str):
            return self.encode_text(*args, **kwargs)
        if len(args) == 1 and isinstance(args[0], torch.Tensor):
            return self.encode_tokens(*args, **kwargs)
        if len(args) == 1 and isinstance(args[0], list):
            a0, *args = args
            return torch.concat([self.forward(x, *args, **kwargs) for x in a0], dim=0)

    def encode_text(self, text: str, encode_in_single_concept = False) -> torch.Tensor:
        # tokenize with BERT
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            pad_to_multiple_of=self.n_tokens_per_concept
        )

        inputs = inputs.to(self.model.device)

        return self._encode(inputs, encode_in_single_concept = encode_in_single_concept)

    def encode_tokens(self, tokens: torch.Tensor, attention_mask=None, encode_in_single_concept=False) -> torch.Tensor:
        tokens = tokens.to(self.model.device)

        # Ensure tokens have at least 2 dimensions
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)

        if attention_mask is not None:
            attention_mask = attention_mask.to(self.model.device)
        else:
            attention_mask = torch.ones_like(tokens, device=self.model.device)

        inputs = {
            'input_ids': tokens,
            'attention_mask': attention_mask,
        }

        return self._encode(inputs, encode_in_single_concept=encode_in_single_concept)

    def _encode(self, inputs: dict[str, torch.Tensor], encode_in_single_concept=False) -> torch.Tensor:
        original_shape = inputs['input_ids'].shape

        if not encode_in_single_concept:
            # Pad input_ids and attention_mask to multiple of n_tokens_per_concept
            seq_length = inputs['input_ids'].size(1)
            remainder = seq_length % self.n_tokens_per_concept
            if remainder != 0:
                pad_length = self.n_tokens_per_concept - remainder
                pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
                padding = torch.full((inputs['input_ids'].size(0), pad_length), pad_token_id,
                                     device=inputs['input_ids'].device)
                inputs['input_ids'] = torch.cat([inputs['input_ids'], padding], dim=1)
                attention_padding = torch.zeros((inputs['attention_mask'].size(0), pad_length),
                                                device=inputs['attention_mask'].device)
                inputs['attention_mask'] = torch.cat([inputs['attention_mask'], attention_padding], dim=1)

            # Split tokens into concepts
            inputs = {k: t.reshape(-1, self.n_tokens_per_concept) for k, t in inputs.items()}

        # Encode token groups into concepts
        return self._encode_tokens_into_concepts(inputs, original_shape=original_shape,
                                                 encode_in_single_concept=encode_in_single_concept)

    def _encode_tokens_into_concepts(self, inputs: dict[str, torch.Tensor], original_shape,
                                     no_grad=True, encode_in_single_concept = False) -> torch.Tensor:
        if no_grad:
            with torch.no_grad():
                output = self.model(**inputs).last_hidden_state.mean(dim=-2)
        else:
            output = self.model(**inputs).last_hidden_state.mean(dim=-2)

        return output.reshape(*original_shape[:-1], -1, output.shape[-1]) if not encode_in_single_concept else output

if __name__ == '__main__':
    encoder = Encoder(n_tokens_per_concept=4)
    text = ", 50 million tons of soil were blown"
    concepts = encoder(text)
    print(concepts)
    """tensor([[[-0.8756, -0.2732, -0.0387, ..., -0.5001, -1.2640, 0.7711],
             [-1.2281, -0.8402, -0.2846, ..., -0.6541, 0.0603, 0.6012],
             [-0.1364, -0.1694, 0.3193, ..., -0.4664, -0.5452, -0.1143]]])"""

    tokens = torch.tensor([[101, 1996, 4248, 2829, 4419, 14523, 2058, 1996, 13971, 3899, 1012, 102]])
    concepts = encoder(tokens)
    print(concepts)
