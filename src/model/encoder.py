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

    def forward(self, *args):
        # had to take out the kwargs from the forward method because python thought it was an init method instead
        """
        Accepts input in the form of a string, a list of strings, or a tensor of tokens (batched or not).

        - If a string is passed, it will be tokenized and encoded into concepts.
        - If a list of strings is passed, each string will be tokenized separately and encoded into concepts,
            with the results concatenated into a single tensor.
        - If a tensor of tokens is passed, it will be encoded into concepts.
        """
        if len(args) == 1 and isinstance(args[0], str):
            return self.encode_text(*args)
        if len(args) == 1 and isinstance(args[0], torch.Tensor):
            return self.encode_tokens(*args)
        if len(args) == 1 and isinstance(args[0], list):
            a0, *args = args
            return torch.concat([self.forward(x, *args) for x in a0], dim=0)

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

    def _encode(self, inputs: dict[str, torch.Tensor], encode_in_single_concept = False) -> torch.Tensor:
        original_shape = inputs['input_ids'].shape

        if not encode_in_single_concept:
            # split tokens into concepts
            inputs = {k: t.reshape(-1, self.n_tokens_per_concept) for k, t in inputs.items()}

        # encode token groups into concepts
        return self._encode_tokens_into_concepts(inputs, original_shape=original_shape, encode_in_single_concept = encode_in_single_concept)

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
    text = "The quick brown fox jumps over the lazy dog."
    concepts = encoder(text)
    print(concepts)

    tokens = torch.tensor([[101, 1996, 4248, 2829, 4419, 14523, 2058, 1996, 13971, 3899, 1012, 102]])
    concepts = encoder(tokens)
    print(concepts)
