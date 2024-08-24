# this module takes text or tokens and returns the corresponding concepts
# for now, the concepts are all the same number of tokens. This is a limitation that will be removed in the future

import numpy as np
import torch
from torch.nn import CosineSimilarity
from transformers import BertTokenizer, BertModel


class Encoder:
    def __init__(self, n_tokens_per_concept: int = 5, tokenizer=None, model=None):
        self.n_tokens_per_concept = n_tokens_per_concept
        self.tokenizer = tokenizer if tokenizer else BertTokenizer.from_pretrained(
            'bert-base-uncased', clean_up_tokenization_spaces=True)
        self.model = model if model else BertModel.from_pretrained('bert-base-uncased')

    def split_tokens_in_batches(self, tokens: list[int], n_tokens_per_concept: int) -> list[list[int]]:
        n_batches = len(tokens) // n_tokens_per_concept
        token_groups = []
        for i in range(n_batches):
            token_groups.append(tokens[i * n_tokens_per_concept:(i + 1) * n_tokens_per_concept])
        # add last batch
        token_groups.append(tokens[n_batches * n_tokens_per_concept:])
        return token_groups

    def encode_tokens_to_concepts(self, tokens: list[int], no_grad=True) -> list[torch.tensor]:
        """
        Encodes a list of tokens into a list of concepts
        :param tokens: torch.tensor of tokens with size [1,n_tokens]
        :param no_grad: if True, the model will not store gradients
        :return: list of concepts with size [n_concepts, n_features_per_concept]
        """
        token_groups = self.split_tokens_in_batches(tokens, self.n_tokens_per_concept)
        concepts = []
        for token_group in token_groups:
            inputs = {'input_ids': token_group}
            if no_grad:
                with torch.no_grad():
                    concept = self.model(**inputs).last_hidden_state.mean(dim=1)
            else:
                concept = self.model(**inputs).last_hidden_state.mean(dim=1)
            concepts.append(concept)
        return concepts

    def encode_text_to_concepts(self, text: str) -> list[torch.tensor]:
        tokens = self.tokenizer.encode(text, return_tensors='pt')
        return self.encode_tokens_to_concepts(tokens)

    def get_similarity_between_sentences(self, sentence1: str, sentence2: str) -> np.ndarray:
        concepts1 = self.encode_text_to_concepts(sentence1)
        concepts2 = self.encode_text_to_concepts(sentence2)
        similarity = np.zeros([len(concepts1), len(concepts2)])

        for i, concept1 in enumerate(concepts1):
            for j, concept2 in enumerate(concepts2):
                similarity[i][j] = CosineSimilarity(dim=1)(concept1, concept2).item()

        return similarity


if __name__ == '__main__':
    def encode_text_to_concepts():
        encoder = Encoder()
        text = "The quick brown fox jumps over the lazy dog."
        concepts = encoder.encode_text_to_concepts(text)
        print(concepts)


    def get_similarity_between_sentences():
        encoder = Encoder()
        sentence1 = "The quick brown fox jumps over the lazy dog."
        sentence2 = "The quick brown fox jumps over the lazy cat."
        similarity = encoder.get_similarity_between_sentences(sentence1, sentence2)
        print(similarity)


    def split_tokens_in_batches():
        encoder = Encoder()
        tokens = list(range(100))
        n_tokens_per_concept = 10
        token_groups = encoder.split_tokens_in_batches(tokens, n_tokens_per_concept)
        print(token_groups)


    def encode_tokens_to_concepts():
        encoder = Encoder()
        tokens = list(range(100))
        concepts = encoder.encode_tokens_to_concepts(tokens)
        print(len(concepts), concepts[0].shape)

    # split_tokens_in_batches()
    # encode_tokens_to_concepts()
    # encode_text_to_concepts()
    # get_similarity_between_sentences()
