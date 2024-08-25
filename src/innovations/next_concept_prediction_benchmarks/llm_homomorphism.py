# this script outputs the next concept from a given llm, decoding strategy, encoder, and input text
import torch
from transformers import GPT2TokenizerFast, GPT2LMHeadModel

from src.model.config import DEVICE
from src.model.encoder import Encoder
from llm_batch_creator import get_n_tokens_inference

def llm_next_concept_prediction(model, encoder, tokenizer, input_text, n_tokens_per_concept) -> torch.Tensor:
    """
    Predict the next concept given an input text, an encoder, and an llm.

    Args:
        llm: The language model to use for generating the response.
        encoder (Encoder): The encoder to use for encoding the text.
        tokenizer: The tokenizer to use for tokenizing the input.
        input_text (str): The input text to use for generating the response.
        n_tokens_per_concept (int): The number of tokens per concept.
    Returns:
        list[torch.Tensor]: The next concept.
    """
    encoded_input = tokenizer(input_text, return_tensors='pt').to(DEVICE)

    next_tokens, perplexity = get_n_tokens_inference(
        encoded_text=encoded_input,
        model=model,
        tokenizer=tokenizer,
        new_tokens_to_generate=n_tokens_per_concept,
        modality="greedy_rollout",
        print_sentence=True
    )
    next_concept, perplexity = encoder.encode_tokens_to_concepts(next_tokens, no_grad=True)

    return next_concept


if __name__ == '__main__':
    encoder = Encoder()
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2').to(DEVICE)

    input_text = "The quick brown fox"
    n_tokens_per_concept = 10
    next_concept = llm_next_concept_prediction(model, encoder, tokenizer, input_text, n_tokens_per_concept)
    print(next_concept)