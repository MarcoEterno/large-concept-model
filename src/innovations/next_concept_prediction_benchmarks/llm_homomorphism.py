# this script outputs the next concept from a given llm, decoding strategy, encoder, and input text

from src.model.encoder import Encoder
from llm_batch_creator import get_n_tokens_inference

def llm_next_concept_prediction(llm, encoder, tokenizer, input_text, n_tokens_per_concept, n_concepts):
    """
    Predict the next concept given an input text, an encoder, and an llm.

    Args:
        llm: The language model to use for generating the response.
        encoder (Encoder): The encoder to use for encoding the text.
        tokenizer: The tokenizer to use for tokenizing the input.
        input_text (str): The input text to use for generating the response.
        n_tokens_per_concept (int): The number of tokens per concept.
        n_concepts (int): The number of concepts to generate.

    Returns:
        list[torch.Tensor]: The next concept.
    """
    encoded_text = tokenizer(input_text, return_tensors='pt')
    tokens = get_n_tokens_inference(encoded_text, llm, tokenizer, n_tokens_per_concept * n_concepts)
    return encoder.encode_tokens_to_concepts(tokens)