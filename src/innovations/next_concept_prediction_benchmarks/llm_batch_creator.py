import numpy as np
from collections import defaultdict
import torch
from transformers import GPT2TokenizerFast, GPT2LMHeadModel

from src.model.config import DEVICE, TOP_K, HIGHER_THAN_P
from src.utils.utils import timer



@timer
def get_llm_probabilities(encoded_text, model, return_type, device=DEVICE):
    """
    Get the probabilities of the next word given the input text.

    Args:
        encoded_text (dict): The input text encoded by the tokenizer. It should be a dictionary, with keys:
            - input_ids: The tokens of the text.
            - attention_mask: The attention mask of the text. (all 1)
        model: The model to use for generating the response.
        return_type (str): The type of return. It can be 'top_k', 'top_p','all' to return the top k probabilities,
            the probabilities higher than TOP_P, or the probabilities for the whole vocabulary.
            never use top k, it is 20x slower than the other methods in our GPT2 preliminary testing.
        device (torch.device): The device to use for the model.

    Returns:
        torch.Tensor: The probabilities of the next word.
    """
    # avoid saving the gradients to save computation and memory:
    with torch.no_grad():
        model.eval()
        outputs = model(**encoded_text, output_hidden_states=True, return_dict=True)
        probabilities = outputs.logits[:, -1, :].softmax(dim=-1)
        if return_type == 'top_k':
            return probabilities.topk(TOP_K)
        elif return_type == 'top_p':
            return probabilities[probabilities > HIGHER_THAN_P]
        else:
            return probabilities

def get_n_tokens_inference(
        encoded_text: dict[str, torch.Tensor],
        model,
        tokenizer,
        new_tokens_to_generate: int,
        modality: str = "greedy_rollout",
        print_sentence: bool = False
):
    """
    Rollout policy to complete a sentence.

    Args:
        encoded_text (dict[str, torch.Tensor]) : The sentence to complete as returned by the tokenizer.
        model (GPT2LMHeadModel): The model to use for generating the response.
        tokenizer (GPT2Tokenizer): The tokenizer to use for tokenizing the input.
        new_tokens_to_generate (int): The maximum number of tokens that generation will last.
        modality (str): The modality to use for the rollout. It can be 'greedy_rollout' or 'top_p_rollout'.

    Returns:
        float: the perplexity of the completed sentence
        torch.Tensor : the tokens of the completed sentence
    """
    if modality == "greedy_rollout":
        # Sentence is completed by extracting the next token with the highest probability.
        # This modality makes the model deterministic, so calculating the value multiple times does not improve accuracy
        perplexity = 0.0
        new_tokens = 0

        while new_tokens < new_tokens_to_generate:
              # and not encoded_text['input_ids'][0][-1] == tokenizer.eos_token_id):  # attention: eos_token_id is 50256
            # for GPT2 but could be None for other LLMs, so please use is None instead of == in that case.
            probabilities = get_llm_probabilities(encoded_text, model, return_type='all')
            next_token = probabilities.argmax()
            if (print_sentence):
                print(tokenizer.decode(encoded_text['input_ids'][0]))
            encoded_text['input_ids'] = torch.cat(
                (encoded_text['input_ids'], torch.tensor([[next_token]], device=DEVICE)), dim=1)
            encoded_text['attention_mask'] = torch.cat(
                [encoded_text['attention_mask'], torch.tensor([[1]], device=DEVICE)], dim=-1)
            p = probabilities[0][next_token].cpu().numpy()
            perplexity -= p * np.log(p)
            new_tokens += 1
        return encoded_text["input_ids"][0][-new_tokens_to_generate:-1], perplexity

if __name__ == "__main__":

    def a_test_get_n_tokens_inference():
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
        input_text = "basketball is a fun game to"
        encoded_input = tokenizer(input_text, return_tensors="pt").to(DEVICE)
        tokens, perplexity  = get_n_tokens_inference(
            encoded_input,
            model=model,
            tokenizer=tokenizer,
            new_tokens_to_generate=40,
            modality="greedy_rollout",
            print_sentence=False
        )
        print(tokenizer.decode(tokens), perplexity)


    a_test_get_n_tokens_inference()

    # we want the flow to be

"""
- [ ] Create a BERT model that takes in input some text (str) and outputs concepts in a vector of dim D.
- [ ] Create a LCM (GPT2*NP), an LLM that takes in input vectors of dim D and outputs vectors not projected.
      This LLM has no tokenizer, no embedding, but positional embedding.
- [ ] BERTize training text, define loss in concept space (cosine similarity).
- [ ] Choose model size for LLM (GPT2 or smaller - we need to win) and LCM (124M params).
- [ ] Train LCM.
- [ ] Train LLM.
- [ ] Benchmark loss LCM vs LLM + BERT.
"""
