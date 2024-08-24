# In this module, we create a wrapper for every LLM,
# which allows us to easily extract the model probabilities for the next word

from typing import List

import numpy as np
import torch
from transformers import GPT2TokenizerFast, GPT2LMHeadModel

from src.model.config import DEVICE, TOP_K, HIGHER_THAN_P
from src.utils.utils import timer


@timer
def get_GPT2_probabilities(encoded_text, model: GPT2LMHeadModel, return_type, device=DEVICE):
    """
    Get the probabilities of the next word given the input text.

    Args:
        encoded_text (dict): The input text encoded by the tokenizer. It should be a dictionary, with keys:
            - input_ids: The tokens of the text.
            - attention_mask: The attention mask of the text. (all 1)
        model (GPT2LMHeadModel): The model to use for generating the response.
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
            print('top_k')
            return probabilities.topk(TOP_K)
        elif return_type == 'top_p':
            print('top_p')
            return probabilities[probabilities > HIGHER_THAN_P]
        else:
            print('all')
            return probabilities
    # execution times greatly different between top_k, top p and all
    # top_k: 213 ms
    # top_p: 10-13 ms
    # all: 5-13 ms
    # Luckily tree exploration is better served with top_p, as it allows to keep the lost probability mass under control


def rollout_policy(
        previous_tokens: List[int],
        model: GPT2LMHeadModel,
        tokenizer: GPT2TokenizerFast,
        rollout_max_new_tokens: int,
        modality: str,
        print_sentence: bool = False
):
    """
    Rollout policy to complete a sentence.

    Args:
        previous_tokens (List[int]): The sentence to complete.
        model (GPT2LMHeadModel): The model to use for generating the response.
        tokenizer (GPT2Tokenizer): The tokenizer to use for tokenizing the input.
        rollout_max_new_tokens (int): The maximum number of tokens that generation will last.
        modality (str): The modality to use for the rollout. It can be 'greedy_rollout' or 'top_p_rollout'.

    Returns:
        float: the value of the completed sentence
        int: the number of tokens added to the sentence
    """
    if modality == "greedy_rollout":
        # Sentence is completed by extracting the next token with the highest probability.
        # This modality makes the model deterministic, so calculating the value multiple times does not improve accuracy
        encoded_text = convert_token_list_for_inference(previous_tokens)
        value_added = 0
        new_tokens = 0

        while (len(encoded_text['input_ids'][0]) < rollout_max_new_tokens
               and not encoded_text['input_ids'][0][-1] == tokenizer.eos_token_id):  # attention: eos_token_id is 50256
            # for GPT2 but could be None for other LLMs, so please use is None instead of == in that case.
            probabilities = get_GPT2_probabilities(encoded_text, model, return_type='all')
            next_token = probabilities.argmax()
            if (print_sentence):
                print(tokenizer.decode(encoded_text['input_ids'][0]))
            encoded_text['input_ids'] = torch.cat(
                (encoded_text['input_ids'], torch.tensor([[next_token]], device=DEVICE)), dim=1)
            encoded_text['attention_mask'] = torch.cat(
                [encoded_text['attention_mask'], torch.tensor([[1]], device=DEVICE)], dim=-1)
            p = probabilities[0][next_token].cpu().numpy()
            value_added -= p * np.log(p)
            new_tokens += 1
        return value_added, new_tokens
    if modality == "top_p_rollout":
        # Sentence is completed by extracting the next token from the ones with probability > TOP_P
        encoded_text = convert_token_list_for_inference(previous_tokens)
        value_added = 0
        new_tokens = 0

        while (len(encoded_text['input_ids'][0]) < rollout_max_new_tokens
               and not encoded_text['input_ids'][0][-1] == tokenizer.eos_token_id):  # attention: eos_token_id is 50256
            # for GPT2 but could be None for other LLMs, so please use is None instead of == in that case.

            probabilities = get_GPT2_probabilities(encoded_text, model, return_type='all')
            next_token = higher_than_p_sampling(probabilities, top_p=HIGHER_THAN_P)
            if (print_sentence):
                print(tokenizer.decode(encoded_text['input_ids'][0]))
            encoded_text['input_ids'] = torch.cat(
                (encoded_text['input_ids'], torch.tensor([[next_token]], device=DEVICE)), dim=1)
            encoded_text['attention_mask'] = torch.cat(
                [encoded_text['attention_mask'], torch.tensor([[1]], device=DEVICE)], dim=-1)
            p = probabilities[0][next_token].cpu().numpy()
            value_added -= p * np.log(p)
            new_tokens += 1
        return value_added, new_tokens


def top_p_sampling(probabilities: torch.Tensor, top_p: float) -> torch.Tensor:
    """
    Sample a token from the probabilities with probability mass higher than top_p.
    Also known as nucleus sampling.

    Args:
        probabilities (torch.Tensor): The probabilities of the next word.
        top_p (float): The minimum probability mass to keep.

    Returns:
        torch.Tensor: The sampled token.
    """
    sorted_probabilities, sorted_indices = torch.sort(probabilities, descending=True)
    cumulative_probabilities = torch.cumsum(sorted_probabilities, dim=-1)
    sorted_indices_to_remove = cumulative_probabilities > top_p
    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
    sorted_indices_to_remove[0] = False
    probabilities[sorted_indices[sorted_indices_to_remove]] = 0
    probabilities /= probabilities.sum()
    return torch.multinomial(probabilities, 1)


@timer
def convert_token_list_for_inference(token_list: List[int], device: torch.device = DEVICE) -> dict[str, torch.Tensor]:
    """
    Convert a list of tokens to a dictionary of torch tensors for inference.

    Args:
        token_list (List[int]): The list of tokens to convert.

    Returns:
        dict[str, Tensor]: The dictionary containing the encoded text and the attention mask.
    """
    input_ids = torch.tensor([token_list], device=device)

    # Create an attention_mask tensor with the same shape and device
    attention_mask = torch.ones_like(input_ids, device=device)

    # Resulting tensors
    inference_dictionary = {
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }

    return inference_dictionary


if __name__ == "__main__":
    def test_top_p_sampling():
        probabilities = torch.tensor([0.1, 0.2, 0.3, 0.4])
        print(top_p_sampling(probabilities, 0.71))

        tokens= []
        for i in range (100):
            tokens.append(top_p_sampling(probabilities, 0.71))

        #print the frequency of each token
        print(torch.bincount(torch.cat(tokens)).float() /100)


    def test_get_GPT2_probabilities():
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)  # you can bring it to a device here
        input_text = "basketball is a fun game to play"
        encoded_input = tokenizer(input_text, return_tensors="pt").to(DEVICE)
        # print(encoded_input)
        outputs = model(**encoded_input, output_hidden_states=True, return_dict=True)
        # print(outputs.logits.shape)
        # print(get_GPT2_probabilities(encoded_input, model, return_type='all'))


    def test_rollout_policy():
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)  # you can bring it to a device here
        previous_tokens = tokenizer("I like ice").input_ids
        print(previous_tokens)
        print(rollout_policy(previous_tokens, model, tokenizer, 100, "top_p_rollout", print_sentence=True))

    test_top_p_sampling()