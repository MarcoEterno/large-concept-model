# In this module, we create a wrapper for every LLM,
# which allows us to easily extract the model probabilities for the next word

from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

from src.config import DEVICE, TOP_K, TOP_P
from src.utils import timer


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
    # avoiding saving the gradients to save computation and memory:
    with torch.no_grad():
        model.eval()
        outputs = model(**encoded_text, output_hidden_states=True, return_dict=True)
        probabilities = outputs.logits[:, -1, :].softmax(dim=-1)
        if return_type == 'top_k':
            print('top_k')
            return probabilities.topk(TOP_K)
        elif return_type == 'top_p':
            print('top_p')
            return probabilities[probabilities > TOP_P]
        else:
            print('all')
            return probabilities
    # execution times greatly different between top_k, top p and all
    # top_k: 213 ms
    # top_p: 10-13 ms
    # all: 5-13 mss
    # Luckily tree exploration is better served with top_p, as it allows to keep the lost probability mass under control


if __name__ == "__main__":
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)  # you can bring it to a device here
    input_text = "basketball is a fun game to play"
    encoded_input = tokenizer(input_text, return_tensors="pt").to(DEVICE)
    print(encoded_input)
    outputs = model(**encoded_input, output_hidden_states=True, return_dict=True)
    print(outputs.logits.shape)

    print(get_GPT2_probabilities(encoded_input, model, return_type='all'))

