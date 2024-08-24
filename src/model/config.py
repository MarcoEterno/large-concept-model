import torch

DEVICE = 'mps' if torch.backends.mps.is_built() else 'cuda' if torch.cuda.is_available() else 'cpu'
TOP_K = 5
HIGHER_THAN_P = 0.05  # tokens with probability less than this will be discarded.
# if no token is left, the token with the highest probability will be chosen
