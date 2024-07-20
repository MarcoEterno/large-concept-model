import torch
DEVICE = 'mps' if torch.backends.mps.is_built() else 'cuda' if torch.cuda.is_available() else 'cpu'
TOP_K = 5
TOP_P = 0.05