import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

EPS: float = torch.finfo(torch.float32).eps
