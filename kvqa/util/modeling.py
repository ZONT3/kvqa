import torch
from torch import nn


class Module(nn.Module):
    """Необходимая база для всех модулей проекта"""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = detect_cuda_device(args)


def detect_cuda_device(device):
    if not device or device not in ('cuda', 'cpu'):
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        return device
