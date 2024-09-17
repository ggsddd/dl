import torch as t
from torch import nn
from torch.utils import data

class MyDataset(data.Dataset):
    def __init__(self):
        super().__init__()

    def __getitem__(self, index):
        return super().__getitem__(index)

    def __len__(self):
        return super().__len__()


















