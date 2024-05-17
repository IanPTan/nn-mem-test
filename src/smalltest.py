import numpy as np
import torch as pt
from torch.nn import nn
from torch.utils.data import DataLoader
from dataset import WinDataset



class Simple(nn.Module):

  def __init__(self):

    super(Simple, self).__init__()

    self.model = nn.Sequential(
        nn.Linear(18, 128),
        nn.ReLU(),
        nn.Linear(128, 9),
        nn.Softmax(dim=-1),
        )

  def forward(self, x):

    return self.model(x)
