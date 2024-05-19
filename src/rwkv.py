import numpy as np
import torch as pt



class Simple(pt.nn.Module):

  def __init__(self):

    super(Simple, self).__init__()

    self.model = pt.nn.Sequential(
        pt.nn.Linear(18, 128),
        pt.nn.BatchNorm1d(128),
        pt.nn.GELU(),
        pt.nn.Linear(128, 9),
        pt.nn.Softmax(dim=-1),
        )

  def forward(self, x):

    return self.model(x)


