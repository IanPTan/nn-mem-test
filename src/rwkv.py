import numpy as np
import torch as pt



class S_rwkv(pt.nn.Module):

  def __init__(self, in_len):

    super(RWKV, self).__init__()
    
    self.last_x = pt.zeros(in_len)


  def forward(self, x):

    return []

