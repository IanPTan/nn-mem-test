import numpy as np
import torch as pt



class S_rwkv(pt.nn.Module):

  def __init__(self, in_len, mem_len=None):

    super(S_rwkv, self).__init__()

    self.in_len = in_len
    if mem_len == None:
      mem_len = in_len
    self.mem_len = mem_len
    
    self.last_x = pt.zeros(in_len)
    self.mem = pt.zeros((2, mem_len))

    self.decay = pt.nn.Parameter(pt.zeros(mem_len))
    self.rkv_w = pt.nn.Parameter(pt.randn(mem_len * 3, in_len * 2))

    self.out_dense = pt.nn.Sequential(
        pt.nn.Linear(mem_len, mem_len),
        pt.nn.GELU(),
        pt.nn.Linear(mem_len, in_len),
        pt.nn.GELU(),
        )

    self.sigmoid = pt.nn.Sigmoid()
    self.softmax = pt.nn.Softmax()

  def forward(self, x):

    xs = pt.concatenate((x, self.last_x))
    rkv = self.rkv_w @ xs
    r, k, v = rkv.reshape(3, self.mem_len)

    kv = pt.stack((pt.exp(k) * v, pt.exp(k)))
    d = pt.exp(-pt.exp(self.decay))
    self.mem = self.mem * d + kv
    wkv = self.mem[0] / self.mem[1]
    rwkv = self.sigmoid(r) * wkv

    self.last_x = x.clone()
    out = self.out_dense(rwkv)
    out = self.softmax(out + x, dim=-1)

    return out

  def reset(self):

    self.last_x *= 0
    self.meme *= 0


