import numpy as np
import torch as pt



class mrwkv(pt.nn.Module):

  def __init__(self, in_len, mem_len=None, serial=False):

    super(mrwkv, self).__init__()

    self.serial=serial

    self.in_len = in_len
    if mem_len == None:
      mem_len = in_len
    self.mem_len = mem_len
    
    self.decay = pt.nn.Parameter(pt.zeros(mem_len))
    self.rkv_w = pt.nn.Parameter(pt.randn(mem_len * 3, in_len * 2))

    self.out_dense = pt.nn.Sequential(
        pt.nn.Linear(mem_len, mem_len),
        pt.nn.GELU(),
        pt.nn.Linear(mem_len, in_len),
        pt.nn.GELU(),
        )

    self.sigmoid = pt.nn.Sigmoid()
    self.softmax = pt.nn.Softmax(dim=-1)

    self.last_x = None
    self.mem = None

  def forward(self, x):

    if self.last_x == None:
      self.last_x = pt.zeros(x.shape, device=x.device)
    if self.mem == None:
      self.mem = pt.zeros((2, *x.shape[:-1], self.mem_len), device=x.device)

    xs = pt.concatenate((x, self.last_x), dim=-1)
    rkv = pt.tensordot(xs, self.rkv_w, dims=((-1,), (-1,)))
    r, k, v = rkv.reshape(3, *rkv.shape[:-1], self.mem_len)

    kv = pt.stack((pt.exp(k) * v, pt.exp(k)))
    d = pt.exp(-pt.exp(self.decay))

    if self.serial:
      self.mem = self.mem * d + kv
    else:
      self.mem[..., 0, :] = kv[..., 0, :]
      for i in range(1, kv.shape[-2]):
        self.mem[..., i, :] = self.mem[..., i - 1, :] * d + kv[..., i, :]

    wkv = self.mem[0] / self.mem[1]
    rwkv = self.sigmoid(r) * wkv

    self.last_x = x.clone()
    out = self.out_dense(rwkv)
    out = self.softmax(out + x)

    return out

  def reset(self):

    self.last_x = None
    self.mem = None


