import numpy as np
import torch as pt



class DenseNorm(pt.nn.Module):

  def __init__(self, in_len, out_len):

    super(DenseNorm, self).__init__()

    self.dense = pt.nn.Linear(in_len, out_len)

  def forward(self, x):

    x_norm = (x - pt.mean(x)) / pt.std(x)

    out = self.dense(x_norm)

    return out

class Attention(pt.nn.Module):

  def __init__(self, in_len, mem_len=None, serial=False):

    super(Attention, self).__init__()

    self.serial=serial

    self.in_len = in_len
    if mem_len == None:
      mem_len = in_len
    self.mem_len = mem_len
    
    self.decay = pt.nn.Parameter(pt.zeros(mem_len))
    self.rkv_w = pt.nn.Parameter(pt.randn(mem_len * 3, in_len * 2))
    self.out_w = pt.nn.Parameter(pt.randn(in_len, mem_len))

    self.sigmoid = pt.nn.Sigmoid()

    self.last_x = None
    self.mem = None

  def forward(self, x):

    if self.last_x == None and self.serial:
      self.last_x = pt.zeros(x.shape, device=x.device)
    if self.last_x == None and not self.serial:
      self.last_x = pt.cat((pt.zeros((*x.shape[:-2], 1, x.shape[-1]), device=x.device), x[..., :-1, :]), dim=-2)

    xs = pt.concatenate((x, self.last_x), dim=-1)
    rkv = pt.tensordot(xs, self.rkv_w, dims=((-1,), (-1,)))
    r, k, v = rkv.reshape(3, *rkv.shape[:-1], self.mem_len)

    kv = pt.stack((pt.exp(k) * v, pt.exp(k)))
    d = pt.exp(-pt.exp(self.decay))

    if self.mem == None and self.serial:
      self.mem = kv
    if self.mem != None and self.serial:
      self.mem = self.mem * d + kv

    if not self.serial:
      # need to make zero tensor for start or accept existing
      self.mem[..., 0, :] = kv[..., 0, :]
      for i in range(1, kv.shape[-2]):
        self.mem[..., i, :] = self.mem[..., i - 1, :] * d + kv[..., i, :]

    wkv = self.mem[0] / self.mem[1]
    rwkv = self.sigmoid(r) * wkv

    self.last_x = x.clone()
    out = pt.tensordot(rwkv, self.out_w, dims=((-1,), (-1,)))

    return out

  def reset(self):

    self.last_x = None
    self.mem = None

class Block(pt.nn.Module):

  def __init__(self, in_len, mem_len=None, serial=False):

    super(Block, self).__init__()

    self.serial = serial
    if mem_len == None:
      mem_len = in_len
    self.mem_len = mem_len

    self.block1 = pt.nn.Sequential(
        DenseNorm(in_len, in_len),
        Attention(in_len, mem_len = mem_len, serial=serial),
        )

    self.block2 = pt.nn.Sequential(
        DenseNorm(in_len, in_len),
        pt.nn.Linear(in_len, in_len),
        pt.nn.GELU(),
        pt.nn.Linear(in_len, in_len),
        pt.nn.GELU(),
        )

  def forward(self, x):

    x = self.block1(x) + x
    x = self.block2(x) + x

    return x
