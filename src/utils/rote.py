import numpy as np
import torch as pt
import h5py as hp



def save(file_path, data, starts):

  with hp.File(file_path, 'w') as file:
    file.create_dataset('games', data=data)
    file.create_dataset('parts', data=starts)



if __name__ == '__main__':

  length = 3
  combinations = pt.arange(10 ** length)
  inds = pt.zeros((10 ** length, length), dtype=pt.int64)
  for i in range(length):
    order = 10 ** (length - i - 1)
    inds[:, i] = combinations // order
    combinations = combinations % order
  data = pt.nn.functional.one_hot(inds)
  with hp.File('rote.h5', 'w') as file:
    file.create_dataset('codes', data=data)
