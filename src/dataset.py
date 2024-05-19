import numpy as np
import torch as pt
from torch.utils.data import Dataset
from utils.tictactoe import to_states
import h5py as hp



class WinDataset(Dataset):

  def __init__(self, file_path, full_state=False):

    self.file = hp.File(file_path, 'r')
    self.parts = self.file['parts']
    self.data = self.file['games'][:self.parts[1]]
    self.length = self.data.shape[0]
    self.full_state = full_state

  def __len__(self):

    return self.length

  def __getitem__(self, ind):

    raw_data = self.data[ind]
    length = np.where(raw_data == -1)[0]
    if len(length):
      length = length[0]
    else:
      length = len(raw_data)
    data = pt.tensor(raw_data[:length], dtype=pt.int64)

    if self.full_state:
      return to_states(data)
    else:
      one_hot = pt.nn.functional.one_hot(data, num_classes=9)
      return one_hot


  def __del__(self):

    self.file.close()
