import numpy as np


_cross = [[[0, 1, 2], [0, 1, 2]], [[0, 1, 2], [2, 1, 0]]]


def check_won(state):
  sums = np.concatenate([state.sum(axis=0),
                         state.sum(axis=1),
                         state[*_cross].sum(axis = 1),
                         ])
  return 3 in abs(sums)

def check_dumb(state, player):
  split_state = np.stack((state < 0, state > 0), axis=0)
  vert_sums = split_state.sum(axis=1)
  hori_sums = split_state.sum(axis=2)
  cross_sums = split_state[:, *_cross].sum(axis = 2)
  return split_state, vert_sums, hori_sums, cross_sums

def search(state, possible):
  pass





state = np.zeros((3, 3), np.int8)
