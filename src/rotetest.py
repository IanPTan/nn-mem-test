import matplotlib.pyplot as plt
import numpy as np
import torch as pt
from torch.utils.data import DataLoader, random_split
from datasets.rotedataset import RoteDataset
import rwkv



def test(ds, m, ind):
  model.reset()
  i, t = ds[ind]
  p = m(i)[5:]
  o = pt.argmax(p, dim=-1)
  return o, t, p



class Model(pt.nn.Module):

  def __init__(self, in_len, mem_len, lay_len, serial=False):

    super(Model, self).__init__()

    self.serial = serial

    self.in_len = in_len
    self.mem_len = mem_len

    self.rwkv_blocks = pt.nn.Sequential(*((rwkv.Block(mem_len, serial=serial),) * lay_len))

    self.model = pt.nn.Sequential(
        pt.nn.Linear(in_len, mem_len),
        rwkv.DenseNorm(mem_len, mem_len),
        self.rwkv_blocks,
        pt.nn.Linear(mem_len, in_len),
        pt.nn.Softmax(dim=-1),
        )

  def forward(self, x):

    return self.model(x)

  def reset(self):

    for block in self.rwkv_blocks:
      block.reset()

  def set_serial(self, state):

    for block in self.rwkv_blocks:
      block.set_serial(state)


epochs = 100
opt_num = 3
seq_len = 5
test_ratio = 0.1
batch_size = 32
hidden_size = 16
lay_len = 1

model = Model(opt_num + 1, hidden_size, lay_len)
dataset = RoteDataset(opt_num, seq_len)

test_size = int(test_ratio * len(dataset))
train_size = len(dataset) - test_size
batch_len = train_size // batch_size + bool(train_size % batch_size)

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

criterion = pt.nn.CrossEntropyLoss()
optimizer = pt.optim.Adam(model.parameters(), lr=1e-6, weight_decay=1e-6)

all_losses = pt.zeros(epochs)
pt.autograd.set_detect_anomaly(True)
print('Starting training.')
for epoch in range(epochs):
  losses = pt.zeros(batch_len)
  for i, (inputs, targets) in enumerate(train_loader):
    optimizer.zero_grad()
    model.reset()
    outputs = model(inputs)[:, seq_len:].permute(0, 2, 1)
    loss = criterion(outputs, targets)
    losses[i] = loss
    loss.backward()
    optimizer.step()
  all_losses[epoch] = pt.mean(losses)
  print(f'Epoch {epoch + 1}, Loss: {pt.mean(losses)}')

plt.plot(all_losses.detach())
plt.show()
