import matplotlib.pyplot as plt
import numpy as np
import torch as pt
from torch.utils.data import DataLoader, random_split
from datasets.rotedataset import RoteDataset
import rwkv


device = pt.device('cuda' if pt.cuda.is_available() else 'cpu')

def test(ds, m, ind, full=False):

  model.reset()
  i, t = ds[ind]
  i, t = i.to(device), t.to(device)
  p = m(i)[5:]
  o = pt.argmax(p, dim=-1)

  if full:
    return o, t, p
  else:
    print(f'Sequence comparison for datapoint {ind}:')
    for it, io in zip(t, o):
      mark = '✔' if it == io else 'WRONG'
      print(f'\t{it}\t{io}\t{mark}')

class Model(pt.nn.Module):

  def __init__(self, in_len, out_len, mem_len, lay_len, serial=False):

    super(Model, self).__init__()

    self.serial = serial

    self.in_len = in_len
    self.mem_len = mem_len

    self.rwkv_blocks = pt.nn.Sequential(*((rwkv.Block(mem_len, serial=serial),) * lay_len))

    self.model = pt.nn.Sequential(
        pt.nn.Linear(in_len, mem_len),
        rwkv.DenseNorm(mem_len, mem_len),
        self.rwkv_blocks,
        pt.nn.Linear(mem_len, out_len),
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

epochs = 5000
opt_num = 3
seq_len = 10
test_ratio = 0.1
batch_size = 10000
hidden_size = 32
lay_len = 4

model = Model(opt_num + 1, opt_num, hidden_size, lay_len).to(device)
dataset = RoteDataset(opt_num, seq_len)

test_size = int(test_ratio * len(dataset))
train_size = len(dataset) - test_size

batch_len = train_size // batch_size + bool(train_size % batch_size)

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

criterion = pt.nn.CrossEntropyLoss()
optimizer = pt.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

all_losses = pt.zeros(epochs)
pt.autograd.set_detect_anomaly(True)
print(f'Starting training using {device}.')
for epoch in range(epochs):
  losses = pt.zeros(batch_len)
  for i, (inputs, targets) in enumerate(train_loader):
    inputs, targets = inputs.to(device), targets.to(device)
    optimizer.zero_grad()
    model.reset()
    outputs = model(inputs)[:, seq_len:].permute(0, 2, 1)
    loss = criterion(outputs, targets)
    losses[i] = loss
    loss.backward()
    optimizer.step()
  all_losses[epoch] = pt.mean(losses)
  print(f'Epoch {epoch + 1}, Loss: {pt.mean(losses)}')
pt.save(model.state_dict(), 'backup.ckpt')
plt.plot(all_losses.detach())
plt.show()
