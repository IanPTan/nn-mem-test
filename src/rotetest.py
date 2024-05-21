import numpy as np
import torch as pt
from torch.utils.data import DataLoader, random_split
from datasets.rotedataset import RoteDataset
import rwkv



class Model(pt.nn.Module):

  def __init__(self, in_len, mem_len, serial=False):

    super(Model, self).__init__()

    self.serial = serial

    self.in_len = in_len
    self.mem_len = mem_len

    self.model = pt.nn.Sequential(
        pt.nn.Linear(in_len, mem_len),
        rwkv.DenseNorm(mem_len, mem_len),
        rwkv.Block(mem_len, serial=serial),
        rwkv.Block(mem_len, serial=serial),
        rwkv.Block(mem_len, serial=serial),
        rwkv.Block(mem_len, serial=serial),
        pt.nn.Linear(mem_len, in_len),
        pt.nn.Softmax(dim=-1),
        )

  def forward(self, x):

    return self.model(x)



opt_num = 3
seq_len = 5
model = Model(opt_num + 1, 16)
dataset = RoteDataset(opt_num, seq_len)

test_ratio = 0.1
test_size = int(test_ratio * len(dataset))
train_size = len(dataset) - test_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

criterion = pt.nn.CrossEntropyLoss()
optimizer = pt.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

pt.autograd.set_detect_anomaly(True)
for epoch in range(100):
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)[:, seq_len:].permute(0, 2, 1)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')
