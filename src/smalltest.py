import numpy as np
import torch as pt
from torch.utils.data import DataLoader, random_split
from datasets.tttdatset import WinDataset



class Simple(pt.nn.Module):

  def __init__(self):

    super(Simple, self).__init__()

    self.model = pt.nn.Sequential(
        pt.nn.Linear(18, 128),
        pt.nn.BatchNorm1d(128),
        pt.nn.GELU(),
        pt.nn.Linear(128, 9),
        pt.nn.Softmax(dim=-1),
        )

  def forward(self, x):

    return self.model(x)



model = Simple()
dataset = WinDataset('dataset.h5')

test_ratio = 0.1
test_size = int(test_ratio * len(dataset))
train_size = len(dataset) - test_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

criterion = pt.nn.CrossEntropyLoss()
optimizer = pt.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

for epoch in range(100):
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')
