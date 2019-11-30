import numpy as np
import torch
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import random_split, DataLoader

from datagen import RoundsDataset


class Feedforward(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.LeakyReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, self.hidden_size // 4)
        self.relu2 = torch.nn.LeakyReLU()
        self.fc3 = torch.nn.Linear(self.hidden_size // 4, 1)
        # self.sigmoid = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        h1 = self.fc2(relu)
        r2 = self.relu2(h1)
        output = self.fc3(r2)
        output = self.sigmoid(output)
        return output


rd = RoundsDataset('/home/zhukov/clients/unicorn/unicorn/supercleanrounds.jsonl',
                   '/home/zhukov/clients/unicorn/unicorn/fund_industries.json',
                   '/home/zhukov/clients/unicorn/Starspace/fund2inv2.tsv',
                   '/home/zhukov/clients/unicorn/unicorn/startups.jsonl',
                   '/home/zhukov/clients/unicorn/Starspace/industries.tsv')

train_size = len(rd) * 80 // 100
validation_size = len(rd) - train_size
datasets = random_split(rd, [train_size, validation_size])
trainset, valset = datasets
print(f"train dataset size {len(trainset)}")
print(f"validation dataset size {len(valset)}")
trainloader = DataLoader(trainset, shuffle=True, batch_size=32)
valloader = DataLoader(valset, shuffle=False, batch_size=32)

model = Feedforward(200, 64)
lossf = MSELoss()
optimizer = Adam(params=model.parameters())
model.train()
epoch = 20
for epoch in range(epoch):
    for i, (vec, money) in enumerate(trainloader):
        optimizer.zero_grad()
        # Forward pass
        money_pred = model(vec)
        # Compute Loss
        loss = lossf(money_pred.squeeze(), money)

        print(f'Epoch {epoch}/{i}: train loss: {loss.item()}')
        # Backward pass
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        vlosses = []
        for i, (vec, money) in enumerate(valloader):
            money_pred = model(vec)
            expected_money = int(money[0].item() * rd.money_norm)
            actual_money = int(money_pred[0].item() * rd.money_norm)
            d = expected_money - actual_money
            print(f'expected ${expected_money} got ${actual_money} delta ${d} {int(d * 100 / (actual_money + 1))}%')
            vloss = lossf(money_pred.squeeze(), money)
            vlosses.append(vloss.item())
        vlosses = np.array(vlosses)
        print(f'Epoch {epoch}: validation loss: {np.median(vlosses)} {np.mean(vlosses)}')

    model.train()
