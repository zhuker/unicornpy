import numpy as np
import torch
from torch.nn import MSELoss, L1Loss
from torch.optim import Adam, SGD
from torch.utils.data import random_split, DataLoader

from datagen import RoundsDataset


# 64/64 b128
# Epoch 19: validation loss: 4.8198673539445736e-05 0.00017801336915519123
# 64/32 b128
# Epoch 19: validation loss: 2.2544363673659973e-05 6.383612474434943e-05
# 64/16 b128
# Epoch 19: validation loss: 4.469220402825158e-05 0.00019022614963302788
# 64/8 b128
# Epoch 19: validation loss: 4.868810538027901e-05 6.405707244994119e-05
# 128/16 b128
# Epoch 19: validation loss: 3.378191286174115e-05 5.9035281927728125e-05
# 256/32 b128
# Epoch 19: validation loss: 8.525216617272235e-05 0.00013184916945517346 $24M $82M
# 256/16 b128
# Epoch 19: validation loss: 4.464450103114359e-05 6.003693982035786e-05 $29M $77M
# 256/64 b128
# Epoch 19: validation loss: 2.6775723199534696e-05 5.0759616994712686e-05 $27M $65M
# 256/128 b128
# Epoch 14: validation loss: 4.23220972152194e-05 0.00010678049258266193 $20M $73M
# Epoch 19: validation loss: 4.108249777345918e-05 0.00010589461596412417 $23M $74M
# 256/128 b256
# Epoch 19: validation loss: 5.418228101916611e-05 6.14224770418202e-05 $18M $67M
# 256/128 b512
# Epoch 19: validation loss: 7.874847324274015e-05 8.169555439963005e-05 $12M $28M $86M

# 256/128 b256 onehot
# Epoch 19: validation loss: 0.000085 0.000086 $12M $13M $266M $59M
# Epoch 19: validation loss: 0.000109 0.000184 $12M 86% 2432% 438%
# Epoch 39: validation loss: 0.000054 0.000050 $13M 81% 520% 182%
# Epoch 99: validation loss: 0.000061 0.000055 $12M 83% 1738% 346%

# 256/128 b256 onehot+country
# Epoch 19: validation loss: 0.000024 0.000052 $12M $12M $205M $54M
#Epoch 19: validation loss: 0.000056 0.000176 $12M 95% 7530% 1039%
#Epoch 39: validation loss: 0.000034 0.000048 $12M 98% 6214% 983%

# 512/256 b256 onehot+country
# expected $249M got $0M delta $249M 865730%
# Epoch 19: validation loss: 0.000034 0.000046 $13M $13M $192M $54M

class Feedforward(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.LeakyReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, self.hidden_size // 2)
        self.relu2 = torch.nn.LeakyReLU()
        self.fc3 = torch.nn.Linear(self.hidden_size // 2, 1)
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


rd = RoundsDataset('dataset/supercleanrounds.jsonl',
                   'dataset/fund_industries.json',
                   'dataset/fund_industries.tsv',
                   'dataset/startups.jsonl',
                   'dataset/startup_industries.tsv',
                   'dataset/funds.jsonl')

train_size = len(rd) * 80 // 100
validation_size = len(rd) - train_size
datasets = random_split(rd, [train_size, validation_size])
trainset, valset = datasets
print(f"train dataset size {len(trainset)}")
print(f"validation dataset size {len(valset)}")
trainloader = DataLoader(trainset, shuffle=True, batch_size=256)
valloader = DataLoader(valset, shuffle=False, batch_size=256)
(vec, money) = rd[0]
print(f"investment round vec shape {vec.shape}")

model = Feedforward(vec.shape[0], 256)
dev = torch.device('cuda')
model.to(dev)
lossf = MSELoss()
optimizer = Adam(params=model.parameters(), lr=0.001)
# optimizer = SGD(params=model.parameters(), lr=0.001)
model.train()
epoch = 20
for epoch in range(epoch):
    for i, (vec, money) in enumerate(trainloader):
        optimizer.zero_grad()
        # Forward pass
        money_pred = model(vec.to(dev))
        # Compute Loss
        loss = lossf(money_pred.squeeze(), money.to(dev))

        print(f'Epoch {epoch}/{i}: train loss: {loss.item():.6f}')
        # Backward pass
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        vlosses = []
        dmoney = torch.tensor([]).to(dev)
        vmoney = torch.tensor([])
        for i, (vec, money) in enumerate(valloader):
            money_pred = model(vec.to(dev))
            norm = rd.money_norm
            expected_money = int(money[0].item() * norm)
            actual_money = int(money_pred[0].item() * norm)
            emoney = money.to(dev) * norm
            amoney = money_pred.squeeze() * norm
            dd = torch.abs(emoney - amoney)
            dd = dd * 100 / (amoney + 1)
            dmoney = torch.cat([dmoney, dd])
            vmoney = torch.cat([vmoney, money * norm])
            d = expected_money - actual_money
            print(
                f'expected ${int(expected_money / 1000000)}M got ${int(actual_money / 1000000)}M delta ${int(d / 1000000)}M {int(d * 100 / (actual_money + 1))}%')
            vloss = lossf(money_pred.squeeze(), money.to(dev))
            vlosses.append(vloss.item())
        vlosses = np.array(vlosses)
        mvloss = np.median(vlosses)
        meanvloss = np.mean(vlosses)
        print(
            f'Epoch {epoch}: validation loss: {mvloss:.6f} {meanvloss:.6f} ${int(vmoney.median() / 1000000)}M {int(dmoney.median())}% {int(dmoney.std())}% {int(dmoney.mean())}%')

    model.train()
