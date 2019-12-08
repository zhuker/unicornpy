# encoder
import torch
from torch import nn

import datagen
from constants import UNIQ_INDUSTRIES, UNIQ_FUNDINGTYPES, INDUSTRIES, FUNDINGTYPES
from fundvae import Encoder, INPUT_DIM, HIDDEN_DIM, LATENT_DIM, Decoder, VAE
import matplotlib.pyplot as plt

assert len(UNIQ_INDUSTRIES) == INDUSTRIES
assert len(UNIQ_FUNDINGTYPES) == FUNDINGTYPES


def showplot(expected, actual):
    plt.figure()
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)
    ax1.imshow(expected.view(FUNDINGTYPES, INDUSTRIES).cpu())
    ax2.imshow(actual.view(FUNDINGTYPES, INDUSTRIES).cpu())
    plt.show()


encoder = Encoder(INPUT_DIM, HIDDEN_DIM, LATENT_DIM)

# decoder
decoder = Decoder(LATENT_DIM, HIDDEN_DIM, INPUT_DIM)

# vae
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = VAE(encoder, decoder).to(device)

model.load_state_dict(torch.load('bestvae.pth', map_location=torch.device(device)))

model.eval()
funds_dataset = datagen.FundsDataset.from_json('dataset/investor_profiles.json')
train_dataset, test_dataset = funds_dataset.split(80)
test_dataset.fundprofiles = sorted(test_dataset.fundprofiles, key=lambda x: len(x[1]), reverse=True)

r = []
with torch.no_grad():
    mse = nn.MSELoss()
    for i, expected in enumerate(test_dataset):
        expected = expected.to(device)
        actual, _, _ = model(expected.view(-1, INPUT_DIM))
        actual = actual.view(FUNDINGTYPES, INDUSTRIES)
        _mse = mse(expected, actual)
        nz = expected.nonzero()
        expected_ = expected != 0
        e = expected[expected_]
        a = actual[expected_]
        _msenz = mse(e, a)
        r.append([_mse.item(), _msenz.item(), expected, actual])

rs = sorted(r, key=lambda x: x[0])
m, mnz, expected, actual = rs[0]
showplot(expected, actual)
# for m, mnz, expected, actual in rs:
#     e = expected[expected != 0]
#     a = actual[expected != 0]
#     print(m, mnz)
#     print(e)
#     print(a)
