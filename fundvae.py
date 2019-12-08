import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from datagen import FundsDataset
from constants import FUNDINGTYPES, INDUSTRIES, SORTED_INDUSTRIES_IDXS, UNIQ_INDUSTRIES

# LATENT_DIM = 20
# Epoch 19, Train Loss: 13.13, Test Loss: 11.86
# Epoch 50, Train Loss: 7.36, Test Loss (5): 7.25
# Epoch 116, Train Loss: 0.093, Test Loss (10): 0.083 Best Loss: 0.063
# Epoch 119, Train Loss: 0.143, Test Loss (10): 0.132 Best Loss: 0.062
# Epoch 170, Train Loss: 0.042, Test Loss (10): 0.037 Best Loss: 0.035
# Epoch 233, Train Loss: 0.068, Test Loss (20): 0.143 Best Loss: 0.025

# LATENT_DIM = 32
# Epoch 140, Train Loss: 0.140, Test Loss (20): 0.051 Best Loss: 0.043

# LATENT_DIM = 10
# Epoch 32, Train Loss: 12.31, Test Loss (5): 11.20

# Epoch 56, Train Loss: 7.25, Test Loss (5): 6.98
# Epoch 59, Train Loss: 6.84, Test Loss (1): 7.83

# 512/64
# Epoch 92, Train Loss: 0.137, Test Loss (20): 0.032 Best Loss: 0.022

# 256/16
# Epoch 121, Train Loss: 0.033, Test Loss (20): 0.026 Best Loss: 0.021

# 128/16
# Epoch 156, Train Loss: 0.064, Test Loss (20): 0.039 Best Loss: 0.021
# Epoch 303, Train Loss: 0.024, Test Loss (40): 0.018 Best Loss: 0.017

# 128/8
# Epoch 375, Train Loss: 0.022, Test Loss (40): 0.016 Best Loss: 0.015

# 64/16
# Epoch 163, Train Loss: 0.028, Test Loss (20): 0.024 Best Loss: 0.022

BATCH_SIZE = 64  # number of data points in each batch
N_EPOCHS = 1000  # times to run the model on complete data
INPUT_DIM = FUNDINGTYPES * INDUSTRIES  # size of each input
HIDDEN_DIM = 1024  # hidden dimension
LATENT_DIM = 96  # latent vector dimension
PATIENCE = 40
lr = 0.0005  # learning rate


class Encoder(nn.Module):
    ''' This the encoder part of VAE
    '''

    def __init__(self, input_dim, hidden_dim, z_dim):
        '''
        Args:
            input_dim: A integer indicating the size of input (in case of MNIST 28 * 28).
            hidden_dim: A integer indicating the size of hidden dimension.
            z_dim: A integer indicating the latent dimension.
        '''
        super().__init__()

        self.linear = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.linear3 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.linear4 = nn.Linear(hidden_dim // 4, hidden_dim // 8)
        self.mu = nn.Linear(hidden_dim // 8, z_dim)
        self.var = nn.Linear(hidden_dim // 8, z_dim)

    def forward(self, x):
        # x is of shape [batch_size, input_dim]

        hidden = F.relu(self.linear4(self.linear3(self.linear2(self.linear(x)))))
        # hidden is of shape [batch_size, hidden_dim]
        z_mu = self.mu(hidden)
        # z_mu is of shape [batch_size, latent_dim]
        z_var = self.var(hidden)
        # z_var is of shape [batch_size, latent_dim]

        return z_mu, z_var


class Decoder(nn.Module):
    ''' This the decoder part of VAE
    '''

    def __init__(self, z_dim, hidden_dim, output_dim):
        '''
        Args:
            z_dim: A integer indicating the latent size.
            hidden_dim: A integer indicating the size of hidden dimension.
            output_dim: A integer indicating the output dimension (in case of MNIST it is 28 * 28)
        '''
        super().__init__()

        self.linear = nn.Linear(z_dim, hidden_dim // 8)
        self.linear1 = nn.Linear(hidden_dim // 8, hidden_dim // 4)
        self.linear2 = nn.Linear(hidden_dim // 4, hidden_dim // 2)
        self.linear3 = nn.Linear(hidden_dim // 2, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x is of shape [batch_size, latent_dim]

        hidden = F.relu(self.linear3(self.linear2(self.linear1(self.linear(x)))))
        # hidden is of shape [batch_size, hidden_dim]

        predicted = torch.sigmoid(self.out(hidden))
        # predicted is of shape [batch_size, output_dim]

        return predicted


class VAE(nn.Module):
    ''' This the VAE, which takes a encoder and decoder.
    '''

    def __init__(self, enc, dec):
        super().__init__()

        self.enc = enc
        self.dec = dec

    def forward(self, expected):
        # encode
        z_mu, z_var = self.enc(expected)

        # sample from the distribution having latent parameters z_mu, z_var
        # reparameterize
        std = torch.exp(z_var / 2)
        eps = torch.randn_like(std)
        x_sample = eps * std - z_mu

        # decode
        predicted = self.dec(x_sample)

        return predicted, z_mu, z_var


reconstruction_lossf = nn.BCELoss(reduction='none')


# reconstruction_lossf = nn.MSELoss(reduction='none')


def test():
    # set the evaluation mode
    model.eval()

    # test loss for the data
    test_loss = 0

    # we don't need to track the gradients, since we are not updating the parameters during evaluation / testing
    with torch.no_grad():
        for i, x in enumerate(test_iterator):
            # reshape the data
            x = x.view(-1, INPUT_DIM)
            x = x.to(device)

            # forward pass
            x_sample, z_mu, z_var = model(x)

            # reconstruction loss
            recon_loss = reconstruction_lossf(x_sample, x)
            mask = (((x.detach() != 0) * 1) + 0.1).clamp(0, 1)
            recon_loss = recon_loss * mask
            recon_loss = recon_loss.mean()

            # kl divergence loss
            kl_loss = 0.5 * torch.sum(torch.exp(z_var) + z_mu ** 2 - 1.0 - z_var)

            # total loss
            loss = recon_loss + kl_loss
            test_loss += loss.item()

    return test_loss


def train():
    # set the train mode
    model.train()

    # loss of the epoch
    train_loss = 0

    for i, x in enumerate(train_iterator):
        # reshape the data into [batch_size, 784]
        x = x.view(-1, INPUT_DIM)
        x = x.to(device)

        # update the gradients to zero
        optimizer.zero_grad()

        # forward pass
        x_sample, z_mu, z_var = model(x)

        # reconstruction loss
        recon_loss = reconstruction_lossf(x_sample, x)
        mask = (((x.detach() != 0) * 1) + 0.1).clamp(0, 1)
        recon_loss = recon_loss * mask
        recon_loss = recon_loss.mean()

        # kl divergence loss
        kl_loss = 0.5 * torch.sum(torch.exp(z_var) + z_mu ** 2 - 1.0 - z_var)

        # total loss
        loss = recon_loss + kl_loss

        # backward pass
        loss.backward()
        train_loss += loss.item()

        # update the weights
        optimizer.step()

    return train_loss


def showplot(expected, actual):
    expected = expected.view(FUNDINGTYPES, INDUSTRIES).cpu()
    actual = actual.view(FUNDINGTYPES, INDUSTRIES).cpu()
    plt.figure(figsize=(8, 6))
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)
    ax1.imshow(expected)
    ax2.imshow(actual)
    plt.show()


def showplot2(expected, actual):
    expected = expected.view(FUNDINGTYPES, INDUSTRIES)
    actual = actual.view(FUNDINGTYPES, INDUSTRIES)
    s = torch.cat((expected, actual))
    plt.figure()
    plt.imshow(s.cpu())
    plt.show()


if __name__ == '__main__':
    for i, name in enumerate(np.array(UNIQ_INDUSTRIES)[SORTED_INDUSTRIES_IDXS]):
        print(i, name)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    funds_dataset = FundsDataset.from_json('dataset/investor_profiles.json')
    train_dataset, test_dataset = funds_dataset.split(80)
    print('funds_dataset', len(funds_dataset), 'train_dataset', len(train_dataset), 'test_dataset', len(test_dataset))

    # encoder
    encoder = Encoder(INPUT_DIM, HIDDEN_DIM, LATENT_DIM)

    # decoder
    decoder = Decoder(LATENT_DIM, HIDDEN_DIM, INPUT_DIM)

    # vae
    model = VAE(encoder, decoder).to(device)
    print(model)

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_iterator = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_iterator = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    best_test_loss = float('inf')
    patience_counter = 1
    save_path = 'best4layer_vae.pth'
    for e in range(N_EPOCHS):
        train_loss = train()
        test_loss = test()

        train_loss /= len(train_dataset)
        test_loss /= len(test_dataset)

        print(
            f'Epoch {e}, Train Loss: {train_loss * 100000:.3f}, Test Loss ({patience_counter}): {test_loss * 100000:.3f} Best Loss: {best_test_loss * 100000:.3f}')

        if best_test_loss > test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), save_path)
            patience_counter = 1
        else:
            patience_counter += 1

        if patience_counter > PATIENCE:
            print("overfit detected")
            break

    mse = nn.MSELoss()


    def reconstruct(expected):
        actual, _, _ = model(expected.view(-1, INPUT_DIM))
        view = actual.view(FUNDINGTYPES, INDUSTRIES)
        showplot(expected, actual)
        for ftype, industries in enumerate(view):
            nonzero = (expected[ftype] != 0).sum().item()
            if nonzero > 0:
                print(ftype, nonzero)
                nz = expected[ftype].nonzero()
                e = expected[ftype][nz]
                a = industries[nz]
                _mse = mse(e, a)
                print('expected:', e.squeeze().cpu().numpy())
                print('actual  :', a.squeeze().cpu().numpy())
                print('mse     :', _mse.item())
                print('expected:', expected[ftype].median().item())
                print('actual  :', industries.median().item())


    model.load_state_dict(torch.load(save_path))

    model.eval()

    r = []
    with torch.no_grad():
        for i, expected in enumerate(test_dataset):
            expected = expected.to(device)
            actual, _, _ = model(expected.view(-1, INPUT_DIM))
            actual = actual.view(FUNDINGTYPES, INDUSTRIES)
            _mse = reconstruction_lossf(expected, actual).mean()
            nz = expected.nonzero()
            expected_ = expected != 0
            e = expected[expected_]
            a = actual[expected_]
            _msenz = mse(e, a)
            r.append([_mse.item(), _msenz.item(), expected, actual])

    rs = sorted(r, key=lambda x: x[0])
    print('reconstruction loss: best medium worst', rs[0][0], rs[len(rs) // 2][0], rs[-1][0])
    m, mnz, expected, actual = rs[0]
    showplot(expected, actual)
    m, mnz, expected, actual = rs[len(rs) // 2]
    showplot(expected, actual)
    m, mnz, expected, actual = rs[-1]
    showplot(expected, actual)
