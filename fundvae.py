import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from datagen import FundsDataset

# LATENT_DIM = 20
# Epoch 19, Train Loss: 13.13, Test Loss: 11.86
# Epoch 50, Train Loss: 7.36, Test Loss (5): 7.25
# Epoch 116, Train Loss: 0.093, Test Loss (10): 0.083 Best Loss: 0.063
# Epoch 119, Train Loss: 0.143, Test Loss (10): 0.132 Best Loss: 0.062

# LATENT_DIM = 10
# Epoch 32, Train Loss: 12.31, Test Loss (5): 11.20

# Epoch 56, Train Loss: 7.25, Test Loss (5): 6.98
# Epoch 59, Train Loss: 6.84, Test Loss (1): 7.83

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
funds_dataset = FundsDataset.from_json('dataset/fundprofiles1.json')
FTYPES = funds_dataset.ftypes
INDUSTRIES = funds_dataset.industries

BATCH_SIZE = 64  # number of data points in each batch
N_EPOCHS = 1000  # times to run the model on complete data
INPUT_DIM = FTYPES * INDUSTRIES  # size of each input
HIDDEN_DIM = 256  # hidden dimension
LATENT_DIM = 20  # latent vector dimension
PATIENCE = 10
lr = 1e-3  # learning rate

train_dataset, test_dataset = funds_dataset.split(80)
print('funds_dataset', len(funds_dataset), 'train_dataset', len(train_dataset), 'test_dataset', len(test_dataset))


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
        self.mu = nn.Linear(hidden_dim, z_dim)
        self.var = nn.Linear(hidden_dim, z_dim)

    def forward(self, x):
        # x is of shape [batch_size, input_dim]

        hidden = F.relu(self.linear(x))
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

        self.linear = nn.Linear(z_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x is of shape [batch_size, latent_dim]

        hidden = F.relu(self.linear(x))
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

    def forward(self, x):
        # encode
        z_mu, z_var = self.enc(x)

        # sample from the distribution having latent parameters z_mu, z_var
        # reparameterize
        std = torch.exp(z_var / 2)
        eps = torch.randn_like(std)
        x_sample = eps.mul(std).add_(z_mu)

        # decode
        predicted = self.dec(x_sample)
        return predicted, z_mu, z_var


# encoder
encoder = Encoder(INPUT_DIM, HIDDEN_DIM, LATENT_DIM)

# decoder
decoder = Decoder(LATENT_DIM, HIDDEN_DIM, INPUT_DIM)

# vae
model = VAE(encoder, decoder).to(device)

# optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)

train_iterator = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_iterator = DataLoader(test_dataset, batch_size=BATCH_SIZE)


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
            recon_loss = F.binary_cross_entropy(x_sample, x, reduction='mean')

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
        recon_loss = F.binary_cross_entropy(x_sample, x, reduction='mean')

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


best_test_loss = float('inf')

patience_counter = 1
for e in range(N_EPOCHS):
    train_loss = train()
    test_loss = test()

    train_loss /= len(train_dataset)
    test_loss /= len(test_dataset)

    print(
        f'Epoch {e}, Train Loss: {train_loss * 100000:.3f}, Test Loss ({patience_counter}): {test_loss * 100000:.3f} Best Loss: {best_test_loss * 100000:.3f}')

    if best_test_loss > test_loss:
        best_test_loss = test_loss
        torch.save(model.state_dict(), 'bestvae.pth')
        patience_counter = 1
    else:
        patience_counter += 1

    if patience_counter > PATIENCE:
        print("overfit detected")
        break

model.load_state_dict(torch.load('bestvae.pth'))

model.eval()
with torch.no_grad():
    expected = test_dataset[42].to(device)
    actual, _, _ = model(expected.view(-1, INPUT_DIM))
    view = actual.view(FTYPES, INDUSTRIES)
    for ftype, industries in enumerate(view):
        nonzero = (expected[ftype] != 0).sum().item()
        if nonzero > 0:
            print(ftype, nonzero)
            nz = expected[ftype].nonzero()
            e = expected[ftype][nz]
            a = industries[nz]
            print('expected:', e.squeeze().cpu().numpy())
            print('actual  :', a.squeeze().cpu().numpy())
            print('expected:', expected[ftype].median().item())
            print('actual  :', industries.median().item())
