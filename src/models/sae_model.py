import torch.nn as nn

class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim=1024, expansion_factor=64):
        super().__init__()
        hidden_dim = input_dim * expansion_factor
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(hidden_dim, input_dim))

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
