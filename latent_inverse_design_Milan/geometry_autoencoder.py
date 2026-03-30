import torch
import torch.nn as nn

class GeometryAE(nn.Module):

    def __init__(self, latent_dim=32):

        super().__init__()

        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(100,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim,64),
            nn.ReLU(),
            nn.Linear(64,128),
            nn.ReLU(),
            nn.Linear(128,100),

        )

    def encode(self,x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self,x):
        z = self.encode(x)
        x_rec = self.decode(z)
        return x_rec


def train_autoencoder(model, train_loader, valid_loader, epochs=80, lr=1e-5, device="cpu"):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):

        model.train()
        train_loss = 0

        for x_batch, _ in train_loader:

            x_batch = x_batch.to(device)

            x_batch.requires_grad_(True)

            optimizer.zero_grad()

            z = model.encode(x_batch)
            x_rec = model.decode(z)

            recon_loss = criterion(x_rec, x_batch)
            latent_loss = torch.mean(z**2)

            loss = recon_loss + 1e-3 * latent_loss


            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for x_batch_val, _ in valid_loader:
                x_batch_val = x_batch_val.to(device)
                z_val = model.encode(x_batch_val)
                x_rec_val = model.decode(z_val)
                recon_loss_val = criterion(x_rec_val, x_batch_val)
                latent_loss_val = torch.mean(z_val**2)
                loss_val = recon_loss_val + 1e-3 * latent_loss_val
                valid_loss += loss_val.item()
        valid_loss /= len(valid_loader)

        print(f"AE Epoch {epoch} train loss: {train_loss:.6f}, valid loss: {valid_loss:.6f}")