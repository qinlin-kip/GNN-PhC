import torch
import torch.nn as nn
import torch.nn.functional as F


class ForwardNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(100, 512), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 50)
        )
    def forward(self, x):
        return self.net(x)


def train_forward_net(model, train_loader, valid_loader, epochs=80, lr=1e-4,
                      device="cpu", lambda_smooth=0.3, lambda_mono=0.1, lambda_smooth2=0.0):

      optimizer = torch.optim.Adam(model.parameters(), lr=lr)

      criterion = nn.SmoothL1Loss()

      train_losses = []
      valid_losses = []

      for epoch in range(epochs):

          model.train()
          total_loss = 0

          for x_batch, y_batch in train_loader:

              x_batch = x_batch.to(device)
              y_batch = y_batch.to(device)

              optimizer.zero_grad()

              pred = model(x_batch)


              # MSE/SmoothL1
              recon_loss = criterion(pred, y_batch)

              # 1. Smoothness
              diff1 = pred[:,1:] - pred[:,:-1]
              smooth1 = torch.mean(diff1**2)

              # Monotonie: Spektren sollten nicht wild oszillieren
              mono_loss = torch.mean(torch.abs(diff1[:,1:] * diff1[:,:-1]) *
                                  (diff1[:,1:].sign() != diff1[:,:-1].sign()))

              # 2 Smoothness
              diff2 = diff1[:,1:] - diff1[:,:-1]
              smooth2 = torch.mean(diff2**2)

              loss = recon_loss + lambda_smooth * smooth1 + lambda_mono * mono_loss + lambda_smooth2 * smooth2

              loss.backward()

              optimizer.step()

              total_loss += loss.item()

          train_loss = total_loss / len(train_loader)
          train_losses.append(train_loss)

          print(f"Epoch {epoch} train loss: {train_loss:.6f}")


          # Validation
          model.eval()
          val_loss = 0
          with torch.no_grad():
              for x_batch, y_batch in valid_loader:
                  x_batch = x_batch.to(device)
                  y_batch = y_batch.to(device)
                  pred = model(x_batch)

                  # Kalkuliere smoothness loss für validierung
                  diff1_val = pred[:,1:] - pred[:,:-1]
                  smooth1_val = torch.mean(diff1_val**2)

                  diff2_val = diff1_val[:,1:] - diff1_val[:,:-1]
                  smooth2_val = torch.mean(diff2_val**2)

                  loss = criterion(pred, y_batch) + lambda_smooth * smooth1_val + lambda_smooth2 * smooth2_val
                  val_loss += loss.item()
          val_loss /= len(valid_loader)
          valid_losses.append(val_loss)

          print(f"Epoch {epoch} valid loss: {val_loss:.6f}")

      return train_losses, valid_losses