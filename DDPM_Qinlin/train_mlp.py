import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from data_utils import PhCDataset, prepare_data
from models import MLP_MODELS


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, default="v1", help="Version of model")
    args = parser.parse_args()

    # Locate data next to the provided h5 file.
    project_root = Path(__file__).resolve().parent.parent
    h5_path = project_root / "new_gnn_data" / "phc_out_profile.h5"

    stats = prepare_data(h5_path)
    print(f"Prepared splits. Train={len(stats['train_design_names'])} Test={len(stats['test_design_names'])}")

    dataset = PhCDataset(h5_path, split="train")
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

    device = get_device()
    model = MLP_MODELS[args.version]().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(1, 151):
        model.train()
        epoch_loss = 0.0
        for s11, n_norm, loss_norm, y_width_norm, _ in loader:
            s11 = s11.to(device)
            n_norm = n_norm.to(device)
            loss_norm = loss_norm.to(device)
            y_width_norm = y_width_norm.to(device)

            x = torch.cat([s11, n_norm, loss_norm], dim=1)
            pred = model(x)
            loss = loss_fn(pred, y_width_norm)

            optim.zero_grad()
            loss.backward()
            optim.step()

            epoch_loss += loss.item() * s11.size(0)

        epoch_loss /= len(loader.dataset)
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Loss {epoch_loss:.6f}")

    out_path = Path(__file__).resolve().parent / f"mlp_weights_{args.version}.pth"
    torch.save(model.state_dict(), out_path)
    print(f"Saved weights to {out_path}")

    # Save a small training summary for reference.
    summary_path = Path(__file__).resolve().parent / f"mlp_training_summary_{args.version}.json"
    with open(summary_path, "w", encoding="utf-8") as fp:
        json.dump({"epochs": 150, "final_loss": epoch_loss, "version": args.version}, fp, indent=2)
    print(f"Saved summary to {summary_path}")
