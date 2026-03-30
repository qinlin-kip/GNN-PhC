import torch
import torch.nn as nn
import torch.nn.functional as F

def inverse_design_latent(
    train_loader,
    forward_net,
    autoencoder,
    y_target,
    steps=1000,
    lr=5e-3,
    n_starts=5,
    lambda_mono_geom=0.0,
    device="cpu"
):
    autoencoder.eval()
    forward_net.eval()

    best_global_loss = float('inf')
    best_global_x = None
    all_ensemble_losses_for_target = []

    for start in range(n_starts):

        latent_dim = autoencoder.latent_dim

        with torch.no_grad():
            x_init, _ = next(iter(train_loader))
            x_init = x_init[0:1].to(device)
            initial_z = autoencoder.encode(x_init)

        z = nn.Parameter(initial_z + 0.1 * torch.randn_like(initial_z))

        delta_x = nn.Parameter(0.01 * torch.randn((1, 100), device=device))

        opt = torch.optim.Adam([z, delta_x], lr=lr)

        best_loss_current_start = float('inf')
        best_x_current_start = None

        for step in range(steps):

            opt.zero_grad()

            x_base = autoencoder.decode(z)
            x = x_base + delta_x

            x = torch.clamp(x, 0.0, 2.0)

            pred = forward_net(x)

            mse_loss = F.mse_loss(pred, y_target)

            reg_z = 1e-4 * torch.mean(z**2)
            reg_dx = 1e-4 * torch.mean(delta_x**2)

            diff_geom = x[:,1:] - x[:,:-1]
            geom_mono_loss = torch.mean(torch.abs(diff_geom[:,1:] * diff_geom[:,:-1]) *
                                    (diff_geom[:,1:].sign() != diff_geom[:,:-1].sign()))

            loss = mse_loss + reg_z + reg_dx + lambda_mono_geom * geom_mono_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_([z, delta_x], 0.5)
            opt.step()

            if loss.item() < best_loss_current_start:
                best_loss_current_start = loss.item()
                best_x_current_start = x.clone().detach()

        print(f"[Start {start+1}] Best Loss: {best_loss_current_start:.6f}")
        all_ensemble_losses_for_target.append(best_loss_current_start)

        if best_loss_current_start < best_global_loss:
            best_global_loss = best_loss_current_start
            best_global_x = best_x_current_start

    print(f"\n=== BEST OVERALL LOSS (Latent Ensemble): {best_global_loss:.6f} ===")

    return best_global_x, best_global_loss, all_ensemble_losses_for_target