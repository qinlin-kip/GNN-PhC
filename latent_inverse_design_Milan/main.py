import torch
from torch.utils import data
from torch.utils.data import random_split
import torch.nn as nn

# Import your custom modules
from phc_data import PhCdata
from geometry_ae import GeometryAE, train_autoencoder
from forward_net import ForwardNet, train_forward_net
from latent_inverse_design import inverse_design_latent
from plotting_utils import (
    plot_Trainings_spektrum,
    plot_forward_model_losses,
    plot_true_vs_predictet,
    spectrum_debug_stats,
    plot_finale_results,
    plot_geometry,
    plot_ensemble_losses_scatter
)
from utils import (
    reset_seeds,
    get_real_targets,
    save_results_to_txt,
    denormalize_data,
    save_denormalized_results_to_txt
)


class Args:
    seed = 42
    path_to_h5 = "Adjust/this/path/as/you/need" # Adjust this path 
    batchsize = 64
    # Autoencoder Parameter
    ae_epochs = 130
    ae_lr = 0.0001
    ae_latentDim = 16
    # Forward Parameter
    lr_forward = 0.001
    forward_epochs = 500
    lambda_smooth = 6.6
    lambda_mono = 12.4
    lambda_smooth2 = 2.2
    # Inverse Parameter
    inverse_lr = 5e-3
    inverse_epochs = 2000
    n_starts = 10
    lambda_mono_geom = 0.02

def main(args):

    reset_seeds(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    h5file = args.path_to_h5 + "/phc_out_profile.h5"

    dataset = PhCdata(h5file)

    # Extract mean and std for denormalization later
    x_mean = dataset.x_data.mean(axis=0).cpu().numpy()
    x_std = dataset.x_data.std(axis=0).cpu().numpy()
    y_mean = dataset.y_data.mean(axis=0).cpu().numpy()
    y_std = dataset.y_data.std(axis=0).cpu().numpy()

    plot_Trainings_spektrum(dataset)

    nsam = len(dataset)
    print("Dataset size:", len(dataset))

    ntrain = int(0.7 * nsam)
    nvalid = int(0.15 * nsam)
    ntest = nsam - ntrain - nvalid

    train_dataset, valid_dataset, test_dataset = random_split(
        dataset,
        [ntrain, nvalid, ntest],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = data.DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True)
    valid_loader = data.DataLoader(valid_dataset, batch_size=args.batchsize)
    test_loader = data.DataLoader(test_dataset, batch_size=args.batchsize)

    forward_net = ForwardNet().to(device)
    autoencoder = GeometryAE(latent_dim=args.ae_latentDim).to(device)

    # Train Autoencoder
    print("\n--- Training Autoencoder ---")
    train_autoencoder(autoencoder, train_loader, valid_loader, epochs=args.ae_epochs, lr=args.ae_lr, device=device)

    # Train ForwardNet
    print("\n--- Training Forward Network ---")
    train_losses_fwd, valid_losses_fwd = train_forward_net(
        forward_net, train_loader, valid_loader, epochs=args.forward_epochs, lr=args.lr_forward,
        device=device, lambda_smooth=args.lambda_smooth, lambda_mono=args.lambda_mono, lambda_smooth2=args.lambda_smooth2
    )

    # Plot forward model loss
    plot_forward_model_losses(train_losses_fwd, valid_losses_fwd)

    # Test ForwardNet
    test_loss = 0
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            pred = forward_net(x_batch)
            loss = nn.MSELoss()(pred, y_batch)
            test_loss += loss.item()
    test_loss /= len(test_loader)
    print(f"\nTest loss for Forward Network: {test_loss:.6f}")

    plot_true_vs_predictet(test_loader, forward_net, device)
    spectrum_debug_stats(forward_net, test_loader, device)

    forward_net.eval()
    autoencoder.eval()

    # Freeze parameters
    for p in forward_net.parameters():
        p.requires_grad = False
    for p in autoencoder.parameters():
        p.requires_grad = False

    # Inverse Design
    print("\n--- Starting Inverse Design ---")
    targets = get_real_targets(test_dataset, device, n=10)
    results_latent = []
    all_targets_ensemble_losses = []

    for i, t in enumerate(targets):
        y_target = t["y"]
        name = t["name"]

        x_opt, loss, ensemble_losses_for_current_target = inverse_design_latent(
            train_loader,
            forward_net,
            autoencoder,
            y_target,
            steps=args.inverse_epochs,
            lr=args.inverse_lr,
            lambda_mono_geom=args.lambda_mono_geom,
            n_starts=args.n_starts,
            device=device
        )

        pred_s12 = forward_net(x_opt.to(device))

        results_latent.append((
            name,
            y_target.cpu(),
            pred_s12.detach().cpu(),
            x_opt.detach().cpu()
        ))

        all_targets_ensemble_losses.append({
            "target_name": name,
            "losses": ensemble_losses_for_current_target
        })

        print(f"Target {i+1} MSE: {nn.functional.mse_loss(pred_s12, y_target).item():.4f}")

    plot_finale_results(results_latent)
    plot_ensemble_losses_scatter(all_targets_ensemble_losses, target_names_to_plot=['real_8', 'real_2', 'real_4'])

    print("\n=== Optimized Geometries from Latent Inverse Design ===")
    for i, (name, y_target, pred_s12, x_opt_to_plot) in enumerate(results_latent):
        x_opt_physical_for_plot = denormalize_data(
            x_opt_to_plot,
            torch.from_numpy(x_mean).to(x_opt_to_plot.device),
            torch.from_numpy(x_std).to(x_opt_to_plot.device)
        )

        if x_opt_physical_for_plot.min() < 0:
            print(f"ACHTUNG: {name} - Geometrie enthält negative Werte! Minimum: {x_opt_physical_for_plot.min():.4f}")
        if x_opt_physical_for_plot.max() > 2.0:
            print(f"ACHTUNG: {name} - Geometrie enthält Werte größer 2.0! Maximum: {x_opt_physical_for_plot.max():.4f}")
        else:
            print(f"Die Geometrie {name} hat keine Werte, die größer als 2 oder kleiner als 0 sind.")
        plot_geometry(x_opt_physical_for_plot, title=f"Optimized Geometry for Target: {name} (Latent Inverse Design - Physical)")

    save_results_to_txt(results_latent)
    save_denormalized_results_to_txt(results_latent, x_mean, x_std, y_mean, y_std)

if __name__ == '__main__':
    args = Args()
    main(args)
