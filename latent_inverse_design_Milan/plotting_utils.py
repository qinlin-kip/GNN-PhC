import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib.ticker as mticker
import torch.nn.functional as F # Needed for F.mse_loss
import torch # Needed for torch.from_numpy and device conversion in plot_geometry

def plot_Trainings_spektrum (dataset):
    for i in range(5):
        _, y = dataset[i]
        plt.plot(y.numpy())

    plt.xlabel("Wavelength index")
    plt.ylabel("S12")
    plt.title("Sample Spectra")
    plt.show()


def plot_forward_model_losses(train_losses, valid_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(valid_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Forward Model Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_true_vs_predictet(test_loader, forward_net, device):
    x_batch, y_batch = next(iter(test_loader))

    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)

    with torch.no_grad():
        pred = forward_net(x_batch)

    true_spec = y_batch[0].cpu().numpy()
    pred_spec = pred[0].cpu().numpy()

    plt.plot(true_spec, label="True")
    plt.plot(pred_spec, label="Predicted")
    plt.xlabel("Wavelength index")
    plt.ylabel("S12")
    plt.legend()
    plt.title("True vs Predicted Spectrum")
    plt.show()


def spectrum_debug_stats(forward_net, loader, device):
    forward_net.eval()
    all_true = []
    all_pred = []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            pred = forward_net(x_batch)
            all_true.append(y_batch.cpu().numpy())
            all_pred.append(pred.cpu().numpy())

    all_true = np.concatenate(all_true, axis=0)
    all_pred = np.concatenate(all_pred, axis=0)

    diff = all_pred - all_true

    print("=== Spectrum Debug Stats ===")
    print(f"Mean True: {all_true.mean():.4f}, Std True: {all_true.std():.4f}")
    print(f"Mean Pred: {all_pred.mean():.4f}, Std Pred: {all_pred.std():.4f}")
    print(f"MAE: {np.abs(diff).mean():.4f}")
    print(f"Max Abs Error: {np.max(np.abs(diff)):.4f}")
    print(f"Min True: {all_true.min():.4f}, Max True: {all_true.max():.4f}")
    print(f"Min Pred: {all_pred.min():.4f}, Max Pred: {all_pred.max():.4f}")

    diff_slope = np.diff(all_pred, axis=1)
    print(f"Mean slope change (Pred): {np.mean(np.abs(diff_slope)):.4f}")
    print(f"Max slope change (Pred): {np.max(np.abs(diff_slope)):.4f}")


def plot_finale_results(results):
    num_results = len(results)
    num_cols = 3
    num_rows = math.ceil(num_results / num_cols)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 4, num_rows * 3))

    if num_results > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    for i, (name, y_target, pred_s12, _) in enumerate(results):
        ax = axes[i]

        ax.plot(y_target.flatten(), 'b-', linewidth=3, label="Target S12")
        ax.plot(pred_s12.flatten(), 'r--', linewidth=3, label="Predicted S12")

        ax.text(
            0.02, 0.98,
            f'{name}\nLoss: {F.mse_loss(pred_s12, y_target).item():.6f}',
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )

        ax.set_xlabel("Wavelength index")
        ax.set_ylabel("Normalized S12")
        ax.legend()
        ax.grid(True, alpha=0.3)

    for j in range(num_results, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(
        "Inverse Design Results (Normalized Spectra)\nTarget vs Predicted",
        fontsize=16,
        fontweight='bold'
    )

    plt.tight_layout()
    plt.show()


def plot_geometry(geometry, title="Optimized Geometry"):
    plt.figure(figsize=(10, 4))
    plt.plot(geometry.cpu().numpy().flatten())
    plt.title(title)
    plt.xlabel("Index")
    plt.ylabel("Width (Normalized)")
    plt.grid(True)
    plt.ylim(0.0, 2.5)
    plt.show()


def plot_ensemble_losses_scatter(all_ensemble_losses, target_names_to_plot=None):
    fig, ax = plt.subplots(figsize=(12, 7))

    if target_names_to_plot is None:
        target_data_to_plot = all_ensemble_losses
    else:
        target_data_to_plot = [
            td for td in all_ensemble_losses
            if td['target_name'] in target_names_to_plot
        ]

    x_positions = np.arange(len(target_data_to_plot))

    np.random.seed(42)

    ax.set_yscale('log')

    ax.yaxis.set_major_locator(mticker.LogLocator(base=10))
    ax.yaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10))

    for i, target_data in enumerate(target_data_to_plot):
        losses = np.array(target_data['losses'], dtype=float)
        x = x_positions[i]

        jitter = (np.random.rand(len(losses)) - 0.5) * 0.08
        x_jittered = x + jitter

        ax.scatter(
            x_jittered,
            losses,
            color='black',
            s=45,
            alpha=0.7,
            zorder=2
        )

        min_idx = np.argmin(losses)
        max_idx = np.argmax(losses)

        min_loss = losses[min_idx]
        max_loss = losses[max_idx]

        min_x = x_jittered[min_idx]
        max_x = x_jittered[max_idx]

        ax.scatter(
            min_x,
            min_loss,
            s=120,
            facecolors='none',
            edgecolors='blue',
            linewidths=2,
            label='Min Loss' if i == 0 else "",
            zorder=3
        )

        ax.scatter(
            max_x,
            max_loss,
            s=120,
            facecolors='none',
            edgecolors='red',
            linewidths=2,
            label='Max Loss' if i == 0 else "",
            zorder=3
        )

        ax.annotate(
            f'{min_loss:.6f}',
            xy=(min_x, min_loss),
            xytext=(-12, -2),
            textcoords='offset points',
            ha='right',
            va='center',
            color='blue',
            fontsize=12,
            fontweight='bold'
        )

        ax.annotate(
            f'{max_loss:.6f}',
            xy=(max_x, max_loss),
            xytext=(12, -2),
            textcoords='offset points',
            ha='left',
            va='center',
            color='red',
            fontsize=12,
            fontweight='bold'
        )

    ax.set_xlabel('Inverse Design Target', fontsize=12)
    ax.set_ylabel('Loss (Log Scale)', fontsize=12)
    ax.set_title('Ensemble Losses per Selected Target', fontsize=14)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [td['target_name'] for td in target_data_to_plot],
        rotation=45,
        ha='right'
    )

    ax.grid(True, which='major', axis='y', linestyle='--', alpha=0.4)
    ax.grid(False, which='minor', axis='y')
    ax.grid(False, axis='x')

    ax.set_xlim(-0.5, len(target_data_to_plot) - 0.5)

    ax.legend()
    plt.tight_layout()
    plt.show()