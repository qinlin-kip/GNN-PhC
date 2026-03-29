import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import random
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

from data_utils import denormalize_1d, draw_2d_mask, prepare_data
from models import MLP_MODELS, UNET_MODELS, DDPM_MODELS


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sample_ddpm(ddpm, cond, shape):
    """Reverse-sample widths from noise when the DDPM class lacks .sample()."""
    device = cond.device
    x = torch.randn(shape, device=device)
    betas = ddpm.betas
    alphas = 1.0 - betas
    alphas_cumprod = ddpm.alphas_cumprod

    for t in reversed(range(ddpm.timesteps)):
        t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
        eps = ddpm.model(x, t_batch, cond)
        beta_t = betas[t]
        alpha_t = alphas[t]
        alpha_bar_t = alphas_cumprod[t]

        sqrt_recip_alpha = torch.sqrt(1.0 / alpha_t)
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar_t)
        model_mean = sqrt_recip_alpha * (x - beta_t / sqrt_one_minus_alpha_bar * eps)

        if t > 0:
            noise = torch.randn_like(x)
            x = model_mean + torch.sqrt(beta_t) * noise
        else:
            x = model_mean
    return x


def parse_model_specs(spec_list):
    """
    Parse specs like ["mlp:v1", "ddpm:v2"].
    Returns list of dicts: {"kind": "mlp"|"ddpm", "version": "v1"}
    """
    parsed = []
    for spec in spec_list:
        if ":" not in spec:
            raise ValueError(f"Model spec '{spec}' must be kind:version (e.g., ddpm:v2)")
        kind, version = spec.split(":", 1)
        kind = kind.lower()
        version = version.lower()
        if kind not in {"mlp", "ddpm"}:
            raise ValueError(f"Unknown model kind '{kind}' in spec '{spec}'.")
        parsed.append({"kind": kind, "version": version})
    return parsed


def load_and_predict(spec, cond, device, guidance_scale=3.0):
    """
    Load the requested model/version, run prediction, and return physical widths.
    Returns (label, width_um, width_interp_490, mse_to_truth, mask)
    """
    kind = spec["kind"]
    version = spec["version"]
    weights_dir = Path(__file__).resolve().parent

    if kind == "mlp":
        model_cls = MLP_MODELS.get(version)
        if model_cls is None:
            raise ValueError(f"MLP version '{version}' not registered.")
        model = model_cls().to(device)
        weights_path = weights_dir / f"mlp_weights_{version}.pth"
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.eval()

        with torch.no_grad():
            y_norm = model(cond).cpu().numpy().reshape(-1)
        y_um = denormalize_1d(y_norm)
        y_um = np.clip(y_um, 0.0, 2.0)
        label = f"MLP {version}"
        return label, y_um

    if kind == "ddpm":
        unet_cls = UNET_MODELS.get(version)
        ddpm_cls = DDPM_MODELS.get(version)
        if unet_cls is None or ddpm_cls is None:
            raise ValueError(f"DDPM version '{version}' not registered.")
        unet = unet_cls().to(device)
        ddpm = ddpm_cls(unet).to(device)
        weights_path = weights_dir / f"ddpm_weights_{version}.pth"
        ddpm.load_state_dict(torch.load(weights_path, map_location=device))
        ddpm.eval()

        with torch.no_grad():
            if hasattr(ddpm, "sample"):
                if version == "v3":
                    y_norm = ddpm.sample(cond, (1, 1, 100), guidance_scale=guidance_scale).cpu().numpy().reshape(-1)
                else:
                    y_norm = ddpm.sample(cond, (1, 1, 100)).cpu().numpy().reshape(-1)
            else:
                y_norm = sample_ddpm(ddpm, cond, (1, 1, 100)).cpu().numpy().reshape(-1)
        y_um = denormalize_1d(y_norm)
        y_um = np.clip(y_um, 0.0, 2.0)
        label = f"DDPM {version}"
        return label, y_um

    raise ValueError(f"Unsupported model kind '{kind}'.")


# -----------------------------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------------------------

def plot_geom(ax, x_cols, true_arr, pred_arr, color, title):
    true_top = true_arr / 2.0
    true_bot = -true_arr / 2.0
    pred_top = pred_arr / 2.0
    pred_bot = -pred_arr / 2.0

    ax.fill_between(x_cols, true_bot, true_top, color="#d3d3d3", label="GT")
    ax.plot(x_cols, pred_top, color=color, linewidth=2, label="Pred")
    ax.plot(x_cols, pred_bot, color=color, linewidth=2)
    ax.set_xlim(0, 490)
    ax.set_ylim(-1.0, 1.0)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=8)
    ax.set_title(title)


def plot_masks(ax, mask, title, extent):
    im_kwargs = dict(aspect="auto", origin="lower", extent=extent, cmap="gray")
    ax.imshow(mask, **im_kwargs)
    ax.set_title(title)


# -----------------------------------------------------------------------------
def run_compare(models, guidance_scale=3.0):
    # Font and size: 15 cm x 20 cm -> inches
    plt.rcParams.update({"font.size": 10})

    project_root = Path(__file__).resolve().parent.parent
    h5_path = project_root / "Data2_new_gnn_data" / "phc_out_profile.h5"

    prepare_data(h5_path)
    split_path = h5_path.parent / "test_splits.json"
    norm_path = h5_path.parent / "norm_stats.json"

    with open(split_path, "r", encoding="utf-8") as fp:
        test_designs = json.load(fp)["test_design_names"]
    with open(norm_path, "r", encoding="utf-8") as fp:
        max_loss = float(json.load(fp)["max_loss"])
        if max_loss <= 0:
            max_loss = 1.0

    design_name = random.choice(test_designs)
    n_choice = random.choice([10, 20])

    with h5py.File(h5_path, "r") as f:
        grp = f[design_name]
        wavelengths = f["wavelengths_nm"][:]
        s11 = grp[f"S_n_{n_choice}/s11_power"][:].astype(np.float32)
        y_true = grp["y_width_arrays"][:].astype(np.float32)
        avg_loss = float(grp["average_scattering_loss_dB"][()])
        img_true = grp["image_matrices"][:].astype(np.float32)

    n_norm = np.array([n_choice / 20.0], dtype=np.float32)
    loss_norm = np.array([avg_loss / max_loss], dtype=np.float32)
    cond_np = np.concatenate([s11, n_norm, loss_norm], axis=0)
    cond = torch.from_numpy(cond_np).unsqueeze(0).to(get_device())

    specs = parse_model_specs(models)
    device = get_device()

    results = []
    x_cols = np.arange(490)
    x_orig = np.linspace(0, 489, 100)
    true_interp = np.interp(x_cols, x_orig, np.clip(y_true, 0.0, 2.0))

    for spec in specs:
        label, y_pred_um = load_and_predict(spec, cond, device, guidance_scale=guidance_scale)
        y_pred_um = np.clip(y_pred_um, 0.0, 2.0)
        pred_interp = np.interp(x_cols, x_orig, y_pred_um)
        mse = float(np.mean((y_pred_um - y_true) ** 2))
        mask = draw_2d_mask(y_pred_um)
        results.append(
            {
                "label": label,
                "kind": spec["kind"],
                "version": spec["version"],
                "y_pred_um": y_pred_um,
                "pred_interp": pred_interp,
                "mse": mse,
                "mask": mask,
            }
        )

    rows = 1 + len(results)
    width_in = 15 / 2.54
    height_in = (20 / 2.54) * (rows / 3)
    fig, axes = plt.subplots(rows, 2, figsize=(width_in, height_in), constrained_layout=True)
    if rows == 1:
        axes = np.array([[axes]])
    elif rows == 2:
        axes = axes.reshape(2, 2)

    extent = [0, 490, -1.0, 1.0]
    plot_masks(axes[0, 0], img_true, "Ground Truth 2D mask", extent)
    axes[0, 1].plot(wavelengths, s11, color="black", linewidth=2)
    axes[0, 1].set_title('Target reflection spectrum')
    axes[0, 1].set_xlabel("Wavelength (nm)")
    axes[0, 1].set_ylabel("|S11|^2")

    for idx, res in enumerate(results, start=1):
        plot_masks(axes[idx, 0], res["mask"], f"{res['label']} 2D mask", extent)
        title = f"{res['label']} geometry | MSE={res['mse']:.4f}"
        if res["kind"] == "ddpm" and res["version"] == "v3":
            title = f"{res['label']} (w={guidance_scale}) | MSE={res['mse']:.4f}"
        plot_geom(
            axes[idx, 1],
            x_cols,
            true_interp,
            res["pred_interp"],
            color="red" if "MLP" in res["label"] else "blue",
            title=title,
        )

    axes[-1, 0].set_xlabel("Propagation pixel (0-490)")
    axes[-1, 0].set_ylabel("Transverse (um)")
    axes[-1, 1].set_xlabel("Propagation pixel")
    axes[-1, 1].set_ylabel("Transverse (um)")
    for r in range(rows - 1):
        axes[r, 0].set_xlabel("")
        axes[r, 0].set_ylabel("")
        axes[r, 1].set_xlabel("")
        axes[r, 1].set_ylabel("")

    out_name = "comparison_" + "_".join([f"{s['kind']}{s['version']}" for s in specs]) + ".png"
    out_path = Path(__file__).resolve().parent / out_name
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved comparison plot to {out_path}")


def run_ddpm_multi(models, seeds=5, guidance_scale=3.0):
    """
    For each requested DDPM model, sample 5 times with different seeds on the same target S11.
    Overlays the 5 curves plus GT on one subplot per model. Figure width 15 cm, font 10.
    """
    plt.rcParams.update({"font.size": 10})

    project_root = Path(__file__).resolve().parent.parent
    h5_path = project_root / "Data2_new_gnn_data" / "phc_out_profile.h5"

    prepare_data(h5_path)
    split_path = h5_path.parent / "test_splits.json"
    norm_path = h5_path.parent / "norm_stats.json"

    with open(split_path, "r", encoding="utf-8") as fp:
        test_designs = json.load(fp)["test_design_names"]
    with open(norm_path, "r", encoding="utf-8") as fp:
        max_loss = float(json.load(fp)["max_loss"])
        if max_loss <= 0:
            max_loss = 1.0

    design_name = random.choice(test_designs) #design_0030'
    print(design_name)
    n_choice = 20#random.choice([10, 20])

    with h5py.File(h5_path, "r") as f:
        grp = f[design_name]
        wavelengths = f["wavelengths_nm"][:]
        s11 = grp[f"S_n_{n_choice}/s11_power"][:].astype(np.float32)
        y_true = grp["y_width_arrays"][:].astype(np.float32)
        avg_loss = float(grp["average_scattering_loss_dB"][()])

    n_norm = np.array([n_choice / 20.0], dtype=np.float32)
    loss_norm = np.array([avg_loss / max_loss], dtype=np.float32)
    cond_np = np.concatenate([s11, n_norm, loss_norm], axis=0)
    device = get_device()
    cond = torch.from_numpy(cond_np).unsqueeze(0).to(device)

    specs = parse_model_specs(models)
    for spec in specs:
        if spec["kind"] != "ddpm":
            raise ValueError("--models for run_ddpm_multi must be DDPM-only (e.g., ddpm:v1 ddpm:v2)")

    x_cols = np.arange(490)
    x_orig = np.linspace(0, 489, 100)
    true_interp = np.interp(x_cols, x_orig, np.clip(y_true, 0.0, 2.0))

    rows = len(specs)
    width_in = 7 / 2.54
    height_in = (18/ 2.54) * max(1, rows / 3)
    fig, axes = plt.subplots(rows, 1, figsize=(width_in, height_in), constrained_layout=True)
    if rows == 1:
        axes = np.array([axes])

    colors = plt.cm.tab10(np.linspace(0, 1, seeds))

    for idx, spec in enumerate(specs):
        label = f"DDPM {spec['version']}"
        ax = axes[idx]

        # Load model once per spec
        unet_cls = UNET_MODELS.get(spec["version"])
        ddpm_cls = DDPM_MODELS.get(spec["version"])
        if unet_cls is None or ddpm_cls is None:
            raise ValueError(f"DDPM version '{spec['version']}' not registered.")
        unet = unet_cls().to(device)
        ddpm = ddpm_cls(unet).to(device)
        weights_path = Path(__file__).resolve().parent / f"ddpm_weights_{spec['version']}.pth"
        ddpm.load_state_dict(torch.load(weights_path, map_location=device))
        ddpm.eval()

        # Plot GT
        ax.fill_between(x_cols, -true_interp / 2, true_interp / 2, color="#d3d3d3", label="GT")

        for s in range(seeds):
            torch.manual_seed(s)
            np.random.seed(s)
            if hasattr(ddpm, "sample"):
                if spec["version"] == "v3":
                    y_norm = ddpm.sample(cond, (1, 1, 100), guidance_scale=guidance_scale).cpu().detach().numpy().reshape(-1)
                else:
                    y_norm = ddpm.sample(cond, (1, 1, 100)).cpu().detach().numpy().reshape(-1)
            else:
                y_norm = sample_ddpm(ddpm, cond, (1, 1, 100)).cpu().detach().numpy().reshape(-1)
            y_um = denormalize_1d(y_norm)
            y_um = np.clip(y_um, 0.0, 2.0)
            pred_interp = np.interp(x_cols, x_orig, y_um)
            ax.plot(x_cols, pred_interp / 2, color=colors[s], linewidth=1.2, label=f"seed{s}")
            ax.plot(x_cols, -pred_interp / 2, color=colors[s], linewidth=1.2)

        ax.set_xlim(0, 490)
        ax.set_ylim(-1.0, 1.0)
        title = f"{label}"
        if spec["version"] == "v3":
            title += f" (w={guidance_scale})"
        ax.set_title(title)
        if idx == rows - 1:
            ax.set_xlabel("Propagation pixel")
            ax.set_ylabel("Transverse (um)")
        else:
            ax.set_xlabel("")
            ax.set_ylabel("")
        handles, labels_leg = ax.get_legend_handles_labels()
        by_label = dict(zip(labels_leg, handles))
        ax.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=8)

    out_name = "ddpm_multi_" + "_".join([f"{s['kind']}{s['version']}" for s in specs]) + ".png"
    out_path = Path(__file__).resolve().parent / out_name
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved DDPM multi-sample overlay to {out_path}")


def run_save_preds(models, guidance_scale=3.0, output_path=None):
    """
    Save target S11 / true widths and predicted widths from requested DDPM models to a JSON file.
    """
    plt.rcParams.update({"font.size": 10})

    project_root = Path(__file__).resolve().parent.parent
    h5_path = project_root / "Data2_new_gnn_data" / "phc_out_profile.h5"

    prepare_data(h5_path)
    split_path = h5_path.parent / "test_splits.json"
    norm_path = h5_path.parent / "norm_stats.json"

    with open(split_path, "r", encoding="utf-8") as fp:
        test_designs = json.load(fp)["test_design_names"]
    with open(norm_path, "r", encoding="utf-8") as fp:
        max_loss = float(json.load(fp)["max_loss"])
        if max_loss <= 0:
            max_loss = 1.0

    design_name = 'design_0563'#random.choice(test_designs)
    n_choice = 20 #random.choice([10, 20])

    with h5py.File(h5_path, "r") as f:
        grp = f[design_name]
        s11 = grp[f"S_n_{n_choice}/s11_power"][:].astype(np.float32)
        y_true = grp["y_width_arrays"][:].astype(np.float32)
        avg_loss = float(grp["average_scattering_loss_dB"][()])

    n_norm = np.array([n_choice / 20.0], dtype=np.float32)
    loss_norm = np.array([avg_loss / max_loss], dtype=np.float32)
    cond_np = np.concatenate([s11, n_norm, loss_norm], axis=0)
    cond = torch.from_numpy(cond_np).unsqueeze(0).to(get_device())

    specs = parse_model_specs(models)
    for spec in specs:
        if spec["kind"] != "ddpm":
            raise ValueError("run_save_preds expects DDPM-only models, e.g., ddpm:v1 ddpm:v3")

    device = get_device()

    results = []
    for spec in specs:
        label, y_pred_um = load_and_predict(spec, cond, device, guidance_scale=guidance_scale)
        results.append(
            {
                "label": label,
                "kind": spec["kind"],
                "version": spec["version"],
                "y_pred_um": y_pred_um.tolist(),
            }
        )

    payload = {
        "design_name": design_name,
        "n_choice": n_choice,
        "guidance_scale": guidance_scale,
        "target": {
            "s11": s11.tolist(),
            "y_width_um_true": y_true.tolist(),
        },
        "predictions": results,
    }

    if output_path is None:
        tag = "_".join([f"{s['version']}" for s in specs])
        output_path = Path(__file__).resolve().parent / f"predictions_ddpm_{tag}.json"
    else:
        output_path = Path(output_path)

    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)
    print(f"Saved predictions to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["compare", "ddpm-multi", "save-preds"],
        default="ddpm-multi",
        help="compare: multi-model compare; ddpm-multi: overlay 5 seeds per DDPM; save-preds: write preds to JSON",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=['ddpm:v1','ddpm:v2','ddpm:v3'],
        help="Model specs (kind:version). For ddpm-multi, use DDPM-only.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=5,
        help="Number of random seeds for ddpm-multi mode",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.0,
        help="Classifier-free guidance scale for DDPM v3 sampling",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output JSON path for save-preds mode",
    )
    args = parser.parse_args()

    if args.mode == "compare":
        run_compare(args.models, guidance_scale=args.guidance_scale)
    elif args.mode == "ddpm-multi":
        run_ddpm_multi(args.models, seeds=args.seeds, guidance_scale=args.guidance_scale)
    else:
        run_save_preds(args.models, guidance_scale=args.guidance_scale, output_path=args.output_path)


if __name__ == "__main__":
    main()
