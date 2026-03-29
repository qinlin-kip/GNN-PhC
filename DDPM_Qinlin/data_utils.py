import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import math


def prepare_data(h5_path):
    """
    Build a deterministic 90/10 train/test split from design groups.

    Saves:
      - test_splits.json : test design names
      - norm_stats.json  : max average_scattering_loss_dB from TRAIN only
    """
    h5_path = Path(h5_path)
    out_dir = h5_path.parent
    split_path = out_dir / "test_splits.json"
    norm_path = out_dir / "norm_stats.json"

    with h5py.File(h5_path, "r") as f:
        design_names = sorted(k for k in f.keys() if k.startswith("design_"))

        # Fixed RNG seed keeps the split deterministic across runs.
        rng = np.random.default_rng(42)
        shuffled = rng.permutation(design_names).tolist()

        n_test = max(1, int(len(design_names) * 0.1))
        test_designs = sorted(shuffled[:n_test])
        test_set = set(test_designs)
        train_designs = [d for d in design_names if d not in test_set]

        # Normalization stat must come from the training split only.
        train_losses = []
        for d in train_designs:
            val = float(f[d]["average_scattering_loss_dB"][()])
            if math.isfinite(val):
                train_losses.append(val)
        max_loss = max(train_losses) if train_losses else 1.0
        # Safety margin to avoid division by zero.
        if max_loss <= 0 or not math.isfinite(max_loss):
            max_loss = 1.0

    with open(split_path, "w", encoding="utf-8") as fp:
        json.dump({"test_design_names": test_designs}, fp, indent=2)

    with open(norm_path, "w", encoding="utf-8") as fp:
        json.dump({"max_loss": float(max_loss)}, fp, indent=2)

    return {
        "train_design_names": train_designs,
        "test_design_names": test_designs,
        "max_loss": float(max_loss),
        "test_splits_path": str(split_path),
        "norm_stats_path": str(norm_path),
    }


def denormalize_1d(y_norm):
    """Reverse diffusion scaling: y_width_um = y_norm + 1.0."""
    return np.asarray(y_norm, dtype=np.float32) + 1.0


def draw_2d_mask(y_widths_um):
    """
    Convert a 100-point width profile (um) into a binary mask (2000, 490).

    Geometry convention:
      - columns (490): propagation axis x
      - rows    (2000): transverse axis y spanning exactly 2.0 um
      - row 1000 is the centerline
    """
    y_widths_um = np.asarray(y_widths_um, dtype=np.float32).reshape(-1)
    if y_widths_um.size != 100:
        raise ValueError(f"Expected 100 width points, got {y_widths_um.size}.")

    # Interpolate 100 design points to the 490 propagation columns.
    x_orig = np.linspace(0, 489, 100, dtype=np.float32)
    x_new = np.arange(490, dtype=np.float32)
    widths_490 = np.interp(x_new, x_orig, y_widths_um).astype(np.float32)

    mask = np.zeros((2000, 490), dtype=np.float32)
    um_per_px = 2.0 / 2000.0
    center_row = 1000

    # Fill symmetric waveguide core around center for each x-column.
    for i, w_um in enumerate(widths_490):
        half_w_px = int((w_um / 2.0) / um_per_px)
        r0 = max(0, center_row - half_w_px)
        r1 = min(2000, center_row + half_w_px + 1)  # +1 because slice end is exclusive
        mask[r0:r1, i] = 1.0

    return mask


class PhCDataset(Dataset):
    """
    PyTorch dataset that yields TWO samples per design: n=10 and n=20.

    Return tuple:
      s11 (50,), n_norm (1,), loss_norm (1,), y_width_norm (1, 100), design_name
    """

    def __init__(self, h5_path, split="train"):
        self.h5_path = Path(h5_path)
        self.split = split.lower()
        if self.split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'.")

        self.split_path = self.h5_path.parent / "test_splits.json"
        self.norm_path = self.h5_path.parent / "norm_stats.json"

        # If split files are missing, create them once.
        if not self.split_path.exists() or not self.norm_path.exists():
            prepare_data(self.h5_path)

        with open(self.split_path, "r", encoding="utf-8") as fp:
            test_designs = set(json.load(fp)["test_design_names"])

        with open(self.norm_path, "r", encoding="utf-8") as fp:
            self.max_loss = float(json.load(fp)["max_loss"])
        if self.max_loss <= 0:
            self.max_loss = 1.0

        with h5py.File(self.h5_path, "r") as f:
            all_designs = sorted(k for k in f.keys() if k.startswith("design_"))

        if self.split == "train":
            design_names = [d for d in all_designs if d not in test_designs]
        else:
            design_names = [d for d in all_designs if d in test_designs]

        # Expand each design into two condition samples: n=10 and n=20.
        self.samples = [(d, 10) for d in design_names] + [(d, 20) for d in design_names]

        # Lazy-open HDF5 handle on first __getitem__ call.
        self._h5 = None

    def __len__(self):
        return len(self.samples)

    def _get_h5(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __getitem__(self, idx):
        design_name, n_val = self.samples[idx]
        f = self._get_h5()
        grp = f[design_name]

        # Target spectrum: S11 power from selected period subgroup. Clean NaNs.
        s11 = grp[f"S_n_{n_val}/s11_power"][:].astype(np.float32)
        s11 = np.nan_to_num(s11, nan=0.0, posinf=0.0, neginf=0.0)

        # Diffusion input normalization: [0,2] um -> [-1,1] via subtracting 1.0.
        y_width = grp["y_width_arrays"][:].astype(np.float32)
        y_width = np.nan_to_num(y_width, nan=1.0, posinf=2.0, neginf=0.0)
        y_width_norm = (y_width - 1.0)[None, :]  # (1, 100)

        # Condition scalars normalized to [0,1]-ish ranges.
        n_norm = np.array([n_val / 20.0], dtype=np.float32)
        avg_loss_raw = float(grp["average_scattering_loss_dB"][()])
        if not math.isfinite(avg_loss_raw):
            avg_loss_raw = 0.0
        loss_norm = np.array([avg_loss_raw / self.max_loss], dtype=np.float32)

        return (
            torch.from_numpy(s11),
            torch.from_numpy(n_norm),
            torch.from_numpy(loss_norm),
            torch.from_numpy(y_width_norm),
            design_name,
        )

    def __del__(self):
        # Close file handle if dataset object is garbage-collected.
        if getattr(self, "_h5", None) is not None:
            try:
                self._h5.close()
            except Exception:
                pass
