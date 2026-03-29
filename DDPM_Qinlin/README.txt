DDPM_Qinlin quickstart

Overview
- Train an MLP baseline and DDPM-based 1D U-Net to generate PhC width profiles conditioned on S11 spectra and scalar descriptors.
- All scripts now read the dataset from ../Data2_new_gnn_data/phc_out_profile.h5 relative to this folder.

Prerequisites
- Python 3.9+ with PyTorch, numpy, h5py, matplotlib (see requirements in repository root if available).
- GPU is optional but recommended; scripts auto-select CUDA, then Apple MPS, then CPU.

Data layout
- Expected HDF5: ../Data2_new_gnn_data/phc_out_profile.h5
- The first run creates ../Data2_new_gnn_data/test_splits.json and norm_stats.json deterministically (90/10 split, loss normalization).

Training
- MLP baseline: python train_mlp.py --version v1
  - Saves weights to mlp_weights_v1.pth and summary to mlp_training_summary_v1.json in this folder.
- DDPM + U-Net: python train_ddpm.py --version v3
  - Saves weights to ddpm_weights_v3.pth in this folder.

Evaluation and visualization
- Compare multiple models and save a plot: python evaluate.py --mode compare --models ddpm:v1 ddpm:v2 ddpm:v3
- Overlay multiple DDPM samples for randomness: python evaluate.py --mode ddpm-multi --models ddpm:v3 --seeds 5 --guidance_scale 3.0
- Save predictions to JSON: python evaluate.py --mode save-preds --models ddpm:v3 --output_path preds.json

Key files in this folder
- train_mlp.py        : trains the MLP baseline
- train_ddpm.py       : trains DDPM models
- evaluate.py         : plotting and JSON export utilities
- data_utils.py       : dataset, normalization, mask drawing
- models.py           : MLP, U-Net backbones, DDPM wrappers

Notes
- Batch sizes are small by default to fit modest GPUs; adjust as needed.
- If you move the HDF5 file again, update the h5_path definitions at the top of the train/evaluate scripts.
