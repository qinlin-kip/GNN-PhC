import random
import torch
import numpy as np
import os

def reset_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_real_targets(dataset, device, n=4):

    targets = []

    for i in range(n):
        x, y = dataset[i]

        print(i, y.min().item(), y.max().item(), y.mean().item())

        targets.append({
            "name": f"real_{i+1}",
            "y": y.unsqueeze(0).to(device)
        })

    return targets


def save_results_to_txt(results, output_dir="./inverse_design_results"):
    os.makedirs(output_dir, exist_ok=True)

    for i, (name, y_target, pred_s12, x_opt) in enumerate(results):
        y_filename = os.path.join(output_dir, f"{name}_target_y.txt")
        np.savetxt(y_filename, y_target.cpu().numpy().flatten())
        print(f"Saved target spectrum for {name} to {y_filename}")

        pred_s12_filename = os.path.join(output_dir, f"{name}_predicted_s12.txt")
        np.savetxt(pred_s12_filename, pred_s12.cpu().numpy().flatten())
        print(f"Saved predicted spectrum for {name} to {pred_s12_filename}")

        x_filename = os.path.join(output_dir, f"{name}_optimized_x.txt")
        np.savetxt(x_filename, x_opt.cpu().numpy().flatten())
        print(f"Saved optimized geometry for {name} to {x_filename}")


def denormalize_data(normalized_data, mean, std):
    return (normalized_data * std + mean).clamp(min=0.0)

def save_denormalized_results_to_txt(results, x_mean, x_std, y_mean, y_std, output_dir="./inverse_design_results_physical"):
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nSaving denormalized results to {output_dir}")

    for i, (name, y_target_norm, pred_s12_norm, x_opt_norm) in enumerate(results):
        y_target_denorm = denormalize_data(y_target_norm, y_mean, y_std)
        y_filename = os.path.join(output_dir, f"{name}_target_y_physical.txt")
        np.savetxt(y_filename, y_target_denorm.cpu().numpy().flatten())
        print(f"Saved physical target spectrum for {name} to {y_filename}")

        pred_s12_denorm = denormalize_data(pred_s12_norm, y_mean, y_std)
        pred_s12_filename = os.path.join(output_dir, f"{name}_predicted_s12_physical.txt")
        np.savetxt(pred_s12_filename, pred_s12_denorm.cpu().numpy().flatten())
        print(f"Saved physical predicted spectrum for {name} to {pred_s12_filename}")

        x_opt_denorm = denormalize_data(x_opt_norm, x_mean, x_std)
        x_filename = os.path.join(output_dir, f"{name}_optimized_x_physical.txt")
        np.savetxt(x_filename, x_opt_denorm.cpu().numpy().flatten())
        print(f"Saved physical optimized geometry for {name} to {x_filename}")