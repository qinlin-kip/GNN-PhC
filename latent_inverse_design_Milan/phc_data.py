from torch.utils.data import Dataset
import numpy as np
import h5py
import torch 

class PhCdata(Dataset):

    def __init__(self, path_to_h5_file, input_size=100):

        self.input_size = input_size

        x_data = []
        y_data = []

        with h5py.File(path_to_h5_file, 'r') as f:

            members = [k for k in f.keys() if k.startswith("design_")]

            for memb in members:
                geom = f[f'{memb}/y_width_arrays'][:]
                s12 = f[f'{memb}/S_n_10/s12_power'][:]

                if np.any(np.isnan(s12)):
                    continue

                x_data.append(geom)
                y_data.append(s12)

        x_data = np.array(x_data).astype("float32")
        y_data = np.array(y_data).astype("float32")

        # Input normalisieren (feature-wise)
        x_mean = x_data.mean(axis=0)
        x_std = x_data.std(axis=0) + 1e-8
        x_data = (x_data - x_mean) / x_std

        # Output normalisieren
        y_mean = y_data.mean(axis=0)
        y_std = y_data.std(axis=0) + 1e-8
        y_data = (y_data - y_mean) / y_std

        self.x_data = torch.from_numpy(x_data)
        self.y_data = torch.from_numpy(y_data)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]