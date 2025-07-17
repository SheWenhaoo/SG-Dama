import torch
from torch.utils.data import Dataset
import h5py 
import numpy as np

class CGANDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, apply_noise=True):
        super(CGANDataset, self).__init__()
        self.h5_path = data_path
        self.apply_noise = apply_noise
    
        with h5py.File(data_path, 'r') as f:
            self.keys = list(f.keys())
    def __len__(self):
        return len(self.keys)
    def __getitem__(self, idx):
        group_name = self.keys[idx]
        with h5py.File(self.h5_path, 'r') as f:
            group = f[group_name]
            sensor = group['sensor'][()]           # shape: (200,)
            strain = group['strain_tensor'][()]    # shape: (151, 151)

        if self.apply_noise:
            noise_level = 0.005
            noise = np.random.normal(0, noise_level, size=sensor.shape)
            sensor = sensor + noise

   
        min_val = sensor.min()
        max_val = sensor.max()
        sensor = (sensor - min_val) / (max_val - min_val + 1e-8)  

    
        sensor = torch.tensor(sensor, dtype=torch.float32)  # (200,)
        strain = torch.tensor(strain, dtype=torch.float32).unsqueeze(0)  # (1, 151, 151)

        return sensor, strain, group_name
