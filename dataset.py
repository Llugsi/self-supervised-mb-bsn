# ===================================================================================================
# GITHUB REPOSITORY MODULE: DATASET UTILITIES
# CENTRALIZED HYDRODYNAMIC DATA ENGINE IMPLEMENTATION (CALTECH & SYNTHETIC DATASETS)
# ===================================================================================================

import os
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset

class NavierStokesFluidDataset(Dataset):
    """
    Custom HDF5 Dataset parser designed for physical continuum field simulations.
    Extracts isolated 2D spatial dimensions from v7.3 MATLAB fluid tensor grids.
    """
    def __init__(self, mat_path="ns_V1e-3_N5000_T50.mat", num_samples=400):
        if not os.path.exists(mat_path):
            raise FileNotFoundError(f"[CRITICAL] True file '{mat_path}' not found in directory.")
            
        print(f"-> [PARSING] Opening v7.3 HDF5 structure using h5py engine from '{mat_path}'...")
        with h5py.File(mat_path, "r") as f:
            u_dataset = f['u']
            raw_data = u_dataset[0, :, :, :num_samples]  # Slicing active time-step t=0
            self.data = np.moveaxis(raw_data, -1, 0).astype(np.float32)
            
        print(f"-> [SUCCESS] Memory mapping secure. Shape anchored: {self.data.shape}")
        print(f"-> [SUCCESS] Extracted {num_samples} genuine fluid profiles in 2D space.")

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        sample = torch.tensor(self.data[idx], dtype=torch.float32).unsqueeze(0)
        return sample, sample


class PDENavierStokesDataset(Dataset):
    """
    Custom HDF5 Dataset parser designed for physical continuum field simulations.
    Extracts isolated 2D spatial dimensions from local synthetic fluid tensor grids.
    """
    def __init__(self, file_path="2D_CFD_Rand_M0.1_Eta0.01_Zeta0.01_periodic_128_Test.hdf5", time_steps_per_sample=1, channel_idx=0):
        super().__init__()
        self.file_path = file_path
        self.channel_idx = channel_idx
        self.time_steps_per_sample = time_steps_per_sample
        
        # Trigger dynamic synthesis if the local data container is absent
        if not os.path.exists(self.file_path):
            print(f"-> [DATA SYNTHESIS] Local HDF5 missing. Synthesizing advanced diagonal vortex grids...")
            num_sims, time_steps, grid_x, grid_y, num_channels = 10, 20, 64, 64, 1
            tensor_climatico = np.zeros((num_sims, time_steps, grid_x, grid_y, num_channels), dtype=np.float32)

            x = np.linspace(-np.pi, np.pi, grid_x)
            y = np.linspace(-np.pi, np.pi, grid_y)
            X, Y = np.meshgrid(x, y)
            base_estructurada = np.sin(X) * np.cos(Y)

            for s in range(num_sims):
                for t in range(time_steps):
                    fase = t * 0.15 + s * 0.5
                    vortice_diagonal = np.sin(X + Y + fase) * np.cos(X - Y - fase)
                    tensor_climatico[s, t, :, :, 0] = base_estructurada + vortice_diagonal

            with h5py.File(self.file_path, 'w') as f:
                f.create_dataset('tensor', data=tensor_climatico)
            print(f"[DATA ENGINE] HDF5 file successfully preserved at: '{self.file_path}'")

        with h5py.File(self.file_path, 'r') as f:
            self.num_batches = f['tensor'].shape[0]
            self.time_length = f['tensor'].shape[1]
        self.total_samples = self.num_batches * (self.time_length - self.time_steps_per_sample + 1)

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        steps_per_sim = self.time_length - self.time_steps_per_sample + 1
        sim_idx = idx // steps_per_sim
        t_idx = idx % steps_per_sim
        with h5py.File(self.file_path, 'r') as f:
            grid_2d = f['tensor'][sim_idx, t_idx, :, :, self.channel_idx].astype(np.float32)
        
        min_val, max_val = grid_2d.min(), grid_2d.max()
        if max_val - min_val > 1e-8:
            grid_2d = (grid_2d - min_val) / (max_val - min_val)
        else:
            grid_2d = np.zeros_like(grid_2d)
        
        tensor_2d = torch.from_numpy(grid_2d).unsqueeze(0)
        return tensor_2d, tensor_2d


def create_j_invariant_mask(inputs):
    """
    Genera una partición estricta J-invariante (malla de ajedrez 2x2)
    para el algoritmo Noise2Self con datos sintéticos.
    """
    B, C, H, W = inputs.size()
    mask = torch.zeros_like(inputs)
    offset_y, offset_x = np.random.randint(0, 2), np.random.randint(0, 2)
    mask[:, :, offset_y::2, offset_x::2] = 1.0
    return mask


def create_j_invariant_mask_real(inputs_tensor):
    """
    Genera una máscara binaria basada en una cuadrícula periódica (tablero de ajedrez)
    para el algoritmo Noise2Self con datos reales de Caltech.
    """
    B, C, H, W = inputs_tensor.shape
    grid_y = torch.arange(H, device=inputs_tensor.device).view(1, 1, H, 1)
    grid_x = torch.arange(W, device=inputs_tensor.device).view(1, 1, 1, W)
    mask = ((grid_y % 2 == 0) & (grid_x % 2 == 0)).float()
    return mask.expand(B, C, H, W)
