# ===================================================================================================
# GITHUB REPOSITORY MODULE: PARAMETRIC ARCHITECTURAL ABLATION ENGINE
# RIGOROUS SPATIAL CHANNEL RESTRICTION FOR EXPERIMENTS I, II, AND III (IEEE COMPLIANT)
# ===================================================================================================

import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import numpy as np

# Strict imports from your core repository files
from models import TNNLS_BlindSpotNet
from dataset import NavierStokesFluidDataset, PDENavierStokesDataset

# Reproducibility anchors
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AblationWrapper(nn.Module):
    """
    Architectural Wrapper designed to execute structural parameter zeroing across concurrent streams.
    Isolates specific receptive fields to evaluate localized spatial partial differential degradation.
    """
    def __init__(self, base_model, experiment_id="IV"):
        super(AblationWrapper, self).__init__()
        self.base_model = base_model
        self.experiment_id = str(experiment_id).upper()
        
    def forward(self, x):
        # 1. Branch A: Ultra-Local context (3x3 grid)
        mask_a = torch.ones_like(self.base_model.conv_branch_a.weight)
        mask_a[:, :, 1, 1] = 0.0
        weight_a = self.base_model.conv_branch_a.weight * mask_a
        h_a = F.conv2d(x, weight_a, bias=self.base_model.conv_branch_a.bias, padding=1)
        
        # 2. Branch B: Meso-Scale global context (5x5 grid)
        mask_b = torch.ones_like(self.base_model.conv_branch_b.weight)
        mask_b[:, :, 2, 2] = 0.0
        weight_b = self.base_model.conv_branch_b.weight * mask_b
        h_b = F.conv2d(x, weight_b, bias=self.base_model.conv_branch_b.bias, padding=2)
        
        # 3. Branch C: Macro-Scale sparse dilated context (3x3 grid, d=2)
        mask_c = torch.ones_like(self.base_model.conv_branch_c.weight)
        mask_c[:, :, 1, 1] = 0.0
        weight_c = self.base_model.conv_branch_c.weight * mask_c
        h_c = F.conv2d(x, weight_c, bias=self.base_model.conv_branch_c.bias, padding=2, dilation=2)
        
        # --- STRUCTURAL FEATURE RESTRICTION (MANUSCRIPT MATRIX COMPLIANCE) ---
        if self.experiment_id == "I":
            # Experiment I: Local Only -> Suppress Meso-Scale and Macro-Scale streams
            h_b = torch.zeros_like(h_b)
            h_c = torch.zeros_like(h_c)
        elif self.experiment_id == "II":
            # Experiment II: Dilated Only -> Suppress Ultra-Local and Meso-Scale streams
            h_a = torch.zeros_like(h_a)
            h_b = torch.zeros_like(h_b)
        elif self.experiment_id == "III":
            # Experiment III: Dense Mix (Local + Global) -> Suppress Macro-Scale sparse stream
            h_c = torch.zeros_like(h_c)
        # Experiment IV remains unconstrained (Full Proposed Model)
            
        multi_scale_features = torch.cat([h_a, h_b, h_c], dim=1)
        return torch.sigmoid(self.base_model.fusion(multi_scale_features))


def train_ablation_pipeline(data_domain="synthetic", experiment_id="IV", epochs=250):
    print("\n" + "="*78)
    print(f"[ABLATION ENGINE] Initializing Experiment {experiment_id} on [{data_domain.upper()}] Domain")
    print("="*78)
    
    if data_domain == "real":
        dataset = NavierStokesFluidDataset("ns_V1e-3_N5000_T50.mat", num_samples=400)
        loader = DataLoader(dataset, batch_size=256, shuffle=True)
    else:
        dataset = PDENavierStokesDataset("2D_CFD_Rand_M0.1_Eta0.01_Zeta0.01_periodic_128_Test.hdf5")
        loader = DataLoader(dataset, batch_size=64, shuffle=True)
        
    log_csv_path = f"ablation_exp_{experiment_id}_{data_domain}_logs.csv"
    if os.path.exists(log_csv_path):
        os.remove(log_csv_path)
        
    raw_bsn = TNNLS_BlindSpotNet(in_channels=1, out_channels=1)
    model = AblationWrapper(raw_bsn, experiment_id=experiment_id).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-6)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.5)
    criterion = nn.MSELoss()
    
    headers = ['Epoch', 'Loss_MSE', 'Learning_Rate', 'Eval_RMSE', 'Eval_R2_Score']
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        rmse_accum, r2_accum, batch_counter = 0.0, 0.0, 0
        
        for inputs, _ in loader:
            inputs = inputs.to(device)
            optimizer.zero_grad()
            
            noise = torch.randn_like(inputs) * 0.15
            degraded_inputs = torch.clamp(inputs + noise, 0.0, 1.0)
            
            outputs = model(degraded_inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
            with torch.no_grad():
                rmse_b = torch.sqrt(torch.mean((outputs - inputs) ** 2)).item()
                target_mean = torch.mean(inputs)
                ss_res = torch.sum((inputs - outputs) ** 2)
                ss_tot = torch.sum((inputs - target_mean) ** 2)
                r2_b = (1.0 - (ss_res / (ss_tot + 1e-12))).item()
                
                rmse_accum += rmse_b
                r2_accum += r2_b
                batch_counter += 1
                
        scheduler.step()
        epoch_mean_loss = running_loss / len(dataset)
        epoch_mean_rmse = rmse_accum / batch_counter
        epoch_mean_r2 = r2_accum / batch_counter
        
        # CORREGIDO: Acceso seguro al índice 0 para evitar el colapso del optimizador en consola
        current_lr = optimizer.param_groups[0]['lr']
        
        file_exists = os.path.exists(log_csv_path)
        with open(log_csv_path, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            if not file_exists: writer.writeheader()
            writer.writerow({'Epoch': epoch + 1, 'Loss_MSE': round(epoch_mean_loss, 6), 
                             'Learning_Rate': current_lr, 'Eval_RMSE': round(epoch_mean_rmse, 6), 'Eval_R2_Score': round(epoch_mean_r2, 4)})
            
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"Exp {experiment_id} | Epoch [{epoch+1:03d}/{epochs}] | Loss: {epoch_mean_loss:.6f} | R2: {epoch_mean_r2:.4f}")
            
    print(f"-> [SUCCESS] Ablation logs safely written to: '{log_csv_path}'")

if __name__ == "__main__":
    # Parametric triggering configuration for user evaluation
    # train_ablation_pipeline(data_domain="synthetic", experiment_id="I", epochs=250)
    print("\n[READY] Select an experiment ID inside the main function block to compute ablation statistics.")
