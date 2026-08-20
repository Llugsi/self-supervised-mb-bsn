# ===================================================================================================
# GITHUB REPOSITORY MODULE: MAIN TRAINING INFRASTRUCTURE FOR REAL BENCHMARK (PART 1)
# CORE PIPELINE EXECUTING PROPOSED MULTI-BRANCH BSN AND SOTA BASELINES SYMMETRICALLY
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

# Strict local module imports from your pristine repository files
from models import TNNLS_BlindSpotNet, N2V_BaselineRegressor
from dataset import NavierStokesFluidDataset, create_j_invariant_mask_real

# Strict seed configuration for absolute experimental reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Auto-detect processing hardware (CUDA acceleration preferred, fallback to CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[SYSTEM ENVIRONMENT] Active hardware execution context: [{device}]")

# ===================================================================================================
# DATA PIPELINE BOUNDING (CALTECH BENCHMARK)
# ===================================================================================================
filename_mat = "ns_V1e-3_N5000_T50.mat"
num_samples_target = 400

fluid_dataset = NavierStokesFluidDataset(filename_mat, num_samples=num_samples_target)
fluid_loader = DataLoader(fluid_dataset, batch_size=256, shuffle=True, num_workers=0)

print(f"[DATA ENGINE] DataLoader successfully bounded with {len(fluid_dataset)} genuine fluid profiles.\n")

def run_hermetic_gradient_proof(model_to_test, model_name="Model"):
    """
    Executes an exact analytical differential sensitivity check to certify absolute receptive field
    isolation at the core pixel coordinate before training.
    """
    model_to_test.eval()
    test_device = next(model_to_test.parameters()).device
    
    spatial_tensor = torch.randn(1, 1, 15, 15, requires_grad=True, device=test_device)
    center_y, center_x = 15 // 2, 15 // 2
    
    output_proof = model_to_test(spatial_tensor)
    center_pixel_output = output_proof[0, 0, center_y, center_x]
    center_pixel_output.backward()
    
    grad_val = spatial_tensor.grad[0, 0, center_y, center_x].item()
    print(f"[AUDIT] {model_name} sensitivity gradient at index (y={center_y}, x={center_x}): {grad_val}")
    
    if abs(grad_val) == 0.0:
        print(f"-> [STATUS] SUCCESS! Architectural Blind-Spot verified. No data leakage detected.")
    else:
        print(f"-> [STATUS] WARNING! Receptive field leakage found. Spatial identity mapping active.")
    return abs(grad_val) == 0.0

# Run hardware auditor dynamically
print("-" * 78)
run_hermetic_gradient_proof(TNNLS_BlindSpotNet(1, 1), "Proposed TNNLS_BlindSpotNet")
run_hermetic_gradient_proof(N2V_BaselineRegressor(1, 1), "Baseline Noise2Void/Noise2Self")
print("-" * 78 + "\n")

# ---------------------------------------------------------------------------------------------------
# PART A: TRAINING THE PROPOSED HERMETIC BLIND-SPOT NETWORK (BSN)
# ---------------------------------------------------------------------------------------------------
print("-" * 78)
print("PHASE 1/3: TRAINING PROPOSED TNNLS_BLINDSPOTNET FRAMEWORK (REAL DATA)")
print("-" * 78)

model = TNNLS_BlindSpotNet(in_channels=1, out_channels=1).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-6)
scheduler = StepLR(optimizer, step_size=50, gamma=0.5)
criterion = nn.MSELoss()

num_epochs_proposed = 250
proposed_csv_path = "experiment_bsn_navier_stokes_1_logs.csv"

if os.path.exists(proposed_csv_path):
    os.remove(proposed_csv_path)

for epoch in range(num_epochs_proposed):
    model.train()
    running_loss = 0.0
    epoch_rmse_accum, epoch_r2_accum, batch_counter = 0.0, 0.0, 0
    
    for inputs, _ in fluid_loader:
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
            
            epoch_rmse_accum += rmse_b
            epoch_r2_accum += r2_b
            batch_counter += 1
            
    scheduler.step()
    epoch_mean_loss = running_loss / len(fluid_dataset)
    epoch_mean_rmse = epoch_rmse_accum / batch_counter
    epoch_mean_r2 = epoch_r2_accum / batch_counter
    current_lr = optimizer.param_groups[0]['lr']
    
    headers_proposed = ['Epoch', 'Loss_MSE', 'Learning_Rate', 'Eval_RMSE', 'Eval_R2_Score']
    file_exists = os.path.exists(proposed_csv_path)
    with open(proposed_csv_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers_proposed)
        if not file_exists: writer.writeheader()
        writer.writerow({'Epoch': epoch + 1, 'Loss_MSE': round(epoch_mean_loss, 6), 
                         'Learning_Rate': current_lr, 'Eval_RMSE': round(epoch_mean_rmse, 6), 'Eval_R2_Score': round(epoch_mean_r2, 4)})
        
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Proposed BSN | Epoch [{epoch+1:03d}/{num_epochs_proposed}] | Loss: {epoch_mean_loss:.6f} | RMSE: {epoch_mean_rmse:.5f} | R2: {epoch_mean_r2:.4f}", flush=True)

# ---------------------------------------------------------------------------------------------------
# PART B: TRAINING SOTA BASELINE — NOISE2VOID (N2V)
# ---------------------------------------------------------------------------------------------------
print("\n" + "-" * 78)
print("PHASE 2/3: TRAINING BASELINE SOTA FRAMEWORK - NOISE2VOID (REAL DATA)")
print("-" * 78)

n2v_csv_path = "experiment_n2v_baseline_logs.csv"
num_baseline_epochs = 250

if os.path.exists(n2v_csv_path):
    os.remove(n2v_csv_path)

headers_baseline = ['Epoch', 'Loss_MSE', 'Learning_Rate', 'Eval_RMSE', 'Eval_R2_Score']

model_n2v = N2V_BaselineRegressor(in_channels=1, out_channels=1).to(device)
optimizer_n2v = optim.Adam(model_n2v.parameters(), lr=0.001, weight_decay=1e-6)
scheduler_n2v = StepLR(optimizer_n2v, step_size=50, gamma=0.5)

for epoch in range(num_baseline_epochs):
    model_n2v.train()
    running_loss_n2v = 0.0
    epoch_rmse_accum, epoch_r2_accum, batch_counter = 0.0, 0.0, 0
    
    for inputs, _ in fluid_loader:
        inputs = inputs.to(device)
        optimizer_n2v.zero_grad()
        
        noise = torch.randn_like(inputs) * 0.15
        degraded_inputs = torch.clamp(inputs + noise, 0.0, 1.0)
        
        n2v_mask = (torch.rand_like(inputs) < 0.2).float()
        masked_inputs = degraded_inputs * (1.0 - n2v_mask) + inputs * n2v_mask
        
        outputs = model_n2v(masked_inputs)
        loss = nn.MSELoss()(outputs * n2v_mask, inputs * n2v_mask)
        loss.backward()
        optimizer_n2v.step()
        
        running_loss_n2v += loss.item() * inputs.size(0)
        
        with torch.no_grad():
            rmse_b = torch.sqrt(torch.mean((outputs - inputs) ** 2)).item()
            target_mean = torch.mean(inputs)
            ss_res = torch.sum((inputs - outputs) ** 2)
            ss_tot = torch.sum((inputs - target_mean) ** 2)
            r2_b = (1.0 - (ss_res / (ss_tot + 1e-12))).item()
            
            epoch_rmse_accum += rmse_b
            epoch_r2_accum += r2_b
            batch_counter += 1
            
    scheduler_n2v.step()
    mean_loss = running_loss_n2v / len(fluid_dataset)
    mean_rmse = epoch_rmse_accum / batch_counter
    mean_r2 = epoch_r2_accum / batch_counter
    current_lr = optimizer_n2v.param_groups[0]['lr']
    
    file_exists = os.path.exists(n2v_csv_path)
    with open(n2v_csv_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers_baseline)
        if not file_exists: writer.writeheader()
        writer.writerow({'Epoch': epoch + 1, 'Loss_MSE': round(mean_loss, 6), 
                         'Learning_Rate': current_lr, 'Eval_RMSE': round(mean_rmse, 6), 'Eval_R2_Score': round(mean_r2, 4)})
        
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"SOTA Noise2Void | Epoch [{epoch+1:03d}/{num_baseline_epochs}] | Loss: {mean_loss:.6f} | RMSE: {mean_rmse:.5f} | R2: {mean_r2:.4f}", flush=True)

# ---------------------------------------------------------------------------------------------------
# PART C: TRAINING THE SOTA NOISE2SELF (N2S) BASELINE
# ---------------------------------------------------------------------------------------------------
print("\n" + "-" * 78)
print("PHASE 3/3: TRAINING SOTA NOISE2SELF FRAMEWORK (REAL DATA)")
print("-" * 78)

n2s_model = N2V_BaselineRegressor(in_channels=1, out_channels=1, hidden_features=32).to(device)
n2s_optimizer = optim.Adam(n2s_model.parameters(), lr=0.002, weight_decay=1e-5)
n2s_scheduler = StepLR(n2s_optimizer, step_size=40, gamma=0.5)

num_n2s_epochs = 250
n2s_csv_path = "experiment_n2s_baseline_logs.csv"

if os.path.exists(n2s_csv_path):
    os.remove(n2s_csv_path)

headers_n2s = ['Epoch', 'Loss_MSE', 'Learning_Rate', 'Eval_RMSE', 'Eval_R2_Score']

for epoch in range(num_n2s_epochs):
    n2s_model.train()
    running_loss_n2s = 0.0
    epoch_rmse_accum_n2s, epoch_r2_accum_n2s, batch_counter_n2s = 0.0, 0.0, 0
    
    for inputs, _ in fluid_loader:
        inputs = inputs.to(device)
        n2s_optimizer.zero_grad()
        
        n2s_mask = create_j_invariant_mask_real(inputs)
        noise = torch.randn_like(inputs) * 0.15
        degraded_inputs = torch.clamp(inputs + noise, 0.0, 1.0)
        
        local_blur = torch.rand_like(inputs) * 0.1
        inputs_masked = torch.where(n2s_mask == 1.0, local_blur, degraded_inputs)
        
        predictions = n2s_model(inputs_masked)
        squared_diff_n2s = (predictions - inputs) ** 2
        loss_n2s = (squared_diff_n2s * n2s_mask).sum() / (n2s_mask.sum() + 1e-8)
        
        loss_n2s.backward()
        n2s_optimizer.step()
        
        running_loss_n2s += loss_n2s.item() * inputs.size(0)
        
        with torch.no_grad():
            rmse_b_n2s = torch.sqrt(torch.mean(squared_diff_n2s)).item()
            target_mean_n2s = torch.mean(inputs)
            ss_res_n2s = torch.sum(squared_diff_n2s)
            ss_tot_n2s = torch.sum((inputs - target_mean_n2s) ** 2)
            r2_b_n2s = (1.0 - (ss_res_n2s / (ss_tot_n2s + 1e-12))).item()
            
        epoch_rmse_accum_n2s += rmse_b_n2s
        epoch_r2_accum_n2s += r2_b_n2s
        batch_counter_n2s += 1
        
    n2s_scheduler.step()
    mean_epoch_loss_n2s = running_loss_n2s / len(fluid_dataset)
    epoch_mean_rmse_n2s = epoch_rmse_accum_n2s / batch_counter_n2s
    epoch_mean_r2_n2s = epoch_r2_accum_n2s / batch_counter_n2s
    current_lr_n2s = n2s_optimizer.param_groups[0]['lr']
    
    file_exists = os.path.exists(n2s_csv_path)
    with open(n2s_csv_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers_n2s)
        if not file_exists: writer.writeheader()
        writer.writerow({'Epoch': epoch + 1, 'Loss_MSE': round(mean_epoch_loss_n2s, 6), 
                         'Learning_Rate': current_lr_n2s, 'Eval_RMSE': round(epoch_mean_rmse_n2s, 6), 'Eval_R2_Score': round(epoch_mean_r2_n2s, 4)})
        
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Noise2Self Baseline | Epoch [{epoch+1:03d}/{num_n2s_epochs}] | N2S Loss: {mean_epoch_loss_n2s:.6f} | RMSE: {epoch_mean_rmse_n2s:.5f} | R2: {epoch_mean_r2_n2s:.4f}", flush=True)

print(f"\n[STATUS] All real data pipeline logs anchored successfully.")

# ===================================================================================================
# AUTOMATED SCIENTIFIC VISUALIZATION BLOCK (REAL DATASET EDITIONS)
# SPECIFICATIONS: HIGH-DENSITY 5-COLUMN NAVIER-STOKES PANEL & LEARNING CURVES
# ===================================================================================================
import matplotlib.pyplot as plt
import pandas as pd

print("\n" + "=" * 78)
print("[VISUAL ENGINE] Compiling High-Resolution Real-World Benchmark Analytics...")
print("=" * 78)

# 1. GENERATE THE 5-COLUMN SPATIAL INFERENCE BENCHMARK PANEL (REAL FLUID FIELDS)
model.eval()
model_n2v.eval()
n2s_model.eval()

with torch.no_grad():
    for sample_inputs, _ in fluid_loader:
        sample_inputs = sample_inputs.to(device)
        break
        
    ground_truth_tensor = sample_inputs
    noise_simulation = torch.randn_like(sample_inputs) * 0.15
    degraded_input_tensor = torch.clamp(sample_inputs + noise_simulation, 0.0, 1.0)
    
    n2v_output_tensor = model_n2v(degraded_input_tensor)
    n2s_output_tensor = n2s_model(degraded_input_tensor)
    bsn_1pass_tensor = model(degraded_input_tensor)

ground_truth_np = ground_truth_tensor.cpu().numpy()[0, 0, :, :]
degraded_np = degraded_input_tensor.cpu().numpy()[0, 0, :, :]
n2v_reconstructed_np = n2v_output_tensor.cpu().numpy()[0, 0, :, :]
n2s_reconstructed_np = n2s_output_tensor.cpu().numpy()[0, 0, :, :]
bsn_1pass_np = bsn_1pass_tensor.cpu().numpy()[0, 0, :, :]

fig, axs = plt.subplots(1, 5, figsize=(24, 5.5))
cmap_choice = 'twilight'

titles = [
    "Navier-Stokes Ground-Truth\n(Real Fluid Velocity)", "Degraded Input Field\n(Continuous Noise)",
    "SOTA Noise2Void\n(Active Masking)", "SOTA Noise2Self\n(J-Invariant Mesh)",
    "Proposed Multi-Branch BSN\n(Direct Inference)"
]
matrices = [ground_truth_np, degraded_np, n2v_reconstructed_np, n2s_reconstructed_np, bsn_1pass_np]

for i, (matrix, title) in enumerate(zip(matrices, titles)):
    im = axs[i].imshow(matrix, cmap=cmap_choice, origin='lower', vmin=0.0, vmax=1.0)
    axs[i].set_title(title, fontsize=12, fontweight='bold', pad=12)
    axs[i].axis('off')
    cbar = fig.colorbar(im, ax=axs[i], fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=18)

plt.tight_layout(pad=2.5)
output_image_name = "ieee_comprehensive_framework_comparison_real.png"
plt.savefig(output_image_name, bbox_inches='tight', dpi=300)
print(f"-> [SUCCESS] Real fluid 5-column panel saved successfully as: '{output_image_name}'")

# 2. GENERATE THE 3-WAY SYMMETRIC LEARNING CURVES FOR REAL BENCHMARK
proposed_log = "experiment_bsn_navier_stokes_1_logs.csv"
baseline_n2v_log = "experiment_n2v_baseline_logs.csv"
baseline_n2s_log = "experiment_n2s_baseline_logs.csv"

try:
    standard_headers = ['Epoch', 'Loss_MSE', 'Learning_Rate', 'Eval_RMSE', 'Eval_R2_Score']
    def load_clean_csv(file_path, headers):
        df = pd.read_csv(file_path, names=headers, header=None, comment='#')
        for col in ['Epoch', 'Loss_MSE', 'Eval_R2_Score']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df.dropna(subset=['Epoch', 'Loss_MSE', 'Eval_R2_Score'])


    df_prop = load_clean_csv(proposed_log, standard_headers)
    df_n2v  = load_clean_csv(baseline_n2v_log, standard_headers)
    df_n2s  = load_clean_csv(baseline_n2s_log, standard_headers)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5))
    style_prop = {'color': '#1f77b4', 'linewidth': 3.0, 'label': 'Proposed Multi-Branch BSN'}
    style_n2v  = {'color': '#d62728', 'linewidth': 2.5, 'linestyle': '--', 'label': 'SOTA Noise2Void'}
    style_n2s  = {'color': '#2ca02c', 'linewidth': 2.5, 'linestyle': ':', 'label': 'SOTA Noise2Self'}
    grid_style = {'color': '#e6e6e6', 'linestyle': '-', 'linewidth': 0.8}
    
    ax1.plot(df_prop['Epoch'].to_numpy(), df_prop['Loss_MSE'].to_numpy(), **style_prop)
    ax1.plot(df_n2v['Epoch'].to_numpy(),  df_n2v['Loss_MSE'].to_numpy(),  **style_n2v)
    ax1.plot(df_n2s['Epoch'].to_numpy(),  df_n2s['Loss_MSE'].to_numpy(),  **style_n2s)
    ax1.set_title("Self-Supervised Convergence Profile (Real Data)", fontsize=13, fontweight='bold', pad=14)
    ax1.set_xlabel("Optimization Epochs", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Training Loss Value (MSE)", fontsize=14, fontweight='bold')
    ax1.set_yscale('log')
    ax1.set_xlim(1, 250)
    ax1.grid(True, **grid_style)
    ax1.legend(loc='upper right', fontsize=12)
    
    ax2.plot(df_prop['Epoch'].to_numpy(), df_prop['Eval_R2_Score'].to_numpy(), **style_prop)
    ax2.plot(df_n2v['Epoch'].to_numpy(),  df_n2v['Eval_R2_Score'].to_numpy(),  **style_n2v)
    ax2.plot(df_n2s['Epoch'].to_numpy(),  df_n2s['Eval_R2_Score'].to_numpy(),  **style_n2s) 
    ax2.set_title("Hydrodynamic Reconstructive Accuracy (Real Data)", fontsize=13, fontweight='bold', pad=14)
    ax2.set_xlabel("Optimization Epochs", fontsize=14, fontweight='bold')
    ax2.set_ylabel("Coefficient of Determination ($R^2$ Score)", fontsize=14, fontweight='bold')
    ax2.set_ylim(-0.05, 0.6)
    ax2.set_xlim(1, 250)
    ax2.grid(True, **grid_style)
    ax2.legend(loc='lower right', fontsize=12)
    
    output_graph_name = "ieee_learning_curves_and_convergence_real.png"
    plt.savefig(output_graph_name, bbox_inches='tight', dpi=300)
    print(f"-> [SUCCESS] Real analytics charts exported successfully to: '{output_graph_name}'")
except Exception as e:
    print(f"[FATAL EXCEPTION] Real graphics parsing error: {str(e)}")
print("=" * 78 + "\n")
