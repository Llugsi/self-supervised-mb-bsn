# ===================================================================================================
# GITHUB REPOSITORY MODULE: CENTRAL PLATFORM ORCHESTRATOR / LAUNCHER (LIVE JUPYTER LAB EDITION)
# ===================================================================================================

import os
import sys
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import subprocess

# Strict imports from your pristine repository files
from models import TNNLS_BlindSpotNet
from dataset import PDENavierStokesDataset

def clear_terminal():
    """Clears the console screen across operating systems."""
    os.system('cls' if os.name == 'nt' else 'clear')

def execute_standalone_script(script_name):
    """
    CORREGIDO: Ejecuta el script secundario leyendo el flujo de salida 
    línea por línea en tiempo real para romper el estado 'Busy' de Jupyter Lab.
    """
    if not os.path.exists(script_name):
        print(f"\n[ERROR] File '{script_name}' not found in the current directory.")
        input("\nPress Enter to return to the Main Menu...")
        return
        
    print(f"\n" + "="*78)
    print(f"[ORCHESTRATOR] Initializing real-time training logs for: '{script_name}'")
    print("="*78 + "\n")
    
    # Abrimos el proceso redirigiendo stdout en vivo y forzando texto decodificado
    process = subprocess.Popen(
        [sys.executable, script_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1  # Forzar el búfer de línea por línea
    )
    
    # Leer el flujo continuamente a medida que el script de entrenamiento imprime texto
    for line in process.stdout:
        print(line, end='', flush=True)
        
    process.wait()
    
    print(f"\n" + "="*78)
    print(f"[ORCHESTRATOR] Execution of '{script_name}' completed with code {process.returncode}.")
    print("="*78)
    input("\nPress Enter to return to the Main Menu...")

def run_parametric_ablation(experiment_id, mode_label):
    """Executes dynamic channel-wise ablation over the native synthetic data loop."""
    clear_terminal()
    print("="*78)
    print(f"[ABLATION ENGINE] Launching Experiment {experiment_id} ({mode_label.upper()})")
    print("="*78)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    filename = "2D_CFD_Rand_M0.1_Eta0.01_Zeta0.01_periodic_128_Test.hdf5"
    
    dataset = PDENavierStokesDataset(filename, time_steps_per_sample=1, channel_idx=0)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    log_csv_path = f"ablation_exp_{experiment_id}_synthetic_logs_a.csv"
    if os.path.exists(log_csv_path):
        os.remove(log_csv_path)
        
    model = TNNLS_BlindSpotNet(in_channels=1, out_channels=1, ablation_mode=mode_label).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-6)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.5)
    criterion = nn.MSELoss()
    
    headers = ['Epoch', 'Loss_MSE', 'Learning_Rate', 'Eval_RMSE', 'Eval_R2_Score']
    num_epochs = 250
    
    for epoch in range(num_epochs):
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
        current_lr = optimizer.param_groups[0]['lr']
        
        with open(log_csv_path, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            if epoch == 0: writer.writeheader()
            writer.writerow({'Epoch': epoch + 1, 'Loss_MSE': round(epoch_mean_loss, 6), 
                             'Learning_Rate': current_lr, 'Eval_RMSE': round(epoch_mean_rmse, 6), 'Eval_R2_Score': round(epoch_mean_r2, 4)})
            
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"Exp {experiment_id} | Epoch [{epoch+1:03d}/{num_epochs}] | Loss: {epoch_mean_loss:.6f} | R2: {epoch_mean_r2:.4f}", flush=True)
            
    print(f"\n[STATUS] Experiment {experiment_id} logs successfully compiled at: '{log_csv_path}'")
    input("\nPress Enter to return to the Ablation Board...")

def ablation_submenu():
    while True:
        clear_terminal()
        print("="*78)
        print("  ARCHITECTURAL ABLATION MATRIX BOARD — METRIC REDUCTION TRACKER")
        print("="*78)
        print("  1. Run Experiment I: Ultra-Local Context Isolation (Local Stream Only)")
        print("  2. Run Experiment II: Macro-Scale Sparse Isolation (Dilated Stream Only)")
        print("  3. Run Experiment III: Dense Multi-Scale Mixture (Excluding Dilation)")
        print("  4. Return to Main Orchestrator Panel")
        print("-"*78)
        
        choice = input("Select ablation configuration target [1-4]: ").strip()
        if choice == '1':
            run_parametric_ablation("I", "local_only")
        elif choice == '2':
            run_parametric_ablation("II", "dilated_only")
        elif choice == '3':
            run_parametric_ablation("III", "dense_mix")
        elif choice == '4':
            break

def main_menu():
    while True:
        clear_terminal()
        print("="*78)
        print("  DETECTION ENGINE: SELF-SUPERVISED MULTI-BRANCH BSN BENCHMARK RESTORATION")
        print("  SYSTEM ARCHITECTURE CONTROL BOARD FOR EMBEDDED EDGE COMPUTING PLATFORMS")
        print("="*78)
        print("  1. Run Full Framework on Genuine Real Dataset (Caltech - 250 Epochs)")
        print("  2. Run Full Framework on Structured Synthetic Simulation (250 Epochs)")
        print("  3. Open Architectural Ablation Control Board (Experiments I, II, III)")
        print("  4. Compile Log Statistics & Generate LaTeX Tables Code")
        print("  5. Verify Local Workspace Integrity & File Matrix Verification")
        print("  6. Exit System Controller")
        print("-"*78)
        
        choice = input("Select an option target [1-6]: ").strip()
        if choice == '1':
            execute_standalone_script("train_real.py")
        elif choice == '2':
            execute_standalone_script("train_synthetic.py")
        elif choice == '3':
            ablation_submenu()
        elif choice == '4':
            execute_standalone_script("compile_scientific_tables.py")
        elif choice == '5':
            clear_terminal()
            print("="*78)
            print("[INTEGRITY AUDITOR] Validating physical workspace alignment indices...")
            print("=" * 78)
            required_files = ["models.py", "dataset.py", "train_real.py", "train_synthetic.py", "compile_scientific_tables.py"]
            missing = 0
            for f in required_files:
                status = "FOUND" if os.path.exists(f) else "MISSING"
                print(f" -> Structural Module Target: {f:<22} | Status: [{status}]")
                if status == "MISSING": missing += 1
            print("-" * 78)
            if missing == 0:
                print("[STATUS] Integrity verified. System contains all required repository elements.")
            else:
                print(f"[WARNING] Local mismatch detected. {missing} core file(s) missing.")
            input("\nPress Enter to return to the Main Menu...")
        elif choice == '6':
            print("\nExiting System Orchestrator. Platform operational states released safely.\n")
            break

if __name__ == "__main__":
    main_menu()
