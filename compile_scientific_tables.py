# ===================================================================================================
# GITHUB REPOSITORY MODULE: AUTOMATED SCIENTIFIC METRIC METADATA COMPILER
# EXTRACTION ENGINE FOR REAL, SYNTHETIC, AND ABLATION EXPERIMENTAL TRAJECTORIES (IEEE COMPLIANT)
# ===================================================================================================

import os
import pandas as pd
import numpy as np

def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')

def process_log_metrics(file_path):
    """
    Safely opens any tracking CSV log and extracts the stable statistical convergence
    profile derived exclusively from the final 10 optimization epochs.
    """
    if not os.path.exists(file_path):
        return None
        
    standard_headers = ['Epoch', 'Loss_MSE', 'Learning_Rate', 'Eval_RMSE', 'Eval_R2_Score']
    try:
        # Read dataframe discarding potential messy text headers safely
        df = pd.read_csv(file_path, names=standard_headers, header=None, comment='#')
        if pd.to_numeric(df.iloc[0, 0], errors='coerce') is np.nan or str(df.iloc[0, 0]).lower() == 'epoch':
            df = df.iloc[1:].reset_index(drop=True)
            
        # Hard force cellular numerical allocation
        for col in ['Loss_MSE', 'Eval_RMSE', 'Eval_R2_Score']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        # Isolate the final 10 convergence landmarks
        last_epochs = df.dropna(subset=['Loss_MSE', 'Eval_RMSE', 'Eval_R2_Score']).iloc[-10:]
        if len(last_epochs) == 0:
            return None
            
        return {
            'loss_mean': last_epochs['Loss_MSE'].mean(),
            'rmse_mean': last_epochs['Eval_RMSE'].mean(),
            'r2_mean': last_epochs['Eval_R2_Score'].mean(),
            'r2_std': last_epochs['Eval_R2_Score'].std()
        }
    except Exception as e:
        print(f"[ERROR] Failed parsing '{file_path}': {str(e)}")
        return None

def main():
    clear_terminal()
    print("="*78)
    print("  METADATA ENGINE: AUTOMATED METRIC COMPILER FOR MANUSCRIPT DATA")
    print("="*78)

    # -----------------------------------------------------------------------------------------------
    # DATA EXTRACTION DICTIONARIES MAPPING
    # -----------------------------------------------------------------------------------------------
    real_logs = {
        "Proposed Multi-Branch BSN": "experiment_bsn_navier_stokes_1_logs.csv",
        "SOTA Noise2Void Baseline": "experiment_n2v_baseline_logs.csv",
        "SOTA Noise2Self Baseline": "experiment_n2s_baseline_logs.csv"
    }

    synthetic_logs = {
        "Proposed Multi-Branch BSN": "experiment_bsn_navier_stokes_1_logs_a.csv",
        "SOTA Noise2Void Baseline": "experiment_n2v_baseline_logs_a.csv",
        "SOTA Noise2Self Baseline": "experiment_n2s_baseline_logs_a.csv"
    }

    ablation_logs = {
        "Exp I: Local Stream Only": "ablation_exp_I_synthetic_logs_a.csv",
        "Exp II: Dilated Stream Only": "ablation_exp_II_synthetic_logs_a.csv",
        "Exp III: Dense Multi-Scale Mixture": "ablation_exp_III_synthetic_logs_a.csv",
        "Exp IV: Full Proposed Framework": "experiment_bsn_navier_stokes_1_logs_a.csv"
    }

    # ===============================================================================================
    # TABLE I COMPILATION: GENUINE REAL-WORLD BENCHMARK (CALTECH)
    # ===============================================================================================
    print("\n[COMPILING] Processing Table I: Genuine Real-World Benchmark (Caltech)...")
    latex_rows_real = ""
    for name, path in real_logs.items():
        metrics = process_log_metrics(path)
        if metrics:
            is_proposed = "Proposed" in name
            bp, bs = ("\\textbf{", "}") if is_proposed else ("", "")
            latex_rows_real += f"        {name} & {metrics['loss_mean']:.5f} & {metrics['rmse_mean']:.5f} & {bp}{metrics['r2_mean']:.4f} \\pm {metrics['r2_std']:.4f}{bs} \\\\\n"
        else:
            latex_rows_real += f"        {name} & N/A & N/A & N/A \\\\\n"

    # ===============================================================================================
    # TABLE II COMPILATION: STRUCTURED SYNTHETIC HYDRODYNAMICS
    # ===============================================================================================
    print("[COMPILING] Processing Table II: Structured Synthetic Hydrodynamics...")
    latex_rows_synth = ""
    for name, path in synthetic_logs.items():
        metrics = process_log_metrics(path)
        if metrics:
            is_proposed = "Proposed" in name
            bp, bs = ("\\textbf{", "}") if is_proposed else ("", "")
            latex_rows_synth += f"        {name} & {metrics['loss_mean']:.5f} & {metrics['rmse_mean']:.5f} & {bp}{metrics['r2_mean']:.4f} \\pm {metrics['r2_std']:.4f}{bs} \\\\\n"
        else:
            latex_rows_synth += f"        {name} & N/A & N/A & N/A \\\\\n"

    # ===============================================================================================
    # TABLE III COMPILATION: ARCHITECTURAL ABLATION ANALYSIS
    # ===============================================================================================
    print("[COMPILING] Processing Table III: Architectural Ablation Matrix...")
    latex_rows_ablation = ""
    for name, path in ablation_logs.items():
        metrics = process_log_metrics(path)
        if metrics:
            is_full = "Full" in name
            bp, bs = ("\\textbf{", "}") if is_full else ("", "")
            latex_rows_ablation += f"        {name} & {metrics['loss_mean']:.5f} & {metrics['rmse_mean']:.5f} & {bp}{metrics['r2_mean']:.4f} \\pm {metrics['r2_std']:.4f}{bs} \\\\\n"
        else:
            latex_rows_ablation += f"        {name} & N/A & N/A & N/A \\\\\n"

    # ===============================================================================================
    # OUTPUT FORMATTING MANIFOLD FOR LATEX OVERLEAF CODES
    # ===============================================================================================
    print("\n" + "="*78)
    print(" LATEX CODE GENERATED FOR OVERLEAF (PASTE INSIDE YOUR MANUSCRIPT)")
    print("="*78)

    print(f"""
% --- TABLE I: REAL DATASET BENCHMARK ---
\\begin{{table}}[htbp]
    \\caption{{Quantitative Framework Performance on Genuine 2D Navier-Stokes Fluid Dataset.}}
    \\label{{tab:real_benchmark_comparison}}
    \\centering
    \\begin{{tabular}}{{lcccc}}
        \\hline
        \\textbf{{Framework Architecture}} & \\textbf{{Final MSE Loss}} & \\textbf{{Eval RMSE}} & \\textbf{{Eval $R^2$ Score}} \\\\
        \\hline
{latex_rows_real}        \\hline
    \\end{{tabular}}
\\end{{table}}

% --- TABLE II: SYNTHETIC DATASET BENCHMARK ---
\\begin{{table}}[htbp]
    \\caption{{Quantitative Framework Performance on Structured Synthetic Hydrodynamics Dataset.}}
    \\label{{tab:synthetic_benchmark_comparison}}
    \\centering
    \\begin{{tabular}}{{lcccc}}
        \\hline
        \\textbf{{Framework Architecture}} & \\textbf{{Final MSE Loss}} & \\textbf{{Eval RMSE}} & \\textbf{{Eval $R^2$ Score}} \\\\
        \\hline
{latex_rows_synth}        \\hline
    \\end{{tabular}}
\\end{{table}}

% --- TABLE III: ARCHITECTURAL ABLATION MATRIX ---
\\begin{{table}}[htbp]
    \\caption{{Architectural Ablation Matrix and Structural Scale Convergence Analysis.}}
    \\label{{tab:manuscript_ablation_matrix}}
    \\centering
    \\begin{{tabular}}{{lcccc}}
        \\hline
        \\textbf{{Ablation Variant Stream}} & \\textbf{{Final MSE Loss}} & \\textbf{{Eval RMSE}} & \\textbf{{Eval $R^2$ Score}} \\\\
        \\hline
{latex_rows_ablation}        \\hline
    \\end{{tabular}}
\\end{{table}}
""")
    print("="*78)
    print("[STATUS] Scientific summary extraction closed. Ready for Overleaf submission.")
    print("="*78 + "\n")

if __name__ == "__main__":
    main()
