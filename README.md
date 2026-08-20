# Self-Supervised Multi-Branch Blind-Spot Networks for Embedded Field Restoration

Official repository for the paper: **"Self-Supervised Multi-Branch Blind-Spot Networks With Strict Lattice Continuity and Verified Hermeticity For Embedded Field Restoration"** (IEEE Compliant Format).

---

## 📌 Project Overview
This repository delivers a mathematically hermetic Self-Supervised Multi-Branch Blind-Spot Network (MB-BSN) tailored for real-time spatial restoration of continuous hydrodynamic fields under severe sensor distortion. By hard-coding strict topological masking operators directly into parallel multi-scale convolutional streams, our framework achieves complete mathematical hermeticism against intra-channel data leakage.

The execution ecosystem is specifically optimized to avoid random GPU thread drop stalls, running seamlessly on resource-constrained general-purpose edge CPUs (such as the mobile Intel Core i7-8665U architecture).

---

## ⚙️ Mathematical Metrics & Design Paradigm
To align with embedded hardware limitations, the Coefficient of Determination ($R^2$) uses an online batch-wise aggregation paradigm rather than global post-epoch calculations, ensuring edge memory frugality and stochastic variance tracking.

---

## 📊 Dataset Availability & Reference Citation
The evaluations utilize the 2D Navier-Stokes fluid mechanics velocity streams retrieved from the Kaggle Digital Repository:

```latex
\(\bibitem{ref_kaggle_dataset}\)
W. Wang, ``Navier-Stokes 2D Fluid Mechanics Benchmark Dataset (N5000nse)\(,'' \textit{Kaggle Digital Repository}, 2023, Available online: \url{https://kaggle.com}. \%\%\)MAGIT_PARSER_PROTECT%%```

---

## 📂 Repository Inventory
The repository includes core files for execution and analysis:
* `main_launcher.py`: Interactive command-line terminal interface controlling execution regimes.
* `models.py`: Core multi-branch blind-spot model topology implementing element-wise Hadamard masking operators.
* `dataset.py`: Fluid mechanics data loader featuring an automated continuous Navier-Stokes synthesis engine fallback.
* `train_real.py`: Production training infrastructure utilizing genuine physics observation samples.
* `train_synthetic.py`: Numerical optimization stream operating under controlled high-frequency noise profiles.
* `run_ablation.py`: Parametric restriction engine executing selective feature suppression steps.
* `compile_scientific_tables.py`: Analytical log processor to auto-generate ready-to-paste LaTeX tables.
* `graphics.py`: Active visualization pipeline tracking model optimization trajectories.

---

## 🚀 Execution & Deployment Guide
Clone the repository and install requirements:
```bash
git clone 
