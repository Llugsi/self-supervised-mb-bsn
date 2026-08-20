# Self-Supervised Multi-Branch Blind-Spot Networks for Embedded Field Restoration

Official repository for the Multi-Branch Blind-Spot Network (MB-BSN) framework tailored for real-time spatial restoration of hydrodynamic fields.

---

## 📌 Project Overview
The framework implements structured topological masking operators to prevent data leakage and is fully optimized to run efficiently on resource-constrained embedded edge CPUs.

---

## 📊 Dataset Availability & Reference
The 2D Navier-Stokes fluid mechanics velocity dataset can be downloaded manually from the following link:
* **Dataset:** [Kaggle Repository](https://www.kaggle.com/datasets/wenwangou/n5000nse?resource=download)

---

## 📂 Repository Inventory
* `main_launcher.py`: Command-line terminal interface controlling execution regimes.
* `models.py` and `dataset.py`: Core model topology and physical data loader components.
* `train_real.py` and `train_synthetic.py`: Production training pipelines for real and synthetic domains.
* `run_ablation.py` and `compile_scientific_tables.py`: Parametric ablation tests and automated LaTeX table compiler.

---

## 🚀 Execution Guide

### 1. Environment Setup & Cloning
Clone the repository and install the standard dependency stack:
```bash
git clone https://github.com/Llugsi/
cd self-supervised-mb-bsn
pip install -r requirements.txt
```

### 2. Launching the System Control Panel
Initiate the main central orchestrator:
```bash
python main_launcher.py
```
Select the respective terminal options to execute model training, perform ablation studies, or compile raw LaTeX tables.

---

## 📜 License
Distributed under the terms of the open-source **MIT License**.
