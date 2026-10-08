# Graph-Enhanced Vehicle Re-Identification

**A PyTorch research project comparing graph-enhanced CNN and Vision Transformer models for vehicle re-identification.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Documentation updated**: October 8, 2026
> **Status**: Research code with selected archived experiment results

## 📋 Overview

Vehicle re-identification retrieves images of the same vehicle across different cameras. This project studies how graph-based spatial feature modeling affects retrieval accuracy and robustness under synthetic occlusion.

The implementation combines a global feature branch with a GCN or GAT branch built from spatial image features. It supports both **ResNet50-IBN-a** and **Vision Transformer (ViT)** backbones, with experiments on **VeRi-776** and **VehicleID**.

Research questions include the effect of backbone choice, graph depth, graph topology, pooling, and feature fusion. The repository includes model implementations, training configurations, evaluation scripts, and selected experiment outputs.

> Previously named `GCN_project`. The name `graph-enhanced-vehicle-reid` makes the application and the graph-enhanced approach explicit while covering both CNN and ViT experiments.

### Key Features

- **Dual Backbone Support**: ResNet50-IBN-a and ViT-Base with native 768-dim features
- **Graph-Based Enhancement**: Multi-layer GCN/GAT for spatial relationship modeling
- **Flexible Graph Construction**: Grid-based (4-neighbor, 8-neighbor) and k-NN dynamic graphs
- **Occlusion Analysis**: Archived VeRi-776 measurements at 11 occlusion levels (0–30%)
- **Two Dataset Workflows**: Training configurations for VeRi-776 and VehicleID
- **Ablation Studies**: Compare backbone, graph depth, and graph topology; see source-linked results below

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/SakuraUltra/graph-enhanced-vehicle-reid.git
cd graph-enhanced-vehicle-reid

# Create virtual environment
python -m venv venv_t4
source venv_t4/bin/activate  # Linux/Mac
# or: venv_t4\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

Run all commands below from the repository root. The entrypoints select CUDA when available and otherwise use CPU; they do not select Apple MPS. Dependencies are not pinned to an exact reproducible environment.

### Dataset Preparation

**VeRi-776:**
```bash
# Download from https://vehiclereid.github.io/VeRi/
# Extract to data/dataset/776_DataSet/
# Expected structure:
# 776_DataSet/
# ├── image_train/
# ├── image_query/
# ├── image_test/
# ├── train_label.xml
# └── ...
```

**VehicleID:**
```bash
# Download from https://pkuml.org/resources/pku-vehicleid.html
# Extract to data/dataset/VehicleID_V1.0/
# Expected structure:
# VehicleID_V1.0/
# ├── image/
# ├── train_test_split/
# │   ├── train_list.txt
# │   ├── test_list_800.txt
# │   ├── test_list_1600.txt
# │   └── test_list_2400.txt
# └── attribute/
```

Datasets and `.pth` checkpoints are not included in this repository. Obtain the datasets from their providers, then update `DATA.ROOT` in the chosen configuration.

### Training

Check initialization before running: the training script loads a local baseline checkpoint from top-level `MODEL.PRETRAINED_PATH` when it exists. A missing file changes initialization. The ViT configuration instead places a path under `MODEL.BACKBONE.PRETRAINED_PATH`; that nested path is not consumed by the top-level checkpoint loader. ViT backbone initialization uses the configured `PRETRAINED` setting. Do not assume these two paths are interchangeable.

**VeRi-776:**
```bash
# Train ResNet50 + GCN (1 layer, 4-neighbor)
python scripts/training/train_bot_gcn.py \
    --config configs/gcn_transformer_configs/abl_cnn_gcn_4nb_l1.yaml

# Train ViT-Base + GCN (1 layer, 4-neighbor)
python scripts/training/train_bot_gcn.py \
    --config configs/gcn_transformer_configs/abl_vit_gcn_4nb_l1.yaml
```

**VehicleID:**
```bash
# Train ResNet50 + kNN GCN (1 layer)
python scripts/training/train_bot_gcn.py \
    --config configs/gcn_transformer_configs/abl_vehicleid_cnn_gcn_knn_l1.yaml

# Train ViT-Base + kNN GCN (1 layer)
python scripts/training/train_bot_gcn.py \
    --config configs/gcn_transformer_configs/abl_vehicleid_vit_gcn_knn_l1.yaml
```

### Evaluation

**VeRi-776 occlusion evaluation:** prepare the clean dataset, the trained checkpoints listed in `ABLATION_MODELS` in the evaluator, and directories `query_00pct`, `query_03pct`, …, `query_30pct` beneath the occlusion root. Preserve the original VeRi query filenames. The prepared occlusion images and their generation script are not included in this checkout.

```bash
python scripts/testing/evaluate_occlusion_abl19.py \
    --occ_root outputs/occlusion_tests_v2 \
    --output_dir outputs/ablation_occlusion_results_local
```

The script accepts `--occ_root` and `--output_dir`; the clean dataset path is the `DATASET_ROOT` constant in the script. It selects CUDA or CPU automatically. Its current model list contains **7 models**, omitting ABL-12, whereas the archive below contains **8 CSV files**. Missing checkpoints or occlusion directories are skipped; a successful process exit alone does not confirm complete evaluation. Use a new output directory to preserve the archive.

**VehicleID-Small evaluation:** requires the standard `test_list_800.txt` split and the checkpoints listed in the VehicleID evaluator.

```bash
python scripts/testing/evaluate_occlusion_vehicleid.py \
    --dataset_root data/dataset/VehicleID_V1.0 \
    --output_dir outputs/vehicleid_occlusion_results_local \
    --test_size small \
    --device cuda:0
```

Current limitations:
- `--test_size small`, `medium`, and `large` select `test_list_800.txt`, `test_list_1600.txt`, and `test_list_2400.txt`, respectively. The evaluator stops with an explicit error if the selected file is missing. Older versions could fall back to the 800-ID split; rerun any affected medium/large evaluations in a fresh output directory.
- This evaluator applies random occlusion to both query and gallery. The VeRi evaluator uses prepared query images and a clean gallery, so their robustness protocols differ.
- A seeded query/gallery split does not seed every random occlusion transform. The separate offline VehicleID image generator uses a different split procedure and is not a prerequisite for this evaluator.
- Existing result CSV files may be reused by the evaluator. Choose a fresh output directory for a new run.

See the [script guide](scripts/README.md) for available entrypoints and cluster setup notes. These commands were checked against the source; training and checkpoint-based evaluation have not been rerun for this documentation update.

## 📊 Results

### VeRi-776: archived occlusion-run snapshot

Both tables use the **same eight individual CSV files** linked below. Clean performance is the `occlusion_level = 0` row, and Occ30 is the `occlusion_level = 30` row. These are saved measurements, not newly rerun experiments. The [older ablation report](docs/ablation/ABLATION_RESULTS_FINAL.md) records a separate historical result set and should not be mixed into this snapshot.

#### Clean performance

| Model / source CSV | Backbone | GCN layers | mAP | Rank-1 | Rank-5 | Rank-10 |
|---|---|---:|---:|---:|---:|---:|
| [ABL-01](outputs/ablation_occlusion_results/individual_results/ABL01_CNN_4nb_L1.csv) | ResNet50-IBN | 1 | 74.52% | 93.62% | 97.26% | 98.09% |
| [ABL-04](outputs/ablation_occlusion_results/individual_results/ABL04_CNN_4nb_L2.csv) | ResNet50-IBN | 2 | 74.20% | 94.22% | 97.14% | 98.57% |
| [ABL-05](outputs/ablation_occlusion_results/individual_results/ABL05_CNN_4nb_L3.csv) | ResNet50-IBN | 3 | 74.37% | 94.28% | 97.62% | 98.33% |
| [ABL-08](outputs/ablation_occlusion_results/individual_results/ABL08_CNN_kNN_L1.csv) | ResNet50-IBN + kNN | 1 | 73.61% | 93.15% | 97.20% | 98.63% |
| [ABL-02](outputs/ablation_occlusion_results/individual_results/ABL02_ViT_4nb_L1.csv) | ViT-Base-768 | 1 | 72.82% | 94.70% | 97.91% | 98.99% |
| [ABL-11](outputs/ablation_occlusion_results/individual_results/ABL11_ViT_4nb_L2.csv) | ViT-Base-768 | 2 | 72.34% | 93.92% | 97.56% | 98.81% |
| [ABL-12](outputs/ablation_occlusion_results/individual_results/ABL12_ViT_4nb_L3.csv) | ViT-Base-768 | 3 | 72.40% | 94.04% | 97.20% | 98.51% |
| [ABL-15](outputs/ablation_occlusion_results/individual_results/ABL15_ViT_kNN_L1.csv) | ViT-Base + kNN | 1 | 72.67% | 94.28% | 97.74% | 98.87% |

#### Performance at the archived 30% occlusion level

| Model | Clean mAP | Occ30 mAP | Absolute drop (pp) | Relative drop |
|---|---:|---:|---:|---:|
| ABL-01 | 74.52% | 64.71% | 9.81 | 13.16% |
| ABL-04 | 74.20% | 64.67% | 9.53 | 12.85% |
| ABL-05 | 74.37% | 65.09% | 9.28 | 12.47% |
| ABL-08 | 73.61% | 63.64% | 9.98 | 13.55% |
| ABL-02 | 72.82% | 65.13% | 7.69 | 10.56% |
| ABL-11 | 72.34% | 64.76% | 7.58 | 10.47% |
| ABL-12 | 72.40% | 64.86% | 7.53 | 10.41% |
| ABL-15 | 72.67% | 64.91% | 7.76 | 10.68% |

`Absolute drop = clean mAP − Occ30 mAP` in **percentage points (pp)**. `Relative drop = 100 × (clean mAP − Occ30 mAP) / clean mAP`. Calculations use unrounded CSV values; displayed values are rounded to two decimals.

In this snapshot, ABL-01 has the highest clean mAP (74.52%), ABL-02 has the highest Occ30 mAP (65.13%), and ABL-12 has the smallest relative drop (10.41%). Among CNN variants, ABL-05 has the highest Occ30 mAP (65.09%). These are different criteria; no single graph depth wins every comparison. The archive does not establish statistical significance or state-of-the-art performance.

### VehicleID-Small Performance

**Historical README figures:** retained from the original report. The corresponding raw VehicleID result CSV files and checkpoints are not included, so these values have not been independently verified in this update. They are not part of the source-linked VeRi snapshot above.

| Model | Backbone | GCN Layers | mAP | Rank-1 | Rank-5 | Params |
|-------|----------|------------|-----|--------|--------|--------|
| **CNN Models** | | | | | | |
| VID-01 | ResNet50-IBN | 1 | **89.04%** | 95.87% | 97.97% | 63.4M |
| VID-02 | ResNet50-IBN | 2 | 88.92% | 95.82% | 97.92% | 63.9M |
| VID-03 | ResNet50-IBN | 3 | 88.68% | 95.72% | 97.87% | 64.4M |
| VID-04 | ResNet50-IBN + kNN | 1 | 89.18% | 95.96% | 98.03% | 63.4M |
| **ViT Models** | | | | | | |
| VID-05 | ViT-Base | 1 | 87.45% | 94.92% | 97.54% | 135.2M |
| VID-06 | ViT-Base | 2 | 86.87% | 94.65% | 97.38% | 136.1M |
| VID-07 | ViT-Base | 3 | 85.23% | 93.71% | 96.89% | 137.0M |
| VID-08 | ViT-Base + kNN | 1 | **89.18%** | 95.94% | 98.01% | 135.2M |

## 🏗️ Architecture

```
Input Image (256×256 or 224×224)
    ↓
Backbone (ResNet50-IBN / ViT-Base)
    ↓
Spatial Features (H×W×C)
    ↓
Grid Construction (4×4 nodes)
    ↓
Graph Adjacency (4-neighbor / 8-neighbor / k-NN)
    ↓
Multi-layer GCN/GAT (1-3 layers)
    ↓
Graph Pooling (mean / max / attention)
    ↓
Feature Fusion (concat / add)
    ↓
BNNeck + Classifier
    ↓
ID Loss + Triplet Loss
```

## 📂 Project Structure

```text
graph-enhanced-vehicle-reid/
├── configs/                  # Baseline, graph model, dataset and augmentation configs
├── models/                   # Backbones, graph modules, fusion and BoT models
├── train/                    # Training utilities and schedulers
├── eval/                     # Retrieval evaluation
├── losses/                   # Identification and triplet losses
├── utils/                    # Sampling, augmentation, metrics and reproducibility
├── scripts/                  # Training, evaluation, graph preparation and SLURM jobs
├── experiments/              # Historical stage-specific notes
├── docs/ablation/            # Earlier ablation report and CSV
├── outputs/                  # Selected committed CSVs and training logs
└── requirements.txt
```

Datasets and model checkpoints must be supplied separately; generated output directories shown in configuration files may not yet exist.

## 🔬 Ablation Studies

The archived VeRi comparison covers CNN depth (ABL-01/04/05), ViT depth (ABL-02/11/12), and one-layer k-NN variants (ABL-08/15). The implementation also supports 8-neighbor graphs, but they are not represented in the eight CSVs summarized here. Lower retrieval scores alone do not establish over-smoothing as their cause.

## 📈 Training Details

The following settings describe the historical experiment report. Consult each YAML configuration for the settings of a new run; runtimes depend on hardware and data access.

### Hardware & Environment
- **GPU**: NVIDIA H100 PCIe (80GB VRAM)
- **Training Time**: 
  - VeRi-776: ~2 hours per model (120 epochs)
  - VehicleID: ~3-4 hours per model (120-180 epochs)
- **Precision**: FP32 (AMP disabled for stability)

### Hyperparameters

| Parameter | VeRi-776 CNN | VeRi-776 ViT | VehicleID CNN | VehicleID ViT |
|-----------|--------------|--------------|---------------|---------------|
| **Batch Size** | 64 (P=16, K=4) | 64 (P=16, K=4) | 64 (P=16, K=4) | **128** (P=32, K=4) |
| **Optimizer** | AdamW | AdamW | AdamW | AdamW |
| **Base LR** | 3.5e-4 | **3.5e-5** | 3.5e-4 | **6.0e-5** |
| **Weight Decay** | 5e-4 | 1e-4 | 5e-4 | 5e-4 |
| **Warmup Epochs** | 10 | 10 | 10 | 10 |
| **Total Epochs** | 120 | 120 | 120 | 180 |
| **Early Stopping** | ❌ | ❌ | ❌ | ✓ (patience=50) |
| **Scheduler** | WarmupCosineAnnealingLR | WarmupCosineAnnealingLR | WarmupCosineAnnealingLR | WarmupCosineAnnealingLR |

**Important Notes:**
- ViT models use **5.8-10× smaller learning rate** than CNN (standard practice for Transformers)
- VehicleID ViT uses **2× larger batch size** (128 vs 64) for better stability
- No re-ranking used in evaluation (all results are direct cosine distance)

### Data Augmentation
- **Training**: Random Horizontal Flip + Random Erasing (p=0.5, area=2%-20%, r ∈ [0.3, 3.3])
- **Occlusion Testing**: Dataset-specific protocols; see evaluation limitations above. A fully seeded occlusion rerun is not established by the committed scripts.
- **Resize**: 256×256 (CNN), 224×224 (ViT)
- **Normalization**: ImageNet statistics

## 🔬 Experiment Reproduction

Configurations and scripts provide starting points for reruns, subject to the dataset, checkpoint, initialization, and evaluator limitations above.

**VeRi-776: individual cluster jobs**
```bash
sbatch scripts/experiments/ablation/run_abl_cnn_4nb_l1.sh
sbatch scripts/experiments/ablation/run_abl_vit_4nb_l1.sh
```

**VehicleID: individual cluster jobs**
```bash
sbatch scripts/experiments/ablation/run_vid_cnn_4nb_l1.sh
sbatch scripts/experiments/ablation/run_vid_vit_4nb_l1.sh
```

Before submitting, edit cluster-specific working directories, environment activation, partitions, and GPU requests. Some scripts retain the original `/users/sl3753/scratch/GCN_project` checkout path; renaming the GitHub repository does not rename a cluster checkout. For local runs, use the Python training commands above instead of `sbatch`.

**Pre-trained Weights:** Available upon request (contact via GitHub Issues)

## 📄 Citation

If you find this work useful, please consider citing:

```bibtex
@article{gcnreid2026,
  title={Graph-Enhanced Vision Transformer for Robust Vehicle Re-Identification},
  author={Sakura Ultra},
  journal={Under Review},
  year={2026}
}
```

## 📜 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- **VeRi-776 Dataset**: [VeRi Dataset](https://vehiclereid.github.io/VeRi/) - X. Liu et al.
- **VehicleID Dataset**: [PKU-VehicleID](https://pkuml.org/resources/pku-vehicleid.html) - H. Liu et al.
- **PyTorch**: Deep learning framework
- **Timm Library**: ViT-Base pre-trained weights
- **ResNet-IBN**: IBN-Net architecture for domain generalization

## 📧 Contact

For questions, collaboration, or pre-trained weights:
- Email: sl3753@york.ac.uk
- GitHub Issues: [Create an issue](https://github.com/SakuraUltra/graph-enhanced-vehicle-reid/issues)
- Repository: [https://github.com/SakuraUltra/graph-enhanced-vehicle-reid](https://github.com/SakuraUltra/graph-enhanced-vehicle-reid)

---

⭐ **If you find this work helpful, please star this repository!**

**Documentation updated**: October 8, 2026 | Selected experiment results archived
