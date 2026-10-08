# Script guide

Run commands from the repository root. Install `requirements.txt`, obtain the datasets, and check the configuration and checkpoint paths first. See the [main README](../README.md) for dataset layout, result sources, and evaluation limitations.

## Training

| Entrypoint | Purpose |
|---|---|
| `scripts/training/train_bot_gcn.py` | Train the global + graph model using CNN or ViT configurations |
| `scripts/training/train_bot_baseline.py` | Train the BoT baseline |
| `scripts/training/monitor_training.py` | Inspect training progress |

```bash
python scripts/training/train_bot_gcn.py \
    --config configs/gcn_transformer_configs/abl_cnn_gcn_4nb_l1.yaml

python scripts/training/train_bot_gcn.py \
    --config configs/gcn_transformer_configs/abl_vit_gcn_4nb_l1.yaml
```

The graph trainer accepts `--output_dir`, `--resume`, `--eval_only`, and graph configuration overrides. Check `--help` after installing dependencies. Baseline checkpoints are not included. The local checkpoint loader reads top-level `MODEL.PRETRAINED_PATH`, not the nested ViT `MODEL.BACKBONE.PRETRAINED_PATH`; check initialization before attempting a historical rerun.

## Occlusion evaluation

| Entrypoint | Inputs and behavior |
|---|---|
| `scripts/testing/evaluate_occlusion_abl19.py` | VeRi dataset, prepared query images, and model checkpoints; current list has 7 models |
| `scripts/testing/evaluate_occlusion_vehicleid.py` | VehicleID dataset and model checkpoints; applies occlusion during evaluation |
| `scripts/testing/generate_vehicleid_occlusion_dataset.py` | Separate offline image generation workflow with its own split procedure |

```bash
python scripts/testing/evaluate_occlusion_abl19.py \
    --occ_root outputs/occlusion_tests_v2 \
    --output_dir outputs/ablation_occlusion_results_local

python scripts/testing/evaluate_occlusion_vehicleid.py \
    --dataset_root data/dataset/VehicleID_V1.0 \
    --output_dir outputs/vehicleid_occlusion_results_local \
    --test_size small \
    --device cuda:0
```

The VeRi evaluator accepts only `--occ_root` and `--output_dir`; change its `DATASET_ROOT` constant if the clean dataset is elsewhere. It expects `query_00pct` through `query_30pct` in increments of 3, with original VeRi filenames, and skips unavailable checkpoints/directories. Prepared VeRi occlusion images and their generator are not included. Its current list omits ABL-12 even though that model has an archived CSV.

The VehicleID evaluator currently requests textual split filenames such as `test_list_small.txt`, while standard files are named `test_list_800.txt`, `test_list_1600.txt`, and `test_list_2400.txt`. Missing files trigger a fallback to the 800-ID split; **do not interpret medium/large runs as correctly selecting those splits**. The small command above relies on that fallback with a standard dataset layout. The training script has a separate numeric mapping.

VehicleID applies random occlusion to both query and gallery; VeRi uses prepared occluded queries and a clean gallery. Split seeding alone does not make random occlusion fully reproducible. The offline VehicleID generator is not required by the on-the-fly evaluator and uses a different query/gallery split. Use a fresh output directory for new evaluations so existing CSVs are neither overwritten nor reused.

## Graph preparation

The historical offline feature workflow lives in `scripts/graph_preparation/`:

1. `extract_features.py` extracts image features from a trained baseline.
2. `generate_graph_nodes.py` pools spatial features into nodes.
3. `build_graph_edges.py` builds grid adjacency; `build_knn_graph.py` builds feature-based adjacency.

Inspect each script's arguments and file paths before use. This separate workflow is not a prerequisite for the end-to-end `train_bot_gcn.py` examples above, which build spatial graphs within the model.

## Cluster jobs

Individual ablation jobs are under `scripts/experiments/ablation/`; other backbone jobs are under `scripts/slurm_jobs/`.

```bash
sbatch scripts/experiments/ablation/run_abl_cnn_4nb_l1.sh
sbatch scripts/experiments/ablation/run_abl_vit_4nb_l1.sh
sbatch scripts/experiments/ablation/run_vid_cnn_4nb_l1.sh
sbatch scripts/experiments/ablation/run_vid_vit_4nb_l1.sh
```

Adapt the working directory, environment activation, SLURM partition, and GPU requests to your cluster. Paths containing `/users/sl3753/scratch/GCN_project` refer to the original cluster checkout; a repository rename does not update that local directory. Cluster setup examples are in `scripts/setup/`. Local training uses the Python entrypoints directly and does not require SLURM.

These instructions were checked against the committed entrypoints. They are not evidence of a fresh training or checkpoint-based evaluation run.
