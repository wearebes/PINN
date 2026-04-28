# Run Guide

## 0. One-Command Pipeline

You can run the full workflow with:

```bash
bash job.sh
```

This script runs, in order:

- training dataset generation
- test dataset generation
- model training
- curvature evaluation

Common overrides:

```bash
DEVICE=cuda USE_SWANLAB=1 TRAIN_EXPERIMENT_NAME=my-train EVAL_EXPERIMENT_NAME=my-eval bash job.sh
```

## 1. Data Generation

Use the following command to generate the training dataset:

```bash
python -m traingenerate.generate
```

This command uses the current default configuration in the codebase. The defaults are now split into:

- `DataConfig` in `traingenerate/config.py`: what data is generated
- `GenerationConfig` in `traingenerate/config.py`: generation workers, generation batch size, output directory, and dataset file name
- `ReinitConfig` in `traingenerate/config.py`: reinitialization numerics
- `TrainConfig` in `model/config.py`: training-only settings such as training batch size

If you want config-first experiments, edit `traingenerate/config.py` and rerun generation.

The generated HDF5 path is controlled by:

- `GenerationConfig.output_dir`
- `GenerationConfig.dataset_name`

If needed, you can still override them from the CLI:

```bash
python -m traingenerate.generate \
  --output-dir dataset \
  --dataset-name train_custom.h5
```

You can also optionally store extra per-sample fields in the trajectory splits:

```bash
python -m traingenerate.generate \
  --stored-sample-fields phi,phi_x,phi_y
```

Each generated dataset now writes a sidecar JSON summary next to the HDF5 file. You can inspect it with:

```bash
python -m traingenerate.inspect \
  --dataset dataset/train_custom.h5
```

## 2. Training With SwanLab

The training entrypoint supports optional SwanLab logging. When enabled, it records:

- dataset path and dataset metadata
- train/data/reinit configuration
- split blueprint counts and sample counts
- model parameter count and device
- per-epoch train/validation losses
- effective IC/Snapshot/PDE weights
- best epoch and final validation summary

Example:

```bash
python -m model.train \
  --dataset-output dataset/train_0.5_setting1.h5 \
  --batch-size 81920 \
  --use-swanlab \
  --swanlab-project PINN \
  --swanlab-experiment-name version1_Curvature \
  --swanlab-tags Curvature,stage2
```

Training batch size is controlled only by `TrainConfig.batch_size` in `model/config.py` or by `model.train --batch-size`. It is no longer inherited from the dataset generation config or stored HDF5 generation batch size.

## 3. Curvature Evaluation With SwanLab

```bash
python -m evaluate --geometry flower --use-swanlab
```

For the independent flower test set:

```bash
python -m evaluate \
  --geometry flower \
  --test-data test_data/test_FP0_DynSign_CFL0.5_EPS2.5_RK3_WENO5_CIN_ST9_APCN.h5 \
  --model out/best_reinit_pinn.pt \
  --device cuda \
  --batch-size 4096 \
  --use-swanlab \
  --swanlab-project PINN \
  --swanlab-experiment-name curvature-eval \
  --swanlab-tags curvature,eval,hkappa \
  --swanlab-logdir swanlog
```

For the circle test split stored inside the training dataset:

```bash
python -m evaluate \
  --geometry circle \
  --circle-dataset dataset/train_0.5_setting1.h5 \
  --model out/curvature-v1_layer4.pt \
  --device cuda \
  --batch-size 4096 \
  --use-swanlab
```

For fixed-reference circle evaluation, there are two recommended sampling modes:

- `true_interface`: use the zero-level-set nodes of the true circle interface as a fixed reference. The sampled nodes do not move across iterations; only the numerical/model `phi`, derivatives, and `h*kappa` values change with `s=n*CFL`.
- `reference_band`: use a fixed narrow band around the true interface. The recommended setting is `|\phi_true| <= 3h`, which corresponds to `--circle-band-width-cells 3.0`.

Recommended SwanLab experiment names:

- `curvature-eval-trueinterface-curvature`
- `curvature-eval-trueinterface-noncurvature`
- `curvature-eval-referenceband-3h-curvature`
- `curvature-eval-referenceband-3h-noncurvature`

If you run inside WSL with your `conda` environment `jq`, use the same CLI after activating that environment:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate jq

python -m evaluate \
  --geometry flower \
  --test-data test_data/test_FP0_DynSign_CFL0.5_EPS2.5_RK3_WENO5_CIN_ST9_APCN.h5 \
  --model out/best_reinit_pinn.pt \
  --device cuda \
  --batch-size 4096 \
  --use-swanlab \
  --swanlab-project PINN \
  --swanlab-experiment-name curvature-eval \
  --swanlab-tags curvature,eval,hkappa \
  --swanlab-logdir swanlog
```

For the fixed `true_interface` evaluation:

```bash
python -m evaluate \
  --geometry circle \
  --circle-dataset dataset/train_0.5_setting1.h5 \
  --circle-sampling-mode true_interface \
  --model out/curvature-v1_layer4.pt \
  --device cpu \
  --batch-size 8192 \
  --num-threads 14 \
  --use-swanlab \
  --swanlab-project PINN \
  --swanlab-experiment-name curvature-eval-trueinterface-curvature \
  --swanlab-tags curvature,eval,hkappa,circle,true_interface \
  --swanlab-logdir swanlog
```

For the fixed `reference_band` evaluation with `3h`:

```bash
python -m evaluate \
  --geometry circle \
  --circle-dataset dataset/train_0.5_setting1.h5 \
  --circle-sampling-mode reference_band \
  --circle-band-width-cells 3.0 \
  --model out/noncurvature-v1_layer4.pt \
  --device cpu \
  --batch-size 8192 \
  --num-threads 14 \
  --use-swanlab \
  --swanlab-project PINN \
  --swanlab-experiment-name curvature-eval-referenceband-3h-noncurvature \
  --swanlab-tags curvature,eval,hkappa,circle,reference_band,3h \
  --swanlab-logdir swanlog
```
