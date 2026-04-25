# Run Guide

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
  --batch-size 8192 \
  --use-swanlab \
  --swanlab-project PINN \
  --swanlab-experiment-name nonsdf-stage2predict \
  --swanlab-tags nonsdf,stage2
```

Training batch size is controlled only by `TrainConfig.batch_size` in `model/config.py` or by `model.train --batch-size`. It is no longer inherited from the dataset generation config or stored HDF5 generation batch size.

## 3. Curvature Evaluation With SwanLab

```bash
python -m evaluate --use-swanlab
```

```bash
python -m evaluate \
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
