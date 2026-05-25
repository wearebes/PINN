# PINN MLP-Field Research

Physics-Informed Neural Networks (PINN) research project learning field predictions (h_kappa) from PDE data using MLP and CNN models.

## Setup

```bash
conda create -n pinn python=3.10 && conda activate pinn
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements-min.txt
```

## Quick Start

```bash
# Generate training data
python -m train_generate

# Train model
python -m model [--lr 5e-4] [--hidden_units 256] [--max_epochs 500]

# Generate test data & evaluate
python -m testdata_generate
python evaluate/flower.py
python evaluate/flower_diagnostics.py
```

## Structure

- `model/` — MLP/CNN architectures, training loop, configs
- `train_generate/` — Training data generation
- `testdata_generate/` — Test data generation
- `evaluate/` — Flower analysis, curvature plotting, diagnostics
- `dataset/256.h5` — Main dataset (HDF5)
- `out/best_stencil_hkappa.pt` — Best trained model

## Key Commands

| Command | Purpose |
|---------|---------|
| `python -m model` | Train MLP model |
| `python evaluate/flower.py` | Main evaluation |
| `python evaluate/training_curvature.py` | Track curvature during training |
| `python smoke_test_model.py` | Quick sanity check |

## Notes

- Configs are dataclasses in `model/config.py` — use `create_train_config()` to build
- Data: HDF5 format, see `train_generate/io.py` for I/O patterns
- Optional SwanLab logging (uncomment in code if needed)
- GPU recommended; CPU is very slow
- Use `patience=30` for early stopping (default)

## Environment

- Windows + WSL or native Linux
- Conda for package management
- Use bash syntax in suggestions
