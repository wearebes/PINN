from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from model.config import default_dataset_path
from testdata_generate.config import DEFAULT_DATASET_NAME, DEFAULT_OUTPUT_DIR


@dataclass(frozen=True)
class EvalConfig:
    eval_geometry: str = "flower"
    test_data_path: Path = DEFAULT_OUTPUT_DIR / DEFAULT_DATASET_NAME
    circle_dataset_path: Path = default_dataset_path()
    circle_sampling_mode: str = "moving_interface"
    circle_band_width_cells: float = 1.0
    model_path: Path = Path("out") / "pinn-1.pt"
    hidden_units: int = 128
    activation: str = "tanh"
    batch_size: int = 4096
    device: str = "cuda"
    num_threads: int = 14
    max_samples: int | None = None
    use_swanlab: bool = False
    swanlab_project: str = "PINN"
    swanlab_experiment_name: str = "curvature-eval"
    swanlab_description: str = "Autograd curvature evaluation on flower or circle test data."
    swanlab_tags: tuple[str, ...] = ("curvature", "eval", "hkappa")
    swanlab_group: str = ""
    swanlab_workspace: str = ""
    swanlab_logdir: str = "swanlog"
    swanlab_mode: str = "cloud"
