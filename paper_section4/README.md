# Section 4 reproducible source-data package

This package recomputes the four Section 4 figure datasets from the current
local `baseline_{32,64,128,256,512}_hgradient` checkpoints.  It does not claim
that those checkpoints reproduce the archived manuscript figures numerically.

Run from the PINN repository with the project environment active:

```bash
python -m paper_section4.recompute --figure all --device cpu --resume
python -m paper_section4.validate_package
python -m paper_section4.publish --cfd-root /path/to/cfd
```

`--resume` reuses completed Figure 1 test-resolution shards, Figure 2 `R/h`
shards, and a contract-compatible flower HDF5.  Published CSV files and their
provenance records are byte-for-byte copies of `paper_section4/results/`.

The CFD checkout is intentionally the rendering boundary.  After publishing:

```bash
python figures/static/plot_all.py
```

The plotting scripts require only the adjacent source-data files and
`figures/cfd_style.py`; they do not import this package or PyTorch.
