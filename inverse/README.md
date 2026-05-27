# Inverse Experiment Pipeline

This folder is the shared interface for all inverse methods.  The main rule:
methods do not choose validation samples or corruptions themselves.  Everyone
uses the same saved case file.

## 1. Create shared cases once

```bash
python3 scripts/make_inverse_cases.py \
  --data-source notebooks/data/kolmogorov_cache \
  --operator sparse_grid \
  --output-path inverse_cases/sparse_grid_val16_seed0.npz \
  --num-samples 16 \
  --sample-seed 0 \
  --visualization-count 16 \
  --visualization-seed 0 \
  --corruption-seed 0 \
  --noise-sigma 0.0
```

Operators:

```text
sparse_grid
downsample
blur
```

The `.npz` stores `x_true_raw`, `y_raw`, `sample_ids`, and JSON metadata.
Commit or share this file path inside the team.  Running different methods
against the same file guarantees identical validation images and corruptions.

`num_samples` controls how many examples are used for metrics.  The first
`visualization_count` examples are selected by `visualization_seed` and placed at
the front of the file, so the visualized examples stay identical when different
people run different metric sample counts.

`case_batch_size` controls how many samples are corrupted at once while building
the case file.

## 2. Implement a method

Each method owns one file:

```text
inverse/methods/dps.py
inverse/methods/ddnm.py
inverse/methods/repaint.py
inverse/methods/ddrm.py
```

Every file must expose:

```python
def sample(checkpoint, operator, y_norm, params) -> torch.Tensor:
    ...
```

Return normalized reconstruction `[B, C, H, W]`.  Do not load data, create
corruptions, compute metrics, or save plots inside method code.

## 3. Run a method

```bash
python3 scripts/run_inverse_experiment.py \
  --checkpoint-path checkpoints/best_score.pt \
  --case-file inverse_cases/sparse_grid_val16_seed0.npz \
  --method dps \
  --output-dir runs_inverse/dps_sparse_grid_val16 \
  --batch-size 8 \
  --visualization-sample-count 16 \
  --steps 256 \
  --seed 0
```

Outputs:

```text
runs_inverse/.../run_config.json
runs_inverse/.../metrics.csv
runs_inverse/.../figures/*.png
runs_inverse/.../reconstructions/x_hat_raw.pt
```

CSV columns are fixed for all methods:

```text
sample_id, seed, method, operator, noise_sigma,
rel_l2, rmse, measurement_error, divergence, vorticity_rmse
```
