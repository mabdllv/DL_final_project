# Kolmogorov Flow 64x64 10k Colab Variant

Отдельный вариант для генерации fluid-like датасета ближе к статье:

```text
forced 2D incompressible Navier-Stokes in vorticity form
periodic domain
Kolmogorov-style forcing
256x256 simulation grid
average-pooling downscale to 64x64
10_000 raw velocity-field snapshots
64x64 resolution
FP32 storage
```

Notebook:

```text
generate_kolmogorov_64_10k_colab.ipynb
```

Это не полная реплика NeurIPS SDA paper. В статье использовался Kolmogorov flow / Navier-Stokes
velocity field. Здесь динамика тоже интегрируется в vorticity form, но в `.npz` сохраняется
восстановленное velocity field `(u_x, u_y)`, а previews строятся по vorticity. Симуляция идет на
`256x256`, после чего velocity и preview-vorticity огрубляются до `64x64` через average pooling.

Формат:

```python
images  # [N, 2, 64, 64], float32, raw velocity (u_x, u_y)
```

Приблизительно:

```python
velocity = images
```

Current preset is tuned for more paper-like, higher-contrast states than the first quick version:

```text
dt = 0.005
simulation grid = 256x256
saved grid = 64x64
coarsening = average pooling
viscosity = 3e-4
drag = 0.025
forcing_amp = 0.55
burn_in_steps = 5000
dtype = float32
output_field = velocity
normalize_output = false
```

Preview images use plain `imshow(..., cmap="RdBu_r")` and show derived raw vorticity, matching
the visualization convention used in the paper. The saved arrays are raw velocity values; normalize
them later using train-set statistics during model training.
