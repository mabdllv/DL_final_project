# Kolmogorov Velocity 256->64 100k Dataset

Final larger dataset preset:

```text
simulate forced 2D incompressible Navier-Stokes at 256x256
recover velocity channels (u_x, u_y)
average-pool velocity to 64x64
save individual snapshots, not trajectory tensors
100_000 samples total
80_000 train / 10_000 val / 10_000 test
float32 storage
```

Notebook:

```text
generate_kolmogorov_256_to_64_100k_colab.ipynb
```

The simulation is integrated in vorticity form for numerical convenience, but the saved ML dataset
matches the paper's state representation more closely:

```python
images  # [N, 2, 64, 64], float32
images[:, 0]  # raw u_x
images[:, 1]  # raw u_y
```

Preview images show vorticity derived during simulation. Split shards are written as:

```text
data/kolmogorov_velocity_256_to_64_100k_splits/train/train_000.npz
...
data/kolmogorov_velocity_256_to_64_100k_splits/val/val_000.npz
data/kolmogorov_velocity_256_to_64_100k_splits/test/test_000.npz
```

Each split shard contains:

```python
images
trajectory_id
snapshot_index
step
viscosity
drag
forcing_amp
split
```
