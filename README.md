# PDE Score Priors for Inverse Problems

Репозиторий для генерации synthetic PDE snapshots и обучения unconditional score-based prior, который
дальше будет использоваться для zero-shot inverse problems:

```text
single PDE state reconstruction from sparse/noisy/low-resolution observations
```

Идея проекта: сначала обучить prior на состояниях PDE без conditioning, затем во время sampling
добавить data-consistency и physics-consistency. Для сравнения планируются четыре zero-shot метода:

- `DPS`: score/posterior guidance через measurement loss.
- `DDNM`: null-space correction через псевдообратный оператор `A^\dagger`.
- `DDRM`: spectral/SVD conditioning для линейных noisy inverse problems.
- `RePaint`: mask/inpainting sampler с repeated resampling jumps.

Это не полноценная trajectory data assimilation. Текущая задача проще:

```text
reconstruct one PDE state x from observation y = A(x) + noise
```

## Что уже сделано

В репозитории уже есть две основные части.

1. Генерация PDE datasets:

```text
grayscott_dataset/
kolmogorov_dataset/
scripts/generate_grayscott.py
scripts/generate_kolmogorov.py
notebooks/*/generate_*_colab.ipynb
```

Основной датасет для inverse-problem экспериментов сейчас:

```text
Kolmogorov velocity snapshots: images [N, 2, 64, 64]
```

Snapshots сохраняются как raw velocity field `(u_x, u_y)`. Нормализация не зашивается в `.npz`;
она считается по train split во время обучения и сохраняется вместе с checkpoint.

2. Score-based VP-SDE prior:

```text
score_training/
scripts/train_score_vp_coords.py
notebooks/train_score_vp_coords_a100_bf16_colab.ipynb
```

Это основной вариант для дальнейших inverse-problem экспериментов. Модель обучается как epsilon
predictor для continuous VP SDE:

```text
mu(t) = cos(arccos(1e-3) * t)
sigma(t) = sqrt(1 - mu(t)^2)
t in [0, 1]
```

К noisy velocity channels добавляются clean coordinate channels. По умолчанию используется
`coordinate_mode="fourier"`, то есть четыре периодических канала:

```text
sin(x), cos(x), sin(y), cos(y)
```

Они не шумятся и не входят в loss. Это важно для periodic Kolmogorov fields.

Во время sampling уже используется deterministic VP reverse update через prediction чистого поля:

```text
x_t -> pred_noise -> pred_x0 -> x_{t_next}
```

Именно этот `pred_x0` является общей точкой входа для DDNM, DDRM и RePaint. DPS будет добавляться
как gradient guidance к score/reverse update.

## Как использовать скачанную модель

После обучения Colab notebook скачивает `best_*.pt`. Положите файл, например, сюда:

```text
checkpoints/best_score.pt
```

Установите зависимости:

```bash
pip install -r requirements.txt
```

Минимальный пример загрузки checkpoint и unconditional sampling:

```python
import math

import torch

from score_training import DiffusersUNet, VPCosineSDE


def make_coord_grid(height: int, width: int, device: torch.device, mode: str) -> torch.Tensor:
    if mode == "fourier":
        x_angles = torch.arange(width, device=device, dtype=torch.float32) * (2.0 * math.pi / width)
        y_angles = torch.arange(height, device=device, dtype=torch.float32) * (2.0 * math.pi / height)
        yy, xx = torch.meshgrid(y_angles, x_angles, indexing="ij")
        return torch.stack((torch.sin(xx), torch.cos(xx), torch.sin(yy), torch.cos(yy)), dim=0).unsqueeze(0)
    if mode == "linear":
        y = torch.linspace(-1.0, 1.0, height, device=device)
        x = torch.linspace(-1.0, 1.0, width, device=device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack((xx, yy), dim=0).unsqueeze(0)
    raise ValueError(f"Unknown coordinate mode: {mode}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load("checkpoints/best_score.pt", map_location=device)

config = ckpt["config"]
stats = ckpt["data_stats"]

channels = int(stats["channels"])
height = int(stats["height"])
width = int(stats["width"])
coordinate_mode = ckpt.get("coordinate_mode", config.get("coordinate_mode", "fourier"))
coords = make_coord_grid(height, width, device, coordinate_mode)
coord_channels = int(coords.shape[1])

model = DiffusersUNet(
    in_channels=channels + coord_channels,
    out_channels=channels,
    channels_per_level=tuple(int(x) for x in config["channels_per_level"].split(",")),
    num_res_blocks=int(config["num_res_blocks"]),
    image_size=height,
    dropout=float(config.get("dropout", 0.0)),
    attention_head_dim=int(config["attention_head_dim"]),
    padding_mode=config.get("padding_mode", "circular"),
).to(device)

state = ckpt.get("ema_model", ckpt["model"])
model.load_state_dict(state)
model.eval()

sde = VPCosineSDE().to(device)

with torch.no_grad():
    samples_norm = sde.sample(
        model=model,
        shape=(8, channels, height, width),
        coords=coords,
        steps=int(config.get("sample_steps", 256)),
        device=device,
        time_embedding_scale=float(ckpt.get("time_embedding_scale", config.get("time_embedding_scale", 999.0))),
        clip_pred_x0=float(config.get("clip_pred_x0", 0.0)),
    )

mean = torch.tensor(stats["mean"], device=device).view(1, channels, 1, 1)
std = torch.tensor(stats["std"], device=device).view(1, channels, 1, 1)
samples_raw = samples_norm * std + mean

# samples_raw has shape [8, 2, 64, 64] for Kolmogorov velocity fields.
```

Для inverse problems важно работать последовательно в двух пространствах:

- модель и sampler работают в normalized space;
- метрики, визуализация и физические величины удобнее считать в raw space;
- observation `y` нужно нормализовать теми же `mean/std`, если conditioning применяется внутри sampler.

Пока отдельного inference CLI для inverse methods нет. Его нужно добавить следующим этапом.

## Дальнейший план действий

Цель следующего этапа: не переобучать модель под каждую inverse problem, а использовать один
скачанный unconditional score/VP checkpoint как prior и сравнить четыре zero-shot samplers.

### 1. Вынести общие inference primitives

Добавить модуль:

```text
inverse/
  checkpoint.py
  operators.py
  samplers.py
  metrics.py
  physics.py
```

Минимальные функции:

```python
load_score_checkpoint(path, device)
predict_x0_from_eps(x_t, pred_eps, mu_t, sigma_t)
vp_ddim_step(x_t, pred_eps, pred_x0, t, t_next)
normalize_observation(y_raw, mean, std)
denormalize_sample(x_norm, mean, std)
```

В текущем `score_training/sde.py` формула уже есть внутри `VPCosineSDE.sample()`:

```text
pred_x0 = (x_t - sigma_t * pred_noise) / mu_t
x_{t_next} = mu_{t_next} * pred_x0 + sigma_{t_next} * pred_noise
```

Ее нужно сделать переиспользуемой, чтобы inverse methods могли менять `pred_x0` или `pred_noise`
перед reverse step.

### 2. Описать forward operators

Начать с операторов, которые хорошо подходят для Kolmogorov velocity fields:

```text
SparseSensorMask: known values on a subset of grid points
RandomMask: inpainting-style missing field regions
Downsample: low-resolution observation, e.g. 64x64 -> 16x16
GaussianBlur: blurred observation
```

Единый интерфейс:

```python
class LinearOperator:
    def forward(self, x): ...
    def pinv(self, y): ...
```

Для DDNM нужен `pinv`. Для DPS достаточно `forward`, если он differentiable. Для DDRM нужен либо
SVD/diagonalization, либо ограниченный набор операторов, где spectral representation известна.

### 3. Реализовать DPS

DPS добавляется как score/posterior guidance:

```text
L_y = ||A(pred_x0(x_t)) - y||_2^2
pred_noise or score update <- guided by grad_{x_t} L_y
```

Практически это отдельный sampler:

```text
inverse/samplers.py::sample_dps(...)
```

Он должен поддерживать:

```text
guidance_scale
guidance schedule over t
measurement noise level
optional physics loss
```

Для проекта это главный general-purpose baseline, потому что он работает и с нелинейным `A`, если
можно сделать backprop.

### 4. Реализовать DDNM

DDNM добавляется как correction of `pred_x0`:

```text
pred_x0_ddnm = A^\dagger y + (I - A^\dagger A) pred_x0
```

Для mask это просто:

```text
pred_x0_ddnm = mask * y + (1 - mask) * pred_x0
```

Файл:

```text
inverse/samplers.py::sample_ddnm(...)
```

Это самый простой и важный baseline для sparse sensors и inpainting. Для noisy observations стоит
сразу добавить soft version:

```text
pred_x0_soft = pred_x0 + lambda_t * A^\dagger (y - A pred_x0)
```

В отчете можно считать soft DDNM ablation, а основным методом оставить DDNM.

### 5. Реализовать RePaint

RePaint нужен для mask/inpainting постановки:

```text
x_t_known = mu_t * y + sigma_t * noise
x_t = mask * x_t_known + (1 - mask) * x_t_generated
```

Отдельно нужен jump schedule:

```text
t -> t-1 -> t -> t-1
```

Файл:

```text
inverse/samplers.py::sample_repaint(...)
```

Сравнивать RePaint лучше не на blur/downsample, а именно на sparse/masked observations. Это будет
честное сравнение против DDNM и DPS на той же inpainting задаче.

### 6. Реализовать DDRM

DDRM стоит делать после DPS/DDNM/RePaint, потому что он требует больше operator-specific кода.
Начать лучше с одного линейного оператора:

```text
Downsample 64x64 -> 16x16
```

или с blur, если будет удобная Fourier diagonalization.

Идея:

```text
A = U Sigma V^T
y_bar = U^T y
pred_x0_bar = V^T pred_x0
```

Дальше по компонентам смешивать measurement и prior prediction с учетом singular values,
measurement noise и текущего diffusion noise.

Файл:

```text
inverse/samplers.py::sample_ddrm(...)
inverse/operators.py::SVDOperator
```

Если времени мало, DDRM можно ограничить одной задачей low-resolution reconstruction и явно
написать в отчете, что метод operator-specific.

### 7. Добавить physics-informed sampling

Для Kolmogorov velocity fields естественный constraint:

```text
div(u) = d u_x / dx + d u_y / dy = 0
```

Два варианта:

1. Hard projection для periodic boundary:

```text
u <- P_divfree(u)
```

через FFT Helmholtz projection.

2. Soft guidance:

```text
L_phys = ||div(u)||_2^2
```

и добавить его к DPS loss:

```text
L = lambda_y L_y + lambda_div L_phys
```

Рекомендуемый минимум:

```text
DPS + div penalty
DDNM/RePaint + FFT div-free projection after pred_x0 correction
```

Так получится отдельная physics-informed ablation:

```text
method
method + div-free projection/guidance
```

### 8. Экспериментальный протокол

Для каждого test field `x_true` генерировать observations:

```text
y = A(x_true) + noise
```

Сравнивать:

```text
Unconditional prior sample
DPS
DDNM
DDRM
RePaint
```

Метрики:

```text
relative L2 error in raw velocity space
measurement consistency ||A(x_hat) - y||
divergence error ||div(u_hat)||
vorticity MAE / RMSE
visual comparison of vorticity fields
```

Минимальный набор задач для отчета:

```text
Sparse sensors / mask reconstruction: DPS, DDNM, RePaint
Low-resolution reconstruction: DPS, DDNM or DDRM
Noisy observations: DPS, soft DDNM, DDRM if implemented
Physics-informed ablation: with/without div(u)=0 constraint
```

### 9. Что не нужно делать на первом этапе

Не нужно переобучать отдельную модель для каждого inverse problem. Текущая score-based VP модель
достаточна как unconditional prior.

Не нужно сразу реализовывать универсальный DDRM для любых `A`. Достаточно одного корректного
linear operator case.

Не нужно менять формат датасета. Важно только аккуратно применять `mean/std` из checkpoint при
переходе между raw velocity space и normalized model space.

## Доступные датасеты

- `grayscott_dataset/`: быстрый reaction-diffusion датасет с attractor-like patterns.
- `kolmogorov_dataset/`: более близкий к статье fluid-like датасет, forced 2D incompressible
  Navier-Stokes in vorticity form with Kolmogorov-style forcing. Текущий Colab preset хранит
  raw velocity snapshots `(u_x, u_y)` в `float32`, без scaling/clipping, а preview строит по raw vorticity через обычный `imshow`.
  Симуляция идет на `256x256`, затем average-pooling downscale до `64x64`. Для более контрастных
  vortex-like states используется stronger forcing / lower damping.

Готовые Colab variants:

```text
notebooks/grayscott_128_10k/generate_grayscott_128_10k_colab.ipynb
notebooks/kolmogorov_64_10k/generate_kolmogorov_64_10k_colab.ipynb
notebooks/kolmogorov_256_to_64_100k/generate_kolmogorov_256_to_64_100k_colab.ipynb
```

## Обучение score-based prior

Код обучения находится в:

```text
score_training/
scripts/train_score_vp_coords.py
notebooks/train_score_vp_coords_a100_bf16_colab.ipynb
```

Пайплайн обучает unconditional VP-SDE score model на сохраненных PDE snapshots. Перед
обучением датасет целиком загружается в RAM, по train split считаются channel-wise `mean` и `std`,
после чего train/val нормализуются по этим статистикам. Каждую эпоху считается validation loss,
сохраняются unconditional samples для наблюдения за прогрессом и checkpoint с `model`, `ema_model`,
optimizer state, config, dataset stats и history. Если validation loss улучшился, сохраняется
`best_*.pt`; в Colab notebook этот файл дополнительно скачивается через `files.download`.

Используется `diffusers.UNet2DModel` с channels `(96, 192, 384)`, `3` residual blocks per level,
attention на нижних разрешениях, `attention_head_dim=32`, SiLU, batch size `128`. К входу модели
добавляются clean coordinate channels; текущий default `coordinate_mode="fourier"`
использует четыре периодических канала `sin(x), cos(x), sin(y), cos(y)`, а legacy mode `linear`
использует два канала `(x, y)` в диапазоне `[-1, 1]`. Координатные каналы никогда не шумятся и не
входят в loss.

`--data-source` может быть:

- публичной ссылкой Yandex Disk на `.zip`, `.npz` или папку;
- локальным `.zip`/`.npz`;
- локальной папкой с `.npz` chunks или split shards.

Для публичной папки Yandex Disk загрузчик скачивает `.npz` и `.json` файлы по отдельности в
`cache_dir` и пропускает уже скачанные файлы с совпадающим размером. Эта ссылка работает напрямую:

```text
https://disk.yandex.ru/d/rrjDGzzX5cfFnA
```

В этой папке сейчас есть `train_000..015.npz`, `val_000..001.npz` и `test_000.npz`. Для обучения
используются только train/val, поэтому отсутствующий `test_001.npz` не блокирует training pipeline.

Статистики нормализации сохраняются в JSON. Если указать `--stats-cache-path`, то при следующем
запуске `mean/std` будут загружены из этого файла, а не пересчитаны:

Пример локального запуска:

```bash
python3 scripts/train_score_vp_coords.py \
  --data-source https://disk.yandex.ru/d/rrjDGzzX5cfFnA \
  --stats-cache-path data/download_cache/kolmogorov_velocity_256_to_64_train_stats.json \
  --dataset-tag kolmogorov_velocity_256_to_64_coords \
  --epochs 1024 \
  --batches-per-epoch 0 \
  --batch-size 128 \
  --precision bf16 \
  --time-embedding-scale 999
```

Для Colab/A100 используйте:

```text
notebooks/train_score_vp_coords_a100_bf16_colab.ipynb
```

В notebook нужно указать `DATA_SOURCE` как публичную ссылку Yandex Disk или путь к архиву/папке.
Preset оптимизирован под `64x64` Kolmogorov velocity snapshots `[N, 2, 64, 64]`, A100 40GB и bf16.
Если памяти не хватает, уменьшите `batch_size` до `64`.

В Colab notebook процесс разделен на отдельные ячейки:

- `Configure training`: создает `TrainConfig`, ничего не скачивает и не обучает.
- `Download and load dataset`: скачивает или проверяет cache, грузит train/val в RAM, считает или
  читает `mean/std`, нормализует данные.
- `Start training`: запускает обучение модели на уже подготовленном `dataset`.

Физические velocity channels шумятся по VP SDE:

```text
mu(t) = cos(arccos(1e-3) * t)
sigma(t) = sqrt(1 - mu(t)^2)
```

Модель предсказывает epsilon только для физических каналов. Непрерывное VP-SDE время `t in [0, 1]`
передается в diffusers как `t * 999`. Эпоха означает полный проход по train loader;
`batches_per_epoch=0` и `val_batches=0`.

## Что генерируется

Gray-Scott system:

```text
u_t = D_u Laplacian(u) - u v^2 + F(1 - u)
v_t = D_v Laplacian(v) + u v^2 - (F + k) v
```

Быстрый preset по умолчанию:

- grid: `64x64`
- channels: один канал `v`
- trajectories: `500`
- max trajectories if quality filtering rejects samples: `2000`
- snapshots per trajectory: `20`
- total images: `10_000`
- burn-in: `2500` explicit Euler steps
- save interval: `30` steps
- solver substeps: `1`
- chunk size: `1000` images
- preview: PNG-сетка samples после каждого chunk
- sequence preview: подряд идущие snapshots одной trajectory
- quality filter: отбрасывает почти однородные snapshots
- boundary condition: periodic
- saved values: raw concentrations in `[0, 1]`

Для diffusion training обычно использовать:

```python
x = 2.0 * images - 1.0
```

## Быстрый запуск

```bash
pip install -r requirements.txt
python scripts/generate_grayscott.py \
  --output-dir data/grayscott_64 \
  --total-images 10000 \
  --num-trajectories 500 \
  --max-trajectories 2000 \
  --snapshots-per-trajectory 20 \
  --grid-size 64 \
  --burn-in-steps 2500 \
  --save-interval 30 \
  --chunk-size 1000 \
  --sim-batch-size 500 \
  --solver-substeps 1 \
  --save-previews \
  --preview-every-chunks 1 \
  --save-sequence-previews \
  --sequence-preview-count 16 \
  --min-image-std 0.025 \
  --min-image-range 0.15 \
  --num-threads 12 \
  --device auto
```

Файлы будут сохранены как:

```text
data/grayscott_64/grayscott_chunk_000.npz
data/grayscott_64/grayscott_chunk_001.npz
...
data/grayscott_64/preview_chunk_000.png
data/grayscott_64/preview_chunk_001.png
...
data/grayscott_64/sequence_preview_batch_000.png
data/grayscott_64/sequence_preview_chunk_000.png
...
data/grayscott_64/manifest.json
```

Каждый chunk содержит:

- `images`: shape `[N, C, H, W]`
- `trajectory_id`
- `snapshot_index`
- `step`
- `F`
- `k`
- `regime_id`
- `split`: `train`, `val`, `test`

Split делается по `trajectory_id`, а не по отдельным кадрам. Это важно, чтобы соседние snapshots
одной траектории не попадали одновременно в train и test.

## Higher-quality preset

Для лучшего разрешения и более аккуратного explicit Euler решения используйте `128x128` и
`solver_substeps=2`. Это не меняет сохраненные timestep labels, но каждый solver step считается
двумя внутренними шагами размера `dt / 2`.

```bash
python scripts/generate_grayscott.py \
  --output-dir data/grayscott_128 \
  --total-images 10000 \
  --num-trajectories 500 \
  --max-trajectories 2000 \
  --snapshots-per-trajectory 20 \
  --grid-size 128 \
  --burn-in-steps 3000 \
  --save-interval 25 \
  --chunk-size 1000 \
  --sim-batch-size 250 \
  --solver-substeps 2 \
  --save-previews \
  --save-sequence-previews \
  --sequence-preview-count 16 \
  --min-image-std 0.025 \
  --min-image-range 0.15 \
  --device auto
```

Colab notebook сейчас настроен именно на этот higher-quality preset.

## Colab

Откройте notebook:

```text
notebooks/grayscott_128_10k/generate_grayscott_128_10k_colab.ipynb
```

Notebook уже указывает на этот repo:

```python
REPO_URL = "https://github.com/SadreevAmir/DL_final_project.git"
```

Если repository private, Colab не сможет клонировать его обычной HTTPS-командой и выдаст ошибку
`could not read Username for 'https://github.com'`. В этом случае:

1. Создайте GitHub personal access token с доступом к repo.
2. В Colab откройте Secrets.
3. Добавьте secret с именем `GITHUB_TOKEN`.
4. Перезапустите clone cell.

Если сделать repository public, token не нужен. Notebook клонирует репозиторий, запускает генерацию
и скачивает каждый `.npz` chunk после сохранения.

## Оценка времени

Грубая оценка для `10_000` хороших картинок:

- Colab A100, `sim_batch_size=500`: обычно `5-20` минут.
- Colab T4/L4: обычно `15-45` минут.
- CPU 12 cores с `--num-threads 12`: может быть `1-3` часа, потому что код векторизован, но PyTorch CPU здесь заметно медленнее GPU.

Для higher-quality `128x128`, `solver_substeps=2`, `sim_batch_size=250`:

- Colab A100: примерно `25-70` минут.
- T4/L4: примерно `1-3` часа.
- CPU не рекомендуется.

Если preview показывает пустые/однородные квадраты, это trajectories, которые пришли к почти
homogeneous fixed point. По умолчанию они отбрасываются фильтром:

```text
min_image_std = 0.025
min_image_range = 0.15
```

Если фильтр слишком строгий и генерация не добирает `total_images`, увеличьте `--max-trajectories`
или ослабьте thresholds.

## Качество и корректность

Получающиеся структурные паттерны являются корректными численными состояниями Gray-Scott model
для выбранной дискретизации, параметров `F, k` и periodic boundary conditions. Почти однородные
поля тоже являются физически допустимым fixed-point outcome, но для diffusion-prior датасета они
обычно вредны, поэтому фильтруются.

Этот генератор не является high-precision PDE benchmark. Это controlled synthetic dataset для ML:
finite-difference Laplacian, explicit Euler, bounded concentrations. Для строгой numerical analysis
нужно делать convergence study по grid size и timestep. Для проекта по reconstruction/diffusion
этого уровня достаточно, особенно с `128x128` и `solver_substeps=2`.

Если нужно быстрее для первого smoke test:

```bash
python scripts/generate_grayscott.py \
  --output-dir data/test \
  --total-images 1000 \
  --num-trajectories 50 \
  --snapshots-per-trajectory 20 \
  --burn-in-steps 1000 \
  --sim-batch-size 50
```

## Насколько это "аттрактор"

Практически это snapshots из long-time regime: начальные случайные blobs не сохраняются, система
сначала прогоняется `burn_in_steps`, и только затем кадры попадают в датасет.

Строго математически мы не доказываем, что samples идеально распределены по инвариантной мере
аттрактора. Для учебного проекта это корректная и честная формулировка:

```text
we sample Gray-Scott states after burn-in as an approximation of the system's long-time attractor distribution
```

Если хотите более однородный датасет, используйте `--param-mode fixed`. Если хотите больше визуального
разнообразия, оставьте default `--param-mode mixed`.
