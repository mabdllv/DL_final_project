# Gray-Scott 128x128 10k Colab Variant

Отдельный Colab-вариант для генерации датасета:

```text
10_000 Gray-Scott snapshots
128x128 resolution
1 channel: v concentration
quality filtering enabled
sample previews + consecutive trajectory previews
```

Notebook:

```text
generate_grayscott_128_10k_colab.ipynb
```

Ожидаемое время на Colab A100: обычно `10-40` минут. Если quality filter отбрасывает много
однородных состояний, время может быть ближе к верхней границе.

Формат данных:

```text
grayscott_chunk_000.npz
grayscott_chunk_001.npz
...
manifest.json
preview_chunk_*.png
sequence_preview_*.png
```

Основной массив внутри `.npz`:

```python
images  # [N, 1, 128, 128], float16, values in [0, 1]
```

Для diffusion training:

```python
x = 2.0 * images - 1.0
```
