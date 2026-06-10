# 3-Phase Fraction Prediction

Predict BCC, FCC, and cementite (Fe3C) phase fractions from HEXRD 2-theta profiles using a pre-trained model.

## Quick start

1) Install dependencies:
```sh
pip install -r requirements.txt
```

2) Get the model file:

Option A: Git LFS (recommended)
```sh
git lfs install
git lfs pull
```

Option B: Direct download (no Git LFS)
```
https://github.com/benrabah2/3ph_cnn_predictor/blob/main/model_3ph_v1.0.0.keras
```
Place the file in the repo folder.

3) Set your model and defaults in `config.py`:
- Update `model_name` to point to your `.keras` file.
- Optional defaults: `Convert_to_q`, `shape_correction`, `nb_of_points_per_profile`, `Q_MIN`, `Q_MAX`, `batch_size`, `plot_data`.

4) Run:
```sh
python predict.py /path/to/your/data_folder -w WAVELENGTH -s START -e END
```
If you omit `-s/--start` and `-e/--end`, the script uses the min and max index found in the folder:
```sh
python predict.py /path/to/your/data_folder -w WAVELENGTH
```
If your data is already in q (not 2-theta), disable conversion and you do not need `-w`:
```sh
python predict.py /path/to/your/data_folder -s START -e END --no-convert-to-q
```
The script writes results to `data_folder/results/predicted_phase_fractions_<DATASET_NAME>.txt`.

Optional overrides:
- `--q-min`, `--q-max`
- `--no-convert-to-q`, `--no-shape-correction`
- `--nb-points`, `--batch-size`
- `--plot`
- `-n/--name` (to filter by dataset prefix)
- `-s/--start` and `-e/--end` (optional; auto-detected if omitted)
- `--out-dir`, `--out-name`, `--out-path`

Recommendation:
- Prefer using `-n`, `-s`, and `-e` to avoid mixing datasets and to control the range explicitly.

## Data format

Each data file must be a text file with at least two columns:
- column 1: 2-theta (or q if using `--no-convert-to-q`)
- column 2: intensity

File naming (two modes):
- If you provide `-n/--name`, filenames must start with that name.
- After the name, any non-digit separator(s) are allowed (`_`, `-`, `.`, etc.).
- Filenames must end with a numeric index and `.dat`.
- If you do **not** provide `-n/--name`, all `.dat` files in the folder that end with a numeric index are used.

## Output

Tab-separated `.txt` with four columns:

```
File Number    BCC (%)    FCC (%)    Cementite (%)
```

- Default output folder: `<data_folder>/results`
- Default file name: `predicted_phase_fractions_<DATASET_NAME>.txt`
- Use `--out-dir`, `--out-name`, or `--out-path` to customize.

## Model file

This repository uses Git LFS for the model file (`model_3ph_v1.0.0.keras`).
If you clone without Git LFS, you will only get a small pointer file, not the real model.
To ensure the model downloads:
```sh
git lfs install
git lfs pull
```

Model versioning:
- Semantic versioning is used, e.g., `model_3ph_v1.0.0.keras`.

## What the script does

1. Loads each profile in the configured index range.
2. Optionally converts 2-theta to q and resamples to 1500 points.
3. Applies log-intensity normalization: `log1p(I) / log1p(1000)`, q normalized to `[0, 1]`.
4. Extracts 5 crystallographic cementite windows (345 pts total) as a second model input.
5. Runs the dual-input model: outputs `cem_present`, `cem_fraction_norm`, `p_bcc`.
6. Reconstructs physical fractions:
   - `cem = cem_present × cem_fraction_norm × 0.1`
   - `f_bcc = (1 − cem) × p_bcc`
   - `f_fcc = (1 − cem) × (1 − p_bcc)`
7. Saves results and optionally plots.

## Model architecture summary

The model is a hierarchical transfer model with two inputs:
- Full spectrum `(N, 1500, 2)` → 3-conv backbone (transferred from 2-phase model) → BCC/FCC head
- CEM windows `(N, 345, 2)` → dedicated 2-conv branch → cementite presence + fraction heads

Three outputs are produced and combined via the reconstruction cascade above.
Cementite fraction is capped at 10% (`CEM_MAX = 0.1`), consistent with the training label definition.

## Citation

If you use this code, please cite:

Benrabah, Imed‐Eddine, Guillaume Geandier, Olha Nakonechna, Benoît Denand, Hugo Van Landeghem, Alexis Deschamps, and Sébastien YP Allain. "Deep Learning for Real‐Time Phase Quantification from X‐ray Diffraction: Toward High‐Throughput Steel Microstructure Mapping." *Advanced Engineering Materials* (2026): e202503172. https://doi.org/10.1002/adem.202503172

Please also reference the model GitHub repository: https://github.com/benrabah2/3ph_cnn_predictor

## Project files

- `predict.py`: command-line entry point.
- `data_utils.py`: preprocessing functions (2θ→q, resampling, log normalization, CEM window extraction).
- `model_utils.py`: model loading, batched prediction, and fraction reconstruction.
- `config.py`: model path, preprocessing defaults, and CEM window definitions.
