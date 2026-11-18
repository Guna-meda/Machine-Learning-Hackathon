# PathSnap - map matching with a classifier + HMM

Noisy GPS points are matched to a road network: a classifier scores candidate roads for each point, then an HMM/Viterbi step builds a connected route. (Phase 1 done: data, features, 6-model comparison. Phase 2: tuning, HMM, API, UI - in progress.)

## Team and ownership (Phase 1)

| Part | Owner | Files |
|---|---|---|
| A. Data understanding, EDA, repo setup | Gunavathi MI | `src/config.py`, `src/data/loader.py`, `src/viz/`, `scripts/01_eda.py`, `scripts/01b_eda_maps.py` |
| B. Label alignment and candidate generation | Gowtham L | `src/data/alignment.py`, `candidates.py`, `dataset.py`, `src/features/geometry.py`, `scripts/02_build_dataset.py` |
| C. Feature engineering and metrics | Gautam Krishna | `src/features/build.py`, `src/models/data.py`, `src/models/evaluate.py` |
| D. Model training and comparison | Guru Preetham | `src/models/zoo.py`, `scripts/03_train_compare.py`, `scripts/04_plots_phase1.py` |

## Setup

```bash
pip install -r requirements.txt
```
Put the data in `data/Student folder/{train,student_test}` (this folder is gitignored). To use another folder: `export PATHSNAP_DATA_DIR=/path/to/folder`.

## Run (Phase 1)

```bash
python3 scripts/01_eda.py
python3 scripts/01b_eda_maps.py
python3 scripts/02_build_dataset.py     # ~20 s, writes cache/
python3 scripts/03_train_compare.py     # ~12 min
python3 scripts/04_plots_phase1.py
```
Results go to `results/`, figures to `reports/figures/`, Folium maps to `reports/maps/`.
