# PathSnap - map matching with a classifier + HMM

Noisy GPS points are matched to a road network. A classifier scores the candidate roads for every GPS point, then an HMM/Viterbi step picks a connected route. The model and API run on **Laptop 1**; the UI runs on **Laptop 2**.

## How it works (short)
1. Each trip has its own road network. GPS is projected to metres; a KD-tree finds candidate arcs near every point (60 m, up to 25).
2. 23 features per (point, candidate) pair -> **HistGradientBoosting** (Optuna-tuned) gives the probability that the candidate is the true road.
3. **HMM/Viterbi** (states = candidate arcs, emission = classifier probability, transition = network distance vs GPS distance) picks the best connected sequence; gaps and disconnected jumps are handled.
4. The API predicts a trip; the **server** then scores it against the reference route it keeps privately. The reference is never sent to the UI.

## Team and ownership
| Part | Owner | Files |
|---|---|---|
| A. Data understanding, EDA, API, UI, deployment | Gunavathi MI | `src/config.py`, `src/data/loader.py`, `src/viz/`, `src/api/`, `src/ui/`, `scripts/01*`, `05`, `10` |
| B. Labels, candidates, HMM core | Gowtham L | `src/data/alignment.py`, `candidates.py`, `dataset.py`, `src/features/geometry.py`, `src/hmm/network.py`, `src/hmm/viterbi.py` |
| C. Features, metrics, decoder tuning | Gautam Krishna | `src/features/build.py`, `src/models/data.py`, `evaluate.py`, `src/hmm/evaluate_hmm.py`, `scripts/08`, `09` |
| D. Models, Optuna tuning, artifacts | Guru Preetham | `src/models/zoo.py`, `scripts/03`, `04`, `06`, `07`, `artifacts/` |

## Install
```bash
pip install -r requirements.txt
```
Python 3.10+. Data goes in `data/Student folder/{train,student_test}` (gitignored). Use `PATHSNAP_DATA_DIR` to point somewhere else.

## Reproduce the experiments
```bash
python3 scripts/01_eda.py                       # data inventory + quality stats
python3 scripts/01b_eda_maps.py                 # EDA maps and plots
python3 scripts/02_build_dataset.py             # alignment, candidates, features   (~20 s)
python3 scripts/03_train_compare.py             # 6 models, GroupKFold             (~12 min)
python3 scripts/04_plots_phase1.py              # comparison plots
python3 scripts/06_tune_optuna.py 40 1200       # Optuna TPE tuning, <=40 trials / 20 min
python3 scripts/07_export_artifacts.py          # final model -> artifacts/
python3 scripts/08_tune_decoder_and_compare.py  # HMM tuning + classifier-only vs +HMM
python3 scripts/09_make_prediction_maps.py      # Folium maps, reference vs predicted (student_test)
python3 scripts/10_robustness_test.py           # needs the API running (see below)
```
Results are written to `results/`, figures to `reports/figures/`, maps to `reports/maps/`.

## Deploy for the evaluation (two laptops, same Wi-Fi / hotspot)

### Laptop 1 - model + API (holds the reference routes)
1. Put the evaluator trip folders in one folder, e.g. `C:\eval_trips\<trip_id>\<trip_id>.track/.nodes/.arcs/.route`.
2. Start the API (the trip ids are discovered from the folder, nothing is hardcoded):
```bash
# Linux / macOS
export PATHSNAP_EVAL_DIR=/path/to/eval_trips
python3 -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```
```powershell
# Windows PowerShell
$env:PATHSNAP_EVAL_DIR = "C:\eval_trips"
python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```
3. Find the laptop's IP (`ipconfig` on Windows, `ip a` / `ifconfig` elsewhere) and allow port 8000 through the firewall (accept the Windows prompt, or run `netsh advfirewall firewall add rule name="PathSnap API" dir=in action=allow protocol=TCP localport=8000` as admin).
4. Request logs appear in this console (and in `logs/api_requests.log`). Show them to the evaluator.

### Laptop 2 - UI
```bash
python3 -m http.server 8080 --directory src/ui
```
Open `http://localhost:8080/?api=http://<LAPTOP1_IP>:8000` (or open it and type the API URL into the box). Click a trip; the UI calls `POST /predict` and shows the route and accuracy. If the API laptop uses another port or address, just change the URL box.

### API
| Endpoint | Purpose |
|---|---|
| `GET /health` | status, model name, decoder settings |
| `GET /trips` | trip ids found in `PATHSNAP_EVAL_DIR` |
| `POST /predict` `{"trip_id": "..."}` | predicted route, per-point arcs, GPS, timings and server-side accuracy |
| `GET /map/{trip_id}` | Folium map: GPS + network + predicted route (never the reference) |

Quick check: `curl http://<LAPTOP1_IP>:8000/health`.

### Environment variables
`PATHSNAP_EVAL_DIR` (trips served by the API; default: student_test folder), `PATHSNAP_ARTIFACTS_DIR` (model files; default `artifacts/`), `PATHSNAP_DATA_DIR` (training data).

## Notes
- The reference route is read only by `src/api/evaluator.py` after the prediction is finished; the prediction code (`src/api/predictor.py`) never opens `.route`.
- Reports are not stored in git. Phase results are in `results/`.
