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
Python 3.10+.

## Data layout (gitignored - never commit `data/`)
```
Machine-Learning-Hackathon/
  data/
    train/          80 trip folders   (from Student folder.zip)   - only needed to re-run training
    student_test/   10 trip folders   (from Student folder.zip)   - used for the trial / rehearsal
    eval_trips/     10 trip folders   (from the evaluator zip)    - used ONLY in the real evaluation
      <trip_id>/<trip_id>.track  .nodes  .arcs  .route
```
The old layout `data/Student folder/{train,student_test}` also works, and a wrapper folder inside `eval_trips` (e.g. `eval_trips/evaluator_test/<trip>/`) is fine.

## Run the system (two laptops, same Wi-Fi / hotspot)

**Laptop 1 - model + API.** Windows PowerShell, from the repo folder:
```powershell
.\scripts\start_api.ps1 test      # TRIAL: serves data\student_test
.\scripts\start_api.ps1 eval      # EVALUATION: serves data\eval_trips
```
(Linux/macOS: `bash scripts/start_api.sh test` or `eval`.) The script prints the address to give Laptop 2, for example `http://192.168.1.20:8000`. Allow Python through the Windows firewall when asked (private network).

**Laptop 2 - UI.**
```powershell
.\scripts\start_ui.ps1 http://192.168.1.20:8000
```
It prints the link to open (`http://localhost:8080/?api=...`). You can also change the API address in the box on the page.

Check from Laptop 2 before the demo: open `http://<LAPTOP1_IP>:8000/health` - it must show `"status":"ok"` and `n_trips` 10.

### API
| Endpoint | Purpose |
|---|---|
| `GET /health` | status, trips folder, model name, decoder settings |
| `GET /trips` | trip ids found in the trips folder |
| `POST /predict` `{"trip_id": "..."}` | predicted route, classifier-only picks, GPS, timings, server-side accuracy |
| `GET /map/{trip_id}` | Folium map: GPS + network + predicted route (never the reference) |

Every request is logged to the console and to `logs/api_requests.log`. Environment variables: `PATHSNAP_EVAL_DIR` (trips folder), `PATHSNAP_ARTIFACTS_DIR` (model files), `PATHSNAP_DATA_DIR` (training data).

## HMM demo (terminal)
```bash
python scripts/12_demo_hmm.py 00000077            # student trip: what the HMM changed, with examples
python scripts/12_demo_hmm.py <trip_id> data/eval_trips    # same on an evaluation trip
```
The UI also shows it: tick/untick "Classifier only" and "With HMM", and read the card "point predictions changed by the HMM".

## Reproduce the experiments
```bash
python3 scripts/01_eda.py ; python3 scripts/01b_eda_maps.py
python3 scripts/02_build_dataset.py             # alignment, candidates, features   (~20 s)
python3 scripts/03_train_compare.py             # 6 models, GroupKFold             (~12 min)
python3 scripts/04_plots_phase1.py
python3 scripts/06_tune_optuna.py 40 1200       # Optuna TPE tuning, <=40 trials / 20 min
python3 scripts/07_export_artifacts.py          # final model -> artifacts/
python3 scripts/08_tune_decoder_and_compare.py  # HMM tuning + classifier-only vs +HMM
python3 scripts/09_make_prediction_maps.py      # Folium maps (student_test, with reference)
python3 scripts/10_robustness_test.py           # needs the API running (trial mode)
```

## Notes
- The reference route is read only by `src/api/evaluator.py` after the prediction is finished; the prediction code (`src/api/predictor.py`) never opens `.route`.
- Reports are not stored in git. Phase results are in `results/`.
