"""Step 12: show what the HMM/Viterbi step does for one trip, in the terminal.

Usage:  python scripts/12_demo_hmm.py <trip_id> [trips_folder] [--examples N] [--no-reference]
        (default folder: data/student_test)

It runs the full prediction pipeline, then prints
  1. how many point predictions the HMM changed compared with the classifier alone,
  2. (if a reference route exists on this machine) how many of those changes fixed or broke a point,
  3. a few concrete examples: candidate probabilities, the classifier's pick, the HMM's pick, and whether
     each pick is connected to the previous point's road in the network.
The reference is used only AFTER prediction, for the explanation - never by the prediction itself.
"""
import sys
sys.path.insert(0, ".")
from pathlib import Path
import numpy as np
from src.config import STUDENT_TEST_DIR, MODEL_ARTIFACTS_DIR
from src.api.predictor import Predictor, clean_track
from src.data.loader import load_trip
from src.data.alignment import align_points_to_route

args = [a for a in sys.argv[1:] if not a.startswith("--")]
n_examples = int(sys.argv[sys.argv.index("--examples") + 1]) if "--examples" in sys.argv else 6
if "--examples" in sys.argv:
    args.remove(str(n_examples))
use_ref = "--no-reference" not in sys.argv
trip_id = args[0] if args else sorted(p.name for p in STUDENT_TEST_DIR.iterdir())[0]
folder = Path(args[1]) if len(args) > 1 else STUDENT_TEST_DIR
trip_dir = folder / trip_id
if not trip_dir.exists():                                    # allow a wrapper folder
    hits = list(folder.glob(f"*/{trip_id}"))
    trip_dir = hits[0] if hits else trip_dir

res = Predictor(MODEL_ARTIFACTS_DIR).predict_trip_dir(trip_dir)
trip, net, feats = res["_trip"], res["_net"], res["_feats"]
clf, hmm = res["point_arcs_clf"], res["point_arcs_hmm"]
n = len(clf)
changed = np.where((clf != hmm) & (clf >= 0))[0]
print(f"Trip {trip_id}: {n} GPS points, {feats.point_idx.nunique()} with candidate roads")
print(f"Classifier alone and HMM disagree on {len(changed)} of {n} points ({100*len(changed)/max(n,1):.1f}%).")

ref = None
if use_ref and (trip_dir / f"{trip_dir.name}.route").exists():
    t2 = load_trip(trip_dir, with_route=True); clean_track(t2)
    _, ref, _ = align_points_to_route(t2)
    fixed = int(((hmm == ref) & (clf != ref)).sum()); broke = int(((clf == ref) & (hmm != ref)).sum())
    print(f"Against the reference route: HMM fixed {fixed} points, broke {broke}  (net {fixed - broke:+d}).")
    print(f"Top-1 accuracy: classifier only {100*(clf == ref).mean():.1f}%  ->  with HMM {100*(hmm == ref).mean():.1f}%")

def connected(a, b):
    return a == b or net.head[a] == net.tail[b]

print("\nExamples where the HMM changed the answer (candidate probabilities = classifier output):")
shown = 0
pool = changed[changed > 0]
if ref is not None:                                           # prefer cases the HMM fixed, if there are any
    fixed_pts = pool[(hmm[pool] == ref[pool]) & (clf[pool] != ref[pool])]
    pool = fixed_pts if len(fixed_pts) else pool
pick = pool[np.linspace(0, len(pool) - 1, min(n_examples, len(pool))).astype(int)] if len(pool) else []   # spread along the trip
for i in pick:
    if shown >= n_examples:
        break
    c = feats[feats.point_idx == i].sort_values("score", ascending=False).head(3)
    prev = hmm[i - 1]
    cand_txt = ", ".join(f"arc {int(r.arc_id)}: p={r.score:.2f}" for r in c.itertuples())
    line = (f"  point {i} (t={trip.track.t.iloc[i]:.0f}s): candidates [{cand_txt}]\n"
            f"     classifier picked arc {clf[i]} (connected to previous road? {'yes' if connected(prev, clf[i]) else 'NO'}); "
            f"HMM picked arc {hmm[i]} (connected? {'yes' if connected(prev, hmm[i]) else 'NO'})")
    if ref is not None:
        line += f"; reference arc {ref[i]} -> {'HMM correct' if hmm[i] == ref[i] else ('classifier correct' if clf[i] == ref[i] else 'both wrong')}"
    print(line); shown += 1
if shown == 0:
    print("  (no changed points to show)")
print("\nWhy: the HMM scores each candidate by its probability AND by how plausible it is to drive there from the previous road\n"
      "(network distance vs GPS distance), then Viterbi picks the best whole sequence.")
