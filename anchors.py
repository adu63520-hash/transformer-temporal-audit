"""
anchors.py  -- Experiment 1: the two analytic anchors, no neural network.

For each task variant and each test-time intervention we report three predictors:
  OLS         : closed-form, must read temporal order
  pooled-bag  : order-blind ridge on the multiset of 72 values (exactly invariant to 'full')
  triple-bag  : order-blind ridge on the set of 24 triples     (exactly invariant to 'time')
Usage:  python anchors.py            (numpy only, ~1 min on CPU)
"""
import json, os, sys
import numpy as np
from tasks import (make_dataset, intervene, ols_beta, feat_pooled_bag, feat_triple_bag,
                   RFFRidge, metrics)

N_TRAIN = int(os.environ.get("N_TRAIN", 20000)); REPS = 30; N_PERM = 10
out = {}
for mode in ("orig", "rand"):
    tr = make_dataset(mode, "train", N_TRAIN, seed=1000)
    tests = {"id": make_dataset(mode, "id", REPS, seed=2000), "ood": make_dataset(mode, "ood", REPS, seed=3000)}
    preds = {"OLS": ols_beta,
             "pooled-bag": None, "triple-bag": None}
    m_pool = RFFRidge(seed=0).fit(feat_pooled_bag(tr["bins"]), np.log(tr["beta"]))
    m_trip = RFFRidge(seed=0).fit(feat_triple_bag(tr["bins"]), np.log(tr["beta"]))
    preds["pooled-bag"] = lambda b: m_pool.predict(feat_pooled_bag(b))
    preds["triple-bag"] = lambda b: m_trip.predict(feat_triple_bag(b))
    out[mode] = {}
    for pname, f in preds.items():
        out[mode][pname] = {}
        for split, d in tests.items():
            res = {"real": metrics(f(d["bins"]), d["beta"])}
            for kind in ("time", "full"):
                rng = np.random.default_rng(7)
                ms = [metrics(f(intervene(d["bins"], kind, rng)), d["beta"]) for _ in range(N_PERM)]
                res[kind] = {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}
                res[f"R_{kind}"] = res[kind]["mre"] / res["real"]["mre"]
                res[f"Rlog_{kind}"] = res[kind]["mae_log"] / res["real"]["mae_log"]
            out[mode][pname][split] = res
            r = res
            print(f"[{mode}] {pname:10s} {split:3s} | real: MRE {r['real']['mre']:.3f} medRE {r['real']['medre']:.3f} "
                  f"r_log {r['real']['pearson_log']:.3f} | time-shuf MRE {r['time']['mre']:.3f} (R={r['R_time']:.1f}) "
                  f"| full-perm MRE {r['full']['mre']:.3f} (R={r['R_full']:.1f})")
os.makedirs("results", exist_ok=True)
json.dump(out, open("results/anchors.json", "w"), indent=1)
print("saved results/anchors.json")
