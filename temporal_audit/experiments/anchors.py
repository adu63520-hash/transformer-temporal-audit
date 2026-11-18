"""Evaluate the OLS and invariant ridge reference estimators."""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from ..tasks import (
    make_dataset, intervene, ols_beta, feat_pooled_bag, feat_triple_bag, RFFRidge, metrics,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--train-size", type=int, default=int(os.environ.get("N_TRAIN", "20000")))
    args = parser.parse_args()
    n_train, reps, n_perm = args.train_size, 30, 10
    out = {}
    for mode in ("orig", "rand"):
        tr = make_dataset(mode, "train", n_train, seed=1000)
        tests = {"id": make_dataset(mode, "id", reps, seed=2000), "ood": make_dataset(mode, "ood", reps, seed=3000)}
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
                    ms = [metrics(f(intervene(d["bins"], kind, rng)), d["beta"]) for _ in range(n_perm)]
                    res[kind] = {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}
                    res[f"R_{kind}"] = res[kind]["mre"] / res["real"]["mre"]
                    res[f"Rlog_{kind}"] = res[kind]["mae_log"] / res["real"]["mae_log"]
                out[mode][pname][split] = res
                r = res
                print(f"[{mode}] {pname:10s} {split:3s} | real: MRE {r['real']['mre']:.3f} medRE {r['real']['medre']:.3f} "
                      f"r_log {r['real']['pearson_log']:.3f} | time-shuf MRE {r['time']['mre']:.3f} (R={r['R_time']:.1f}) "
                      f"| full-perm MRE {r['full']['mre']:.3f} (R={r['R_full']:.1f})")

    output = args.results_dir / "anchors.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
