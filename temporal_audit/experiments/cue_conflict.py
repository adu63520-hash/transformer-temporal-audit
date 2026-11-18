"""Evaluate forcing-period transfer with the fixed recurrence."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from .. import tasks as T

PERIODS = [24, 20, 16, 12, 8]
REPS, STEPS, SEED = 30, 3000, 0


def summarise(pred, beta):
    pred = np.maximum(np.asarray(pred, dtype=np.float64), 1e-6)
    re = np.abs(pred - beta) / beta
    ratio = pred / beta
    return {"mre": float(re.mean()), "medre": float(np.median(re)),
            "median_ratio": float(np.median(ratio)),
            "q25_ratio": float(np.quantile(ratio, .25)), "q75_ratio": float(np.quantile(ratio, .75)),
            "mean_log_ratio": float(np.log(ratio).mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--checkpoints", type=Path)
    parser.add_argument("--anchors-only", action="store_true")
    args = parser.parse_args()
    results_dir = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = args.checkpoints or results_dir / "ckpt"

    def log(message):
        line = time.strftime("%H:%M:%S ") + message
        print(line, flush=True)
        with (results_dir / "log_cue_conflict.txt").open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")

    sets = {p: T.make_period_dataset(p, REPS, seed=4000 + p) for p in PERIODS}
    out = {"periods": PERIODS, "reps": REPS, "anchors": {}, "models": {}}

    # ---- analytic anchors (numpy only) ----
    tr = T.make_dataset("orig", "train", 20000, seed=1000)
    ridge = T.RFFRidge(seed=0).fit(T.feat_triple_bag(tr["bins"]), np.log(tr["beta"]))
    for name, f in (("OLS", T.ols_beta), ("order-free ridge", lambda b: ridge.predict(T.feat_triple_bag(b)))):
        out["anchors"][name] = {str(p): summarise(f(d["bins"]), d["beta"]) for p, d in sets.items()}
        log(f"anchor {name:17s} median pred/true by period: " +
            "  ".join(f"{p}h {out['anchors'][name][str(p)]['median_ratio']:.2f}" for p in PERIODS))
    if args.anchors_only:
        (results_dir / "cue_conflict_anchors.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
        return

    # ---- Transformers ----
    import torch
    from ..train_tf import run, GPTRegressor, predict
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"torch {torch.__version__} on {device}")
    for cond in ("clean", "time", "full"):
        ck = checkpoints / f"orig_{cond}_s{SEED}.pt"
        if not ck.exists():
            log(f"training orig/{cond} (seed {SEED}, {STEPS} steps) ...")
            run("orig", cond, SEED, results_dir / "runs_ckpt" / f"orig_{cond}_s{SEED}.json",
                steps=STEPS, log=log, ckpt_path=ck)
        c = torch.load(ck, map_location=device, weights_only=True)
        model = GPTRegressor(c["d"], c["layers"], c["heads"], c["embed"]).to(device)
        model.load_state_dict(c["model"]); model.eval()
        res = {}
        for p, d in sets.items():
            r = {"ordered": summarise(predict(model, d["bins"], c["mu"], c["sd"], device), d["beta"])}
            rng = np.random.default_rng(7)
            r["hour_shuffled"] = summarise(predict(model, T.intervene(d["bins"], "time", rng), c["mu"], c["sd"], device), d["beta"])
            res[str(p)] = r
        out["models"][cond] = res
        log(f"model  {cond:6s} ordered inputs, median pred/true by period: " +
            "  ".join(f"{p}h {res[str(p)]['ordered']['median_ratio']:.2f}" for p in PERIODS))
        log(f"model  {cond:6s} ordered inputs, MRE by period:              " +
            "  ".join(f"{p}h {res[str(p)]['ordered']['mre']:.3f}" for p in PERIODS))
    with (results_dir / "cue_conflict.json").open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    log(f"Saved {results_dir / 'cue_conflict.json'}")


if __name__ == "__main__":
    main()
