"""
cue_conflict.py -- cue-conflict test (no input is permuted).

Models are trained on the fixed-forcing task (24 h forcing period). At test time the forcing period is
changed. The recurrence that defines beta is untouched, so a predictor that uses consecutive readings
(OLS) stays unbiased; a predictor that uses order-free amplitude statistics is thrown off, because the
mapping from amplitudes to beta depends on the forcing period. We ask on which side each Transformer falls.

    python cue_conflict.py            # trains 3 models (~6 min each, skipped if already saved), then evaluates
Outputs: results/cue_conflict.json, checkpoints in results/ckpt/
"""
import json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); os.chdir(HERE)
import tasks as T  # noqa: E402

PERIODS = [24, 20, 16, 12, 8]
REPS, STEPS, SEED = 30, 3000, 0
RES = os.path.join(HERE, "results"); CK = os.path.join(RES, "ckpt")
LOG = os.path.join(RES, "log_cue_conflict.txt")


def log(msg):
    line = time.strftime("%H:%M:%S ") + str(msg)
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def summarise(pred, beta):
    pred = np.maximum(np.asarray(pred, dtype=np.float64), 1e-6)
    re = np.abs(pred - beta) / beta
    ratio = pred / beta
    return {"mre": float(re.mean()), "medre": float(np.median(re)),
            "median_ratio": float(np.median(ratio)),
            "q25_ratio": float(np.quantile(ratio, .25)), "q75_ratio": float(np.quantile(ratio, .75)),
            "mean_log_ratio": float(np.log(ratio).mean())}


def main():
    sets = {p: T.make_period_dataset(p, REPS, seed=4000 + p) for p in PERIODS}
    out = {"periods": PERIODS, "reps": REPS, "anchors": {}, "models": {}}

    # ---- analytic anchors (numpy only) ----
    tr = T.make_dataset("orig", "train", 20000, seed=1000)
    ridge = T.RFFRidge(seed=0).fit(T.feat_triple_bag(tr["bins"]), np.log(tr["beta"]))
    for name, f in (("OLS", T.ols_beta), ("order-free ridge", lambda b: ridge.predict(T.feat_triple_bag(b)))):
        out["anchors"][name] = {str(p): summarise(f(d["bins"]), d["beta"]) for p, d in sets.items()}
        log(f"anchor {name:17s} median pred/true by period: " +
            "  ".join(f"{p}h {out['anchors'][name][str(p)]['median_ratio']:.2f}" for p in PERIODS))
    if "--anchors-only" in sys.argv:
        json.dump(out, open(os.path.join(RES, "cue_conflict_anchors.json"), "w"), indent=1); return

    # ---- Transformers ----
    import torch
    from train_tf import run, GPTRegressor, predict
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"torch {torch.__version__} on {device}")
    for cond in ("clean", "time", "full"):
        ck = os.path.join(CK, f"orig_{cond}_s{SEED}.pt")
        if not os.path.exists(ck):
            log(f"training orig/{cond} (seed {SEED}, {STEPS} steps) ...")
            run("orig", cond, SEED, os.path.join(RES, "runs_ckpt", f"orig_{cond}_s{SEED}.json"),
                steps=STEPS, log=log, ckpt_path=ck)
        c = torch.load(ck, map_location=device, weights_only=False)
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
    with open(os.path.join(RES, "cue_conflict.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    log("=== CUE CONFLICT DONE -> results/cue_conflict.json")


if __name__ == "__main__":
    main()
