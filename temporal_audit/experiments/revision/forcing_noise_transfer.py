"""Repeat period-transfer evaluation on fresh forcing noise with existing checkpoints."""
import argparse
import json
from pathlib import Path


import numpy as np
import torch

from ... import tasks as T
from ...train_tf import GPTRegressor, predict


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("output", type=Path)
    p.add_argument("--checkpoints", type=Path, help="Checkpoint directory (default: RESULTS_DIR/ckpt)")
    args = p.parse_args()
    checkpoints = args.checkpoints or args.results_dir / "ckpt"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    periods = (24, 20, 16, 12, 8)
    seeds = list(range(5100, 5110))
    data = {period: [T.make_period_dataset(period, 10, seed) for seed in seeds] for period in periods}
    models = {}
    for cond in ("clean", "time", "full"):
        ck = torch.load(checkpoints / f"orig_{cond}_s0.pt",
                        map_location=device, weights_only=True)
        model = GPTRegressor(ck["d"], ck["layers"], ck["heads"], ck["embed"]).to(device)
        model.load_state_dict(ck["model"])
        model.eval()
        models[cond] = (model, ck["mu"], ck["sd"])
    result = {}
    for period, batches in data.items():
        bins = np.concatenate([x["bins"] for x in batches])
        beta = np.concatenate([x["beta"] for x in batches])
        preds = {"OLS": T.ols_beta(bins)}
        for name, (model, mu, sd) in models.items():
            preds[name] = predict(model, bins, mu, sd, device, bs=128)
        summary = {}
        for name, values in preds.items():
            ratios = (np.maximum(values, 1e-6) / beta).reshape(len(seeds), -1)
            medians = np.median(ratios, axis=1)
            summary[name] = {"batch_medians": medians.tolist(), "median_of_medians": float(np.median(medians)),
                             "min": float(medians.min()), "max": float(medians.max())}
        result[str(period)] = summary
        print(period, {k: [round(v["min"], 4), round(v["max"], 4)] for k, v in summary.items()}, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"seeds": seeds, "reps_per_cell": 10, "periods": result}, indent=2)+"\n")


if __name__ == "__main__":
    main()
