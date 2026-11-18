"""Reproduce the additional diagnostics for the first PRL review."""
import argparse
import json
from pathlib import Path


import numpy as np
import torch

from ... import tasks as T
from ...train_tf import GPTRegressor, predict


def paired_changes(predictor, data, kind, draws):
    beta = data["beta"]
    original = np.maximum(predictor(data["bins"]), 1e-6)
    rng = np.random.default_rng(7)
    perturbed = np.stack([
        np.maximum(predictor(T.intervene(data["bins"], kind, rng)), 1e-6)
        for _ in range(draws)
    ])
    return {
        "ordered_mre": np.mean(np.abs(original - beta) / beta),
        "perturbed_mre": np.mean(np.abs(perturbed - beta) / beta),
        "prediction_change": np.mean(np.abs(perturbed - original) / beta),
    }


def successor_accuracy(bins, beta):
    """Rank possible successors using the recurrence and a bag-derived beta."""
    forcing, t1, t2 = T.dequantise(bins).transpose(2, 0, 1)
    b = beta[:, None, None]
    residual1 = t1[:, None, :] - t1[:, :-1, None] - b * (forcing[:, None, :] - t1[:, :-1, None])
    residual2 = t2[:, None, :] - t2[:, :-1, None] - b * (t1[:, :-1, None] - t2[:, :-1, None])
    score = residual1**2 + residual2**2
    current = np.arange(T.N_TIME - 1)
    score[:, current, current] = np.inf
    correct = score[:, current, current + 1]
    # Uniform tie credit makes the score independent of candidate enumeration.
    ties = (score == correct[..., None]).sum(axis=2)
    return np.mean((correct == score.min(axis=2)) / ties)


def rotation_control():
    # Each pair has identical full-state multisets and identical first states.
    # The sign of the oriented cross product detects the valid direction.
    rng = np.random.default_rng(90)
    phases = rng.uniform(0, 2 * np.pi, 1080)
    theta = phases[:, None] + 2 * np.pi * np.arange(24)[None, :] / 24
    plus = np.stack([np.cos(theta), np.sin(theta)], axis=-1)
    minus = plus[:, (-np.arange(24)) % 24]
    states = np.concatenate((plus, minus))
    label = np.repeat([1, -1], len(plus))
    def estimate(paths):
        a, b = paths[:, :-1], paths[:, 1:]
        return np.sign((a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]).sum(axis=1))
    rng = np.random.default_rng(7)
    shuffled = [np.take_along_axis(states, rng.permuted(
        np.tile(np.arange(24), (len(states), 1)), axis=1)[..., None], axis=1)
        for _ in range(5)]
    return {"pairs": len(plus), "ordered_accuracy": np.mean(estimate(states) == label),
            "shuffled_accuracy": np.mean([np.mean(estimate(x) == label) for x in shuffled])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--checkpoints", type=Path, help="Checkpoint directory (default: RESULTS_DIR/ckpt)")
    args = parser.parse_args()
    checkpoints = args.checkpoints or args.results_dir / "ckpt"
    data = T.make_dataset("orig", "id", 30, seed=2000)
    tr = T.make_dataset("orig", "train", 20000, seed=1000)
    ridge = T.RFFRidge(seed=0).fit(T.feat_triple_bag(tr["bins"]), np.log(tr["beta"]))
    def ridge_predict(bins):
        return ridge.predict(T.feat_triple_bag(bins))

    result = {"successor_top1": successor_accuracy(data["bins"], ridge_predict(data["bins"])),
              "successor_chance": 1 / 23, "rotation": rotation_control(),
              "paired": {}, "fixed_seed_aggregate": {}}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    summary = json.loads((args.results_dir / "summary.json").read_text())
    for label, f in (("OLS", T.ols_beta), ("ridge", ridge_predict)):
        result["paired"][label] = {k: paired_changes(f, data, k, 5) for k in ("time", "full")}
        result["fixed_seed_aggregate"][label] = {
            k: {"mean": summary["anchors"][f"orig|{'OLS' if label == 'OLS' else 'triple-bag'}|id"][field]}
            for k, field in (("ordered", "mre"), ("time", "mre_time"), ("full", "mre_full"))
        }
    for cond in ("clean", "time", "full"):
        checkpoint = torch.load(checkpoints / f"orig_{cond}_s0.pt",
                                map_location=device, weights_only=True)
        model = GPTRegressor(checkpoint["d"], checkpoint["layers"], checkpoint["heads"], checkpoint["embed"]).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()

        def infer(bins):
            return predict(model, bins, checkpoint["mu"], checkpoint["sd"], device, bs=128)

        result["paired"][cond] = {k: paired_changes(infer, data, k, 5) for k in ("time", "full")}
        rows = [r["id"] for r in summary["runs"] if r["mode"] == "orig" and r["cond"] == cond
                and r["embed"] == "token+value" and r["steps"] == 3000]
        assert len(rows) == 3
        base = np.array([r["mre"] for r in rows])
        stats = lambda a: {"mean": np.mean(a), "sd": np.std(a)}
        aggregate = {"ordered": stats(base)}
        for kind in ("time", "full"):
            shifted = np.array([r[f"mre_{kind}"] for r in rows])
            aggregate[kind] = stats(shifted)
            aggregate[f"delta_{kind}"] = stats(shifted - base)
        result["fixed_seed_aggregate"][cond] = aggregate
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
