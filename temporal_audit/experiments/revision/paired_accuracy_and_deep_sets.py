"""Run the numerical analyses added for Reviewer 2."""
import argparse
import json
from pathlib import Path


import numpy as np
import torch
from torch import nn

from ... import tasks as T
from ...train_tf import GPTRegressor, predict


def mre(prediction, beta):
    return float(np.mean(np.abs(np.maximum(prediction, 1e-6) - beta) / beta))


def stats(data, results_dir):
    beta = data["beta"]
    errors = {}
    for cond in ("clean", "time", "full"):
        rows = []
        for seed in range(3):
            saved = json.loads((results_dir / "runs" / f"orig_{cond}_s{seed}.json").read_text())
            row = saved["final"]["id"]
            error = np.abs(np.maximum(row["pred_real"], 1e-6) - beta) / beta
            assert np.isclose(error.mean(), row["real"]["mre"])
            rows.append(error)
        errors[cond] = np.array(rows)
    rng = np.random.default_rng(414)
    sampled = rng.integers(0, 36, (10000, 36))
    output = {}
    for cond in ("time", "full"):
        diff = errors[cond] - errors["clean"]
        seed_means = diff.mean(1)
        grouped = diff.reshape(3, 36, 30).mean(axis=(0, 2))
        half_width = 4.3026527297 * seed_means.std(ddof=1) / np.sqrt(3)
        output[cond] = {"mean": diff.mean(), "seed_differences": seed_means.tolist(),
                        "seed_ci": [diff.mean() - half_width, diff.mean() + half_width],
                        "grid_bootstrap_ci": np.quantile(grouped[sampled].mean(1), [.025, .975]).tolist()}
    return output


class DeepSets(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(3, 128), nn.SiLU(), nn.Linear(128, 128),
                                     nn.SiLU(), nn.Linear(128, 128))
        self.readout = nn.Sequential(nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 128),
                                     nn.SiLU(), nn.Linear(128, 1))

    def forward(self, values):
        return self.readout(self.encoder(values).mean(dim=1)).squeeze(-1)


def train_sets(train, id_test, ood_test, device, scaled):
    def values(bins):
        centered = T.dequantise(bins) - 20
        x = np.arcsinh(centered / np.array([8, .5, .2])) if scaled else centered / 10
        return torch.tensor(x, dtype=torch.float32, device=device)

    log_beta = np.log(train["beta"])
    center, scale = log_beta.mean(), log_beta.std()
    x = values(train["bins"])
    y = torch.tensor((log_beta - center) / scale, dtype=torch.float32, device=device)
    shuffled = T.intervene(id_test["bins"], "time", np.random.default_rng(7))
    out = []
    for seed in range(3):
        torch.manual_seed(seed)
        model = DeepSets().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=.001)
        for epoch in range(75):
            for idx in torch.randperm(len(x), device=device).split(256):
                opt.zero_grad()
                nn.functional.mse_loss(model(x[idx]), y[idx]).backward()
                opt.step()
        model.eval()
        def infer(bins):
            with torch.no_grad():
                return np.exp(model(values(bins)).cpu().numpy() * scale + center)
        p = infer(id_test["bins"])
        row = {"seed": seed, "id_mre": mre(p, id_test["beta"]),
               "ood_mre": mre(infer(ood_test["bins"]), ood_test["beta"]),
               "max_relative_perm_change": float(np.max(np.abs(infer(shuffled) - p) / id_test["beta"]))}
        out.append(row)
        print("Deep Sets", row, flush=True)
    return out


def recurrence_terms(series):
    f, t1, t2 = series.transpose(2, 0, 1)
    x = np.concatenate([f[:, 1:] - t1[:, :-1], t1[:, :-1] - t2[:, :-1]], axis=1)
    y = np.concatenate([np.diff(t1, axis=1), np.diff(t2, axis=1)], axis=1)
    return x, y


def estimate_quantised(series, step):
    x, y = recurrence_terms(series)
    ols = (x * y).sum(1) / (x**2).sum(1)
    # If 0 <= beta <= 1 and each roundoff has magnitude <= step/2,
    # the observed residual is bounded by step in each transition.
    active = np.abs(x) > 1e-10
    denominator = np.where(active, x, 1)
    a, b = (y - step) / denominator, (y + step) / denominator
    lo = np.maximum(0, np.where(active, np.minimum(a, b), -np.inf).max(1))
    hi = np.minimum(1, np.where(active, np.maximum(a, b), np.inf).min(1))
    feasible = (lo <= hi + 1e-10) & np.all(active | (np.abs(y) <= step + 1e-10), axis=1)
    return ols, np.where(feasible, np.clip(ols, lo, hi), ols), feasible, lo, hi


def quantisation():
    rng = np.random.default_rng(2000)
    lengths, logs = T.grid_params(False, 30)
    beta = T.beta_of(lengths, logs)
    raw = T.simulate(beta, T.make_forcing(len(beta), "orig", rng), "orig", rng)
    x, _ = recurrence_terms(raw)
    drive = np.sqrt((x**2).mean(1))
    low_beta, high_beta = beta <= np.quantile(beta, .25), beta >= np.quantile(beta, .75)
    low_drive, high_drive = drive <= np.quantile(drive, .25), drive >= np.quantile(drive, .75)
    output = {}
    for step in (0, 0.01, 0.1, 0.5):
        series = raw if not step else np.round(raw / step) * step
        ols, corrected, feasible, lo, hi = estimate_quantised(series, step + 1e-12)
        assert np.all(feasible) and np.all((lo - 1e-10 <= beta) & (beta <= hi + 1e-10))
        output[str(step)] = {"ols": mre(ols, beta), "projected": mre(corrected, beta),
                             "ols_low_beta": mre(ols[low_beta], beta[low_beta]),
                             "ols_high_beta": mre(ols[high_beta], beta[high_beta]),
                             "ols_low_drive": mre(ols[low_drive], beta[low_drive]),
                             "ols_high_drive": mre(ols[high_drive], beta[high_drive]),
                             "beta_cutoffs": np.quantile(beta, [.25, .75]).tolist(),
                             "drive_cutoffs": np.quantile(drive, [.25, .75]).tolist()}
    return output


def shifted_inputs(data, kind, rng):
    if kind == "time":
        return T.intervene(data, "time", rng)
    values = data.copy()
    for channel in (1, 2):
        order = rng.random(data.shape[:2]).argsort(1)
        values[:, :, channel] = np.take_along_axis(data[:, :, channel], order, axis=1)
    return T.intervene(values, "time", rng)


def chronology_check(data, device, checkpoints):
    rng = np.random.default_rng(81)
    ck = torch.load(checkpoints / "orig_time_s0.pt", map_location=device, weights_only=True)
    model = GPTRegressor(ck["d"], ck["layers"], ck["heads"], ck["embed"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    bins, beta = data["bins"], data["beta"]
    results = {}
    for kind in ("time", "unpair"):
        shifted = [shifted_inputs(bins, kind, rng) for _ in range(5)]
        assert all(np.array_equal(np.sort(x, axis=1), np.sort(bins, axis=1)) for x in shifted)
        predictions = [predict(model, x, ck["mu"], ck["sd"], device, bs=128) for x in shifted]
        results[kind] = {"mre": float(np.mean([mre(p, beta) for p in predictions])),
                         "mean_pred": np.mean(predictions, axis=0)}
    change = float(np.mean(np.abs(results["time"]["mean_pred"] - results["unpair"]["mean_pred"]) / beta))
    for row in results.values():
        del row["mean_pred"]
    results["paired_change"] = change
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--checkpoints", type=Path, help="Checkpoint directory (default: RESULTS_DIR/ckpt)")
    args = parser.parse_args()
    checkpoints = args.checkpoints or args.results_dir / "ckpt"
    torch.set_num_threads(2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train = T.make_dataset("orig", "train", 20000, 1000)
    test = T.make_dataset("orig", "id", 30, 2000)
    ood = T.make_dataset("orig", "ood", 30, 3000)
    result = {"statistics": stats(test, args.results_dir), "quantisation": quantisation(),
              "chronology": chronology_check(test, device, checkpoints)}
    print(json.dumps(result, indent=2), flush=True)
    result["deep_sets_raw"] = train_sets(train, test, ood, device, scaled=False)
    result["deep_sets_scaled"] = train_sets(train, test, ood, device, scaled=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
