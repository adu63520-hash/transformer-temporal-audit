"""Fit fixed-start recurrence rollouts to quantised observation intervals."""
import argparse
import json
from pathlib import Path


import numpy as np

from ... import tasks as T


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    rng = np.random.default_rng(2000)
    lengths, logs = T.grid_params(False, 30)
    truth = T.beta_of(lengths, logs)
    series = T.simulate(truth, T.make_forcing(len(truth), "orig", rng), "orig", rng)
    grid = np.r_[0., np.geomspace(1e-5, 1., 2048)]
    results = {}
    for step in (.01, .1, .5):
        rounded = np.round(series / step) * step
        forcing = rounded[:, :, 0]
        target1, target2 = rounded[:, :, 1], rounded[:, :, 2]
        def objective(beta):
            t1 = t2 = 20.
            radius1 = radius2 = 0.
            score = 0.
            for t in range(T.N_TIME):
                t2 = (1-beta)*t2 + beta*t1
                t1 = (1-beta)*t1 + beta*forcing[:, t:t+1]
                radius2 = (1-beta)*radius2 + beta*radius1
                radius1 = (1-beta)*radius1 + beta*step/2
                for estimate, observed, radius in ((t1, target1, radius1), (t2, target2, radius2)):
                    difference = estimate - observed[:, t:t+1]
                    violation = np.maximum(np.abs(difference) - radius - step/2, 0)
                    score = score + violation**2 + .01*difference**2
            return score
        score = np.concatenate([objective(g[None, :]) for g in np.array_split(grid, 16)], axis=1)
        best = score.argmin(axis=1)
        lo = grid[np.maximum(best - 1, 0)][:, None]
        hi = grid[np.minimum(best + 1, len(grid) - 1)][:, None]
        for _ in range(24):
            left = (2*lo + hi)/3
            right = (lo + 2*hi)/3
            take_left = objective(left) <= objective(right)
            lo, hi = np.where(take_left, lo, left), np.where(take_left, right, hi)
        pred = ((lo + hi)/2).ravel()
        results[str(step)] = float(np.mean(np.abs(np.maximum(pred, 1e-6) - truth) / truth))
    print(results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
