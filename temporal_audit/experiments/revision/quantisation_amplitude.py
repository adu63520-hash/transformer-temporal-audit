"""Evaluate excitation amplitude at fixed beta and quantisation resolution."""
import argparse
import json
from pathlib import Path


import numpy as np

from ... import tasks as t


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    rng = np.random.default_rng(2000)
    length, log_alpha = t.grid_params(False, 30)
    beta = t.beta_of(length, log_alpha)
    raw = t.simulate(beta, t.make_forcing(len(beta), "orig", rng), "orig", rng)
    out = {}
    for amp in (.5, 1., 2.):
        series = 20 + amp * (raw - 20)
        pred = t.ols_beta(t.quantise(series))
        out[str(amp)] = t.metrics(pred, beta)["mre"]
    print(out)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
