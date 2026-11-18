"""Combine anchor and Transformer result files into the figure inputs."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    A = json.loads((args.results_dir / "anchors.json").read_text(encoding="utf-8"))
    S = {"anchors": {}, "runs": []}
    for mode in A:
        for pred in A[mode]:
            for split in ("id", "ood"):
                x = A[mode][pred][split]
                S["anchors"][f"{mode}|{pred}|{split}"] = {
                    "mre": x["real"]["mre"], "medre": x["real"]["medre"], "mre_time": x["time"]["mre"],
                    "mre_full": x["full"]["mre"], "R_time": x["R_time"], "R_full": x["R_full"],
                    "r_log": x["real"]["pearson_log"]}
    for f in sorted((args.results_dir / "runs").glob("*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        d = {k: r[k] for k in ("mode", "cond", "seed", "embed", "steps", "n_params")}
        d["tag"] = f.stem
        for split in ("id", "ood"):
            x = r["final"][split]
            d[split] = {"mre": x["real"]["mre"], "medre": x["real"]["medre"], "mre_time": x["time"]["mre"],
                        "mre_full": x["full"]["mre"], "medre_time": x["time"]["medre"], "R_time": x["R_time"],
                        "R_full": x["R_full"], "r_log": x["real"]["pearson_log"], "spearman": x["real"]["spearman"]}
        d["curve"] = r["curve"]
        S["runs"].append(d)

    output = args.results_dir / "summary.json"
    output.write_text(json.dumps(S, indent=1), encoding="utf-8")
    print(f"{output}: {len(S['runs'])} runs, {len(S['anchors'])} anchor entries")


if __name__ == "__main__":
    main()
