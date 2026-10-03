"""Collect results/anchors.json and every run in results/runs/ into results/summary.json
(the single file read by make_paper_figures.py)."""
import glob, json, os

HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
A = json.load(open("results/anchors.json"))
S = {"anchors": {}, "runs": []}
for mode in A:
    for pred in A[mode]:
        for split in ("id", "ood"):
            x = A[mode][pred][split]
            S["anchors"][f"{mode}|{pred}|{split}"] = {
                "mre": x["real"]["mre"], "medre": x["real"]["medre"], "mre_time": x["time"]["mre"],
                "mre_full": x["full"]["mre"], "R_time": x["R_time"], "R_full": x["R_full"],
                "r_log": x["real"]["pearson_log"]}
for f in sorted(glob.glob("results/runs/*.json")):
    r = json.load(open(f))
    d = {k: r[k] for k in ("mode", "cond", "seed", "embed", "steps", "n_params")}
    d["tag"] = os.path.basename(f)[:-5]
    for split in ("id", "ood"):
        x = r["final"][split]
        d[split] = {"mre": x["real"]["mre"], "medre": x["real"]["medre"], "mre_time": x["time"]["mre"],
                    "mre_full": x["full"]["mre"], "medre_time": x["time"]["medre"], "R_time": x["R_time"],
                    "R_full": x["R_full"], "r_log": x["real"]["pearson_log"], "spearman": x["real"]["spearman"]}
    d["curve"] = r["curve"]
    S["runs"].append(d)
json.dump(S, open("results/summary.json", "w"), indent=1)
print(f"results/summary.json: {len(S['runs'])} runs, {len(S['anchors'])} anchor entries")
