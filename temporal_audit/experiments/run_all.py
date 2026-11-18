"""Run and resume the Transformer experiment grid."""
import argparse
import json
from pathlib import Path
import sys
import time
import traceback

def main():
    ap = argparse.ArgumentParser(description="Run or resume the Transformer benchmark.")
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--long", action="store_true",
                    help="follow-up: order-free ceiling on the random-forcing task, 12000 steps")
    args = ap.parse_args()
    results_dir = args.results_dir
    (results_dir / "runs").mkdir(parents=True, exist_ok=True)

    def log(message):
        line = time.strftime("%H:%M:%S ") + message
        print(line, flush=True)
        with (results_dir / "log.txt").open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")

    log(f"python {sys.version.split()[0]} at {sys.executable}")
    log("importing torch (first import can take a minute on Windows) ...")
    import torch
    log(f"torch {torch.__version__} imported, checking CUDA ...")
    cuda_ok = torch.cuda.is_available()
    log(f"cuda available: {cuda_ok}")
    from ..train_tf import run
    log(f"=== start | torch {torch.__version__} | cuda {torch.cuda.is_available()} | "
        f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'} | smoke={args.smoke}")

    if args.smoke:
        try:
            r = run("orig", "clean", 0, results_dir / "smoke.json", steps=300, n_train=2000,
                    eval_every=100, reps=3, n_perm=2, log=log)
            c = r["curve"]
            log(f"SMOKE OK | params {r['n_params']/1e6:.2f}M | {300 / r['train_sec']:.1f} it/s incl. eval | "
                f"loss {c[0]['train_loss']:.3f} -> {c[-1]['train_loss']:.3f}")
        except Exception:
            log("SMOKE FAILED\n" + traceback.format_exc())
            raise
        return

    jobs = []
    for seed in range(args.seeds):                      # seed 0 of everything first
        for mode in ("orig", "rand"):
            for cond in ("clean", "time", "full"):
                jobs.append((mode, cond, seed, "token+value", args.steps, ""))
    for mode in ("orig", "rand"):                       # ablation: pure token embedding
        jobs.append((mode, "clean", 0, "token", args.steps, "_tokenonly"))
    if args.long:                                       # hour-shuffled training x3, then one clean reference
        jobs = [("rand", "time", sd, "token+value", 12000, "_long12k") for sd in range(3)]
        jobs.append(("rand", "clean", 0, "token+value", 12000, "_long12k"))

    t0 = time.time()
    failures = []
    for i, (mode, cond, seed, embed, steps, suffix) in enumerate(jobs):
        tag = f"{mode}_{cond}_s{seed}{suffix}"
        path = results_dir / "runs" / (tag + ".json")
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing["steps"] != steps:
                raise SystemExit(f"{path} contains {existing['steps']} steps. Use a separate --results-dir.")
            log(f"[{i + 1}/{len(jobs)}] {tag}: already done, skip")
            continue
        log(f"[{i + 1}/{len(jobs)}] {tag}: start")
        try:
            r = run(mode, cond, seed, path, steps=steps, embed=embed, log=log)
            f = r["final"]["id"]
            log(f"[{i + 1}/{len(jobs)}] {tag}: DONE in {r['train_sec'] / 60:.1f} min | ID MRE {f['real']['mre']:.3f} "
                f"R_time {f['R_time']:.2f} R_full {f['R_full']:.1f} | elapsed {(time.time() - t0) / 60:.0f} min")
        except Exception:
            log(f"[{i + 1}/{len(jobs)}] {tag}: FAILED\n" + traceback.format_exc())
            failures.append(tag)
    if failures:
        raise SystemExit(f"{len(failures)} runs failed: {', '.join(failures)}")
    log("=== ALL DONE")


if __name__ == "__main__":
    main()
