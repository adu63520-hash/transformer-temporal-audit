"""
run_all.py -- one command runs everything (resumable: finished runs are skipped).

    python run_all.py --smoke      # ~1 min sanity check, run this first
    python run_all.py              # full grid, 3,000 steps per run
Results: results/runs/*.json, progress log: results/log.txt
"""
import argparse, json, os, sys, time, traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(HERE)
RES = os.path.join(HERE, "results")
os.makedirs(os.path.join(RES, "runs"), exist_ok=True)
LOG = os.path.join(RES, "log.txt")


def log(msg):
    line = time.strftime("%H:%M:%S ") + str(msg)
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--long", action="store_true",
                    help="follow-up: order-free ceiling on the random-forcing task, 12000 steps")
    args = ap.parse_args()

    log(f"python {sys.version.split()[0]} at {sys.executable}")
    log("importing torch (first import can take a minute on Windows) ...")
    import torch
    log(f"torch {torch.__version__} imported, checking CUDA ...")
    cuda_ok = torch.cuda.is_available()
    log(f"cuda available: {cuda_ok}")
    from train_tf import run
    log(f"=== start | torch {torch.__version__} | cuda {torch.cuda.is_available()} | "
        f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'} | smoke={args.smoke}")

    if args.smoke:
        try:
            r = run("orig", "clean", 0, os.path.join(RES, "smoke.json"), steps=300, n_train=2000,
                    eval_every=100, reps=3, n_perm=2, log=log)
            c = r["curve"]
            log(f"SMOKE OK | params {r['n_params']/1e6:.2f}M | {300 / r['train_sec']:.1f} it/s incl. eval | "
                f"loss {c[0]['train_loss']:.3f} -> {c[-1]['train_loss']:.3f}")
        except Exception:
            log("SMOKE FAILED\n" + traceback.format_exc())
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

    t0, done = time.time(), 0
    for i, (mode, cond, seed, embed, steps, suffix) in enumerate(jobs):
        tag = f"{mode}_{cond}_s{seed}{suffix}"
        path = os.path.join(RES, "runs", tag + ".json")
        if os.path.exists(path):
            log(f"[{i + 1}/{len(jobs)}] {tag}: already done, skip")
            continue
        log(f"[{i + 1}/{len(jobs)}] {tag}: start")
        try:
            r = run(mode, cond, seed, path, steps=steps, embed=embed, log=log)
            done += 1
            f = r["final"]["id"]
            log(f"[{i + 1}/{len(jobs)}] {tag}: DONE in {r['train_sec'] / 60:.1f} min | ID MRE {f['real']['mre']:.3f} "
                f"R_time {f['R_time']:.2f} R_full {f['R_full']:.1f} | elapsed {(time.time() - t0) / 60:.0f} min")
        except Exception:
            log(f"[{i + 1}/{len(jobs)}] {tag}: FAILED\n" + traceback.format_exc())
    log("=== ALL DONE")


if __name__ == "__main__":
    main()
