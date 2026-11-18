"""Reproduce initial-submission artwork."""
import json, sys, os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt


def main():
    SRC = sys.argv[1] if len(sys.argv) > 1 else "results/summary.json"
    OUT = sys.argv[2] if len(sys.argv) > 2 else "figures/initial"
    os.makedirs(OUT, exist_ok=True)
    S = json.load(open(SRC)); RUNS = [r for r in S["runs"] if r["embed"] == "token+value"]
    C = {"clean": "#2a78d6", "time": "#eb6834", "full": "#1baf7a"}
    INK, SEC, MUT, GRID = "#0b0b0b", "#52514e", "#898781", "#e6e5e1"
    plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "axes.edgecolor": MUT, "axes.labelcolor": SEC,
                         "xtick.color": SEC, "ytick.color": SEC, "text.color": INK, "pdf.fonttype": 42})
    LAB = {"clean": "clean", "time": "hours\nshuffled", "full": "all\npermuted"}
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
    def plain(a, ticks):
        a.yaxis.set_major_locator(FixedLocator(ticks)); a.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}")); a.yaxis.set_minor_formatter(NullFormatter())

    def style(a):
        a.grid(axis="y", color=GRID, lw=.6, zorder=0); a.set_axisbelow(True)
        for s in ("top", "right"): a.spines[s].set_visible(False)

    # ---------------- Figure 1: original task, three training regimes --------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.35))
    A = lambda p, k: S["anchors"][f"orig|{p}|id"][k]
    titles = ["(a) Test error (MRE)", r"(b) Hour-shuffle ratio $R_{\mathrm{time}}$", r"(c) Full-permutation ratio $R_{\mathrm{full}}$"]
    for j, key in enumerate(("mre", "R_time", "R_full")):
        a = ax[j]; a.set_yscale("log")
        for k, cond in enumerate(("clean", "time", "full")):
            v = [r["id"][key] for r in RUNS if r["mode"] == "orig" and r["cond"] == cond and r["steps"] == 3000]
            xs = k + np.linspace(-.13, .13, len(v)); m = float(np.mean(v))
            a.plot([k - .27, k + .27], [m, m], color=C[cond], lw=1.8, zorder=2)
            a.scatter(xs, v, s=26, color=C[cond], edgecolor="white", linewidth=.9, zorder=3)
            a.annotate(f"{m:.3f}" if key == "mre" else (f"{m:.1f}" if m >= 3 else f"{m:.2f}"), (k, max(v) * (1.09 if key == "mre" else 1.28)), ha="center", va="bottom", fontsize=7.5)
        if key == "mre":
            a.axhline(A("OLS", "mre"), color=MUT, lw=.9, ls="--", zorder=1); a.text(2.55, A("OLS", "mre") * 1.04, "OLS", fontsize=6.8, color=SEC, ha="right", va="bottom")
            a.axhline(A("triple-bag", "mre"), color=MUT, lw=.9, ls=":", zorder=1); a.text(-.45, A("triple-bag", "mre") * .96, "order-free ridge", fontsize=6.8, color=SEC, ha="left", va="top")
            a.set_ylim(0.015, 0.2); plain(a, [0.02, 0.05, 0.1, 0.2])
        else:
            a.axhline(1, color=MUT, lw=.9, zorder=1); a.text(-.45, 0.93, "no effect", fontsize=6.8, color=SEC, ha="left", va="top")
            if key == "R_time":
                a.axhline(A("OLS", "R_time"), color=MUT, lw=.9, ls="--", zorder=1); a.text(2.55, A("OLS", "R_time") * 1.06, "OLS", fontsize=6.8, color=SEC, ha="right", va="bottom")
            a.set_ylim(0.4, 150); plain(a, [1, 10, 100])
        a.set_xticks(range(3)); a.set_xticklabels([LAB[c] for c in ("clean", "time", "full")]); a.set_xlim(-.5, 2.6)
        a.set_title(titles[j], fontsize=8.5, loc="left", pad=5); a.set_xlabel("training inputs"); style(a)
    fig.tight_layout(w_pad=1.2)
    fig.savefig(f"{OUT}/fig1_same_accuracy.pdf"); fig.savefig(f"{OUT}/fig1_same_accuracy.png", dpi=220); plt.close(fig)

    # ---------------- Figure 2: random-forcing task, 12k-step training curves -------------------------
    fig, ax = plt.subplots(1, 2, figsize=(3.35, 2.2))
    L = [r for r in RUNS if r["mode"] == "rand" and r["steps"] == 12000]
    for r in sorted(L, key=lambda r: (r["cond"], r["seed"])):
        st = [c["step"] / 1000 for c in r["curve"]]
        if r["cond"] == "clean":
            ax[0].plot(st, [c["id_mre"] for c in r["curve"]], color=C["clean"], lw=1.6, zorder=3)
            ax[1].plot(st, [c["id_R_time"] for c in r["curve"]], color=C["clean"], lw=1.6, zorder=3)
        else:
            ax[0].plot(st, [c["id_mre_time"] for c in r["curve"]], color=C["time"], lw=1.1, alpha=.95, zorder=2)
            ax[1].plot(st, [c["id_R_time"] for c in r["curve"]], color=C["time"], lw=1.1, alpha=.95, zorder=2)
    Ar = lambda p, k: S["anchors"][f"rand|{p}|id"][k]
    ax[0].set_yscale("log"); ax[0].set_ylim(0.025, 0.6); plain(ax[0], [0.03, 0.1, 0.3]); ax[0].axhline(Ar("OLS", "mre"), color=MUT, lw=.9, ls="--", zorder=1)
    ax[0].text(12, Ar("OLS", "mre") * 1.05, "OLS", fontsize=6.5, color=SEC, ha="right", va="bottom")
    ax[0].text(6.6, 0.42, "clean", color=INK, fontsize=7, va="center"); ax[0].plot([5.2, 6.2], [0.42, 0.42], color=C["clean"], lw=1.6)
    ax[0].text(6.6, 0.3, "shuffled", color=INK, fontsize=7, va="center"); ax[0].plot([5.2, 6.2], [0.3, 0.3], color=C["time"], lw=1.1)
    ax[0].set_title("(a) Test error (MRE)", fontsize=8, loc="left", pad=4)
    ax[1].set_yscale("log"); ax[1].set_ylim(0.15, 40); plain(ax[1], [0.3, 1, 3, 10, 30]); ax[1].axhline(1, color=MUT, lw=.9, zorder=1)
    ax[1].axhline(Ar("OLS", "R_time"), color=MUT, lw=.9, ls="--", zorder=1); ax[1].text(12, Ar("OLS", "R_time") * 1.08, "OLS", fontsize=6.5, color=SEC, ha="right", va="bottom")
    ax[1].set_title(r"(b) $R_{\mathrm{time}}$", fontsize=8, loc="left", pad=4)
    for a in ax:
        a.set_xlabel("training steps (k)"); a.set_xticks([0, 4, 8, 12]); a.set_xlim(0, 12.3); style(a)
    fig.tight_layout(w_pad=.8)
    fig.savefig(f"{OUT}/fig2_training_curves.pdf"); fig.savefig(f"{OUT}/fig2_training_curves.png", dpi=220); plt.close(fig)
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
