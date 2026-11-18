"""Reproduce initial-submission artwork."""
import sys, os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def main():
    OUT = sys.argv[1] if len(sys.argv) > 1 else "figures/initial"; os.makedirs(OUT, exist_ok=True)
    INK, SEC, MUT = "#0b0b0b", "#52514e", "#898781"
    RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#256abf"]  # hour 1..8, light -> dark
    CH = ["F", "T$_1$", "T$_2$"]
    plt.rcParams.update({"font.size": 7.5, "font.family": "DejaVu Sans", "text.color": INK, "pdf.fonttype": 42})
    H = 8
    orig = [[(c, h) for h in range(H)] for c in range(3)]                       # grid[row][col] = (channel, hour)
    rng = np.random.default_rng(3)
    perm = rng.permutation(H)
    shuf = [[(c, perm[k]) for k in range(H)] for c in range(3)]
    flat = [(c, h) for c in range(3) for h in range(H)]; idx = rng.permutation(len(flat))
    full = [[flat[idx[r * H + k]] for k in range(H)] for r in range(3)]
    panels = [("Original trajectory", orig, ["temporal order", "simultaneity", "channel identity"], [True, True, True]),
              (r"Hour shuffle $\pi_{\mathrm{time}}$ (matched)", shuf, ["temporal order", "simultaneity", "channel identity"], [False, True, True]),
              (r"Full permutation $\pi_{\mathrm{full}}$ (unmatched)", full, ["temporal order", "simultaneity", "channel identity"], [False, False, False])]
    fig, axes = plt.subplots(3, 1, figsize=(3.35, 3.55))
    for a, (title, grid, props, kept) in zip(axes, panels):
        for r in range(3):
            for k in range(H):
                c, h = grid[r][k]
                a.add_patch(Rectangle((k, 2 - r), .94, .9, facecolor=RAMP[h], edgecolor="white", linewidth=.6))
                a.text(k + .47, 2 - r + .45, f"{CH[c]}", ha="center", va="center", fontsize=6.3,
                       color="white" if h >= 5 else INK)
            a.text(-.25, 2 - r + .45, CH[r] + " slot", ha="right", va="center", fontsize=6.5, color=SEC)
        a.set_xlim(-1.75, H + 5.6); a.set_ylim(-.1, 3.45); a.axis("off")
        a.text(0, 3.12, title, fontsize=7.8, ha="left", va="bottom")
        for i, (p, kp) in enumerate(zip(props, kept)):
            a.text(H + .3, 2.45 - i * .95, ("kept" if kp else "lost"), fontsize=6.5, ha="left", va="center",
                   color=INK, fontweight="bold" if not kp else "normal")
            a.text(H + 1.6, 2.45 - i * .95, p, fontsize=6.5, ha="left", va="center", color=SEC)
    axes[-1].text(0, -.38, "hour 1", fontsize=6.3, color=SEC, ha="left", va="top")
    axes[-1].text(H - .06, -.38, "hour 8", fontsize=6.3, color=SEC, ha="right", va="top")
    for k in range(H):
        axes[-1].add_patch(Rectangle((1.6 + k * .58, -.72), .52, .3, facecolor=RAMP[k], edgecolor="white", linewidth=.4, clip_on=False))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.96, bottom=0.07, hspace=0.32)
    fig.savefig(f"{OUT}/fig0_interventions.pdf"); fig.savefig(f"{OUT}/fig0_interventions.png", dpi=240)
    print("ok")


if __name__ == "__main__":
    main()
