"""Render the three revised figures as PDF and PNG."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
import numpy as np

colors = ["#1565ad", "#bd4e14", "#12775f"]
conditions = ["clean", "time", "full"]
markers = ["o", "s", "D"]


def save(fig, name, output):
    fig.savefig(output / f"{name}.pdf")
    fig.savefig(output / f"{name}.png", dpi=240)
    plt.close(fig)


def schematic(output):
    ramp = ["#e7f1fc", "#d0e4f7", "#b5d4f1", "#95c1e9",
            "#73a8d8", "#4d88c1", "#32689d", "#214872"]
    channels = ["F", "T1", "T2"]
    rng = np.random.default_rng(3)
    original = np.arange(24).reshape(3, 8)
    hour = original[:, rng.permutation(8)]
    full = rng.permutation(original.ravel()).reshape(3, 8)
    panels = [
        ("(a) Original sequence", original, "Chronology, simultaneity and channels retained"),
        ("(b) Shuffle whole hours", hour, "Positions randomised; triples and channels retained"),
        ("(c) Permute individual values", full, "Positions, triples and channel assignments disrupted"),
    ]
    fig = plt.figure(figsize=(3.36, 3.95))
    for i, (title, codes, note) in enumerate(panels):
        axis = fig.add_axes((.025, .681 - i * .294, .95, .275))
        axis.set_xlim(-1.5, 8.15)
        axis.set_ylim(-.65, 3.72)
        axis.set_axis_off()
        axis.text(-1.38, 3.58, title, fontsize=10, fontweight="bold", va="top")
        for row in range(3):
            y = 2 - row
            axis.text(-.15, y + .40, f"{channels[row]} slot", ha="right", va="center", fontsize=8)
            for col in range(8):
                original_channel, original_hour = divmod(codes[row, col], 8)
                axis.add_patch(Rectangle((col, y), .93, .80, color=ramp[original_hour], lw=0))
                axis.text(col + .465, y + .40, channels[original_channel], va="center", ha="center",
                          fontsize=8, color="white" if original_hour >= 5 else "#101010")
        axis.text(-1.38, -.2, note, va="top", fontsize=8)
    legend = fig.add_axes((.045, .012, .91, .053))
    legend.set_xlim(0, 17)
    legend.set_ylim(0, 1)
    legend.set_axis_off()
    legend.text(0, .5, "Original hour", fontsize=8, va="center")
    for i, color in enumerate(ramp):
        legend.add_patch(Rectangle((6 + i * 1.35, .08), 1.15, .84, color=color, lw=0))
        legend.text(6.575 + i * 1.35, .5, str(i + 1), ha="center", va="center", fontsize=8,
                    color="white" if i >= 5 else "#101010")
    save(fig, "fig0_interventions", output)


def log_axis(ax, limits, ticks):
    ax.set_yscale("log")
    ax.set_ylim(limits)
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.grid(axis="y", color="#d9d9d9", lw=.6)
    ax.set_axisbelow(True)


def fixed_forcing(summary, output):
    runs = [r for r in summary["runs"] if r["mode"] == "orig" and r["embed"] == "token+value"
            and r["steps"] == 3000]
    ols = summary["anchors"]["orig|OLS|id"]
    fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.78), layout="constrained", sharey=True)
    for ax, (key, title) in zip(axes, [
        ("mre", "(a) Ordered inputs"), ("mre_time", "(b) Whole-hour shuffle"),
        ("mre_full", "(c) Full permutation"),
    ]):
        log_axis(ax, (.017, 2.7), [.02, .05, .1, .2, .5, 1, 2])
        ax.set_title(title, loc="left", pad=5)
        for i, cond in enumerate(conditions):
            errors = [r["id"][key] for r in runs if r["cond"] == cond]
            mean = np.mean(errors)
            ax.scatter(i + np.linspace(-.13, .13, len(errors)), errors, s=33, color=colors[i],
                       marker=markers[i], edgecolor="white", linewidth=.6, zorder=3)
            ax.hlines(mean, i-.22, i+.22, color=colors[i], linewidth=1.7, zorder=2)
            if key == "mre" and cond == "full":
                ax.text(i, min(errors) / 1.10, f"{mean:.3f}", ha="center", va="top", fontsize=9)
            else:
                ax.text(i, max(errors) * 1.16, f"{mean:.3f}", ha="center", va="bottom", fontsize=9)
        if key != "mre_full":
            ax.axhline(ols[key], color="#505050", ls="--", lw=.85, zorder=1)
            ax.text(2.46 if key == "mre" else 2.48, ols[key] * 1.07, "OLS", ha="right", va="bottom", fontsize=9)
        ax.tick_params(labelsize=9.5)
        ax.set_xlim(-.5, 2.54)
        ax.set_xticks(range(3), ["Clean", "Hour-\nshuffled", "Fully\npermuted"])
    axes[0].set_ylabel("Mean relative error (log scale)")
    fig.supxlabel("Transformer training inputs", fontsize=9)
    save(fig, "fig1_same_accuracy", output)


def training_curves(summary, output):
    runs = [r for r in summary["runs"] if r["mode"] == "rand" and r["embed"] == "token+value"
            and r["steps"] == 12000]
    fig, axes = plt.subplots(2, 1, figsize=(3.36, 3.65), sharex=True)
    fig.subplots_adjust(left=.17, right=.97, bottom=.13, top=.85, hspace=.50)
    for r in sorted(runs, key=lambda r: (r["cond"], r["seed"])):
        clean = r["cond"] == "clean"
        st = [x["step"] / 1000 for x in r["curve"]]
        options = dict(color=colors[0 if clean else 1], ls="-" if clean else "--",
                       lw=1.35 if clean else .95, marker="o" if clean else None, ms=2.5,
                       zorder=3 if clean else 2)
        key = "id_mre" if clean else "id_mre_time"
        axes[0].plot(st, [x[key] for x in r["curve"]], **options)
        axes[1].plot(st, [x["id_R_time"] for x in r["curve"]], **options)
    ar = summary["anchors"]["rand|OLS|id"]
    axes[0].axhline(ar["mre"], color="#555555", lw=.8, ls=":")
    axes[1].axhline(ar["R_time"], color="#555555", lw=.8, ls=":")
    axes[1].axhline(1, color="#a0a0a0", lw=.7)
    log_axis(axes[0], (.025, .65), [.03, .1, .3])
    log_axis(axes[1], (.18, 30), [.3, 1, 3, 10, 30])
    axes[0].set_title("(a) Error on the training input type", loc="left")
    axes[0].set_ylabel("Test MRE")
    axes[1].set_title("(b) Sensitivity to whole-hour shuffling", loc="left")
    axes[1].set_ylabel("Error ratio")
    axes[1].set_xlabel("Training steps (thousands)")
    axes[1].set_xticks([0, 4, 8, 12])
    axes[1].set_xlim(.65, 12.25)
    fig.legend(handles=[
        Line2D([], [], color=colors[0], marker="o", ms=3, label="Clean (1 seed)"),
        Line2D([], [], color=colors[1], ls="--", label="Hour-shuffled (3 seeds)"),
        Line2D([], [], color="#555555", ls=":", label="OLS"),
    ], loc="upper center", ncol=3, fontsize=7.5, frameon=False,
        handlelength=1.35, columnspacing=.85, bbox_to_anchor=(.50, .984))
    save(fig, "fig2_training_curves", output)



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))

    font_dir = Path(r"C:\Windows\Fonts")
    for file in ("times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf"):
        if (font_dir / file).exists():
            font_manager.fontManager.addfont(str(font_dir / file))
    family = "Times New Roman"
    font_manager.findfont(family, fallback_to_default=False)
    plt.rcParams.update({
        "font.family": family, "font.size": 9, "pdf.fonttype": 42,
        "mathtext.fontset": "custom", "mathtext.rm": family, "mathtext.it": family + ":italic",
        "mathtext.bf": family + ":bold", "mathtext.bfit": family + ":bold:italic", "mathtext.fallback": None,
        "axes.spines.right": False, "axes.spines.top": False,
        "axes.edgecolor": "#555555", "axes.titleweight": "bold",
    })

    schematic(args.output)
    fixed_forcing(summary, args.output)
    training_curves(summary, args.output)


if __name__ == "__main__":
    main()

