"""Slide-sized versions of the key figures (larger type). Run after species_network.py."""

import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

spec = importlib.util.spec_from_file_location("sn", Path(__file__).with_name("species_network.py"))
sn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sn)
OUT = Path("results")


def main():
    d, _, _ = sn.load()
    cut = np.load(OUT / f"species_network_cut{sn.SUFFIX}.npz")
    snake, br = cut["snake"], cut["bridge"]
    rng = np.random.default_rng(1)
    x = d.Presacrals.to_numpy() * np.exp(rng.uniform(-0.03, 0.03, len(d)))
    y = d.Digits.to_numpy() + rng.uniform(-0.3, 0.3, len(d))
    side = np.zeros(len(d), int)
    side[snake] = 1
    isb = np.zeros(len(d), bool)
    isb[br] = True
    fig, ax = plt.subplots(figsize=(8.2, 4.75), dpi=200)
    fig.patch.set_facecolor("#fbfbf8")
    ax.set_facecolor("#fbfbf8")
    ax.scatter(x[side == 0], y[side == 0], s=14, color="#a8a7a1", alpha=0.6, lw=0, label=f"Limbed lizards ({(side == 0).sum():,})")
    ax.scatter(x[side == 1], y[side == 1], s=16, color="#2a78d6", alpha=0.85, lw=0, label=f"Snake-like community ({(side == 1).sum()})")
    ax.scatter(x[isb], y[isb], s=60, facecolor="none", edgecolor="#e34948", lw=1.6, label=f"Bridge species ({isb.sum()})")
    ax.axhline(7.5, color="#e34948", lw=1.2, ls=(0, (5, 4)))
    ax.text(16, 7.7, "Cheeger cut", color="#b8302f", fontsize=13)
    ax.set_xscale("log")
    ax.set_xticks([15, 25, 40, 60, 100], ["15", "25", "40", "60", "100"])
    ax.minorticks_off()
    ax.set_yticks([0, 2, 4, 6, 8, 10])
    ax.set_xlabel("Presacral vertebrae (body length)", fontsize=14, color="#0b0b0b")
    ax.set_ylabel("Digits", fontsize=14, color="#0b0b0b")
    ax.tick_params(labelsize=12, colors="#52514e")
    ax.grid(color="#e8e7e3", lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#b5b4ae")
    ax.legend(fontsize=12, frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.95))
    fig.tight_layout()
    fig.savefig(OUT / "slide_cut.png", facecolor=fig.get_facecolor())
    drivers_compact()
    drivers_compact("Moist forest", "paper_drivers.png")


def drivers_compact(forest_label="Rainforest", out="slide_drivers.png"):
    import json
    u = json.loads((OUT / "squamate_uncertainty.json").read_text())["coupling_multipliers"]
    rows = [("Burrowing", "elongation", u["fossorial -> elongate"]), ("Burrowing", "digit reduction", u["fossorial -> digits reduced"])]
    for env, label in (("desert", "Desert"), ("moist_forest", forest_label)):
        c = json.loads((OUT / f"squamate_driver_{env}.json").read_text())["couplings"]
        rows += [(label, "elongation", c[f"{env} -> elongate"]), (label, "digit reduction", c[f"{env} -> digits reduced"])]
    fig, ax = plt.subplots(figsize=(7.16, 4.24), dpi=200)
    fig.patch.set_facecolor("#fbfbf8")
    ax.set_facecolor("#fbfbf8")
    y = np.arange(len(rows))[::-1]
    for yi, (drv, trait, v) in zip(y, rows):
        lo, hi = v["ci95"]
        color = "#2a78d6" if (lo > 1 or hi < 1) else "#a8a7a1"
        ax.plot([lo, hi], [yi, yi], color=color, lw=3, solid_capstyle="round")
        ax.scatter([v["multiplier"]], [yi], s=70, color=color, zorder=3, edgecolor="#fbfbf8", lw=1.2)
    ax.axvline(1, color="#52514e", lw=1, ls="--")
    ax.text(1.08, len(rows) - 0.45, "no effect", fontsize=12, color="#52514e")
    ax.set_xscale("log")
    ax.set_xlim(0.04, 60)
    ticks = [0.1, 0.3, 1, 3, 10, 30]
    ax.set_xticks(ticks, [f"{t:g}×" for t in ticks])
    ax.minorticks_off()
    ax.set_yticks(y, [f"{d} → {t}" for d, t, _ in rows], fontsize=13, color="#0b0b0b")
    ax.set_xlabel("Multiplier on the rate of the change (95% interval)", fontsize=12, color="#52514e")
    ax.tick_params(axis="x", labelsize=12, colors="#52514e")
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#e8e7e3", lw=0.6)
    ax.set_axisbelow(True)
    ax.set_ylim(-0.6, len(rows) - 0.2)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#b5b4ae")
    fig.tight_layout()
    fig.savefig(OUT / out, facecolor=fig.get_facecolor())


if __name__ == "__main__":
    main()
