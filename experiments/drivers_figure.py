"""Forest plot of environmental drivers backed out from the tree, with 95% intervals.

Reads results/squamate_uncertainty.json (burrowing, from the 4-trait model) and
results/squamate_driver_<env>.json (desert, moist forest, 5-trait models).
Run from the repository root after those scripts.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("results")
PLAIN = {"elongate": "elongation", "digits reduced": "digit reduction", "digitless": "losing all digits", "fossorial": "burrowing"}


def rows():
    out = []
    u = json.loads((OUT / "squamate_uncertainty.json").read_text())["coupling_multipliers"]
    for t in ("elongate", "digits reduced"):
        v = u[f"fossorial -> {t}"]
        out.append(("Burrowing", PLAIN[t], v["multiplier"], v["ci95"]))
    for env, label in (("desert", "Desert"), ("moist_forest", "Moist forest")):
        f = OUT / f"squamate_driver_{env}.json"
        if not f.exists():
            continue
        c = json.loads(f.read_text())["couplings"]
        for t in ("elongate", "digits reduced", "digitless", "fossorial"):
            v = c[f"{env} -> {t}"]
            out.append((label, PLAIN[t], v["multiplier"], v.get("ci95", [v["multiplier"], v["multiplier"]])))
    return out


def main():
    r = rows()
    fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(r) + 1.0), dpi=220)
    fig.patch.set_facecolor("#fbfbf8")
    ax.set_facecolor("#fbfbf8")
    y = np.arange(len(r))[::-1]
    for yi, (drv, trait, m, ci) in zip(y, r):
        lo, hi = ci
        clear = lo > 1 or hi < 1
        color = "#2a78d6" if clear else "#a8a7a1"
        ax.plot([lo, hi], [yi, yi], color=color, lw=2.2, solid_capstyle="round")
        ax.scatter([m], [yi], s=36, color=color, zorder=3, edgecolor="#fbfbf8", lw=1)
    ax.axvline(1, color="#52514e", lw=0.9, ls="--")
    ax.set_xscale("log")
    ticks = [0.1, 0.3, 1, 3, 10, 30, 100]
    ax.set_xticks(ticks, [f"{t:g}×" for t in ticks])
    ax.minorticks_off()
    ax.set_yticks(y, [f"{drv} → {trait}" for drv, trait, _, _ in r], fontsize=8, color="#0b0b0b")
    ax.set_xlabel("Multiplier on the gain rate (95% interval); grey = interval includes no effect", fontsize=7.5, color="#52514e")
    ax.tick_params(axis="x", labelsize=7.5, colors="#52514e")
    ax.grid(axis="x", color="#e8e7e3", lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#b5b4ae")
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"drivers.{ext}", facecolor=fig.get_facecolor())
    print(r)


if __name__ == "__main__":
    main()
