"""Draw the squamate body-plan graph and its conductance bottleneck, from the fitted model.

Nodes are the 12 valid body plans, arranged by how many of the four traits they
hold. Arrows are trait gains, with width proportional to the log of the fitted
gain rate. The dashed line is the Cheeger (sweep) cut, which isolates the
ancestral body plan. Each node is labeled with the mean time the walk stays
there before any change, 1 / (total exit rate).

Run from the repository root after experiments/squamate_fit.py.
"""

import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

from evograph import walks

spec = importlib.util.spec_from_file_location("sq", Path(__file__).with_name("squamate_fit.py"))
sq = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sq)

POS = {
    "0": (0, 0),
    "E": (1, 1.25), "R": (1, 0), "F": (1, -1.25),
    "ER": (2, 1.9), "EF": (2, 0.65), "RF": (2, -0.65), "RZ": (2, -1.9),
    "ERZ": (3, 1.25), "ERF": (3, 0), "RZF": (3, -1.25),
    "ERZF": (4, 0),
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#b5b4ae"
BLUE, BLUE_LIGHT, ACCENT, ACCENT_LIGHT = "#2a78d6", "#cde2fb", "#e34948", "#fbe3e1"


def main():
    p = json.loads(Path("results/squamate_fit_params.json").read_text())["P_pairwise"]
    model = walks.TraitModel(p["gain"], np.full(sq.K, p["loss"]), gamma=1.0, D=np.array(p["Gamma"]), keep=sq.KEEP)
    Q, _, _ = sq.restricted_chain(model)
    summary = json.loads(Path("results/squamate_summary.json").read_text())
    labels = [sq.label(x) for x in sq.VALID]
    wait = {labels[i]: 100 / -Q[i, i] for i in range(len(labels))}  # Myr

    fig, ax = plt.subplots(figsize=(10, 5.4), dpi=200)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.set_axis_off()

    rates = [Q[i, j] for i in range(len(labels)) for j in range(len(labels))
             if i != j and Q[i, j] > 0 and sq.VALID[j] > sq.VALID[i] and (sq.VALID[j] & sq.VALID[i]) == sq.VALID[i]]
    lo = np.log10(min(rates))
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            x, y = sq.VALID[i], sq.VALID[j]
            if i == j or Q[i, j] <= 0 or y < x or (x & y) != x:
                continue  # draw gains only
            cut = a == "0"
            w = 0.6 + 1.25 * (np.log10(Q[i, j]) - lo)
            arrow = FancyArrowPatch(POS[a], POS[b], arrowstyle="-|>", mutation_scale=7 + 1.1 * w,
                                    lw=w, color=ACCENT if cut else "#c9c8c3",
                                    shrinkA=20 if cut else 17, shrinkB=17, zorder=1)
            ax.add_patch(arrow)
            if cut:
                mx, my = np.add(POS[a], POS[b]) / 2
                ax.text(mx - 0.06, my + (0.12 if POS[b][1] >= 0 else -0.12), f"{Q[i, j]:.2f}",
                        fontsize=7, color=ACCENT, ha="right", va="center")

    for name, (x, y) in POS.items():
        anc = name == "0"
        ax.scatter(x, y, s=1500 if anc else 900, color=ACCENT_LIGHT if anc else BLUE_LIGHT,
                   edgecolor=ACCENT if anc else BLUE, lw=1.6, zorder=3)
        ax.text(x, y, "ancestor" if anc else name, ha="center", va="center",
                fontsize=7 if anc else 9, color=INK, fontweight="bold", zorder=4)
        t = wait[name]
        above = name == "ERZ"  # keeps the label clear of the steep RZ -> ERZ arrow
        ax.text(x, y + (0.34 if above else -0.42 if anc else -0.34), f"{t:.0f} Myr" if t >= 1 else "<1 Myr",
                ha="center", va="bottom" if above else "top", fontsize=7, color=MUTED)

    ax.plot([0.5, 0.5], [-2.35, 2.35], ls=(0, (4, 3)), lw=1.4, color=ACCENT, zorder=2)
    phi = summary["bottleneck_conductance"]
    e = int(np.floor(np.log10(phi)))
    sup = str(e).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
    ax.text(0.5, 2.5, f"Cheeger cut: conductance {phi / 10**e:.0f} × 10{sup}",
            ha="center", va="bottom", fontsize=8, color=ACCENT)
    ax.text(-0.02, -0.85, f"Mean stay before\nany change:\n≈ {round(wait['0'], -1):.0f} Myr, longer than\nsquamates have existed",
            ha="center", va="top", fontsize=7.5, color=INK, linespacing=1.3)
    share = summary["stationary_top"].get("ERZF", 0)
    ax.text(4, -0.6, f"snake-like,\nburrowing\n({100 * share:.0f}% of long-run mass)", ha="center", va="top", fontsize=7, color=MUTED)

    ax.text(-0.35, 3.05, "Squamate body plans: leaving the ancestral lizard form is the bottleneck",
            fontsize=11, color=INK, ha="left", va="bottom")
    ax.text(-0.35, -2.75, "E elongate (≥35 vertebrae)    R digits reduced    Z digitless    F burrowing.   "
            "Arrows: gains, width ∝ log rate (red numbers per 100 Myr). Grey: mean stay before any change.",
            fontsize=7, color=MUTED, ha="left", va="top")
    ax.set_xlim(-0.45, 4.45)
    ax.set_ylim(-3.0, 3.35)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"results/squamate_isolation.{ext}", facecolor=fig.get_facecolor())


if __name__ == "__main__":
    main()
