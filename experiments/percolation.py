"""How much developmental constraint can the space of body plans absorb?

Keep each edge of the trait hypercube H_k with probability p = lambda / k and
measure the fraction of body plans in the largest connected component. Ajtai,
Komlos and Szemeredi (1982) proved a giant component appears exactly when
lambda > 1, so the curves for growing k should sharpen around lambda = 1.

Run from the repository root:  python experiments/percolation.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evograph import statespace

KS = (8, 12, 16)
LAMBDAS = np.linspace(0.2, 3.0, 29)
REPS = 5
COLORS = {8: "#86b6ef", 12: "#2a78d6", 16: "#104281"}  # ordinal blue ramp, validated


def giant_curve(k, rng):
    out = []
    for lam in LAMBDAS:
        fracs = [
            statespace.giant_fraction(statespace.hypercube(k, statespace.constraint_mask(k, lam / k, rng)))
            for _ in range(REPS)
        ]
        out.append(np.mean(fracs))
    return np.array(out)


def main():
    rng = np.random.default_rng(6850)
    curves = {k: giant_curve(k, rng) for k in KS}

    Path("results").mkdir(exist_ok=True)
    with open("results/percolation.csv", "w") as f:
        f.write("lambda," + ",".join(f"k={k}" for k in KS) + "\n")
        for i, lam in enumerate(LAMBDAS):
            f.write(f"{lam:.2f}," + ",".join(f"{curves[k][i]:.4f}" for k in KS) + "\n")

    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=200)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.axvline(1.0, color="#52514e", lw=1, ls=(0, (3, 3)))
    ax.text(1.04, 0.97, "p = 1/k", color="#52514e", fontsize=9, va="top")
    for k in KS:
        ax.plot(LAMBDAS, curves[k], color=COLORS[k], lw=2, label=f"k = {k} traits ({2**k:,} body plans)")
    ax.set_xlabel("Edges kept per body plan, λ = p·k", color="#0b0b0b")
    ax.set_ylabel("Share of body plans in the giant component", color="#0b0b0b")
    ax.set_title("Body-plan space stays connected only above p = 1/k", color="#0b0b0b", loc="left")
    ax.set_xlim(LAMBDAS[0], LAMBDAS[-1])
    ax.set_ylim(0, 1.02)
    ax.grid(color="#e8e7e3", lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#b5b4ae")
    ax.tick_params(colors="#52514e")
    ax.legend(frameon=False, fontsize=8, loc="center right", bbox_to_anchor=(0.97, 0.35))
    fig.tight_layout()
    fig.savefig("results/percolation.png", facecolor=fig.get_facecolor())
    print("wrote results/percolation.png and results/percolation.csv")


if __name__ == "__main__":
    main()
