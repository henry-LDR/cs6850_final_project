"""Part 2: resistance as a progressive cascade on a trait-dependency graph.

From the 93 distinct M. tuberculosis resistance profiles we estimate how much
each resistance depends on others:

    W[i, j] = P(j present | i present) - P(j present | i absent),   clipped at 0,

so W[i, j] > 0 means carrying i makes j more likely. Distinct profiles are
counted once each, which limits pseudo-replication from clonal clusters; a
tree-based estimate should replace this (see README).

On W we run the cascade models of AGT chapter 24:
  - threshold cascades (Section 2 rule) from each single seed, over q,
  - Independent Cascade spread from each single seed, with W as probabilities,
  - greedy seed selection, which is within 1 - 1/e of optimal for the
    Independent Cascade model (submodular spread).

Run from the repository root after python data/fetch.py.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evograph import cascades

DRUGS = ["INH", "RIF", "PZA", "EMB", "STR", "AMI", "CAP", "MOX", "OFL", "PRO"]
K = len(DRUGS)
OUT = Path("results")


def dependency_weights(X):
    W = np.zeros((K, K))
    for i in range(K):
        has, lacks = X[:, i] == 1, X[:, i] == 0
        if has.any() and lacks.any():
            W[i] = X[has].mean(axis=0) - X[lacks].mean(axis=0)
    np.fill_diagonal(W, 0)
    return np.clip(W, 0, None)


def main():
    OUT.mkdir(exist_ok=True)
    d = pd.read_csv("data/raw/tb_resistance.csv")
    X = np.unique(d[DRUGS].astype(int).to_numpy(), axis=0)  # distinct profiles
    W = dependency_weights(X)

    qs = np.round(np.arange(0.05, 0.55, 0.05), 2)
    threshold = {DRUGS[s]: [int(cascades.threshold_cascade(W, [s], q)[0].sum()) for q in qs] for s in range(K)}

    def ic(seeds, rng):
        return cascades.independent_cascade(W, seeds, rng)

    spread = {DRUGS[s]: cascades.expected_spread(ic, [s], runs=2000, rng=0) for s in range(K)}
    greedy = cascades.greedy_seeds(lambda S: cascades.expected_spread(ic, S, runs=1000, rng=0), range(K), 3)

    summary = {
        "distinct_profiles": int(len(X)),
        "dependency_W": {DRUGS[i]: dict(zip(DRUGS, np.round(W[i], 3).tolist())) for i in range(K)},
        "strongest_dependencies": [
            [DRUGS[i], DRUGS[j], round(float(W[i, j]), 3)]
            for i, j in zip(*np.unravel_index(np.argsort(W, axis=None)[::-1][:8], W.shape))
        ],
        "threshold_q": qs.tolist(),
        "threshold_cascade_size_by_seed": threshold,
        "independent_cascade_expected_spread": {k: round(v, 2) for k, v in spread.items()},
        "greedy_seeds": [DRUGS[s] for s in greedy],
    }
    (OUT / "tb_cascades.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "dependency_W"}, indent=2))

    fig, (a, b) = plt.subplots(1, 2, figsize=(8.4, 3.8), dpi=200, gridspec_kw={"width_ratios": [1.1, 1]})
    fig.patch.set_facecolor("#fcfcfb")
    for ax in (a, b):
        ax.set_facecolor("#fcfcfb")
    from matplotlib.colors import LinearSegmentedColormap
    blues = LinearSegmentedColormap.from_list("ramp", ["#fcfcfb", "#cde2fb", "#86b6ef", "#2a78d6", "#104281"])
    im = a.imshow(W, cmap=blues, vmin=0, vmax=max(0.6, W.max()))
    a.set_xticks(range(K), DRUGS, rotation=90, fontsize=7, color="#52514e")
    a.set_yticks(range(K), DRUGS, fontsize=7, color="#52514e")
    a.set_xlabel("…makes this resistance more likely", fontsize=8, color="#0b0b0b")
    a.set_ylabel("Carrying this resistance…", fontsize=8, color="#0b0b0b")
    a.set_title("Dependency weights W[i, j]", fontsize=9, loc="left", color="#0b0b0b")
    cb = fig.colorbar(im, ax=a, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=7, colors="#52514e")
    cb.outline.set_visible(False)
    for s in a.spines.values():
        s.set_visible(False)

    order = sorted(spread, key=spread.get, reverse=True)
    b.barh(order[::-1], [spread[k] for k in order[::-1]], color="#2a78d6", height=0.6)
    b.set_xlabel("Expected resistances after the cascade", fontsize=8, color="#0b0b0b")
    b.set_title("Independent Cascade spread from one seed", fontsize=9, loc="left", color="#0b0b0b")
    b.grid(axis="x", color="#e8e7e3", lw=0.6)
    b.set_axisbelow(True)
    b.tick_params(labelsize=7, colors="#52514e")
    for s in ("top", "right"):
        b.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        b.spines[s].set_color("#b5b4ae")
    fig.tight_layout()
    fig.savefig(OUT / "tb_cascades.png", facecolor=fig.get_facecolor())


if __name__ == "__main__":
    main()
