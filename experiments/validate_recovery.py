"""Validation: can the fitter recover planted rates and coupling on the real TB tree?

We plant a 4-trait TraitModel with known gain rates, loss rate and coupling
gamma, simulate it down the published M. tuberculosis tree, keep only the 395
tips that have data in the real study, and refit both the independent
(gamma = 0) and coupled models. For each planted gamma we report the fitted
gamma, the likelihood-ratio test's rejection rate at the 5% level (false
positives when gamma = 0, power otherwise) and the error in the gain rates.

Run from the repository root after python data/fetch.py:
  python experiments/validate_recovery.py           (simulate and fit, about 4 minutes)
  python experiments/validate_recovery.py --reuse   (redraw from saved results)
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2

from evograph import fit, trees, walks

K = 4
GAIN = np.array([6.0, 4.0, 2.5, 1.5])
LOSS = 1.0
GAMMAS = [0.0, 0.5, 1.0]
REPS = 20
D = np.ones((K, K)) - np.eye(K)
OUT = Path("results")


def main(reuse=False):
    OUT.mkdir(exist_ok=True)
    if reuse:
        plot(json.loads((OUT / "validate_recovery.json").read_text())["runs"])
        return
    parent, length, names = trees.read_newick(Path("data/raw/tb_tree.nwk").read_text())
    length = trees.rescale(length, parent)
    node = {names[v]: v for v in trees.tips(parent)}
    isolates = pd.read_csv("data/raw/tb_resistance.csv", dtype={"Isolate": str}).Isolate
    observed = [node[i] for i in isolates]

    rows = []
    for gamma in GAMMAS:
        truth = walks.TraitModel(GAIN, np.full(K, LOSS), gamma=gamma, D=D)
        for rep in range(REPS):
            rng = np.random.default_rng(1000 * int(10 * gamma) + rep)
            states = trees.simulate(truth, parent, length, 0, rng=rng)
            data = {v: fit.tip_vector([(states[v] >> i) & 1 for i in range(K)], K) for v in observed}
            ind = fit.fit_trait_model(K, parent, length, data, D=D, fit_gamma=False)
            x0 = np.r_[np.log(ind["gain"]), np.log(ind["loss"]), 0.0]
            cpl = fit.fit_trait_model(K, parent, length, data, D=D, fit_gamma=True, x0=x0)
            lr = max(0.0, 2 * (cpl["loglik"] - ind["loglik"]))
            row = {
                "gamma_true": gamma,
                "rep": rep,
                "gamma_fit": float(cpl["gamma"]),
                "p_value": float(chi2.sf(lr, 1)),
                "loss_fit": cpl["loss"],
                "converged": bool(ind["converged"] and cpl["converged"]),
                "gain_fit": np.asarray(cpl["gain"]).tolist(),
                "mean_traits_at_tips": float(np.mean([bin(states[v]).count("1") for v in observed])),
            }
            rows.append(row)
            print(f"gamma {gamma:.1f} rep {rep}: fitted {row['gamma_fit']:.2f}, p = {row['p_value']:.3g}")

    summary = {"planted": {"gain": GAIN.tolist(), "loss": LOSS, "gammas": GAMMAS, "reps": REPS}, "runs": rows}
    by = {}
    for g in GAMMAS:
        r = [x for x in rows if x["gamma_true"] == g]
        gains = np.array([x["gain_fit"] for x in r])
        by[str(g)] = {
            "gamma_fit_median": float(np.median([x["gamma_fit"] for x in r])),
            "gamma_fit_range": [float(min(x["gamma_fit"] for x in r)), float(max(x["gamma_fit"] for x in r))],
            "lrt_rejects_at_0.05": f"{sum(x['p_value'] < 0.05 for x in r)}/{len(r)}",
            "gain_median_abs_log_error": float(np.median(np.abs(np.log(gains / GAIN)))),
            "loss_fit_median": float(np.median([x["loss_fit"] for x in r])),
            "fits_converged": f"{sum(x['converged'] for x in r)}/{len(r)}",
        }
    summary["by_gamma"] = by
    (OUT / "validate_recovery.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(by, indent=2))
    plot(rows)


def plot(rows):
    fig, (a, b) = plt.subplots(1, 2, figsize=(8.4, 3.6), dpi=200)
    fig.patch.set_facecolor("#fcfcfb")
    rng = np.random.default_rng(0)
    for ax in (a, b):
        ax.set_facecolor("#fcfcfb")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#b5b4ae")
        ax.tick_params(labelsize=7, colors="#52514e")
        ax.grid(color="#e8e7e3", lw=0.6)
        ax.set_axisbelow(True)

    lim = [-0.4, 1.6]
    a.plot(lim, lim, color="#b5b4ae", lw=1, ls="--", label="Fitted = planted")
    for g in GAMMAS:
        r = [x["gamma_fit"] for x in rows if x["gamma_true"] == g]
        a.scatter(g + rng.uniform(-0.05, 0.05, len(r)), r, s=22, color="#2a78d6", edgecolor="#fcfcfb", lw=0.8, zorder=3)
    a.scatter([], [], s=22, color="#2a78d6", label=f"One simulated dataset ({REPS} per value)")
    a.set_xlim(lim)
    a.set_ylim(min(lim[0], min(x["gamma_fit"] for x in rows) - 0.1), max(lim[1], max(x["gamma_fit"] for x in rows) + 0.1))
    a.set_xlabel("Planted coupling γ", fontsize=8, color="#0b0b0b")
    a.set_ylabel("Fitted coupling γ", fontsize=8, color="#0b0b0b")
    a.set_title("Coupling is recovered", fontsize=9, loc="left", color="#0b0b0b")
    a.legend(fontsize=7, frameon=False, loc="upper left")

    true = np.tile(GAIN, len(rows))
    fitted = np.concatenate([x["gain_fit"] for x in rows])
    lo, hi = 0.5, 20
    b.plot([lo, hi], [lo, hi], color="#b5b4ae", lw=1, ls="--")
    b.scatter(true * np.exp(rng.uniform(-0.04, 0.04, len(true))), fitted, s=14, color="#2a78d6", edgecolor="#fcfcfb", lw=0.6, alpha=0.8, zorder=3)
    b.set_xscale("log")
    b.set_yscale("log")
    b.set_xlim(lo, hi)
    b.set_ylim(min(lo, fitted.min() * 0.8), max(hi, fitted.max() * 1.2))
    b.set_xticks(GAIN, [f"{g:g}" for g in GAIN])
    yt = [y for y in (0.3, 1, 3, 10, 30) if b.get_ylim()[0] <= y <= b.get_ylim()[1]]
    b.set_yticks(yt, [f"{y:g}" for y in yt])
    b.minorticks_off()
    b.set_xlabel("Planted gain rate (per tree depth)", fontsize=8, color="#0b0b0b")
    b.set_ylabel("Fitted gain rate", fontsize=8, color="#0b0b0b")
    b.set_title("Gain rates are recovered", fontsize=9, loc="left", color="#0b0b0b")
    fig.tight_layout()
    fig.savefig(OUT / "validate_recovery.png", facecolor=fig.get_facecolor())


if __name__ == "__main__":
    main(reuse="--reuse" in sys.argv)
