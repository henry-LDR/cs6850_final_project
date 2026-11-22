"""Fit the biased trait walk to M. tuberculosis drug resistance on its phylogeny.

Data: 395 isolates typed for resistance to 10 drugs (Casali et al. 2014), on
their published tree. The state graph is the 10-trait hypercube (1,024
resistance profiles), and the root is fully susceptible.

Steps:
  1. Fit per-drug gain rates and a shared loss rate with independent traits
     (gamma = 0), then with all-to-all coupling gamma, and compare them with a
     likelihood-ratio test. gamma > 0 means each resistance already held speeds
     up the next.
  2. Analyze the fitted chain with the course tools: first steps from the
     susceptible state, expected time to multidrug resistance (MDR, INH + RIF)
     and extensive resistance (XDR), the stationary distribution, and the
     conductance bottleneck from the Fiedler sweep cut.

Run from the repository root after python data/fetch.py:
  python experiments/tb_fit.py            (fits, then analyzes)
  python experiments/tb_fit.py --reuse    (analyzes saved fits)
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

from evograph import fit, spectral, trees, walks

DRUGS = ["INH", "RIF", "PZA", "EMB", "STR", "AMI", "CAP", "MOX", "OFL", "PRO"]
K = len(DRUGS)
D = np.ones((K, K)) - np.eye(K)
OUT = Path("results")


def load():
    parent, length, names = trees.read_newick(Path("data/raw/tb_tree.nwk").read_text())
    length = trees.rescale(length, parent)  # root-to-deepest-tip = 1 time unit
    node = {names[v]: v for v in trees.tips(parent)}
    d = pd.read_csv("data/raw/tb_resistance.csv", dtype={"Isolate": str})
    data = {node[r.Isolate]: fit.tip_vector(r[DRUGS].astype(int).tolist(), K) for _, r in d.iterrows()}
    return parent, length, data, d


def bits(x):
    return [DRUGS[i] for i in range(K) if (x >> i) & 1]


def mask(*names):
    return sum(1 << DRUGS.index(n) for n in names)


def main(reuse):
    OUT.mkdir(exist_ok=True)
    parent, length, data, d = load()
    params_path = OUT / "tb_fit_params.json"

    if reuse and params_path.exists():
        saved = json.loads(params_path.read_text())
    else:
        independent = fit.fit_trait_model(K, parent, length, data, D=D, fit_gamma=False)
        print("independent fit:", independent["loglik"])
        x0 = np.r_[np.log(independent["gain"]), np.log(independent["loss"]), 0.0]
        coupled = fit.fit_trait_model(K, parent, length, data, D=D, fit_gamma=True, x0=x0)
        print("coupled fit:", coupled["loglik"])
        saved = {
            name: {key: (np.asarray(r[key]).tolist() if key == "gain" else r[key])
                   for key in ("gain", "loss", "gamma", "loglik", "n_params", "converged")}
            for name, r in (("independent", independent), ("coupled", coupled))
        }
        params_path.write_text(json.dumps(saved, indent=2))

    ind, cpl = saved["independent"], saved["coupled"]
    lr = 2 * (cpl["loglik"] - ind["loglik"])
    p_value = float(chi2.sf(lr, df=1))
    model = walks.TraitModel(cpl["gain"], np.full(K, cpl["loss"]), gamma=cpl["gamma"], D=D)

    # Random walks: first steps, hitting times, stationary distribution.
    P = model.transition_matrix()
    steps_per_time = model.steps_per_unit_time()
    first = model.flip_rates(np.zeros(K, int))[0]
    first = first / first.sum()
    mdr = [x for x in range(2**K) if x & mask("INH", "RIF") == mask("INH", "RIF")]
    xdr = [x for x in mdr if x & mask("MOX", "OFL") and x & mask("AMI", "CAP")]
    t_mdr = walks.hitting_times(P, mdr)[0] / steps_per_time
    t_xdr = walks.hitting_times(P, xdr)[0] / steps_per_time
    sigma = walks.stationary(P)

    # Greedy most probable route from the susceptible state to MDR.
    route, x = [0], 0
    while x & mask("INH", "RIF") != mask("INH", "RIF"):
        rates = model.flip_rates(np.array([(x >> i) & 1 for i in range(K)]))[0]
        rates[[i for i in range(K) if (x >> i) & 1]] = 0  # gains only
        x |= 1 << int(np.argmax(rates))
        route.append(x)

    # Spectral analysis: the conductance bottleneck of the fitted chain.
    W = walks.flow_graph(P, sigma)
    S, phi = spectral.sweep_cut(W, measure=sigma)
    inside = np.zeros(2**K, bool)
    inside[S] = True
    trait_share = {DRUGS[i]: round(float(((np.arange(2**K)[inside] >> i) & 1).mean()), 3) for i in range(K)}
    side = inside if sigma[inside].sum() <= 0.5 else ~inside
    boundary_flow = np.asarray(W[side][:, ~side].sum(axis=1)).ravel()
    choke = np.flatnonzero(side)[np.argsort(boundary_flow)[::-1][:5]]

    observed = d[DRUGS].astype(int).to_numpy() @ (1 << np.arange(K))
    summary = {
        "isolates": int(len(d)),
        "distinct_observed_profiles": int(len(np.unique(observed))),
        "loglik_independent": ind["loglik"],
        "loglik_coupled": cpl["loglik"],
        "gamma": cpl["gamma"],
        "likelihood_ratio": lr,
        "p_value": p_value,
        "gain_rates": dict(zip(DRUGS, np.round(cpl["gain"], 4).tolist())),
        "loss_rate": cpl["loss"],
        "first_step_probability": dict(zip(DRUGS, np.round(first, 3).tolist())),
        "most_probable_route_to_MDR": [bits(x) for x in route],
        "time_to_MDR_tree_depths": float(t_mdr),
        "time_to_XDR_tree_depths": float(t_xdr),
        "integration_index": float(sigma[-1] / walks.independent_prediction(model.r_gain, model.r_loss)),
        "bottleneck_conductance": float(phi),
        "bottleneck_set_size": int(len(S)),
        "bottleneck_trait_share_inside": trait_share,
        "choke_point_states": [bits(x) for x in choke],
    }
    (OUT / "tb_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    # Figure: per-drug gain rates under the coupled model.
    order = np.argsort(cpl["gain"])[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=200)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.bar([DRUGS[i] for i in order], np.asarray(cpl["gain"])[order], color="#2a78d6", width=0.6)
    ax.set_yscale("log")
    ax.set_ylabel("Gain rate per tree depth (log scale)", color="#0b0b0b")
    ax.set_title(f"TB resistance: fitted gain rates (coupling γ = {cpl['gamma']:.2f}, p = {p_value:.1g})",
                 color="#0b0b0b", loc="left", fontsize=10)
    ax.grid(axis="y", color="#e8e7e3", lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#b5b4ae")
    ax.tick_params(colors="#52514e")
    fig.tight_layout()
    fig.savefig(OUT / "tb_gain_rates.png", facecolor=fig.get_facecolor())


if __name__ == "__main__":
    main(reuse="--reuse" in sys.argv)
