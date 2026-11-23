"""Fit the biased trait walk to M. tuberculosis drug resistance on its phylogeny.

Data: 395 isolates typed for resistance to 10 drugs (Casali et al. 2014), on
their published tree. The state graph is the 10-trait hypercube (1,024
resistance profiles), and the root is fully susceptible.

Steps:
  1. Fit three nested models and compare them with likelihood-ratio tests:
       A. independent drugs (gamma = 0), test results taken as exact;
       B. independent drugs, each test result wrong with probability error;
       C. as B, plus all-to-all coupling gamma (the log-fold change in a gain
          rate once all 9 other resistances are held).
     Closely related isolates often differ on 2 of the 10 drugs, which
     model A can only explain with very fast gains and losses; phenotypic
     drug-susceptibility tests are known to misclassify some drugs, which
     B allows. gamma > 0 means each resistance already held speeds up the next.
  2. Analyze the fitted chain with the course tools: first steps from the
     susceptible state, expected time to multidrug resistance (MDR, INH + RIF)
     and extensive resistance (XDR), the stationary distribution, and the
     conductance bottleneck from the Fiedler sweep cut.

Run from the repository root after python data/fetch.py:
  python experiments/tb_fit.py            (fits, then analyzes; about 15 minutes on 2 cores)
  python experiments/tb_fit.py --reuse    (analyzes saved fits)
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor
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
# All-to-all coupling, scaled so gamma is the log-fold change in a drug's gain rate once all
# 9 other resistances are held (each held resistance contributes gamma / 9).
D = (np.ones((K, K)) - np.eye(K)) / (K - 1)
OUT = Path("results")


def load():
    parent, length, names = trees.read_newick(Path("data/raw/tb_tree.nwk").read_text())
    length = trees.rescale(length, parent)  # root-to-deepest-tip = 1 time unit
    node = {names[v]: v for v in trees.tips(parent)}
    d = pd.read_csv("data/raw/tb_resistance.csv", dtype={"Isolate": str})
    obs = {node[r.Isolate]: r[DRUGS].astype(int).tolist() for _, r in d.iterrows()}
    return parent, length, obs, d


def run_fit(spec):
    """One fit, in a worker process. spec = (fit_gamma, fit_error, x0)."""
    fit_gamma, fit_error, x0 = spec
    parent, length, obs, _ = load()
    r = fit.fit_trait_model(K, parent, length, D=D, observations=obs, fit_gamma=fit_gamma,
                            fit_error=fit_error, x0=x0)
    return {key: (np.asarray(r[key]).tolist() if key in ("gain", "theta") else r[key])
            for key in ("gain", "loss", "gamma", "error", "theta", "loglik", "n_params", "converged")}


def bits(x):
    return [DRUGS[i] for i in range(K) if (x >> i) & 1]


def mask(*names):
    return sum(1 << DRUGS.index(n) for n in names)


def main(reuse):
    OUT.mkdir(exist_ok=True)
    _, _, _, d = load()
    params_path = OUT / "tb_fit_params.json"

    if reuse and params_path.exists():
        saved = json.loads(params_path.read_text())
    else:
        cache = OUT / "tb_fit_cache.json"  # finished fits, so an interrupted run resumes
        done = json.loads(cache.read_text()) if cache.exists() else {}
        with ProcessPoolExecutor(2) as pool:
            if "B" not in done:
                done["A"], done["B"] = pool.map(run_fit, [(False, False, None), (False, True, None)])
                cache.write_text(json.dumps(done))
            a, b = done["A"], done["B"]
            print("A independent, exact tests:", a["loglik"])
            print("B independent, test error:", b["loglik"], "error", b["error"])
            tb = np.asarray(b["theta"])
            starts = [np.r_[tb[: K + 1], g, tb[K + 1]] for g in (0.0, 1.0)]
            c = max(pool.map(run_fit, [(True, True, s) for s in starts]), key=lambda r: r["loglik"])
            print("C coupled, test error:", c["loglik"], "gamma", c["gamma"])
        saved = {"A_independent_exact": a, "B_independent_error": b, "C_coupled_error": c}
        params_path.write_text(json.dumps(saved, indent=2))

    a, b, cpl = saved["A_independent_exact"], saved["B_independent_error"], saved["C_coupled_error"]
    lr_error = max(0.0, 2 * (b["loglik"] - a["loglik"]))
    p_error = float(0.5 * chi2.sf(lr_error, df=1))  # error = 0 is on the boundary: 50:50 mixture
    lr = max(0.0, 2 * (cpl["loglik"] - b["loglik"]))
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
        "loglik": {"A_independent_exact": a["loglik"], "B_independent_error": b["loglik"],
                   "C_coupled_error": cpl["loglik"]},
        "A_rates": {"gain": dict(zip(DRUGS, np.round(a["gain"], 3).tolist())), "loss": a["loss"]},
        "test_error_rate": cpl["error"],
        "error_likelihood_ratio": lr_error,
        "error_p_value": p_error,
        "gamma": cpl["gamma"],
        "gamma_likelihood_ratio": lr,
        "gamma_p_value": p_value,
        "gain_rates": dict(zip(DRUGS, np.round(cpl["gain"], 4).tolist())),
        "loss_rate": cpl["loss"],
        "all_fits_converged": bool(a["converged"] and b["converged"] and cpl["converged"]),
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
    ax.set_title(f"TB resistance: fitted gain rates (γ = {cpl['gamma']:.2f}, p = {p_value:.1g}; "
                 f"test error {100 * cpl['error']:.1f}%)",
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
