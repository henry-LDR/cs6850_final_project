"""Biased random walks on the squamate species network: the direction of evolution.

Network: the species network of experiments/species_network.py (919 species; links
to the 6 closest relatives and the 6 closest look-alikes).

Two walks step from a species to one of its neighbours:
  unbiased   every neighbour equally likely (the simple random walk of the course)
  biased     each step is tilted by how evolution actually moved on the dated tree.
             From 100 sampled histories (experiments/squamate_time.py), trait t was
             gained rho_t times as often as it was lost. A step that gains trait t
             is weighted by sqrt(rho_t), one that loses it by 1 / sqrt(rho_t).
Comparing them separates the network's structure from evolution's direction.
We measure the long-run share of time in the snake-like community (the side of
the Cheeger cut), the expected number of steps from an ordinary lizard to reach
it, and the share of walkers there after a given number of steps.

Scenarios (macroevolution under environmental change): we change the burrowing
bias, as if surface habitat loss pushed species underground (burrowing favoured
4x) or soil pollution made burrowing costly (burrowing disfavoured 4x), and rerun
both the species-network walk and the fitted 12-state trait model.

Run from the repository root after experiments/species_network.py and
experiments/squamate_time.py.
"""

import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evograph import walks

here = Path(__file__).parent


def module(name):
    spec = importlib.util.spec_from_file_location(name, here / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


sn = module("species_network")
sq = sn.sq
OUT = Path("results")
STEPS = 300


def species_states(d):
    return np.c_[d.Presacrals >= 35, d.Digits < 10, d.Digits == 0, d.fossorial.fillna(0) == 1].astype(int)


def walk_matrix(G, X, rho):
    """Row-stochastic transition matrix; rho = None gives the unbiased walk."""
    W = G.toarray()
    if rho is not None:
        tilt = np.exp(0.5 * (X[None, :, :] - X[:, None, :]) @ np.log(rho))  # tilt[i, j] for a step i -> j
        W = W * tilt
    return W / W.sum(1, keepdims=True)


def analyze_walk(P, snake, start):
    n = len(P)
    sigma = walks.stationary(P)
    others = np.setdiff1d(np.arange(n), snake)
    pos = {v: i for i, v in enumerate(others)}
    h = np.linalg.solve(np.eye(len(others)) - P[np.ix_(others, others)], np.ones(len(others)))
    p = np.zeros(n)
    p[start] = 1 / len(start)
    share = []
    for _ in range(STEPS):
        p = p @ P
        share.append(float(p[snake].sum()))
    return {"long_run_share_snake_like": float(sigma[snake].sum()),
            "mean_steps_ordinary_to_snake_like": float(np.mean([h[pos[v]] for v in start])),
            "share_snake_like_after_steps": share}


def trait_model_scenario(burrow_factor):
    """Fitted 12-state model with the burrowing gain rate scaled by burrow_factor."""
    p = json.loads((OUT / f"squamate_fit_params{sq.SUFFIX}.json").read_text())["P_pairwise"]
    gain = np.array(p["gain"], float)
    gain[3] *= burrow_factor
    model = walks.TraitModel(gain, np.full(sq.K, p["loss"]), gamma=1.0, D=np.array(p["Gamma"]), keep=sq.KEEP)
    Q, P, lam = sq.restricted_chain(model)
    sigma = walks.stationary(P)
    target = [i for i, x in enumerate(sq.VALID) if (x & 0b101) == 0b101]
    t = walks.hitting_times(P, target)[0] / lam * 100
    order, burrow_first = sq.routes_to_snake(Q, runs=20000)
    return {"mean_time_ancestor_to_snake_like_Myr": float(t),
            "long_run_share_snake_like": float(sigma[target].sum()),
            "routes": order, "burrowing_before_elongation": burrow_first.get("fossorial first", 0.0)}


def main():
    rng = np.random.default_rng(6850)
    d, age, form = sn.load()
    R, L = sn.knn_graph(age, sn.K_NN, rng), sn.knn_graph(form, sn.K_NN, rng)
    G = ((R + L) > 0).astype(float)
    snake = np.load(OUT / f"species_network_cut{sn.SUFFIX}.npz")["snake"]
    X = species_states(d)
    ordinary = np.flatnonzero(X.sum(1) == 0)
    changes = json.loads((OUT / f"squamate_time{sq.SUFFIX}.json").read_text())["trait_changes_per_history"]
    rho = np.array([changes[t]["gain_to_loss_ratio"] for t in sq.TRAITS])

    results = {"gain_to_loss_ratios": dict(zip(sq.TRAITS, np.round(rho, 2).tolist())),
               "ordinary_lizard_starts": int(len(ordinary)), "snake_like_community": int(len(snake))}
    walks_out = {"unbiased": analyze_walk(walk_matrix(G, X, None), snake, ordinary),
                 "biased_by_evolution": analyze_walk(walk_matrix(G, X, rho), snake, ordinary)}
    scenarios = {}
    for name, f in (("burrowing_disfavoured_4x (soil pollution)", 0.25), ("observed", 1.0),
                    ("burrowing_favoured_4x (surface habitat loss)", 4.0)):
        r = rho.copy()
        r[3] *= f
        scenarios[name] = {"species_network_walk": analyze_walk(walk_matrix(G, X, r), snake, ordinary),
                           "trait_model": trait_model_scenario(f)}
    results["walks"] = {k: {kk: vv for kk, vv in v.items() if kk != "share_snake_like_after_steps"} | {"share_after_100_steps": v["share_snake_like_after_steps"][99]}
                        for k, v in walks_out.items()}
    results["scenarios"] = {k: {"species_network_walk": {kk: vv for kk, vv in v["species_network_walk"].items() if kk != "share_snake_like_after_steps"},
                                "trait_model": v["trait_model"]} for k, v in scenarios.items()}
    (OUT / f"species_walk{sn.SUFFIX}.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(json.dumps(results, indent=2, ensure_ascii=False))
    plot(walks_out, scenarios, rho)


def plot(walks_out, scenarios, rho):
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.2, 3.6), dpi=200, gridspec_kw={"width_ratios": [1, 1.3]})
    fig.patch.set_facecolor("#fcfcfb")
    for ax in (a, b):
        ax.set_facecolor("#fcfcfb")
        ax.grid(color="#e8e7e3", lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=7, colors="#52514e")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#b5b4ae")
    order = np.argsort(rho)
    plain = {"elongate": "elongation", "digits reduced": "digit reduction", "digitless": "losing all digits", "fossorial": "burrowing"}
    a.barh([plain[sq.TRAITS[i]] for i in order], rho[order], color="#2a78d6", height=0.6)
    a.axvline(1, color="#52514e", lw=0.8, ls="--")
    for y, i in enumerate(order):
        a.text(rho[i] + 0.2, y, f"{rho[i]:.1f} : 1", va="center", fontsize=7, color="#52514e")
    a.set_xlim(0, rho.max() * 1.25)
    a.set_xlabel("Gains per loss on the dated tree", fontsize=8, color="#0b0b0b")
    a.set_title("Evolution's direction, trait by trait", fontsize=9, loc="left", color="#0b0b0b")
    steps = np.arange(1, STEPS + 1)
    series = [("Unbiased walk", walks_out["unbiased"]["share_snake_like_after_steps"], "#b5b4ae", "-"),
              ("Biased by evolution", walks_out["biased_by_evolution"]["share_snake_like_after_steps"], "#2a78d6", "-"),
              ("…and burrowing favoured 4×", scenarios["burrowing_favoured_4x (surface habitat loss)"]["species_network_walk"]["share_snake_like_after_steps"], "#104281", "--"),
              ("…and burrowing disfavoured 4×", scenarios["burrowing_disfavoured_4x (soil pollution)"]["species_network_walk"]["share_snake_like_after_steps"], "#86b6ef", "--")]
    for name, s, c, ls in series:
        b.plot(steps, s, color=c, lw=2, ls=ls, label=name)
    b.set_xlabel("Steps of the walk, starting from ordinary lizards", fontsize=8, color="#0b0b0b")
    b.set_ylabel("Share of walkers in the\nsnake-like community", fontsize=8, color="#0b0b0b")
    b.set_title("Walks on the species network", fontsize=9, loc="left", color="#0b0b0b")
    b.set_ylim(0, None)
    b.legend(fontsize=7, frameon=False, loc="center right", bbox_to_anchor=(1, 0.36))
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"species_walk{sn.SUFFIX}.{ext}", facecolor=fig.get_facecolor())


if __name__ == "__main__":
    main()
