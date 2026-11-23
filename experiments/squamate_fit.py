"""Squamates: how lizards became snake-like, as a biased walk on a constrained hypercube.

Data: data/processed/squamates_merged.csv (built by data/build_squamates.py from five
sources on the Title et al. 2024 tree; 6,885 species with at least one trait). Four
binary traits per species, unknown values allowed:

    bit 0  elongate        at least 35 presacral vertebrae (fully digited species
                           average 26, sd 4.6; no digitless species has fewer than 37)
    bit 1  digits reduced  fewer than 10 digits
    bit 2  digitless       0 digits
    bit 3  fossorial       burrowing habitat

Digitless implies digits reduced, so the 4 states that are digitless but not
reduced are removed: the state graph is the 4-cube with those nodes deleted,
12 body plans. The root (the squamate ancestor) is 0000, an ordinary lizard.

Tree: Title et al. (2024), branch lengths in units of 100 million years.
SQUAMATE_TREE=zw2016 reruns on the Zheng and Wiens (2016) tree with the original
convergence dataset alone (robustness check; results get a _zw2016 suffix).

Models (nested, compared by likelihood-ratio tests):
  A  independent traits: 4 gain rates, 1 shared loss rate
  C  one coupling gamma on the all-to-all dependency graph
  P  pairwise coupling Gamma[j, i]: holding trait j multiplies trait i's gain rate by
     exp(Gamma[j, i]), a fitted, directed trait-dependency graph (10 identifiable weights)

On the best model we ask: which routes lead from 0000 to the snake-like form
(elongate and digitless), how long that takes, where the stationary
distribution sits, and where the conductance bottleneck of the walk is.

Run from the repository root after python data/build_squamates.py.
"""

import itertools
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import chi2

from evograph import fit, spectral, trees, walks

TRAITS = ["elongate", "digits reduced", "digitless", "fossorial"]
SHORT = ["E", "R", "Z", "F"]
K = 4
RAW = Path("data/raw")
OUT = Path("results")


def valid(x):
    return not ((x >> 2) & 1 and not (x >> 1) & 1)  # digitless needs digits reduced


VALID = [x for x in range(2**K) if valid(x)]
# Couplings the data cannot identify: digitless is only ever gained while digits are reduced,
# so reduced -> digitless is confounded with digitless's own gain rate, and
# digitless -> reduced never applies. Both stay at 0, leaving 10 free couplings.
FREE = ~np.eye(K, dtype=bool)
FREE[1, 2] = FREE[2, 1] = False
KEEP = np.array([[valid(x ^ (1 << i)) and valid(x) for i in range(K)] for x in range(2**K)])


# Which tree: the Title et al. (2024) time tree (default, with the merged trait data from
# data/build_squamates.py) or "zw2016" (the Zheng and Wiens 2016 tree with the original
# convergence dataset alone), for robustness checks. Results for zw2016 get a suffix.
TREE = os.environ.get("SQUAMATE_TREE", "title2024")
SUFFIX = "" if TREE == "title2024" else f"_{TREE}"


def load_tree():
    if TREE == "title2024":
        parent, length, names = trees.read_newick((RAW / "squamate_tree_title2024.tre").read_text())
        return parent, np.asarray(length) / 100.0, names, "Title et al. (2024)"
    parent, length, names = trees.read_nexus((RAW / "squamate_tree_zw2016.nexus").read_text())
    # Keep Lepidosauria (tuatara + squamates); the file also has turtles, birds and crocodilians
    tip = {names[v]: v for v in trees.tips(parent)}
    root = trees.mrca(parent, [tip["Sphenodon_punctatus"], tip["Dibamus_bourreti"], tip["Iguana_iguana"]])
    parent, length, names = trees.subtree(parent, length, names, root)
    return parent, np.asarray(length) / 100.0, names, "Zheng and Wiens (2016)"


def load_species(node):
    """One row per tree tip with data: digits, presacrals, fossorial (0/1), family, snake."""
    if TREE == "title2024":
        d = pd.read_csv(Path("data/processed/squamates_merged.csv"))
        return d[d.tip.isin(node)].reset_index(drop=True)
    d = pd.read_excel(RAW / "squamate_traits.xlsx", sheet_name="Dataset")
    d["tip"] = [tip_name(s, c, node) for s, c in zip(d.Species, d["Changes to Zheng & Wiens (2016)"])]
    d = d[d.tip.apply(lambda x: isinstance(x, str))].drop_duplicates("tip").reset_index(drop=True)
    d["digits"] = pd.to_numeric(d.Digits, errors="coerce")
    d["presacrals"] = pd.to_numeric(d.Presacrals, errors="coerce")
    d["fossorial"] = (d["Habitat (4)"] == "Fossorial").astype(float).where(d["Habitat (4)"].notna())
    d["family"] = d["Clade (14)"]
    d["snake"] = d["Clade (14)"] == "Serpentes"
    return d


def code(d):
    col = lambda s, f: np.where(s.isna(), 2, f(s.fillna(0)).astype(int))
    return np.c_[
        col(d.presacrals, lambda s: s >= 35),
        col(d.digits, lambda s: s < 10),
        col(d.digits, lambda s: s == 0),
        col(d.fossorial, lambda s: s == 1),
    ]


def load():
    parent, length, names, source = load_tree()
    node = {names[v]: v for v in trees.tips(parent)}
    d = load_species(node)
    X = code(d)
    obs = {node[s]: X[i].tolist() for i, s in enumerate(d.tip)}
    return parent, length, obs, d, X, source


def tip_name(species, change, node):
    """Tree tip for a species: itself, or the stand-in the dataset says it used ('Used Genus species')."""
    if species in node:
        return species
    if isinstance(change, str) and change.startswith("Used "):
        alt = change[5:].strip().replace(" ", "_")
        return alt if alt in node else None
    return None


def run_fit(spec):
    kind, x0 = spec
    parent, length, obs, *_ = load()
    r = fit.fit_trait_model(K, parent, length, observations=obs, keep=KEEP, x0=x0,
                            fit_gamma=(kind == "C"), pairwise=FREE if kind == "P" else False)
    return {key: (np.asarray(r[key]).tolist() if key in ("gain", "theta") else r[key])
            for key in ("gain", "loss", "gamma", "Gamma", "theta", "loglik", "n_params", "converged")}


def label(x):
    return "".join(SHORT[i] for i in range(K) if (x >> i) & 1) or "0"


def restricted_chain(model):
    """Generator and lazy chain on the 12 valid body plans."""
    Q = model.generator().toarray()[np.ix_(VALID, VALID)]
    lam = 2 * (-np.diag(Q)).max()
    P = sp.csr_matrix(np.eye(len(VALID)) + Q / lam)
    return Q, P, lam


def routes_to_snake(Q, start=0, runs=40000, seed=0):
    """Simulate the embedded jump chain from start until it is elongate and digitless."""
    rng = np.random.default_rng(seed)
    idx = {x: i for i, x in enumerate(VALID)}
    target = lambda x: (x & 0b101) == 0b101
    jump = Q.copy()
    np.fill_diagonal(jump, 0)
    jump = jump / jump.sum(axis=1, keepdims=True)
    order, first_gain = {}, {}
    for _ in range(runs):
        x, seen = start, []
        while not target(x):
            y = VALID[rng.choice(len(VALID), p=jump[idx[x]])]
            gained = (y & ~x)
            if gained and gained.bit_length() - 1 not in seen:
                seen.append(gained.bit_length() - 1)
            x = y
        key = " → ".join(TRAITS[i] for i in seen if i in (0, 1, 2))  # body-plan order
        order[key] = order.get(key, 0) + 1
        f = "fossorial first" if 3 in seen and seen.index(3) < seen.index(0) else "elongate first"
        first_gain[f] = first_gain.get(f, 0) + 1
    norm = lambda c: {k: round(v / runs, 3) for k, v in sorted(c.items(), key=lambda kv: -kv[1])}
    return norm(order), norm(first_gain)


def main(reuse):
    OUT.mkdir(exist_ok=True)
    parent, length, obs, d, X, source = load()
    params = OUT / f"squamate_fit_params{SUFFIX}.json"
    if reuse and params.exists():
        saved = json.loads(params.read_text())
    else:
        a = run_fit(("A", None))
        ta = np.asarray(a["theta"])
        with ProcessPoolExecutor(2) as pool:
            c, p = pool.map(run_fit, [("C", np.r_[ta, 0.0]), ("P", np.r_[ta, np.zeros(int(FREE.sum()))])])
        saved = {"A_independent": a, "C_one_gamma": c, "P_pairwise": p}
        params.write_text(json.dumps(saved, indent=2))
    a, c, p = saved["A_independent"], saved["C_one_gamma"], saved["P_pairwise"]
    for name, r in saved.items():
        print(name, round(r["loglik"], 2), "converged", r["converged"])

    lr_c = max(0.0, 2 * (c["loglik"] - a["loglik"]))
    lr_p = max(0.0, 2 * (p["loglik"] - a["loglik"]))
    aic = {name: 2 * r["n_params"] - 2 * r["loglik"] for name, r in saved.items()}
    G = np.array(p["Gamma"])
    model = walks.TraitModel(p["gain"], np.full(K, p["loss"]), gamma=1.0, D=G, keep=KEEP)

    Q, P, lam = restricted_chain(model)
    sigma = walks.stationary(P)
    snake = [i for i, x in enumerate(VALID) if (x & 0b101) == 0b101]
    t_snake = walks.hitting_times(P, snake)[0] / lam * 100  # Myr
    order, fossorial_first = routes_to_snake(Q)
    first = Q[0].copy()
    first[0] = 0
    first = first / first.sum()
    S, phi = spectral.sweep_cut(walks.flow_graph(P, sigma), measure=sigma)
    inS = set(int(i) for i in S)
    side = [VALID[i] for i in range(len(VALID)) if i in inS]
    other = [VALID[i] for i in range(len(VALID)) if i not in inS]

    known = (X != 2).all(axis=1)
    summary = {
        "tree": source,
        "species_on_tree": int(len(obs)),
        "species_fully_coded": int(known.sum()),
        "trait_frequencies": {TRAITS[i]: round(float((X[:, i][X[:, i] != 2]).mean()), 3) for i in range(K)},
        "loglik": {k: round(v["loglik"], 3) for k, v in saved.items()},
        "aic": {k: round(v, 2) for k, v in aic.items()},
        "one_gamma": c["gamma"],
        "one_gamma_p_value": float(chi2.sf(lr_c, 1)),
        "pairwise_p_value_vs_independent": float(chi2.sf(lr_p, int(FREE.sum()))),
        "pairwise_p_value_vs_one_gamma": float(chi2.sf(max(0.0, 2 * (p["loglik"] - c["loglik"])), int(FREE.sum()) - 1)),
        "couplings_at_bound": [f"{TRAITS[j]} -> {TRAITS[i]}" for j in range(K) for i in range(K) if FREE[j, i] and abs(G[j, i]) > 5.9],
        "gain_per_100_Myr": dict(zip(TRAITS, np.round(p["gain"], 4).tolist())),
        "loss_per_100_Myr": p["loss"],
        "Gamma_from_row_to_column": {TRAITS[j]: dict(zip(TRAITS, np.round(G[j], 2).tolist())) for j in range(K)},
        "first_step_from_ancestor": {label(VALID[i]): round(float(first[i]), 3) for i in np.argsort(first)[::-1] if first[i] > 0},
        "routes_to_snake_like": order,
        "fossorial_before_elongate": fossorial_first,
        "mean_time_to_snake_like_Myr": float(t_snake),
        "stationary_top": {label(VALID[i]): round(float(sigma[i]), 3) for i in np.argsort(sigma)[::-1][:5]},
        "bottleneck_conductance": float(phi),
        "bottleneck_side": [label(x) for x in side],
        "bottleneck_other_side": [label(x) for x in other],
    }
    (OUT / f"squamate_summary{SUFFIX}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    plot(G, order, p["loglik"] - a["loglik"])


def plot(G, order, gain):
    from matplotlib.colors import LinearSegmentedColormap

    fig, (a, b) = plt.subplots(1, 2, figsize=(9.2, 3.8), dpi=200, gridspec_kw={"width_ratios": [1, 1.25]})
    fig.patch.set_facecolor("#fcfcfb")
    for ax in (a, b):
        ax.set_facecolor("#fcfcfb")
    div = LinearSegmentedColormap.from_list("div", ["#e34948", "#f3b5b0", "#f0efec", "#86b6ef", "#104281"])
    lim = max(1.0, np.abs(G).max())
    show = np.where(FREE, G, np.nan)
    im = a.imshow(show, cmap=div, vmin=-lim, vmax=lim)
    for j in range(K):
        for i in range(K):
            if FREE[j, i]:
                txt = ("≥ +6" if G[j, i] > 5.9 else "≤ −6" if G[j, i] < -5.9 else f"{G[j, i]:+.1f}")
                a.text(i, j, txt, ha="center", va="center", fontsize=7,
                       color="#fcfcfb" if abs(G[j, i]) > 0.6 * lim else "#0b0b0b")
            elif i != j:
                a.text(i, j, "fixed", ha="center", va="center", fontsize=6, color="#8a8984")
    a.set_xticks(range(K), TRAITS, rotation=30, ha="right", fontsize=7, color="#52514e")
    a.set_yticks(range(K), TRAITS, fontsize=7, color="#52514e")
    a.set_xlabel("…changes the gain rate of this trait", fontsize=8, color="#0b0b0b")
    a.set_ylabel("Holding this trait…", fontsize=8, color="#0b0b0b")
    a.set_title("Fitted dependency graph Γ (log rate multiplier)", fontsize=9, loc="left", color="#0b0b0b")
    for s in a.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=a, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=7, colors="#52514e")
    cb.outline.set_visible(False)

    top = list(order.items())[:5]
    names = [k.replace(" → ", " → \n", 0) for k, _ in top][::-1]
    b.barh(range(len(top)), [v for _, v in top][::-1], color="#2a78d6", height=0.6)
    b.set_yticks(range(len(top)), names, fontsize=7, color="#52514e")
    for y, (_, v) in enumerate(top[::-1]):
        b.text(v + 0.01, y, f"{100 * v:.0f}%" if v >= 0.01 else f"{100 * v:.1f}%", va="center", fontsize=7, color="#52514e")
    b.set_xlim(0, max(v for _, v in top) * 1.2)
    b.set_xlabel("Share of simulated lineages", fontsize=8, color="#0b0b0b")
    b.set_title("Order of body-plan changes on the way to snake-like", fontsize=9, loc="left", color="#0b0b0b")
    b.grid(axis="x", color="#e8e7e3", lw=0.6)
    b.set_axisbelow(True)
    b.tick_params(axis="x", labelsize=7, colors="#52514e")
    for s in ("top", "right"):
        b.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        b.spines[s].set_color("#b5b4ae")
    fig.tight_layout()
    fig.savefig(OUT / f"squamate_fit{SUFFIX}.png", facecolor=fig.get_facecolor())


if __name__ == "__main__":
    main(reuse="--reuse" in sys.argv)
