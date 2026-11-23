"""Squamate species network: small-world structure and convergence bottlenecks.

Nodes: the 918 squamate species with both digit and presacral-vertebra counts that
sit on the Zheng and Wiens (2016) tree.

Two kinds of links:
  relatives    each species to its 6 closest relatives on the tree (most recent
               common ancestor), made symmetric
  look-alikes  each species to its 6 most similar body forms (digits and log
               vertebra count, each standardized), made symmetric
A look-alike link between species whose common ancestor is over 100 million years
old, with both species derived (elongate or digit-reduced), is a convergence link.

Small world: relatives alone behave like a lattice (clustered, but split into
pieces with long paths); look-alike links act as shortcuts across the tree.

Convergence bottlenecks: on the body-form graph (Gaussian similarity
w = exp(-d^2 / 2h^2), the standard spectral-clustering construction) we take the
Cheeger sweep cut. We then use the tree to ask how many unrelated lineages are on
each side and which species bridge the cut.

Run from the repository root after python data/fetch.py (and the hand-downloaded
squamate files, see data/README.md).
"""

import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.csgraph as cg
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from evograph import spectral, trees

spec = importlib.util.spec_from_file_location("sq", Path(__file__).with_name("squamate_fit.py"))
sq = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sq)

K_NN = 6
H = 0.3  # kernel bandwidth in standardized units; the smallest round value that connects all species
CONVERGENT_AGE = 100  # Myr
LINEAGE_AGE = 50  # Myr: species whose ancestry splits longer ago count as separate lineages
OUT = Path("results")


def load():
    parent, length, names, _ = sq.load_tree()
    length = np.asarray(length) * 100  # Myr
    d = pd.read_excel(sq.RAW / "squamate_traits.xlsx", sheet_name="Dataset")
    node = {names[v]: v for v in trees.tips(parent)}
    d["tip"] = [sq.tip_name(s, c, node) for s, c in zip(d.Species, d["Changes to Zheng & Wiens (2016)"])]
    for c in ("Digits", "Presacrals"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d[d.tip.apply(lambda x: isinstance(x, str)) & d.Digits.notna() & d.Presacrals.notna()]
    d = d.drop_duplicates("tip").reset_index(drop=True)
    # age of the most recent common ancestor of every pair: shared path length from the root
    rows, cols = [], []
    for r, v in enumerate(node[t] for t in d.tip):
        while parent[v] >= 0:
            rows.append(r)
            cols.append(v)
            v = parent[v]
    A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(d), len(parent)))
    shared = (A @ sp.diags(length) @ A.T).toarray()
    height = shared.diagonal().max()
    age = height - shared
    np.fill_diagonal(age, 0)
    F = np.c_[d.Digits, np.log(d.Presacrals)]
    Z = (F - F.mean(0)) / F.std(0)
    form = np.sqrt(((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1))
    return d, age, form


def knn_graph(dist, k, rng):
    n = len(dist)
    dd = dist + rng.random(dist.shape) * 1e-9  # random tie-breaks among identical forms
    np.fill_diagonal(dd, np.inf)
    nb = np.argsort(dd, axis=1)[:, :k]
    A = sp.csr_matrix((np.ones(n * k), (np.repeat(np.arange(n), k), nb.ravel())), shape=(n, n))
    return ((A + A.T) > 0).astype(float)


def small_world(A):
    A = sp.csr_matrix(A)
    ncomp, lab = cg.connected_components(A, directed=False)
    idx = np.flatnonzero(lab == np.bincount(lab).argmax())
    B = A[idx][:, idx]
    dist = cg.shortest_path(B, unweighted=True, directed=False)
    deg = np.asarray(B.sum(1)).ravel()
    tri = np.asarray((B @ B).multiply(B).sum(1)).ravel() / 2
    clustering = np.mean(np.where(deg > 1, tri / np.maximum(deg * (deg - 1) / 2, 1), 0))
    return {"edges": int(A.sum() / 2), "components": int(ncomp), "giant": int(len(idx)),
            "mean_path_length": float(dist[np.triu_indices(len(idx), 1)].mean()), "clustering": float(clustering)}


def kernel_graph(form, h):
    W = np.exp(-form**2 / (2 * h**2))
    W[form > 3 * h] = 0
    np.fill_diagonal(W, 0)
    return W


def cheeger(W, idx):
    B = sp.csr_matrix(W[np.ix_(idx, idx)])
    m = np.asarray(B.sum(1)).ravel()
    S, phi = spectral.sweep_cut(B, measure=m)
    inS = np.zeros(len(idx), bool)
    inS[np.asarray(S)] = True
    small = idx[inS] if inS.sum() <= len(idx) / 2 else idx[~inS]
    return small, np.setdiff1d(idx, small), float(phi)


def bridge(W, a, b, share=0.8):
    """Species on each side that together carry `share` of the weight across the cut."""
    Wc = W[np.ix_(a, b)]
    out = []
    for ws, ids in ((Wc.sum(1), a), (Wc.sum(0), b)):
        o = np.argsort(ws)[::-1]
        out += list(ids[o[: np.searchsorted(np.cumsum(ws[o]) / ws.sum(), share) + 1]])
    return np.array(out)


def describe(d, idx, lineage):
    s = d.iloc[idx]
    return {"species": int(len(idx)),
            "digits_range": [int(s.Digits.min()), int(s.Digits.max())], "digits_median": float(s.Digits.median()),
            "presacrals_range": [int(s.Presacrals.min()), int(s.Presacrals.max())], "presacrals_median": float(s.Presacrals.median()),
            "fossorial_share": round(float((s["Habitat (4)"] == "Fossorial").mean()), 3),
            "clades": int(s["Clade (14)"].nunique()), "clade_counts": s["Clade (14)"].value_counts().to_dict(),
            "lineages_split_over_50_Myr": int(len(set(lineage[idx])))}


def main():
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(6850)
    d, age, form = load()
    n = len(d)
    lineage = fcluster(linkage(squareform(age, checks=False), "single"), t=LINEAGE_AGE, criterion="distance")

    R, L = knn_graph(age, K_NN, rng), knn_graph(form, K_NN, rng)
    G = ((R + L) > 0).astype(float)
    m = int(G.sum() / 2)
    rand = sp.triu(sp.random(n, n, density=2 * m / (n * (n - 1)), random_state=1, data_rvs=np.ones), 1)
    rand = ((rand + rand.T) > 0).astype(float)
    sw = {"relatives_only": small_world(R), "look_alikes_only": small_world(L), "combined": small_world(G),
          "random_same_density": small_world(rand)}
    derived = ((d.Presacrals >= 35) | (d.Digits < 10)).to_numpy()
    e = sp.triu(L, 1).tocoo()
    ea = age[e.row, e.col]
    conv = (ea > CONVERGENT_AGE) & derived[e.row] & derived[e.col]
    sw["convergence_links"] = {"look_alike_links": int(len(ea)), "between_relatives_over_100_Myr_apart": int((ea > CONVERGENT_AGE).sum()),
                               "of_which_both_species_derived": int(conv.sum()),
                               "their_common_ancestor_age_Myr_median": float(np.median(ea[conv])) if conv.any() else None,
                               "derived_species": int(derived.sum())}

    W = kernel_graph(form, H)
    snake_side, limbed, phi = cheeger(W, np.arange(n))
    br = bridge(W, snake_side, limbed)
    inner_small, inner_rest, phi2 = cheeger(W, snake_side)
    lsmall, lrest, phi_limbed = cheeger(W, limbed)
    robustness = {}
    for h in (0.2, 0.25, 0.3, 0.4, 0.5):
        Wh = kernel_graph(form, h)
        nc, lab = cg.connected_components(sp.csr_matrix(Wh), directed=False)
        idx = np.flatnonzero(lab == np.bincount(lab).argmax())
        a, b, p = cheeger(Wh, idx)
        robustness[str(h)] = {"conductance": round(p, 5), "small_side": int(len(a)),
                              "small_side_digits_max": int(d.Digits.iloc[a].max()),
                              "small_side_clades": int(d["Clade (14)"].iloc[a].nunique()),
                              "species_in_graph": int(len(idx))}

    summary = {
        "species": n,
        "small_world": sw,
        "convergence_bottleneck": {
            "conductance": phi,
            "next_best_cut_among_limbed_conductance": phi_limbed,
            "snake_like_side": describe(d, snake_side, lineage),
            "limbed_side": describe(d, limbed, lineage),
            "bridge_species": describe(d, br, lineage) | {"names": d.Species.iloc[br].tolist()},
            "second_cut_inside_snake_like_side": {"conductance": phi2, "small_side": describe(d, inner_small, lineage)},
        },
        "bandwidth_robustness": robustness,
    }
    (OUT / "species_network.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    np.savez(OUT / "species_network_cut.npz", snake=snake_side, bridge=br)
    plot(d, snake_side, br, phi, phi_limbed, describe(d, snake_side, lineage), describe(d, br, lineage))


def plot(d, snake_side, br, phi, phi_limbed, snake_desc, bridge_desc):
    rng = np.random.default_rng(1)
    x = d.Presacrals.to_numpy() * np.exp(rng.uniform(-0.03, 0.03, len(d)))
    y = d.Digits.to_numpy() + rng.uniform(-0.3, 0.3, len(d))
    side = np.zeros(len(d), int)
    side[snake_side] = 1
    is_bridge = np.zeros(len(d), bool)
    is_bridge[br] = True
    fig, ax = plt.subplots(figsize=(7.6, 4.4), dpi=200)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.scatter(x[side == 0], y[side == 0], s=9, color="#b5b4ae", alpha=0.6, lw=0, label=f"Limbed lizards ({(side == 0).sum()})")
    ax.scatter(x[side == 1], y[side == 1], s=11, color="#2a78d6", alpha=0.8, lw=0,
               label=f"Snake-like community ({snake_desc['species']} species, {snake_desc['clades']} clades)")
    ax.scatter(x[is_bridge], y[is_bridge], s=34, facecolor="none", edgecolor="#e34948", lw=1.2,
               label=f"Bridge species ({bridge_desc['species']}, from {bridge_desc['lineages_split_over_50_Myr']} lineages)")
    ax.set_xscale("log")
    ax.set_xticks([15, 25, 40, 60, 100, 200], ["15", "25", "40", "60", "100", "200"])
    ax.minorticks_off()
    ax.set_yticks(range(0, 11, 2))
    ax.set_xlabel("Presacral vertebrae (body elongation)", fontsize=8, color="#0b0b0b")
    ax.set_ylabel("Digits", fontsize=8, color="#0b0b0b")
    ax.set_title(f"Species network: one sharp bottleneck (Cheeger conductance {phi:.3f}; next best {phi_limbed:.2f})",
                 fontsize=9, loc="left", color="#0b0b0b")
    ax.grid(color="#e8e7e3", lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=7, colors="#52514e")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#b5b4ae")
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"species_network.{ext}", facecolor=fig.get_facecolor())


if __name__ == "__main__":
    main()
