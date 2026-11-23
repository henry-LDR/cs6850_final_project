"""Error bars for the species-network results: resample species.

Each replicate keeps a random 80% of the 1,451 lizards (without replacement, so no
species is duplicated in the network) and recomputes the small-world measures and
the Cheeger convergence bottleneck. We report 2.5%, 50% and 97.5% percentiles, and how
often the replicate's snake-like side matches the full analysis (Jaccard overlap on
the species both contain).

Run from the repository root after experiments/species_network.py.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as cg
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

spec = importlib.util.spec_from_file_location("sn", Path(__file__).with_name("species_network.py"))
sn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sn)
OUT = Path("results")
REPS = 200
KEEP = 0.8


def pct(v):
    return [round(float(q), 4) for q in np.percentile(np.asarray(v, float), [2.5, 50, 97.5])]


def main():
    d, age, form = sn.load()
    n = len(d)
    full_snake = set(np.load(OUT / f"species_network_cut{sn.SUFFIX}.npz")["snake"].tolist())
    rng = np.random.default_rng(6850)
    rec = {k: [] for k in ("conductance", "next_best_conductance", "snake_like_share", "snake_like_families",
                           "snake_like_lineages", "bridge_species_share", "bridge_lineages", "jaccard_with_full_cut",
                           "clustering_ratio", "path_ratio", "convergence_links_per_1000_species")}
    for r in range(REPS):
        idx = np.sort(rng.choice(n, size=int(KEEP * n), replace=False))
        a, f = age[np.ix_(idx, idx)], form[np.ix_(idx, idx)]
        sub = d.iloc[idx].reset_index(drop=True)
        lineage = fcluster(linkage(squareform(a, checks=False), "single"), t=sn.LINEAGE_AGE, criterion="distance")
        W = sn.kernel_graph(f, sn.H)
        nc, lab = cg.connected_components(sp.csr_matrix(W), directed=False)
        giant = np.flatnonzero(lab == np.bincount(lab).argmax())
        s, other, phi = sn.cheeger(W, giant)
        if sub.Digits.iloc[s].median() > sub.Digits.iloc[other].median():  # make s the snake-like side
            s, other = other, s
        _, _, phi2 = sn.cheeger(W, other)
        br = sn.bridge(W, s, other)
        rec["conductance"].append(phi)
        rec["next_best_conductance"].append(phi2)
        rec["snake_like_share"].append(len(s) / len(idx))
        rec["snake_like_families"].append(sub.family.iloc[s].nunique())
        rec["snake_like_lineages"].append(len(set(lineage[s])))
        rec["bridge_species_share"].append(len(br) / len(idx))
        rec["bridge_lineages"].append(len(set(lineage[br])))
        mine = set(idx[s].tolist())
        ref = full_snake & set(idx.tolist())
        rec["jaccard_with_full_cut"].append(len(mine & ref) / len(mine | ref))
        R, L = sn.knn_graph(a, sn.K_NN, rng), sn.knn_graph(f, sn.K_NN, rng)
        G = ((R + L) > 0).astype(float)
        m = int(G.sum() / 2)
        k = len(idx)
        rand = sp.triu(sp.random(k, k, density=2 * m / (k * (k - 1)), random_state=r, data_rvs=np.ones), 1)
        rand = ((rand + rand.T) > 0).astype(float)
        sg, sr = sn.small_world(G), sn.small_world(rand)
        rec["clustering_ratio"].append(sg["clustering"] / sr["clustering"])
        rec["path_ratio"].append(sg["mean_path_length"] / sr["mean_path_length"])
        derived = ((sub.Presacrals >= 35) | (sub.Digits < 10)).to_numpy()
        e = sp.triu(L, 1).tocoo()
        conv = (a[e.row, e.col] > sn.CONVERGENT_AGE) & derived[e.row] & derived[e.col]
        rec["convergence_links_per_1000_species"].append(1000 * conv.sum() / k)
        print(r, {kk: round(float(v[-1]), 4) for kk, v in rec.items()}, flush=True)
    summary = {"replicates": REPS, "species_kept_per_replicate": int(KEEP * n),
               "intervals_95": {k: pct(v) for k, v in rec.items()},
               "share_of_replicates_with_same_cut_jaccard_over_0.9": float(np.mean(np.array(rec["jaccard_with_full_cut"]) > 0.9))}
    (OUT / f"species_network_bootstrap{sn.SUFFIX}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
