"""Part 1: are the body plans we observe a connected, traversable region of the hypercube?

For each dataset, S is the set of distinct trait profiles observed. We take
the subgraph of H_k induced on S (profiles joined when they differ in one
trait) and measure:
  - the share of S in its largest connected component,
  - stretch: mean path length inside that component divided by mean Hamming
    distance for the same pairs (1 means no detours are needed),
  - lambda_2 of that component and the expansion bounds of Claim 2.2.
Two null models give the same number of profiles: uniformly random profiles,
and random profiles with the same count at each number of traits present
(which controls for observed profiles having few or many traits).

Datasets: M. tuberculosis drug resistance (10 traits, 395 isolates) and
bacterial phenotypes (7 traits, 1,545 species with complete data).

Run from the repository root after python data/fetch.py.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp

from evograph import spectral, statespace

OUT = Path("results")
N_NULL = 500
BACTERIA_TRAITS = ["gram_positive", "motile", "spore_forming", "oxygen_tolerant",
                   "rod_shaped", "thermophilic", "host_associated"]
TB_DRUGS = ["INH", "RIF", "PZA", "EMB", "STR", "AMI", "CAP", "MOX", "OFL", "PRO"]


def bacteria_profiles():
    d = pd.read_csv("data/raw/bacteria_traits_gtdb.csv", low_memory=False)

    def binar(col, ones, zeros):
        s = d[col].astype(str).str.strip()
        return np.where(s.isin(ones), 1.0, np.where(s.isin(zeros), 0.0, np.nan))

    shapes_other = ["coccus", "filament", "spiral", "coccobacillus", "vibrio", "pleomorphic", "irregular",
                    "fusiform", "square", "spindle", "star", "branced", "disc"]
    host = d.isolation_source.astype(str).str.startswith("host").astype(float)
    T = pd.DataFrame({
        "gram_positive": binar("gram_stain", ["positive"], ["negative"]),
        "motile": binar("motility", ["yes", "flagella", "gliding", "axial filament"], ["no"]),
        "spore_forming": binar("sporulation", ["yes"], ["no"]),
        "oxygen_tolerant": binar("metabolism", ["aerobic", "obligate aerobic", "facultative", "microaerophilic"],
                                 ["anaerobic", "obligate anaerobic"]),
        "rod_shaped": binar("cell_shape", ["bacillus"], shapes_other),
        "thermophilic": binar("range_tmp", ["thermophilic", "extreme thermophilic", "thermotolerant"],
                              ["mesophilic", "psychrophilic", "psychrotolerant"]),
        "host_associated": np.where(d.isolation_source.isna(), np.nan, host),
    })[BACTERIA_TRAITS].dropna()
    return T.astype(int).to_numpy() @ (1 << np.arange(len(BACTERIA_TRAITS)))


def tb_profiles():
    d = pd.read_csv("data/raw/tb_resistance.csv")
    return d[TB_DRUGS].astype(int).to_numpy() @ (1 << np.arange(len(TB_DRUGS)))


def induced(k, S):
    A = statespace.hypercube(k)
    return A[S][:, S]


def metrics(k, S):
    S = np.sort(np.asarray(S))
    A = induced(k, S)
    labels = statespace.components(A)
    counts = np.bincount(labels)
    giant = np.flatnonzero(labels == counts.argmax())
    G = A[giant][:, giant]
    out = {"profiles": len(S), "components": int(counts.size), "giant_share": counts.max() / len(S)}
    if len(giant) > 1:
        dist = statespace.distances(G)
        x = S[giant]
        ham = np.array([[bin(a ^ b).count("1") for b in x] for a in x])
        off = ~np.eye(len(giant), dtype=bool)
        out["stretch"] = float(dist[off].mean() / ham[off].mean())
        lo, hi = spectral.expansion_bounds(G)
        out["lambda2"] = float(spectral.fiedler(G)[0])
        out["expansion_lower"], out["expansion_upper"] = float(lo), float(hi)
    return out


def null_sets(k, S, rng, matched):
    S = np.asarray(S)
    allx = np.arange(2**k)
    if not matched:
        return rng.choice(2**k, size=len(S), replace=False)
    weight = np.array([bin(x).count("1") for x in allx])
    picks = []
    for w, c in zip(*np.unique([bin(x).count("1") for x in S], return_counts=True)):
        picks.append(rng.choice(allx[weight == w], size=c, replace=False))
    return np.concatenate(picks)


def analyze(name, k, profiles, rng):
    S, counts = np.unique(profiles, return_counts=True)
    obs = metrics(k, S)
    nulls = {}
    for label, matched in (("uniform", False), ("weight_matched", True)):
        draws = [metrics(k, null_sets(k, S, rng, matched)) for _ in range(N_NULL)]
        g = np.array([m["giant_share"] for m in draws])
        st = np.array([m.get("stretch", np.nan) for m in draws])
        nulls[label] = {
            "giant_share_mean": float(g.mean()),
            "giant_share_p_at_least_observed": float((g >= obs["giant_share"] - 1e-12).mean()),
            "stretch_mean": float(np.nanmean(st)),
            "stretch_p_at_most_observed": float((st[~np.isnan(st)] <= obs.get("stretch", np.inf) + 1e-12).mean()),
            "giant_share_draws": g.tolist(),
            "stretch_draws": st[~np.isnan(st)].tolist(),
        }
    top = np.argsort(counts)[::-1][:5]
    return {"dataset": name, "k": k, "observations": int(len(profiles)), "observed": obs, "null": nulls,
            "most_common_profiles": [[int(S[i]), int(counts[i])] for i in top]}


def main():
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(6850)
    results = [
        analyze("M. tuberculosis resistance", len(TB_DRUGS), tb_profiles(), rng),
        analyze("Bacterial phenotypes", len(BACTERIA_TRAITS), bacteria_profiles(), rng),
    ]
    drop = ("giant_share_draws", "stretch_draws")
    slim = [{**r, "null": {k: {kk: vv for kk, vv in v.items() if kk not in drop}
                           for k, v in r["null"].items()}} for r in results]
    (OUT / "observed_states.json").write_text(json.dumps(slim, indent=2))
    print(json.dumps(slim, indent=2))

    # TB: connectivity is the informative statistic. Bacteria (k = 7, 73 of 128
    # profiles): random sets that dense are connected too, so stretch is shown.
    panels = [
        (results[0], "giant_share", "giant_share_draws", np.linspace(0, 1, 26),
         "Share of profiles in the largest connected set"),
        (results[1], "stretch", "stretch_draws", np.linspace(1.0, 1.15, 31),
         "Stretch: path length / Hamming distance"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), dpi=200)
    fig.patch.set_facecolor("#fcfcfb")
    for ax, (r, stat, draws, bins, xlabel) in zip(axes, panels):
        ax.set_facecolor("#fcfcfb")
        ax.hist(r["null"]["weight_matched"][draws], bins=bins, color="#b5b4ae",
                label="Null: random, same trait counts")
        ax.axvline(r["observed"][stat], color="#2a78d6", lw=2, label="Observed profiles")
        ax.set_title(f"{r['dataset']}\n{r['observed']['profiles']} profiles, k = {r['k']} traits",
                     color="#0b0b0b", fontsize=9, loc="left")
        ax.set_xlabel(xlabel, color="#0b0b0b", fontsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#b5b4ae")
        ax.tick_params(colors="#52514e", labelsize=8)
    axes[0].set_ylabel(f"Null draws (of {N_NULL})", color="#0b0b0b", fontsize=8)
    axes[0].legend(frameon=False, fontsize=7, loc="center", bbox_to_anchor=(0.6, 0.6))
    fig.tight_layout()
    fig.savefig(OUT / "observed_states.png", facecolor=fig.get_facecolor())


if __name__ == "__main__":
    main()
