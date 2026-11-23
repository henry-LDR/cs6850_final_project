"""Known-truth check of the trait-model fitter with directed couplings (needs no raw data).

We simulate the fitted 4-trait model (results/squamate_fit_params.json) on a random
3,000-tip tree as deep as the squamate tree, then refit the independent and the
pairwise-coupling models from the tip states alone, exactly as experiments/squamate_fit.py
does. The check: are the couplings recovered, with the right direction and size, and
does the likelihood-ratio test detect them?

Run from the repository root: python experiments/fit_validation.py [seed]
Writes results/fit_validation_<seed>.json.
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import chi2

from evograph import fit, trees, walks

spec = importlib.util.spec_from_file_location("sq", Path(__file__).with_name("squamate_fit.py"))
sq = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sq)


def main(seed):
    p = json.loads(Path("results/squamate_fit_params.json").read_text())["P_pairwise"]
    truth = walks.TraitModel(p["gain"], np.full(sq.K, p["loss"]), gamma=1.0, D=np.array(p["Gamma"]), keep=sq.KEEP)
    parent, length = trees.yule_tree(3000, rng=seed)
    parent = np.asarray(parent)
    length = trees.rescale(np.asarray(length, float), parent, height=2.06)
    state = trees.simulate(truth, parent, length, rng=seed + 100)
    obs = {int(v): [int((state[v] >> i) & 1) for i in range(sq.K)] for v in trees.tips(parent)}
    a = fit.fit_trait_model(sq.K, parent, length, observations=obs, keep=sq.KEEP, fit_gamma=False)
    r = fit.fit_trait_model(sq.K, parent, length, observations=obs, keep=sq.KEEP, pairwise=sq.FREE,
                            x0=np.r_[a["theta"], np.zeros(int(sq.FREE.sum()))])
    G_true, G_fit = np.array(p["Gamma"]), np.array(r["Gamma"])
    out = {"seed": seed, "tips": len(obs),
           "tips_ordinary": int(sum(state[v] == 0 for v in obs)), "tips_snake_like": int(sum((state[v] & 5) == 5 for v in obs)),
           "loglik_independent": a["loglik"], "loglik_pairwise": r["loglik"], "converged": r["converged"],
           "lr_p_value": float(chi2.sf(2 * (r["loglik"] - a["loglik"]), int(sq.FREE.sum()))),
           "loglik_at_truth": float(fit.log_likelihood(truth, parent, length, {v: fit.tip_vector(o, sq.K) for v, o in obs.items()})),
           "gain_true": p["gain"], "gain_fit": np.asarray(r["gain"]).round(4).tolist(), "loss_true": p["loss"], "loss_fit": r["loss"],
           "Gamma": {f"{sq.TRAITS[j]} -> {sq.TRAITS[i]}": {"true": round(float(G_true[j, i]), 2), "fit": round(float(G_fit[j, i]), 2)}
                     for j in range(sq.K) for i in range(sq.K) if sq.FREE[j, i]}}
    Path(f"results/fit_validation_{seed}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
