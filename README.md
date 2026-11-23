# How Traversable Is the Space of Body Plans?

CS/INFO 6850 (Fall 2026) project by Henry Ji and Calvin Zhang.

We model evolution as a biased random walk on a state graph whose nodes are body plans (combinations of k binary traits) and whose edges are single trait changes, so the base graph is the hypercube H_k. Developmental constraints delete edges at random, selection biases the walk, and a phylogeny records how lineages' walks branched. We ask how traversable this space is through the three parts of the course: random graphs and small worlds, cascades, and spectral analysis with random walks. The full plan is in [docs/roadmap.pdf](docs/roadmap.pdf).

## How the course readings map to the code

| Reading | Results we use | Module |
| --- | --- | --- |
| Random Graphs and Expansion | G(n, p); configuration model; expansion α(G) as the minimum surface-to-volume ratio; expansion implies paths of length O((d/α) log n); hypercubes have expansion exactly 1 | `evograph/statespace.py` |
| Spectral Analysis of Graphs | L = D − A; λ₂ as a Rayleigh quotient; Claim 2.2, 2α ≥ λ₂ ≥ α²/(2Δ) | `evograph/spectral.py` |
| Random Walks | lazy walk M′ = I/2 + M/2; mixing in (2d/λ₂)(1.5 ln n + ln 1/δ) steps; irreducible chains, stationary distribution, return time 1/σ | `evograph/walks.py` |
| Cascading Behavior in Networks (AGT ch. 24) | threshold rule from the coordination game; contagion threshold ≤ 1/2; Linear Threshold and Independent Cascade models; greedy seed selection within 1 − 1/e under submodularity | `evograph/cascades.py` |

`tests/test_readings.py` checks that the code reproduces results stated in these readings, for example that α(H₄) = 1, that H_k's Laplacian has eigenvalues 2j with multiplicity C(k, j), and that the lazy walk on H₆ mixes within its bound.

The biased evolutionary walk is a continuous-time chain turned into a lazy discrete-time chain by uniformization, P = I + Q / (2 · maximum exit rate). With equal rates and no constraints, P is exactly the lazy walk M′ on H_k, so every tool above applies. For the biased chain we use conductance (sets weighted by the stationary distribution) in place of expansion.

## Layout

| Path | Contents |
| --- | --- |
| `evograph/statespace.py` | Trait hypercube, constraint masks, G(n, p), configuration model, distances, exact expansion |
| `evograph/spectral.py` | Laplacian, Fiedler vector, sweep cuts, expansion bounds |
| `evograph/walks.py` | Lazy walk, mixing bound, `TraitModel` (gain and loss rates, coupling γ, dependency graph D), stationary distribution, hitting and return times, integration index |
| `evograph/cascades.py` | Threshold, Linear Threshold and Independent Cascade models; greedy seed selection |
| `evograph/trees.py` | Yule trees, Newick parsing, simulation of a `TraitModel` along a tree, counting independent origins |
| `evograph/fit.py` | Likelihood of tip data on a tree (Felsenstein pruning) and maximum-likelihood fitting of `TraitModel` |
| `data/fetch.py` | Downloads the datasets we can reach automatically into `data/raw/` |
| `experiments/percolation.py` | Part 1, simulation: giant component of the constrained hypercube near p = 1/k |
| `experiments/observed_states.py` | Part 1, data: are observed trait profiles a connected, geodesic region of the hypercube? |
| `experiments/tb_cascades.py` | Part 2: TB drug resistance as a cascade on a dependency graph |
| `experiments/tb_fit.py` | Part 3: fit the biased walk to TB resistance on its tree; hitting times, stationary distribution, bottleneck |
| `experiments/squamate_fit.py` | Squamates: elongation, digit loss and burrowing on a constrained 4-cube; pairwise dependency graph, routes to the snake-like form, bottleneck |
| `experiments/squamate_graph.py` | Slide figure: the squamate body-plan graph with its Cheeger cut around the ancestor |
| `experiments/validate_recovery.py` | Validation: recover planted rates and coupling from data simulated on the real TB tree |
| `results/` | Figures and tables from the experiments |
| `data/README.md` | Datasets and where to get them |
| `docs/` | Roadmap (LaTeX and PDF) |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
python data/fetch.py                       # TB, C4 and bacterial data into data/raw/
python experiments/percolation.py          # seconds
python experiments/observed_states.py      # about a minute
python experiments/tb_cascades.py          # seconds
python experiments/validate_recovery.py    # about 4 minutes
python experiments/squamate_fit.py         # about 5 minutes; needs data/raw/squamate_traits.xlsx (see data/README.md)
python experiments/tb_fit.py               # slow (an hour or more) and being reworked; see Next steps
```

## Quick start

```python
import numpy as np
from evograph import spectral, walks

# Four crab traits: wide carapace, folded abdomen, reinforced shell, sideways gait.
D = np.ones((4, 4)) - np.eye(4)            # each trait helps the others
model = walks.TraitModel(r_gain=[0.5] * 4, r_loss=[1.0] * 4, gamma=0.8, D=D)
P = model.transition_matrix()
sigma = walks.stationary(P)

print("integration index:", walks.integration_index(model))         # > 1: linked traits attract
print("steps to evolve the crab form from 0000:", walks.hitting_times(P, [0b1111])[0])
S, phi = spectral.sweep_cut(walks.flow_graph(P, sigma), measure=sigma)
print("bottleneck conductance:", phi)
```

## Prior work we build on

Inferring how traits accumulate on the hypercube from phylogenetic data is an established method. HyperTraPS (Johnston and Williams 2016, *Cell Systems* 2, 101–111; Greenbury, Barahona and Johnston 2020, *Cell Systems* 10, 39–51), HyperHMM (Moen and Johnston 2023, *Bioinformatics* 39, btac803) and HyperTraPS-CT (Aga et al. 2024, *PLOS Computational Biology* 20, e1012393) infer transition probabilities on hypercubic transition graphs, and Williams et al. (2013, *eLife* 2, e00961) did this for C4 photosynthesis. Our TB and C4 data come from the HyperTraPS-CT repository. What we add is the graph-theory layer from the course (expansion, conductance bottlenecks, cascades, mixing and hitting times) applied to the fitted walk, and a comparison across groups.

## Results so far

| Part | Experiment | Finding | Output |
| --- | --- | --- | --- |
| 1 | Constrained hypercube percolation | The giant component appears at p·k ≈ 1 and sharpens as k grows. At p·k = 2 it holds 81% of body plans, about what G(n, p) gives at the same mean degree (80%) | `results/percolation.png` |
| 1 | Observed TB profiles | 88 of 93 resistance profiles (95%) are linked by single-drug steps, against about 25% for random profiles with the same drug counts (0 of 500 null draws as high) | `results/observed_states.png` |
| 1 | Observed bacterial phenotypes | 73 of 128 possible profiles over 7 traits occur, and paths between them are nearly geodesic: stretch 1.006, against about 1.07 for random profiles (0 of 500 as low) | `results/observed_states.png` |
| 2 | TB dependency cascade | Dependency weights recover known biology unprompted: INH comes before RIF (0.72), AMI and CAP cross-resist, as do the two fluoroquinolones. An INH seed has the largest cascade (6.8 of 10 drugs, Independent Cascade) | `results/tb_cascades.png` |
| 2, 3 | Squamates: snake-like body forms | Traits are strongly coupled. A directed dependency graph beats one shared coupling (p = 5×10⁻¹²). Elongation and complete digit loss each multiply the other's rate at least 400-fold. 99.8% of simulated lineages elongate before losing their last digits. The conductance bottleneck isolates the ancestral body plan: leaving it is the rare step, and once one trait changes the rest tend to follow. Provisional tree, 1,246 species | `results/squamate_fit.png` |
| 3 | TB fit on the tree (partial) | Sister isolates differ on 2.1 of 10 drugs despite tiny branch lengths. Allowing for test misclassification improves the fit enormously (log-likelihood +54, about 11% error per test), and the coupling is positive (γ ≈ 0.65, p < 10⁻⁴ conditional on the other estimates). But with resistance allowed to be lost, rates run to the optimizer's bounds, so we hold off on hitting times and bottlenecks for TB until we fit an irreversible model | `results/tb_fit_partial.json` |
| Validation | Planted models on the TB tree | Fitted γ has median 0.05, 0.54 and 1.12 for planted 0, 0.5 and 1. The likelihood-ratio test rejects γ = 0 in 2/20, 16/20 and 20/20 runs | `results/validate_recovery.png` |

## Milestones

| Date | Deliverable |
| --- | --- |
| Nov 16 | Roadmap ([docs/roadmap.pdf](docs/roadmap.pdf)) |
| Nov 23 | Slides: simulations and a first squamate fit (done: `results/`) |
| Week of Nov 30 | Lightning talk and check-in |
| Dec 15 | Final write-up |

## Next steps

- [x] Part 1: giant component of constrained hypercubes
- [x] Part 1: observed profiles against random-profile nulls (TB, bacteria)
- [x] Part 2: dependency-graph cascades and greedy seeds (TB)
- [x] Validation: recover planted rates and γ on the real TB tree
- [x] Part 3: conductance bottleneck, routes and hitting times for squamates
- [ ] Part 3, TB: refit as irreversible accumulation (no loss) with a test-error rate per drug, as HyperTraPS treats these data, then rerun `experiments/tb_fit.py`'s analysis (hitting times to MDR and XDR, bottleneck)
- [ ] Part 1: expansion and diameter of constrained hypercubes against G(n, p) and configuration-model nulls with the same degrees
- [ ] Part 2: estimate W on the tree (pairwise coupling fits) rather than from distinct profiles, so clonal clusters don't count as independent
- [ ] Part 2: Independent Cascade on a bacterial species network for gene transfer
- [ ] C4: the 37 lineages have no tree and many unknowns; use the Part 1 analysis with unknowns treated as wildcards
- [x] Squamates: first fit on a provisional tree
- [ ] Squamates: rerun on the Zheng and Wiens (2016) tree once `squamate_tree_zw2016.nexus` is downloaded (`experiments/squamate_fit.py` picks it up automatically)
- [ ] Squamates: confidence intervals for Γ (couplings at the bound are lower bounds), and sensitivity to the 35-vertebra threshold
- [ ] Validation: recover planted bottlenecks, not only rates

Course readings are not committed (one is a book chapter). Keep local copies in `readings/`, which is git-ignored.
