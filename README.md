# Where evolution converges and stalls: a squamate species network

CS/INFO 6850 (Structure of Information Networks), Fall 2026. Henry Ji and Calvin Zhang.

Snake-like bodies (long trunks, reduced or absent limbs) evolved dozens of times in lizards. We turn squamate body-form data and a dated phylogeny into a **species network** and use the course's network tools to ask where evolution converges and where it stalls:

- **Small world**: species linked to their closest relatives and closest look-alikes, compared with random graphs.
- **Cheeger cuts**: the network's thinnest boundary, and whether unrelated lineages funnel through it (convergence).
- **Random walks on the dated tree**: how long lineages spend at each step (time bottlenecks), and a walk biased by evolution's observed direction.
- **Drivers and what-ifs**: environmental effects backed out from the tree set the walk's parameters under habitat change.

We do not propose a new evolutionary model; a standard trait model (dependent binary traits on a tree, as in Pagel 1994 and HyperTraPS) is used only to date and orient changes for the network analysis. The 3-page progress writeup is [`docs/writeup.pdf`](docs/writeup.pdf).

## Results so far (95% intervals)

| Finding | Result | Output |
| --- | --- | --- |
| Data | Five sources merged on the Title et al. (2024) tree: 2,403 species with digit and vertebra counts (1,451 lizards). Vertebral counts agree across sources at Spearman 0.99 | `results/squamates_data_quality.json` |
| Small world | Clustering 18× that of a random graph with the same density (15–17× in 80% subsamples), mean path 1.39×; 720 convergence links join look-alikes whose ancestors split over 100 Myr ago | `results/species_network.json`, `results/species_network_bootstrap.json` |
| Convergence bottleneck | One Cheeger cut, conductance 0.003 (0.002–0.004) vs 0.38 for the next best, splits 217 snake-like lizards (15 families, about 30 lineages) from 1,234 limbed ones; 26 bridge species (*Lerista*, *Brachymeles*, *Bachia*...) span it. Same cut in 200 of 200 species resamples, with snakes added, and on the Zheng and Wiens (2016) tree | `results/species_network.png` |
| Time bottleneck | Ordinary lizards wait 510 Myr (447–566) before any change, longer than squamates have existed (206 Myr); once started, a lineage becomes snake-like in 36 Myr (33–41). About 47 independent origins (43–54) | `results/squamate_time.json`, `results/squamate_uncertainty.json` |
| Direction | Every trait is gained more often than lost (1.9–7.1×); a walk tilted by those ratios reaches the snake-like community 2.6× faster than an unbiased walk (2.4–2.8) | `results/species_walk.png` |
| Drivers | Burrowing multiplies elongation and digit-reduction rates about 16× (6–41, 7–33); desert and tropical moist forest do not (all intervals include 1) | `results/drivers.png` |
| What-ifs | If lineages took up burrowing twice as often, snake-like bodies arrive 25% sooner (18–31%); half as often, 24% later (16–36%); desert or forest shifts change it by at most 7% | `results/squamate_whatif.json` |

## Setup and run order

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Download the squamate data by hand into `data/raw/` (see [`data/README.md`](data/README.md)), then run from the repository root:

```bash
python data/build_squamates.py               # merge five sources onto the Title et al. (2024) tree
python experiments/squamate_fit.py           # trait model on 6,885 species (about 15 min)
python experiments/squamate_time.py          # stochastic maps: origins, time at each step
python experiments/species_network.py        # species network: small world, Cheeger cut
python experiments/species_walk.py           # unbiased vs biased walks
python experiments/species_network_bootstrap.py   # error bars for the network
python experiments/squamate_uncertainty.py   # error bars for the trait model, times and walks
python experiments/squamate_drivers.py desert         # environmental drivers (about 40 min each)
python experiments/squamate_drivers.py moist_forest
python experiments/squamate_whatif.py        # burrowing what-ifs
python experiments/drivers_figure.py && python experiments/slide_figures.py && python experiments/species_network_drawing.py && python experiments/squamate_graph.py
```

Robustness: `NETWORK_SPECIES=all` keeps snakes in the network; `SQUAMATE_TREE=zw2016` reruns on the Zheng and Wiens (2016) tree with the original convergence dataset.

## Layout

| Path | Contents |
| --- | --- |
| `evograph/statespace.py` | Trait-state hypercubes, random-graph nulls, expansion |
| `evograph/spectral.py` | Laplacians, Fiedler vectors, sweep (Cheeger) cuts |
| `evograph/walks.py` | Lazy walks, `TraitModel` (gain/loss rates with directed couplings), stationary distributions, hitting times, flow graphs |
| `evograph/trees.py` | Newick and NEXUS parsing, subtrees, simulation on trees |
| `evograph/fit.py` | Tree likelihood (pruning; batched expm or uniformization), maximum-likelihood fitting with couplings, constraints and tip error |
| `data/build_squamates.py` | Merges the trait sources and checks their agreement |
| `experiments/` | One script per analysis, as in the run order above |
| `results/` | JSON summaries and figures from the scripts |
| `docs/` | Roadmap (Nov 16) and the progress writeup |
| `tests/` | Tests of the library against the course readings and of the fitter |

Earlier explorations (TB drug resistance, bacterial phenotypes, percolation on hypercubes) were removed from `main`; they remain on the branch `archive/before-squamate-focus`.

## Prior work

Dependent-trait models on phylogenies: Pagel (1994); HyperTraPS and HyperTraPS-CT (Johnston and Williams 2016; Greenbury et al. 2020; Aga et al. 2024). Snake-like convergence: Bergmann and Morinaga (2019); Camaiti et al. (2022). Stochastic mapping: Nielsen (2002); Huelsenbeck, Nielsen and Bollback (2003). Full references are in the writeup.
