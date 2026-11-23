# Data

Raw downloads go in `data/raw/` and cleaned tables in `data/processed/`; both are git-ignored, so record every download here.

`python data/fetch.py` downloads everything marked **fetched** below, pinned to fixed commits so results are reproducible. Our shared compute can only reach GitHub, so datasets hosted on Dryad, figshare or journal sites have to be downloaded by hand into `data/raw/`.

## In use now

| Group | File in `data/raw/` | What it holds | Source |
| --- | --- | --- | --- |
| *M. tuberculosis* (first real fit) | `tb_tree.nwk`, `tb_resistance.csv` | Phylogeny of 1,000+ isolates; resistance to 10 drugs (INH, RIF, PZA, EMB, STR, AMI, CAP, MOX, OFL, PRO) for 395 of them, 93 distinct profiles | Casali et al. (2014), *Nature Genetics* 46, 279–286, as curated in the [HyperTraPS-CT repository](https://github.com/StochasticBiology/hypertraps-ct) — **fetched** |
| C4 plants | `c4_traits.csv`, `c4_trait_names.txt` | 16 C4 traits for 37 lineages, coded 0, 1, or 2 for unknown; no tree | Williams et al. (2013), *eLife* 2, e00961, as curated in the HyperTraPS-CT repository — **fetched** |
| Squamates | `squamate_traits.xlsx` | Digits, phalanges, presacral vertebrae and habitat for 1,386 species, coded on the Zheng and Wiens (2016) tree | Snake-like convergence dataset, [Dryad doi:10.5061/dryad.crjdfn3g7](https://doi.org/10.5061/dryad.crjdfn3g7) (2025) — **download by hand**, save `Snake-like_Convergence_Dataset.xlsx` under this name |
| Squamates | `squamate_tree_zw2016.nexus` | Zheng and Wiens (2016) time tree, 4,162 species | `Dataset_S2_zw2016.nexus` in [Dryad doi:10.5061/dryad.0zpc8671s](https://doi.org/10.5061/dryad.0zpc8671s) — **download by hand**. (Dryad doi:10.5061/dryad.tv055 holds only the alignment.) |
| Squamates (until the above is in place) | `squamate_tree_provisional.tre` | A 4,162-tip squamate time tree; its crown age (174 Ma) differs from Zheng and Wiens (205 Ma), so it is a different published tree, probably Pyron and Burbrink (2014). 1,246 of the 1,386 species match | `datasets/squa.tre` in [github.com/uyedaj/bmr](https://github.com/uyedaj/bmr) — **fetched** |
| Bacteria | `bacteria_traits_gtdb.csv` | 11,265 species with phenotypes and GTDB taxonomy | [Madin et al. (2020), *Scientific Data* 7, 170](https://github.com/bacteria-archaea-traits/bacteria-archaea-traits) — **fetched** |

## Squamate data, merged (used by every squamate experiment)

`python data/build_squamates.py` merges the sources below onto the Title et al. (2024) tree and writes `data/processed/squamates_merged.csv` plus `data/processed/squamates_quality.json`, which records how the sources agree where they overlap. All of these are **downloaded by hand** into `data/raw/` under the names shown.

| File in `data/raw/` | What it holds | Source |
| --- | --- | --- |
| `squamate_tree_title2024.tre` | Time tree of 6,885 squamates (main tree) | Title et al. (2024), shipped in the vertebral dataset below |
| `squamate_vertebrae_2357.csv` | Presacral vertebrae for 2,364 species (radiographs and literature), body length, elongation index, habitat | Dryad doi:10.5061/dryad.4qrfj6qr1 (`Evolution-vertebrae-data-files.zip`) |
| `squambase_1.xlsx` | Limb status, substrate, family and Zheng and Wiens names for all 11,744 squamates | SquamBase 1.0, Meiri (2024), Dryad doi:10.5061/dryad.76hdr7t3b (`Supplementary_Table_S1`) |
| `camaiti2022_s2_morphology.xlsx`, `camaiti2022_s3_raw.xlsx` | Fingers, toes, presacrals and limb proportions of limb-reduced skinks and relatives | Camaiti et al. (2022), *J. Biogeography* 49:1397, doi:10.1111/jbi.14392, Appendices S2 and S3 |
| `bergmann_morinaga2019.csv` | Fingers, toes, presacrals, body and limb lengths for 231 species | Bergmann and Morinaga (2019), Dryad doi:10.5061/dryad.5rh14nc |
| `squamate_traits.xlsx` | Digits, presacrals, habitat for 1,386 species (see above) | Dryad doi:10.5061/dryad.crjdfn3g7 |
| `squamate_tree_zw2016.nexus` | Zheng and Wiens (2016) time tree (robustness checks, `SQUAMATE_TREE=zw2016`) | `Dataset_S2_zw2016.nexus` in Dryad doi:10.5061/dryad.0zpc8671s |

Agreement where sources overlap: vertebral counts, convergence dataset vs radiographs, Spearman 0.99 over 1,052 species (median difference 0); digit counts, convergence dataset vs Camaiti et al., Spearman 0.98 over 240 species. Where no source counts digits, SquamBase's limb status fills in: "4 well developed limbs" → 10 digits (96% exactly 10 and all at least 8 in the counted overlap), "limbless" → 0 for snakes only (13% of lizards SquamBase calls limbless, mostly pygopodids, keep vestigial digits). Result: 2,403 species with both digit and vertebra counts (1,451 lizards, 952 snakes), up from 911.

## Planned

| Group | What we use | Size | Source |
| --- | --- | --- | --- |
| Squamates (main test) | Snake-like body form, digit and limb counts, phylogeny | About 25 origins among 261 species | Wiens, Brandley and Reeder (2006), *Evolution* 60, 123–141; body-part data for six clades in [Bergmann and Morinaga (2019), Dryad](https://datadryad.org/dataset/doi:10.5061/dryad.5rh14nc) |
| C4 plants | 16 C4 traits, presence or absence | 73 species in 18 lineages, so 2¹⁶ = 65,536 states | [Williams et al. (2013), *eLife* 2, e00961](https://elifesciences.org/articles/00961), supplementary data |
| Decapods (headline example) | Crab body-plan characters, coded by hand | At least five origins and seven losses of the crab form | Wolfe, Luque and Bracken-Grissom (2021), *BioEssays* 43, 2100020 |
| Bacteria (cascades) | 14 phenotypic and 5 genomic traits | About 15,000 species-level records | [Madin et al. (2020), figshare](https://doi.org/10.6084/m9.figshare.c.4843290.v3) |
| Bacteria (tree) | Genome taxonomy | 901,341 genomes in 199,923 species clusters | [GTDB release R232](https://data.gtdb.aau.ecogenomic.org/releases/release232/232.0/RELEASE_NOTES.txt) |
| Backbone | Synthetic tree of life | 2,385,875 tips, 159,925 placed by phylogeny | [Open Tree of Life v16.1](https://tree.opentreeoflife.org/about/synthesis-release) |
| Backbone | Divergence times | 137,306 species | [TimeTree 5](https://timetree.org) |
