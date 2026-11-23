# Data

Raw downloads go in `data/raw/` and built tables in `data/processed/`; both are git-ignored. Dryad and journal sites are not reachable from our shared compute, so everything below is downloaded by hand and saved under the names shown.

## Squamate data

`python data/build_squamates.py` merges the sources below onto the Title et al. (2024) tree and writes `data/processed/squamates_merged.csv` plus `data/processed/squamates_quality.json`, which records how the sources agree where they overlap. All of these are **downloaded by hand** into `data/raw/` under the names shown.

| File in `data/raw/` | What it holds | Source |
| --- | --- | --- |
| `squamate_tree_title2024.tre` | Time tree of 6,885 squamates (main tree) | Title et al. (2024), shipped in the vertebral dataset below |
| `squamate_vertebrae_2357.csv` | Presacral vertebrae for 2,364 species (radiographs and literature), body length, elongation index, habitat | Dryad doi:10.5061/dryad.4qrfj6qr1 (`Evolution-vertebrae-data-files.zip`) |
| `squambase_1.xlsx` | Limb status, substrate, family and Zheng and Wiens names for all 11,744 squamates | SquamBase 1.0, Meiri (2024), Dryad doi:10.5061/dryad.76hdr7t3b (`Supplementary_Table_S1`) |
| `camaiti2022_s2_morphology.xlsx`, `camaiti2022_s3_raw.xlsx` | Fingers, toes, presacrals and limb proportions of limb-reduced skinks and relatives | Camaiti et al. (2022), *J. Biogeography* 49:1397, doi:10.1111/jbi.14392, Appendices S2 and S3 |
| `bergmann_morinaga2019.csv` | Fingers, toes, presacrals, body and limb lengths for 231 species | Bergmann and Morinaga (2019), Dryad doi:10.5061/dryad.5rh14nc |
| `squamate_traits.xlsx` | Digits, presacrals, habitat for 1,386 species | Snake-like convergence dataset (2025), Dryad doi:10.5061/dryad.crjdfn3g7 (`Snake-like_Convergence_Dataset.xlsx`) |
| `squamate_tree_zw2016.nexus` | Zheng and Wiens (2016) time tree (robustness checks, `SQUAMATE_TREE=zw2016`) | `Dataset_S2_zw2016.nexus` in Dryad doi:10.5061/dryad.0zpc8671s |

Agreement where sources overlap: vertebral counts, convergence dataset vs radiographs, Spearman 0.99 over 1,052 species (median difference 0); digit counts, convergence dataset vs Camaiti et al., Spearman 0.98 over 240 species. Where no source counts digits, SquamBase's limb status fills in: "4 well developed limbs" → 10 digits (96% exactly 10 and all at least 8 in the counted overlap), "limbless" → 0 for snakes only (13% of lizards SquamBase calls limbless, mostly pygopodids, keep vestigial digits). Result: 2,403 species with both digit and vertebra counts (1,451 lizards, 952 snakes), up from 911.
