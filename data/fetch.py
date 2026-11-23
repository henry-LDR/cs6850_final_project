"""Download the datasets this project uses into data/raw/ (git-ignored).

Every source is pinned to a commit so results are reproducible. Run from the
repository root:  python data/fetch.py
"""

from pathlib import Path
from urllib.request import urlopen

HYPERTRAPS = "https://raw.githubusercontent.com/StochasticBiology/hypertraps-ct/49b0a88729068057e34e664129b356c53f79b20c/RawData/"
BMR = "https://raw.githubusercontent.com/uyedaj/bmr/ed72364d343968b7aaf91dd464164edd7360d8a2/datasets/"
MADIN = "https://raw.githubusercontent.com/bacteria-archaea-traits/bacteria-archaea-traits/396dcd4d98ca20b613ba280f5514611651b6fb91/output/"

FILES = {
    # M. tuberculosis drug resistance (Casali et al. 2014, Nature Genetics), as curated for HyperTraPS-CT
    "tb_tree.nwk": HYPERTRAPS + "ng.2878-S2.txt",
    "tb_resistance.csv": HYPERTRAPS + "tuberculosis-v5-header-19-29.csv",
    # C4 photosynthesis traits (Williams et al. 2013, eLife), as curated for HyperTraPS-CT; 2 = unknown
    "c4_traits.csv": HYPERTRAPS + "c4-curated.csv",
    "c4_trait_names.txt": HYPERTRAPS + "c4-trait-names.txt",
    # Bacterial and archaeal phenotypes with GTDB taxonomy (Madin et al. 2020, Scientific Data)
    "bacteria_traits_gtdb.csv": MADIN + "condensed_species_GTDB.csv",
    # Provisional squamate time tree (4,162 tips) until the Zheng and Wiens (2016) tree is in place
    "squamate_tree_provisional.tre": BMR + "squa.tre",
}

# Download by hand (Dryad is not reachable from our shared compute):
#   squamate_traits.xlsx       Snake-like_Convergence_Dataset.xlsx from doi:10.5061/dryad.crjdfn3g7
#   squamate_tree_zw2016.nexus Dataset_S2_zw2016.nexus from doi:10.5061/dryad.0zpc8671s


def main():
    raw = Path(__file__).resolve().parent / "raw"
    raw.mkdir(exist_ok=True)
    for name, url in FILES.items():
        path = raw / name
        if path.exists():
            print(f"have  {name}")
            continue
        with urlopen(url, timeout=120) as r:
            path.write_bytes(r.read())
        print(f"fetched {name} ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
