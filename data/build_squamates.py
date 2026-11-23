"""Merge the squamate trait sources onto the Title et al. (2024) tree.

Sources (all in data/raw/, see data/README.md):
  squamate_traits.xlsx          snake-like convergence dataset (Dryad 2025): digits, presacrals, habitat; 1,386 species
  squamate_vertebrae_2357.csv   vertebral counts from radiographs (Dryad doi:10.5061/dryad.4qrfj6qr1), keyed to the Title tree
  squambase_1.xlsx              SquamBase 1.0 (Meiri 2024): limb status and substrate for all 11,744 squamates,
                                and each species' name in Zheng and Wiens (2016)
  camaiti2022_s2_morphology.xlsx limb-reduced skinks and relatives (Camaiti et al. 2022): fingers, toes, presacrals
  bergmann_morinaga2019.csv     fingers, toes, presacrals and body measurements for 231 species
  squamate_tree_title2024.tre   Title et al. (2024) time tree, 6,885 species

Each trait takes the first source that has it, in the order listed in PRIORITY. Digits
are fingers + toes on one side (0-10), the convention of the convergence dataset. Where no
source counts digits, SquamBase's limb status fills in: "limbless" snakes -> 0 and
"4 well developed limbs" -> 10. (Limbless lizards are left uncounted: in the overlap,
13% of species SquamBase calls limbless, mostly pygopodids, keep 1-2 vestigial digits.) Both inferences, and every overlap between sources,
are checked against the counted data and reported in data/processed/squamates_quality.json.

Writes data/processed/squamates_merged.csv (one row per tree tip with any data).
Run from the repository root: python data/build_squamates.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evograph import trees  # noqa: E402

RAW = Path("data/raw")
OUT = Path("data/processed")


def us(s):
    return s.astype(str).str.strip().str.replace(" ", "_")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    parent, length, names = trees.read_newick((RAW / "squamate_tree_title2024.tre").read_text())
    tips = {names[v] for v in trees.tips(parent)}

    sb = pd.read_excel(RAW / "squambase_1.xlsx")
    sb["tip"] = us(sb["Species name (Binomial)"])
    zw_to_tip = dict(zip(us(sb["name in Zheng and Wiens (2016)"].dropna()),
                         sb.loc[sb["name in Zheng and Wiens (2016)"].notna(), "tip"]))

    # 1. convergence dataset (Zheng and Wiens names -> current names)
    cd = pd.read_excel(RAW / "squamate_traits.xlsx", sheet_name="Dataset")
    cd["tip"] = [s if s in tips else zw_to_tip.get(s) for s in cd.Species]
    for c in ("Digits", "Presacrals"):
        cd[c] = pd.to_numeric(cd[c], errors="coerce")
    cd = cd[cd.tip.isin(tips)].drop_duplicates("tip").set_index("tip")

    # 2. radiograph vertebral counts
    vt = pd.read_csv(RAW / "squamate_vertebrae_2357.csv").drop_duplicates("tree").set_index("tree")

    # 3. limb-reduced skinks
    cm = pd.read_excel(RAW / "camaiti2022_s2_morphology.xlsx")
    cm["tip"] = us(cm.Species)
    cm = cm[cm.tip.isin(tips)].drop_duplicates("tip").set_index("tip")
    cm["digits"] = cm.Man + cm.Pes

    # 4. Bergmann and Morinaga
    bm = pd.read_csv(RAW / "bergmann_morinaga2019.csv", skiprows=18, na_values=["NA"])
    bm["tip"] = us(bm.Species)
    bm = bm[bm.tip.isin(tips)].drop_duplicates("tip").set_index("tip")
    bm["digits"] = bm.FD + bm.HD

    sbi = sb[sb.tip.isin(tips)].drop_duplicates("tip").set_index("tip")

    rows = pd.DataFrame(index=sorted(tips))
    def take(series_list):
        val = pd.Series(np.nan, index=rows.index)
        src = pd.Series("", index=rows.index)
        for name, s in series_list:
            s = s.reindex(rows.index)
            fill = val.isna() & s.notna()
            val[fill] = s[fill]
            src[fill] = name
        return val, src

    rows["presacrals"], rows["presacrals_source"] = take([
        ("convergence_dataset", cd.Presacrals), ("radiographs", vt.numberPresacralVert),
        ("camaiti2022", cm.PSVn), ("bergmann_morinaga2019", bm.Presacral)])
    limbs = sbi["Limbs"].reindex(rows.index)
    fam = sbi["Family"].reindex(rows.index).fillna(vt.family.reindex(rows.index))
    is_snake = (vt["type"].reindex(rows.index) == "snake") | fam.isin(SNAKE_FAMILIES)
    # "limbless" -> 0 digits only for snakes: limbless lizards such as pygopodids often keep 1-2 vestigial digits
    inferred = pd.Series(np.where((limbs == "limbless") & is_snake, 0.0, np.where(limbs == "4 well developed limbs", 10.0, np.nan)), index=rows.index)
    rows["digits"], rows["digits_source"] = take([
        ("convergence_dataset", cd.Digits), ("camaiti2022", cm.digits), ("bergmann_morinaga2019", bm.digits),
        ("squambase_limb_status", inferred)])
    sub = sbi["substrate"].reindex(rows.index).astype(str)
    sb_fos = pd.Series(np.where(sub.str.contains("Fossorial"), 1.0, np.where(sub == "nan", np.nan, 0.0)), index=rows.index)
    cd_fos = (cd["Habitat (4)"] == "Fossorial").astype(float).where(cd["Habitat (4)"].notna())
    rows["fossorial"], rows["fossorial_source"] = take([("convergence_dataset", cd_fos), ("squambase", sb_fos)])
    rows["family"] = sbi["Family"].reindex(rows.index)
    rows["family"] = rows.family.fillna(vt.family.reindex(rows.index))
    rows["limb_status"] = limbs
    rows["svl_mm"] = vt.completeSVL.reindex(rows.index)
    rows["elongation_index"] = vt.elongationIndex.reindex(rows.index)
    rows["snake"] = is_snake
    rows = rows[rows[["presacrals", "digits", "fossorial"]].notna().any(axis=1)]
    rows.index.name = "tip"
    rows.to_csv(OUT / "squamates_merged.csv")

    # ---- quality checks on overlaps ----
    q = {"tree_tips": len(tips), "species_with_any_trait": int(len(rows)),
         "species_with_digits_and_presacrals": int((rows.digits.notna() & rows.presacrals.notna()).sum()),
         "previous_dataset_alone_with_both": int((cd.Digits.notna() & cd.Presacrals.notna()).sum()),
         "presacrals_by_source": rows.presacrals_source.value_counts().to_dict(),
         "digits_by_source": rows.digits_source.value_counts().to_dict()}
    def agree(a, b, label):
        both = a.notna() & b.reindex(a.index).notna()
        x, y = a[both], b.reindex(a.index)[both]
        if both.sum() < 5:
            return {label: {"overlap": int(both.sum())}}
        return {label: {"overlap": int(both.sum()), "spearman": round(float(spearmanr(x, y)[0]), 3),
                        "median_abs_difference": float(np.median(np.abs(x - y))),
                        "share_within_10_percent": round(float((np.abs(x - y) <= 0.1 * np.maximum(x, y)).mean()), 3)}}
    checks = {}
    checks |= agree(cd.Presacrals, vt.numberPresacralVert, "presacrals: convergence dataset vs radiographs")
    checks |= agree(cd.Presacrals, cm.PSVn, "presacrals: convergence dataset vs Camaiti 2022")
    checks |= agree(cd.Digits, cm.digits, "digits: convergence dataset vs Camaiti 2022")
    checks |= agree(cd.Digits, bm.digits, "digits: convergence dataset vs Bergmann & Morinaga")
    lim = sbi["Limbs"].reindex(cd.index)
    checks["digit inference: counted digits where SquamBase says '4 well developed limbs'"] = {
        "species": int((lim == "4 well developed limbs").sum()),
        "share_with_10": round(float((cd.Digits[lim == "4 well developed limbs"].dropna() == 10).mean()), 3),
        "share_with_8_or_more": round(float((cd.Digits[lim == "4 well developed limbs"].dropna() >= 8).mean()), 3)}
    checks["digit inference: counted digits where SquamBase says 'limbless'"] = {
        "species": int((lim == "limbless").sum()),
        "share_with_0": round(float((cd.Digits[lim == "limbless"].dropna() == 0).mean()), 3)}
    both = cd_fos.notna() & sb_fos.reindex(cd_fos.index).notna()
    checks["fossorial: convergence dataset vs SquamBase substrate"] = {
        "overlap": int(both.sum()), "agreement": round(float((cd_fos[both] == sb_fos.reindex(cd_fos.index)[both]).mean()), 3)}
    q["agreement_checks"] = checks
    (OUT / "squamates_quality.json").write_text(json.dumps(q, indent=2))
    print(json.dumps(q, indent=2))


SNAKE_FAMILIES = {
    "Acrochordidae", "Aniliidae", "Anomalepididae", "Anomochilidae", "Boidae", "Bolyeriidae", "Calabariidae",
    "Candoiidae", "Charinidae", "Colubridae", "Cyclocoridae", "Cylindrophiidae", "Dipsadidae", "Elapidae",
    "Erycidae", "Gerrhopilidae", "Homalopsidae", "Lamprophiidae", "Leptotyphlopidae", "Loxocemidae",
    "Natricidae", "Pareidae", "Prosymnidae", "Psammophiidae", "Pseudaspididae", "Pseudoxyrhophiidae",
    "Pythonidae", "Sanziniidae", "Tropidophiidae", "Typhlopidae", "Uropeltidae", "Viperidae", "Xenodermidae",
    "Xenopeltidae", "Xenophidiidae", "Xenotyphlopidae", "Atractaspididae", "Lamprophiidae", "Colubroidea"}


if __name__ == "__main__":
    main()
