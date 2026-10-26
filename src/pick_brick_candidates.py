"""Pick buildings that are probably brick, to download more brick photos.

Brick is the rarest class (93 photos), so we look for buildings without a photo yet that:
  - have 2-5 floors (almost all brick houses we labelled are 2-5 floors),
  - are surrounded by brick: among the 5 nearest labelled houses, at least 3 are brick
    (old houses were built in whole quarters, so neighbours are often the same type),
  - are NOT in the test districts (Almaly + Jetisu), so the test stays untouched.
Saves the osm_ids to data/brick_candidates.csv, nearest-to-brick first.

Run: python src/pick_brick_candidates.py 300   (300 = how many to keep)
"""
import os
import sys

import numpy as np
import pandas as pd

from train import ROOT, TEST_DISTRICTS

OUT_PATH = os.path.join(ROOT, "data", "brick_candidates.csv")
NEIGHBOURS = 5
MIN_BRICK = 0.6  # 3 of 5 neighbours


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    data = os.path.join(ROOT, "data")
    buildings = pd.read_csv(os.path.join(data, "buildings.csv")).merge(
        pd.read_csv(os.path.join(data, "districts.csv")), on="osm_id")
    labels = pd.read_csv(os.path.join(data, "labels.csv"))
    labels["osm_id"] = labels.photo.str[:-4].astype(int)
    labels["label"] = labels.label.replace({"tile": "brick"})
    labelled = labels[labels.label.isin(["panel", "brick", "monolith"])].merge(buildings, on="osm_id")
    tried = pd.read_csv(os.path.join(data, "photos.csv")).osm_id

    cand = buildings[~buildings.osm_id.isin(tried)
                     & ~buildings.district.isin(TEST_DISTRICTS)
                     & buildings.levels.isin(["2", "3", "4", "5"])].copy()

    # 1 degree of longitude is shorter than 1 degree of latitude here, so scale it
    k = np.cos(np.radians(43.25))
    a = np.c_[cand.lat, cand.lon * k]
    b = np.c_[labelled.lat, labelled.lon * k]
    dist = ((a[:, None, :] - b[None]) ** 2).sum(-1)  # every candidate to every labelled house
    nearest = np.argsort(dist, axis=1)[:, :NEIGHBOURS]
    is_brick = labelled.label.to_numpy(dtype=object) == "brick"
    cand["brick_share"] = is_brick[nearest].mean(axis=1)

    cand = cand[cand.brick_share >= MIN_BRICK].sort_values("brick_share", ascending=False).head(n)
    cand[["osm_id", "district", "levels", "brick_share"]].to_csv(OUT_PATH, index=False)
    print(f"{len(cand)} candidates saved to {OUT_PATH}")
    print(cand.district.value_counts().to_string())


if __name__ == "__main__":
    main()
