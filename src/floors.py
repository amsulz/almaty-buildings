"""Building type by number of floors (floors from OSM).
My label if I have one, otherwise the model's answer.
"""
import os

import pandas as pd

from train import CLASSES, ROOT

pred = pd.read_csv(os.path.join(ROOT, "results", "predictions.csv"))
pred = pred[~pred.hand.isin(["bad", "other"])]
pred["type"] = pred.hand.where(pred.hand.isin(CLASSES), pred.pred)
pred = pred[pred.type.isin(CLASSES)]

buildings = pd.read_csv(os.path.join(ROOT, "data", "buildings.csv"))
df = pred.merge(buildings, on="osm_id")
df["levels"] = pd.to_numeric(df.levels, errors="coerce")
df = df[df.levels.notna()]

df["floors"] = pd.cut(df.levels, [0, 3, 5, 9, 12, 100],
                      labels=["1-3", "4-5", "6-9", "10-12", "13+"])
counts = pd.crosstab(df.floors, df.type)[CLASSES]
percent = (counts.div(counts.sum(axis=1), axis=0) * 100).round(1)
percent.insert(0, "buildings", counts.sum(axis=1))

print(percent)
print(f"\n{len(df)} of {len(pred)} buildings have floors in OSM")
percent.to_csv(os.path.join(ROOT, "results", "floors.csv"))
