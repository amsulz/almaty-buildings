"""Share of panel / brick / monolith houses in every district.
My label if I have one, otherwise the model's answer.
"""
import os

import pandas as pd

from train import CLASSES, ROOT

pred = pd.read_csv(os.path.join(ROOT, "results", "predictions.csv"))
districts = pd.read_csv(os.path.join(ROOT, "data", "districts.csv"))
df = pred.merge(districts, on="osm_id")

# bad photos and "other" houses are not counted
df = df[~df.hand.isin(["bad", "other"])]

# some names in OSM are in Kazakh
df["district"] = df.district.replace({
    "Bostandıq District": "Bostandyk",
    "Türksib District": "Turksib",
    "Медеу ауданы": "Medeu",
    "Jetisu district": "Jetisu",
}).str.replace(" District", "")

df["type"] = df.hand.where(df.hand.isin(CLASSES), df.pred)
df = df[df.type.isin(CLASSES)]

counts = pd.crosstab(df.district, df.type)[CLASSES]
percent = (counts.div(counts.sum(axis=1), axis=0) * 100).round(1)
percent.insert(0, "buildings", counts.sum(axis=1))
percent = percent.sort_values("brick", ascending=False)

print(percent)
percent.to_csv(os.path.join(ROOT, "results", "district_stats.csv"))
