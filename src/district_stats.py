"""Share of panel / brick / monolith houses in every district, from the model's answers.

Takes results/predictions.csv (made by predict_city.py), adds the district of
every building and counts the types. If I labelled the photo by hand, my label
is used instead of the model's answer. Buildings where the model was not sure
("unknown") are not counted. Saves to results/district_stats.csv.
  python src/district_stats.py
"""
import os

import pandas as pd

from train import CLASSES, ROOT

OUT_PATH = os.path.join(ROOT, "results", "district_stats.csv")

# OpenStreetMap gives some names in Kazakh or with special letters
NAMES = {
    "Bostandıq District": "Bostandyk",
    "Türksib District": "Turksib",
    "Медеу ауданы": "Medeu",
    "Jetisu district": "Jetisu",
}


def short_name(name):
    return NAMES.get(name, name.replace(" District", ""))


def main():
    pred = pd.read_csv(os.path.join(ROOT, "results", "predictions.csv"))
    districts = pd.read_csv(os.path.join(ROOT, "data", "districts.csv"))
    df = pred.merge(districts, on="osm_id", how="left")
    df["district"] = df.district.fillna("unknown").map(short_name)

    # my own label is better than the model's guess
    df["type"] = df.hand.where(df.hand.isin(CLASSES), df.pred)

    counts = pd.crosstab(df.district, df.type).reindex(columns=CLASSES, fill_value=0)
    stats = (counts.div(counts.sum(axis=1), axis=0) * 100).round(1)
    stats.columns = [f"{c}_%" for c in CLASSES]
    stats.insert(0, "buildings", counts.sum(axis=1))
    stats = stats.sort_values("brick_%", ascending=False)

    stats.to_csv(OUT_PATH)
    print(stats.to_string())
    print(f"\nsaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
