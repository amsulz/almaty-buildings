"""Map of Almaty with every building as a coloured dot.

Two maps:
  python src/make_map.py          # my hand labels (data/labels.csv) -> results/map_labels.html
  python src/make_map.py --pred   # model answers (results/predictions.csv) -> results/map_predictions.html
Click a dot to see the type, floors and year from OSM and a link to Google Maps.
On the model map the popup also shows how sure the model is and my label, if I have one.
"""
import os
import sys

import folium
import pandas as pd

from train import ROOT

COLORS = {"panel": "#1f77b4", "brick": "#d62728", "monolith": "#2ca02c", "unknown": "#999999"}


def hand_labels():
    labels = pd.read_csv(os.path.join(ROOT, "data", "labels.csv"))
    labels["osm_id"] = labels.photo.str[:-4].astype(int)
    labels["label"] = labels.label.replace({"tile": "brick"})  # 5 = brick under facing
    return labels[labels.label.isin(COLORS)]


def model_answers():
    df = pd.read_csv(os.path.join(ROOT, "results", "predictions.csv"))
    df = df[~df.hand.isin(["bad", "other"])]  # bad photos and non-houses, the model can't know
    return df.rename(columns={"pred": "label"})


def main():
    pred = "--pred" in sys.argv
    df = model_answers() if pred else hand_labels()
    df = df.merge(pd.read_csv(os.path.join(ROOT, "data", "buildings.csv")), on="osm_id")
    map_path = os.path.join(ROOT, "results", "map_predictions.html" if pred else "map_labels.html")

    m = folium.Map(location=[df.lat.mean(), df.lon.mean()], zoom_start=12)
    for label, color in COLORS.items():
        if not (df.label == label).any():
            continue
        layer = folium.FeatureGroup(name=f"{label} ({(df.label == label).sum()})")
        for row in df[df.label == label].itertuples():
            year = "" if pd.isna(row.year) else f", {row.year}"
            floors = "" if pd.isna(row.levels) else f", {row.levels} floors"
            extra = ""
            if pred:
                extra = f"<br>model sure: {row.confidence:.0%}"
                if not pd.isna(row.hand):
                    extra += f", my label: {row.hand}"
            link = f"https://www.google.com/maps/search/?api=1&query={row.lat},{row.lon}"
            folium.CircleMarker(
                [row.lat, row.lon], radius=4, color=color, fill=True, fill_opacity=0.8, weight=0,
                popup=f"{label}{floors}{year}<br>{row.street} {row.number}{extra}<br>"
                      f"<a href='{link}' target='_blank'>Google Maps</a>",
            ).add_to(layer)
        layer.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)  # turn each type on / off

    m.save(map_path)
    print(f"{len(df)} buildings on the map, saved to {map_path}")


if __name__ == "__main__":
    main()
