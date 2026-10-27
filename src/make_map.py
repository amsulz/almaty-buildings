"""Map of Almaty with every labelled building as a coloured dot.

For now only my hand labels (data/labels.csv). Later the same map will show the
model's answers for the whole city. Click a dot to see the type, year and floors
from OSM and a link to Google Maps. Saves results/map_labels.html:
  python src/make_map.py
"""
import os

import folium
import pandas as pd

from train import ROOT

MAP_PATH = os.path.join(ROOT, "results", "map_labels.html")
COLORS = {"panel": "#1f77b4", "brick": "#d62728", "monolith": "#2ca02c"}


def main():
    labels = pd.read_csv(os.path.join(ROOT, "data", "labels.csv"))
    labels["osm_id"] = labels.photo.str[:-4].astype(int)
    labels["label"] = labels.label.replace({"tile": "brick"})  # 5 = brick under facing
    labels = labels[labels.label.isin(COLORS)]
    df = labels.merge(pd.read_csv(os.path.join(ROOT, "data", "buildings.csv")), on="osm_id")

    m = folium.Map(location=[df.lat.mean(), df.lon.mean()], zoom_start=12)
    for label, color in COLORS.items():
        layer = folium.FeatureGroup(name=f"{label} ({(df.label == label).sum()})")
        for row in df[df.label == label].itertuples():
            year = "" if pd.isna(row.year) else f", {row.year}"
            floors = "" if pd.isna(row.levels) else f", {row.levels} floors"
            link = f"https://www.google.com/maps/search/?api=1&query={row.lat},{row.lon}"
            folium.CircleMarker(
                [row.lat, row.lon], radius=4, color=color, fill=True, fill_opacity=0.8, weight=0,
                popup=f"{label}{floors}{year}<br>{row.street} {row.number}<br>"
                      f"<a href='{link}' target='_blank'>Google Maps</a>",
            ).add_to(layer)
        layer.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)  # turn each type on / off

    m.save(MAP_PATH)
    print(f"{len(df)} buildings on the map, saved to {MAP_PATH}")


if __name__ == "__main__":
    main()
