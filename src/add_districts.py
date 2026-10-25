"""Find the city district (Auezov, Bostandyk, ...) of every building.

We need districts to split the photos into train / val / test by place:
the model should be tested on districts it has never seen.
District borders come from OpenStreetMap. Saves data/districts.csv
"""
import os

import pandas as pd
import requests

ROOT = os.path.join(os.path.dirname(__file__), "..")
BUILDINGS_PATH = os.path.join(ROOT, "data", "buildings.csv")
OUT_PATH = os.path.join(ROOT, "data", "districts.csv")

OVERPASS_URLS = ["https://overpass-api.de/api/interpreter",
                 "https://overpass.private.coffee/api/interpreter"]
# the 8 districts of Almaty (admin_level 6 in OSM)
QUERY = """
[out:json][timeout:270];
area["name:en"="Almaty"]["admin_level"="4"]->.city;
rel["boundary"="administrative"]["admin_level"="6"](area.city);
out geom;
"""


def overpass(query):
    for url in OVERPASS_URLS:
        r = requests.post(url, data={"data": query},
                          headers={"User-Agent": "almaty-buildings student project"}, timeout=300)
        if r.ok:
            return r.json()
        print(f"{url} answered {r.status_code}, trying the next one")
    raise RuntimeError("all Overpass servers failed, try again later")


def inside(lat, lon, segments):
    """Ray casting: go from the point to the east and count how many border
    lines we cross. Odd number = inside. Works without joining the border
    pieces into one closed line."""
    crossings = 0
    for (lat1, lon1), (lat2, lon2) in segments:
        if (lat1 > lat) != (lat2 > lat):
            lon_cross = lon1 + (lat - lat1) * (lon2 - lon1) / (lat2 - lat1)
            if lon_cross > lon:
                crossings += 1
    return crossings % 2 == 1


def main():
    districts = {}
    for rel in overpass(QUERY)["elements"]:
        segments = []
        for m in rel["members"]:
            if m["type"] != "way" or "geometry" not in m:
                continue
            pts = [(p["lat"], p["lon"]) for p in m["geometry"]]
            segments += list(zip(pts[:-1], pts[1:]))
        name = rel["tags"].get("name:en", rel["tags"]["name"])
        districts[name] = segments
    print("districts:", ", ".join(districts))

    b = pd.read_csv(BUILDINGS_PATH)
    found = []
    for row in b.itertuples():
        found.append(next((n for n, s in districts.items() if inside(row.lat, row.lon, s)), None))
    b["district"] = found
    b[["osm_id", "district"]].to_csv(OUT_PATH, index=False)
    print(b.district.value_counts(dropna=False))


if __name__ == "__main__":
    main()
