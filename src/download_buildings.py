"""Download all apartment buildings in Almaty from OpenStreetMap (Overpass API).

No key needed. For every building we keep its id, centre point and the few tags
that sometimes help (floors, year, wall material, address).
Saves data/buildings.csv
"""
import os

import pandas as pd
import requests

# the main server is often overloaded, so there is a mirror to try next
OVERPASS_URLS = ["https://overpass-api.de/api/interpreter",
                 "https://overpass.private.coffee/api/interpreter"]
# "out center" gives one point in the middle of each building instead of the whole outline
QUERY = """
[out:json][timeout:180];
area["name:en"="Almaty"]["admin_level"="4"]->.city;
way["building"="apartments"](area.city);
out center tags;
"""
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "buildings.csv")


def overpass(query):
    for url in OVERPASS_URLS:
        # Overpass refuses requests without a User-Agent
        r = requests.post(url, data={"data": query},
                          headers={"User-Agent": "almaty-buildings student project"}, timeout=200)
        if r.ok:
            return r.json()
        print(f"{url} answered {r.status_code}, trying the next one")
    raise RuntimeError("all Overpass servers failed, try again later")


def main():
    rows = []
    for el in overpass(QUERY)["elements"]:
        tags = el.get("tags", {})
        rows.append({
            "osm_id": el["id"],
            "lat": el["center"]["lat"],
            "lon": el["center"]["lon"],
            "levels": tags.get("building:levels"),
            "year": tags.get("start_date"),
            "material": tags.get("building:material"),
            "street": tags.get("addr:street"),
            "number": tags.get("addr:housenumber"),
        })
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"{len(df)} buildings saved to data/buildings.csv")
    print(f"with floors: {df.levels.notna().sum()}, with year: {df.year.notna().sum()}, "
          f"with address: {df.street.notna().sum()}")


if __name__ == "__main__":
    main()
