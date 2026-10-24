"""Download a Street View photo of each building, camera turned towards it.

For every building we first ask the free metadata endpoint for the nearest
panorama, then point the camera from that panorama at the building centre.
Photos go to data/photos/ (not on GitHub), info about them to data/photos.csv.

Run: python src/download_photos.py 10   (10 = how many buildings to try)
"""
import math
import os
import sys

import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = os.path.join(os.path.dirname(__file__), "..")
BUILDINGS_PATH = os.path.join(ROOT, "data", "buildings.csv")
PHOTOS_DIR = os.path.join(ROOT, "data", "photos")
INFO_PATH = os.path.join(ROOT, "data", "photos.csv")

META_URL = "https://maps.googleapis.com/maps/api/streetview/metadata"
PHOTO_URL = "https://maps.googleapis.com/maps/api/streetview"
RADIUS = 50   # metres; a panorama further away shows the house too small
PITCH = 15    # look a bit up, apartment blocks are tall
FOV = 90      # wide enough for the whole facade

load_dotenv(os.path.join(ROOT, ".env"))
KEY = os.getenv("GOOGLE_MAPS_KEY")


def heading(lat1, lon1, lat2, lon2):
    """Compass direction (0 = north, 90 = east) from point 1 to point 2."""
    lat1, lat2 = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def distance(lat1, lon1, lat2, lon2):
    """Distance in metres (flat approximation is fine for < 100 m)."""
    dy = (lat2 - lat1) * 111_320
    dx = (lon2 - lon1) * 111_320 * math.cos(math.radians(lat1))
    return math.hypot(dx, dy)


def main():
    if not KEY:
        raise SystemExit("GOOGLE_MAPS_KEY is empty, put it in .env")
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    os.makedirs(PHOTOS_DIR, exist_ok=True)

    # random_state fixes the sample, so a second run tries the same buildings
    buildings = pd.read_csv(BUILDINGS_PATH).sample(n=n, random_state=42)
    done = pd.read_csv(INFO_PATH) if os.path.exists(INFO_PATH) else pd.DataFrame()
    rows = done.to_dict("records")
    done_ids = set(done.osm_id) if len(done) else set()

    for i, b in enumerate(buildings.itertuples()):
        if b.osm_id in done_ids:
            continue
        # save progress now and then, so a crash doesn't lose everything
        if i % 50 == 0:
            pd.DataFrame(rows).to_csv(INFO_PATH, index=False)
        # metadata is free: it only says whether a panorama exists and where
        try:
            meta = requests.get(META_URL, params={
                "location": f"{b.lat},{b.lon}", "radius": RADIUS,
                "source": "outdoor", "key": KEY}, timeout=30).json()
        except requests.RequestException:
            print(f"{b.osm_id}: network error, skipped (will retry next run)")
            continue
        if meta["status"] != "OK":
            print(f"{b.osm_id}: no panorama ({meta['status']})")
            rows.append({"osm_id": b.osm_id, "status": meta["status"]})
            continue

        # people upload their own panoramas too (cafes inside etc.), keep only Google's car
        if "Google" not in meta.get("copyright", ""):
            print(f"{b.osm_id}: not a Google panorama ({meta.get('copyright')})")
            rows.append({"osm_id": b.osm_id, "status": "NOT_GOOGLE"})
            continue
        p_lat, p_lon = meta["location"]["lat"], meta["location"]["lng"]
        d = distance(p_lat, p_lon, b.lat, b.lon)
        # Google sometimes returns a panorama a bit outside the radius
        if d > RADIUS:
            print(f"{b.osm_id}: panorama too far ({d:.0f} m)")
            rows.append({"osm_id": b.osm_id, "status": "TOO_FAR", "dist_m": round(d)})
            continue
        h = heading(p_lat, p_lon, b.lat, b.lon)
        try:
            photo = requests.get(PHOTO_URL, params={
                "pano": meta["pano_id"], "heading": round(h), "pitch": PITCH,
                "fov": FOV, "size": "640x640", "key": KEY}, timeout=30)
        except requests.RequestException:
            print(f"{b.osm_id}: network error, skipped (will retry next run)")
            continue
        if not photo.ok:
            print(f"{b.osm_id}: photo request failed ({photo.status_code})")
            continue
        with open(os.path.join(PHOTOS_DIR, f"{b.osm_id}.jpg"), "wb") as f:
            f.write(photo.content)

        print(f"{b.osm_id}: ok, {d:.0f} m away, taken {meta.get('date')}")
        rows.append({"osm_id": b.osm_id, "status": "OK", "pano_id": meta["pano_id"],
                     "date": meta.get("date"), "heading": round(h), "dist_m": round(d)})

    pd.DataFrame(rows).to_csv(INFO_PATH, index=False)
    ok = sum(r["status"] == "OK" for r in rows)
    print(f"{ok} photos in data/photos/, info in data/photos.csv")


if __name__ == "__main__":
    main()
