"""Small window for labelling photos by hand.

Shows one photo at a time, you press a key:
  1 = panel     2 = brick     3 = monolith
  4 = other / not sure (unique tower, mixed, private house)
  5 = small tiles on the facade, can't tell what is under them
      (checked later against Krisha and merged into a real class)
  0 = bad photo (trees, fence, wrong house, no house)
  G = open this house in Google Maps (to look around when trees are in the way)
  Backspace = go back one photo, Esc = quit
Every answer is saved at once to data/labels.csv, so you can stop any time
and continue later from the same place.

Run: python src/label.py
"""
import os
import webbrowser

import pandas as pd
import tkinter as tk
from PIL import Image, ImageTk

ROOT = os.path.join(os.path.dirname(__file__), "..")
PHOTOS_DIR = os.path.join(ROOT, "data", "photos")
LABELS_PATH = os.path.join(ROOT, "data", "labels.csv")

KEYS = {"1": "panel", "2": "brick", "3": "monolith", "4": "other", "5": "tile", "0": "bad"}


class Labeller:
    def __init__(self, root):
        self.photos = sorted(f for f in os.listdir(PHOTOS_DIR) if f.endswith(".jpg"))
        # address, floors and year from OSM help to check a house in 2GIS / Krisha
        b = pd.read_csv(os.path.join(ROOT, "data", "buildings.csv")).set_index("osm_id")
        self.about = {}
        self.coords = {f"{osm_id}.jpg": (r.lat, r.lon) for osm_id, r in b.iterrows()}
        for osm_id, r in b.iterrows():
            parts = [f"{r.street} {r.number}" if pd.notna(r.street) else "no address"]
            if pd.notna(r.levels):
                parts.append(f"{r.levels} floors")
            if pd.notna(r.year):
                parts.append(f"built {r.year}")
            self.about[f"{osm_id}.jpg"] = ", ".join(parts)
        self.labels = {}
        if os.path.exists(LABELS_PATH):
            df = pd.read_csv(LABELS_PATH)
            self.labels = dict(zip(df.photo, df.label))
        # start from the first photo that has no label yet
        self.i = next((k for k, p in enumerate(self.photos) if p not in self.labels),
                      len(self.photos))

        self.root = root
        self.img_label = tk.Label(root)
        self.img_label.pack()
        self.info = tk.Label(root, font=("Arial", 14))
        self.info.pack(pady=5)
        tk.Label(root, font=("Arial", 12),
                 text="1 panel   2 brick   3 monolith   4 other   5 tile   0 bad   "
                      "G open in Google Maps   Backspace back   Esc quit").pack(pady=5)
        root.bind("<Key>", self.on_key)
        self.show()

    def show(self):
        if self.i >= len(self.photos):
            self.img_label.config(image="")
            self.info.config(text=f"All {len(self.photos)} photos labelled!")
            return
        name = self.photos[self.i]
        img = Image.open(os.path.join(PHOTOS_DIR, name))
        self.tk_img = ImageTk.PhotoImage(img)  # keep a reference or tkinter drops it
        self.img_label.config(image=self.tk_img)
        done = len(self.labels)
        old = self.labels.get(name, "")
        self.info.config(text=f"{self.i + 1} / {len(self.photos)}   (labelled {done})   "
                              f"{name}   {old}\n{self.about.get(name, '')}")

    def save(self):
        df = pd.DataFrame({"photo": list(self.labels), "label": list(self.labels.values())})
        df.to_csv(LABELS_PATH, index=False)

    def on_key(self, event):
        if event.keysym == "Escape":
            self.root.destroy()
        elif event.keysym == "BackSpace":
            self.i = max(self.i - 1, 0)
            self.show()
        elif event.keysym.lower() == "g" and self.i < len(self.photos):
            # open the same house in Google Maps Street View to look around by hand
            lat, lon = self.coords[self.photos[self.i]]
            webbrowser.open("https://www.google.com/maps/@?api=1&map_action=pano"
                            f"&viewpoint={lat},{lon}")
        elif event.char in KEYS and self.i < len(self.photos):
            self.labels[self.photos[self.i]] = KEYS[event.char]
            self.save()
            self.i += 1
            self.show()


if __name__ == "__main__":
    window = tk.Tk()
    window.title("Almaty buildings: label photos")
    Labeller(window)
    window.mainloop()
