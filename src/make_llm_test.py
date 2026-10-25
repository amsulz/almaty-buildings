"""Prepare the test photos for labelling by other AI models (Codex, Claude, Gemini).

Test = districts the model never sees in training (Almaly + Jetisu).
Photos are copied to a folder OUTSIDE the project and renamed 001.jpg, 002.jpg ...
so the AI can't see the address, the OSM id or my labels.
The key (which number is which building) stays in data/llm_test_key.csv.
"""
import os
import shutil

import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
PHOTOS_DIR = os.path.join(ROOT, "data", "photos")
KEY_PATH = os.path.join(ROOT, "data", "llm_test_key.csv")
OUT_DIR = os.path.join(ROOT, "..", "llm_test", "photos")

TEST_DISTRICTS = ["Almaly District", "Jetisu district"]


def main():
    labels = pd.read_csv(os.path.join(ROOT, "data", "labels.csv"))
    labels["osm_id"] = labels.photo.str[:-4].astype(int)
    # 5 (tile) means brick under facing, see LABELING.md
    labels["label"] = labels.label.replace({"tile": "brick"})
    labels = labels[labels.label.isin(["panel", "brick", "monolith"])]
    districts = pd.read_csv(os.path.join(ROOT, "data", "districts.csv"))
    test = labels.merge(districts, on="osm_id")
    test = test[test.district.isin(TEST_DISTRICTS)]

    # shuffle, so the order says nothing about the street or the type
    test = test.sample(frac=1, random_state=0).reset_index(drop=True)
    test["number"] = [f"{i + 1:03d}" for i in range(len(test))]

    os.makedirs(OUT_DIR, exist_ok=True)
    for row in test.itertuples():
        shutil.copy(os.path.join(PHOTOS_DIR, row.photo), os.path.join(OUT_DIR, f"{row.number}.jpg"))
    test[["number", "photo", "district", "label"]].to_csv(KEY_PATH, index=False)
    print(f"{len(test)} photos copied to llm_test/photos, key in data/llm_test_key.csv")
    print(test.label.value_counts())


if __name__ == "__main__":
    main()
