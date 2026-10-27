"""Label every downloaded photo with CLIP + logistic regression.

The model is trained on train + val only (the test districts are never used for
training). Every photo in data/photos.csv gets a type and a confidence; if the
model is less than 60% sure, the answer is "unknown" instead of a guess.
Photos I labelled myself keep my label (column "hand").
Saves results/predictions.csv for src/make_map.py:
  python src/predict_city.py
"""
import os

import pandas as pd

from clip_logreg import clip_features, make_model
from train import ROOT, load_split

PREDICTIONS_PATH = os.path.join(ROOT, "results", "predictions.csv")
C = 1000            # best C from clip_logreg.py
MIN_CONFIDENCE = 0.6


def main():
    train, val, _ = load_split()
    dev = pd.concat([train, val])

    photos = pd.read_csv(os.path.join(ROOT, "data", "photos.csv"))
    photos = photos[photos.status == "OK"]
    photos["photo"] = photos.osm_id.astype(str) + ".jpg"

    x_dev = clip_features(list(dev.photo))
    x_all = clip_features(list(photos.photo))  # new photos go through CLIP once, then cached
    model = make_model(C).fit(x_dev, dev.label.values)

    proba = model.predict_proba(x_all)
    photos["pred"] = model.classes_[proba.argmax(1)]
    photos["confidence"] = proba.max(1).round(3)
    photos.loc[photos.confidence < MIN_CONFIDENCE, "pred"] = "unknown"

    labels = pd.read_csv(os.path.join(ROOT, "data", "labels.csv"))
    labels["hand"] = labels.label.replace({"tile": "brick"})
    photos = photos.merge(labels[["photo", "hand"]], on="photo", how="left")

    photos[["osm_id", "pred", "confidence", "hand"]].to_csv(PREDICTIONS_PATH, index=False)
    print(f"{len(photos)} photos, saved to {PREDICTIONS_PATH}")
    print(photos.pred.value_counts().to_string())
    unlabelled = photos[photos.hand.isna()]
    print(f"\nnot labelled by hand: {len(unlabelled)}")
    print(unlabelled.pred.value_counts().to_string())


if __name__ == "__main__":
    main()
