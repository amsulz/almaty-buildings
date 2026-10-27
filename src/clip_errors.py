"""Which photos CLIP + logreg gets wrong in cross-validation over districts.

Only train + val (the "dev" photos), the test districts are not touched.
Every photo is predicted by a model that never saw its district, the same way
as in clip_logreg.py. Saves the mistakes, most confident first, to
results/clip_cv_errors.csv, so they can be opened in label.py / Google Maps
and checked: is the model wrong, or my label?
  python src/clip_errors.py
"""
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict

from clip_logreg import clip_features, make_model
from train import CLASSES, ROOT, load_split

ERRORS_PATH = os.path.join(ROOT, "results", "clip_cv_errors.csv")
C = 1000  # best C from clip_logreg.py


def main():
    train, val, _ = load_split()
    dev = pd.concat([train, val]).reset_index(drop=True)
    x = clip_features(list(dev.photo))

    model = make_model(C)
    proba = cross_val_predict(model, x, dev.label.values, groups=dev.district,
                              cv=LeaveOneGroupOut(), method="predict_proba")
    order = sorted(CLASSES)  # sklearn keeps classes in alphabetical order
    dev["pred"] = [order[i] for i in proba.argmax(1)]
    dev["confidence"] = proba.max(1).round(3)

    print("errors by district (share of photos wrong):")
    print(dev.groupby("district").apply(lambda d: (d.pred != d.label).mean())
          .sort_values(ascending=False).round(2).to_string())
    print("\nbrick photos the model called something else, by what:")
    print(dev[(dev.label == "brick") & (dev.pred != "brick")].pred.value_counts().to_string())

    # confident mistakes first: these are the best candidates for a wrong label
    errors = dev[dev.pred != dev.label].sort_values("confidence", ascending=False)
    errors[["photo", "district", "label", "pred", "confidence"]].to_csv(ERRORS_PATH, index=False)
    print(f"\n{len(errors)} of {len(dev)} wrong, saved to {ERRORS_PATH}")
    print(f"mistakes with confidence > 0.9: {(errors.confidence > 0.9).sum()}")


if __name__ == "__main__":
    main()
