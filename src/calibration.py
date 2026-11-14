"""Can I trust the confidence of CLIP + logreg?

If the model says "80% sure", it should be right about 80% of the time.
I check this with the same cross-validation by district as in clip_logreg.py,
for different C. Saves results/calibration.csv
"""
import os

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict

from clip_logreg import clip_features, make_model
from train import ROOT, load_split

C_VALUES = [1, 10, 30, 100, 1000, 10000]
BINS = [0, 0.6, 0.8, 0.95, 1.01]  # confidence groups
NAMES = ["below 60%", "60-80%", "80-95%", "above 95%"]


def main():
    train, val, _ = load_split()
    dev = pd.concat([train, val])  # test photos are not used here
    x = clip_features(list(dev.photo))
    y = dev.label.values

    rows = []
    for c in C_VALUES:
        proba = cross_val_predict(make_model(c), x, y, groups=dev.district,
                                  cv=LeaveOneGroupOut(), method="predict_proba")
        classes = np.unique(y)  # same order as the columns of proba
        right = classes[proba.argmax(1)] == y
        confidence = proba.max(1)
        row = {"C": c, "accuracy": round(right.mean(), 3),
               # log loss is small when the model is right AND not too sure when it is wrong
               "log_loss": round(log_loss(y, proba, labels=classes), 3)}
        group = pd.cut(confidence, BINS, labels=NAMES, right=False)
        for name in NAMES:
            part = right[group == name]
            row[name + " photos"] = len(part)
            row[name + " right"] = round(part.mean(), 2) if len(part) else None
        rows.append(row)

    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    table.to_csv(os.path.join(ROOT, "results", "calibration.csv"), index=False)


if __name__ == "__main__":
    main()
