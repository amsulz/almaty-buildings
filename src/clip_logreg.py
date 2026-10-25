"""CLIP features + logistic regression, instead of fine-tuning a whole network.

CLIP (ViT-L/14) is not trained here at all: it only turns every photo into 768 numbers.
On these numbers we train a simple logistic regression, which can't overfit 500 photos
as easily as ResNet-50 did.

Model choice by cross-validation over districts (train + val): train on all districts
but one, predict the one left out, repeat for every district. The test districts
(Almaly + Jetisu) are not touched unless you run with --test, once, at the very end:
  python src/clip_logreg.py          # cross-validation only
  python src/clip_logreg.py --test   # final score on test + results/answers_clip.csv
"""
import os
import sys

import numpy as np
import open_clip
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict

from train import CLASSES, PHOTOS_DIR, ROOT, load_split

FEATURES_PATH = os.path.join(ROOT, "models", "clip_features.npz")
ANSWERS_PATH = os.path.join(ROOT, "results", "answers_clip.csv")
C_VALUES = [0.1, 1, 10, 100, 1000]  # smaller C = simpler model

device = "cuda" if torch.cuda.is_available() else "cpu"


def clip_features(photos):
    """768 numbers per photo, saved to disk so CLIP runs only once."""
    if os.path.exists(FEATURES_PATH):
        saved = np.load(FEATURES_PATH, allow_pickle=True)
        cache = dict(zip(saved["photos"], saved["features"]))
    else:
        cache = {}
    missing = [p for p in photos if p not in cache]
    if missing:
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="openai", device=device)
        model.eval()
        with torch.no_grad():
            for i in range(0, len(missing), 32):
                batch = missing[i:i + 32]
                x = torch.stack([preprocess(Image.open(os.path.join(PHOTOS_DIR, p)).convert("RGB"))
                                 for p in batch]).to(device)
                f = model.encode_image(x)
                f = f / f.norm(dim=1, keepdim=True)  # same length for every photo
                cache.update(zip(batch, f.cpu().numpy()))
        os.makedirs(os.path.dirname(FEATURES_PATH), exist_ok=True)
        np.savez(FEATURES_PATH, photos=np.array(list(cache)), features=np.stack(list(cache.values())))
    return np.stack([cache[p] for p in photos])


def make_model(c):
    # brick is rare, "balanced" makes its mistakes cost more (like the weights in train.py)
    return LogisticRegression(C=c, class_weight="balanced", max_iter=5000)


def report(true, pred):
    acc = (true == pred).mean()
    # mean of per-class accuracy: brick counts as much as panel, even though it's rare
    per_class = [(pred[true == c] == c).mean() for c in CLASSES]
    print(f"  accuracy {acc:.3f}, balanced {np.mean(per_class):.3f}  "
          + "  ".join(f"{c} {a:.2f}" for c, a in zip(CLASSES, per_class)))


def main():
    train, val, test = load_split()
    dev = pd.concat([train, val])  # val is just one more district for cross-validation
    x = clip_features(list(dev.photo) + list(test.photo))
    x_dev, x_test = x[:len(dev)], x[len(dev):]
    y_dev = dev.label.values
    print(f"dev {len(dev)} photos in {dev.district.nunique()} districts, test {len(test)}")

    best_c, best_score = None, -1
    for c in C_VALUES:
        pred = cross_val_predict(make_model(c), x_dev, y_dev, groups=dev.district,
                                 cv=LeaveOneGroupOut())
        print(f"C = {c}")
        report(y_dev, pred)
        score = np.mean([(pred[y_dev == k] == k).mean() for k in CLASSES])
        if score > best_score:
            best_c, best_score, best_pred = c, score, pred

    print(f"\nbest C = {best_c} (by balanced accuracy). Rows = true, columns = predicted:")
    print(pd.DataFrame(confusion_matrix(y_dev, best_pred, labels=CLASSES),
                       index=CLASSES, columns=CLASSES))

    if "--test" in sys.argv:
        model = make_model(best_c).fit(x_dev, y_dev)
        test = test.copy()
        test["pred"] = model.predict(x_test)
        print("\nTEST")
        report(test.label.values, test.pred.values)
        key = pd.read_csv(os.path.join(ROOT, "data", "llm_test_key.csv"), dtype={"number": str})
        answers = key[["number", "photo"]].merge(test[["photo", "pred"]], on="photo")
        answers.rename(columns={"number": "photo", "photo": "file", "pred": "label"})[
            ["photo", "label"]].to_csv(ANSWERS_PATH, index=False)


if __name__ == "__main__":
    main()
