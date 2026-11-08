"""What the models mix up on the test photos (Almaly and Jetisu).
Rows are the real type, columns are the answer.
"""
import os

import pandas as pd

from train import CLASSES, ROOT

# current labels, not the old key: a few test photos were relabelled later
key = pd.read_csv(os.path.join(ROOT, "data", "llm_test_key.csv"), dtype={"number": str})
labels = pd.read_csv(os.path.join(ROOT, "data", "labels.csv"))
labels["label"] = labels.label.replace({"tile": "brick"})
key = key.drop(columns="label").merge(labels, on="photo")
key = key[key.label.isin(CLASSES)]

tables = []
for name in ["resnet", "clip", "codex"]:
    answers = pd.read_csv(os.path.join(ROOT, "results", f"answers_{name}.csv"), dtype={"photo": str})
    answers.columns = ["number", "answer"]
    df = key.merge(answers, on="number")

    table = pd.crosstab(df.label, df.answer).reindex(index=CLASSES, columns=CLASSES, fill_value=0)
    print(f"\n{name}: {(df.label == df.answer).mean():.1%} right on {len(df)}")
    print(table)
    tables.append(table.assign(model=name))

pd.concat(tables).to_csv(os.path.join(ROOT, "results", "confusion.csv"))
