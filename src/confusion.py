"""What the models mix up on the test photos (Almaly and Jetisu).
Rows are the real type, columns are what the model said.
  python src/confusion.py
"""
import os

import pandas as pd

from train import CLASSES, ROOT

key = pd.read_csv(os.path.join(ROOT, "data", "llm_test_key.csv"), dtype={"number": str})
key["label"] = key.label.replace({"tile": "brick"})  # tile is brick in training

tables = []
for name in ["resnet", "clip", "codex"]:
    answers = pd.read_csv(os.path.join(ROOT, "results", f"answers_{name}.csv"), dtype={"photo": str})
    answers.columns = ["number", "answer"]
    df = key.merge(answers, on="number")
    df = df[df.label.isin(CLASSES)]

    table = pd.crosstab(df.label, df.answer).reindex(index=CLASSES, columns=CLASSES, fill_value=0)
    print(f"\n{name}: {(df.label == df.answer).mean():.1%} right")
    print(table)
    tables.append(table.assign(model=name))

pd.concat(tables).to_csv(os.path.join(ROOT, "results", "confusion.csv"))
