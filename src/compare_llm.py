"""Compare answers of another AI (Codex, Claude, Gemini ...) with my labels on the test photos.

Run: python src/compare_llm.py ../llm_test/answers_codex.csv codex

Accuracy is shown twice:
- "original": against the key made when the test folder was prepared
- "cleaned": against the current labels.csv, where construction sites and
  hidden houses were later fixed by the labeling rules (they drop out of the test)
"""
import os
import sys

import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
KEY_PATH = os.path.join(ROOT, "data", "llm_test_key.csv")
LABELS_PATH = os.path.join(ROOT, "data", "labels.csv")
RESULTS_PATH = os.path.join(ROOT, "results", "llm_comparison.csv")


def report(df, title):
    ok = df.label_me == df.label_ai
    print(f"\n{title}: {ok.mean() * 100:.1f}% ({ok.sum()}/{len(df)})")
    for c in ["panel", "brick", "monolith"]:
        part = df[df.label_me == c]
        print(f"  {c:9s} {(part.label_me == part.label_ai).sum()}/{len(part)}")
    print(pd.crosstab(df.label_me, df.label_ai, rownames=["me"], colnames=["ai"]))
    return round(ok.mean() * 100, 1), len(df)


def main():
    answers_path, name = sys.argv[1], sys.argv[2]
    key = pd.read_csv(KEY_PATH, dtype={"number": str})
    answers = pd.read_csv(answers_path, dtype={"photo": str})
    answers["photo"] = answers.photo.str.zfill(3)
    df = key.merge(answers.rename(columns={"photo": "number", "label": "label_ai"}), on="number")
    df = df.rename(columns={"label": "label_me"})

    acc_orig, n_orig = report(df, "original")

    labels = pd.read_csv(LABELS_PATH)
    labels["label"] = labels.label.replace({"tile": "brick"})
    df = df.drop(columns="label_me").merge(labels.rename(columns={"label": "label_me"}), on="photo")
    df = df[df.label_me.isin(["panel", "brick", "monolith"])]
    acc_clean, n_clean = report(df, "cleaned")

    # keep one line per model, so the table can be rebuilt any time
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    old = pd.read_csv(RESULTS_PATH) if os.path.exists(RESULTS_PATH) else pd.DataFrame()
    row = pd.DataFrame([{"model": name, "acc_original": acc_orig, "n_original": n_orig,
                         "acc_cleaned": acc_clean, "n_cleaned": n_clean}])
    if len(old):
        old = old[old.model != name]
    pd.concat([old, row]).to_csv(RESULTS_PATH, index=False)


if __name__ == "__main__":
    main()
