"""Re-run the notebook's pipeline as originally written, then without target leakage.

The notebook built drug_avg_rating and condition_avg_rating from every review before the
train/test split, so each review's own rating (test reviews included) went into a feature used
to predict it. This script runs the same data, split, features and models twice:

  original   averages over all rows, as in notebooks/main.ipynb
  leak_free  the split first, then averages from training rows only (prepare_features(fit_index=...))

plus a run without the two averages at all, and XGBoost accuracy broken down by how
many reviews each drug has. Results go to results/metrics/leak_free_rerun.json.

    python -m src.models.leak_free_rerun        (from the repository root)

It runs on pandas 2 (requirements.txt) or pandas 3; two shims keep pandas 3 identical to the
pandas 2 run, see below.
"""
import json
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score

from src.data.processing import (load_data, clean_data, create_effectiveness_label,
                                 extract_temporal_features, prepare_train_test_split)
from src.nlp.utils import extract_text_features, preprocess_reviews
from src.features.engineering import prepare_features
from src.models.train import ModelTrainer

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "metrics" / "leak_free_rerun.json"
t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)


def features(df, fit_index=None):
    X, _ = prepare_features(df, text_col="review_processed", include_tfidf=True,
                            max_tfidf_features=500, categorical_cols=["drugName", "condition"],
                            fit_index=fit_index)
    # pandas 3 builds the sparse TF-IDF columns with NaN as the fill value; pandas 2 used 0.
    tfidf = [c for c in X.columns if c.startswith("tfidf_")]
    X[tfidf] = X[tfidf].apply(lambda col: col.sparse.to_dense() if hasattr(col, "sparse") else col).fillna(0.0)
    return X


def summarise(cls, reg):
    out = {"classification": {k: {m: float(v[m]) for m in ("accuracy", "f1_score", "roc_auc") if v.get(m) is not None}
                              for k, v in cls.items()}}
    if reg:
        out["regression"] = {k: {m: float(v[m]) for m in ("rmse", "mae", "r2_score")} for k, v in reg.items()}
    return out


def main():
    df = load_data(str(ROOT / "data/processed/medVe_data_final_version.xlsx"))
    df = clean_data(df)
    df = create_effectiveness_label(df, threshold=7)
    df = extract_temporal_features(df, date_col="date")
    # pandas 2's astype(str) made the one missing condition the string "nan"; pandas 3 keeps it missing.
    df["condition"] = df["condition"].fillna("nan")
    log("text features (the slow step)")
    df = extract_text_features(df, text_col="review")
    df = preprocess_reviews(df, text_col="review", preprocessed_col="review_processed")

    train_df, test_df = prepare_train_test_split(df, test_size=0.2, random_state=42)
    tr, te = train_df.index, test_df.index
    y_cls, y_reg = df["is_effective"], df["rating"]
    results = {"n_reviews": len(df), "n_train": len(tr), "n_test": len(te)}
    preds = {}

    def run(name, X, regression=True):
        log(f"{name}: training")
        trainer = ModelTrainer(random_state=42)
        cls = trainer.train_classification_models(X.loc[tr], y_cls.loc[tr], X.loc[te], y_cls.loc[te])
        reg = trainer.train_regression_models(X.loc[tr], y_reg.loc[tr], X.loc[te], y_reg.loc[te]) if regression else {}
        results[name] = summarise(cls, reg)
        preds[name] = np.asarray(cls["XGBoost"]["predictions"])

    run("original", features(df))
    X_fixed = features(df, fit_index=tr)
    run("leak_free", X_fixed)
    run("without_averages", X_fixed.drop(columns=["drug_avg_rating", "condition_avg_rating"]), regression=False)

    counts = df["drugName"].map(df["drugName"].value_counts()).loc[te].to_numpy()
    truth = y_cls.loc[te].to_numpy()
    rows = []
    for lo, hi, label in ((1, 1, "1"), (2, 5, "2-5"), (6, 50, "6-50"), (51, 500, "51-500"), (501, 10**9, "501+")):
        m = (counts >= lo) & (counts <= hi)
        rows.append({"drug_reviews": label, "test_reviews": int(m.sum()),
                     **{k: float(accuracy_score(truth[m], p[m])) for k, p in preds.items()}})
    results["xgboost_accuracy_by_drug_review_count"] = rows
    results["single_review_drugs"] = int((df["drugName"].value_counts() == 1).sum())

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    log(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
