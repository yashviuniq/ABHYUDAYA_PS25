"""Produce SHAP-based explanations (reason codes) for predictions.

Functions:
- load_pipeline
- explain_dataset -> returns top contributors for each example

Note: expects a pipeline saved by `model_trainer.py` where the pipeline is
Pipeline([('preprocess', ColumnTransformer), ('clf', RandomForestClassifier)])
"""
from __future__ import annotations

import json
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import shap


def load_pipeline(path: str):
    return joblib.load(path)


def _get_feature_names(pipeline) -> List[str]:
    """Extract feature names after preprocessing (works with sklearn >=1.0)"""
    preproc = pipeline.named_steps["preprocess"]

    # ColumnTransformer exposes get_feature_names_out
    try:
        names = preproc.get_feature_names_out()
        return list(names)
    except Exception:
        # Fallback: create approximate names
        num = preproc.transformers_[0][2]
        cat = preproc.transformers_[1][2]
        cat_ohe = pipeline.named_steps["preprocess"].named_transformers_["cat"].named_steps[
            "ohe"
        ]
        cat_names = list(cat_ohe.get_feature_names_out(cat)) if hasattr(cat_ohe, "get_feature_names_out") else list(cat)
        return list(num) + list(cat_names)


def explain_dataset(pipeline, X: pd.DataFrame, top_k: int = 3) -> pd.DataFrame:
    """Return top-k positive contributing features (reason codes) per sample.

    Returns a DataFrame with columns: index, prob, top_reasons (list of tuples)
    where each tuple is (feature_name, shap_value)
    """
    # Extract model and transformed data
    clf = pipeline.named_steps["clf"]
    preproc = pipeline.named_steps["preprocess"]

    X_trans = preproc.transform(X)

    # SHAP TreeExplainer on RandomForest
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_trans)

    # For binary classification shap_values is a list [neg, pos]
    # We focus on class 1 (index 1) contributions
    if isinstance(shap_values, list):
        shap_vals_pos = np.array(shap_values[1])
    else:
        shap_vals_pos = np.array(shap_values)

    feature_names = _get_feature_names(pipeline)

    results = []

    probs = pipeline.predict_proba(X)[:, 1]

    for i, row in enumerate(shap_vals_pos):
        # positive contributions sorted descending
        pos_idx = np.argsort(-row)[:top_k]
        top = [(feature_names[j], float(row[j])) for j in pos_idx]
        results.append({"index": i, "prob": float(probs[i]), "top_reasons": top})

    return pd.DataFrame(results)


def main(pipeline_path: str, data_csv: str, out_json: str = "explanations.json") -> None:
    pipeline = load_pipeline(pipeline_path)
    X = pd.read_csv(data_csv)[pipeline.named_steps["preprocess"].feature_names_in_]

    df = explain_dataset(pipeline, X, top_k=3)
    df.to_json(out_json, orient="records", lines=False)
    print(f"Wrote explanations to {out_json}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate SHAP-based reason codes for a dataset")
    parser.add_argument("--pipeline", default="models/rf_pipeline.joblib")
    parser.add_argument("--data", default="data/patients.csv")
    parser.add_argument("--out", default="explanations.json")
    args = parser.parse_args()
    main(args.pipeline, args.data, args.out)
