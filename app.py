"""Streamlit dashboard for JijaArogyaCare: hospital-themed no-show risk viewer.

- Loads a saved pipeline (`models/rf_pipeline.joblib`)
- Allows generating sample data or uploading CSV
- Shows risk-level indicators: Green/Yellow/Red
- Displays reason codes (from SHAP) for selected patient
"""
from __future__ import annotations

import os
from typing import Optional

import joblib
import pandas as pd
import shap
import streamlit as st

from data_gen import generate_synthetic_data
from explainer import explain_dataset, load_pipeline

MODEL_PATH = "models/rf_pipeline.joblib"


RISK_THRESHOLDS = {
    "green": 0.4,
    "yellow": 0.7,  # >= yellow is yellow; >= red is red
}


def risk_label(prob: float) -> tuple[str, str]:
    if prob >= RISK_THRESHOLDS["yellow"]:
        return "High", "red"
    if prob >= RISK_THRESHOLDS["green"]:
        return "Medium", "orange"
    return "Low", "green"


@st.cache_data
def load_model(path: str = MODEL_PATH):
    return joblib.load(path)


def style_badge(label: str, color: str) -> str:
    return f"<span style='background:{color};color:white;padding:6px 10px;border-radius:6px;font-weight:600'>{label}</span>"


def main():
    st.set_page_config(page_title="JijaArogyaCare — No-Show Risk", layout="wide", page_icon="🏥")

    st.markdown("# 🏥 JijaArogyaCare — No-Show Risk Dashboard")
    st.markdown("---")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Data")
        uploaded = st.file_uploader("Upload patient CSV (optional)", type=["csv"])

        if uploaded is not None:
            df = pd.read_csv(uploaded)
        else:
            n = st.slider("Generate sample patients", 10, 5000, 200)
            if st.button("Generate sample data"):
                df = generate_synthetic_data(n_samples=n)
            else:
                st.info("Upload a CSV or press 'Generate sample data' to create a dataset.")
                df = None

    with col2:
        st.subheader("Model")
        if os.path.exists(MODEL_PATH):
            st.success("Model found")
            if st.button("Reload model"):
                model = load_model()
                st.experimental_rerun()
        else:
            st.warning("Model not found. Run `python model_trainer.py` first.")

    if df is None:
        return

    # Keep a copy of original
    original = df.copy()

    model = load_model()

    # Select features expected by model
    expected = model.named_steps["preprocess"].feature_names_in_
    missing = [f for f in expected if f not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        return

    X = df[expected]
    probs = model.predict_proba(X)[:, 1]
    df["no_show_prob"] = probs

    # Risk category
    df["risk"], df["risk_color"] = zip(*[risk_label(p) for p in probs])

    st.subheader("Predictions")
    st.markdown("**Summary**")
    counts = df["risk"].value_counts().to_dict()
    c1, c2, c3 = st.columns(3)
    c1.markdown(style_badge("Low", "green"), unsafe_allow_html=True)
    c1.caption(f"{counts.get('Low',0)} patients")
    c2.markdown(style_badge("Medium", "orange"), unsafe_allow_html=True)
    c2.caption(f"{counts.get('Medium',0)} patients")
    c3.markdown(style_badge("High", "red"), unsafe_allow_html=True)
    c3.caption(f"{counts.get('High',0)} patients")

    st.dataframe(df.head(200))

    st.subheader("Inspect individual patient")
    idx = st.number_input("Row index", min_value=0, max_value=len(df) - 1, value=0)
    patient = X.iloc[[idx]]
    prob = float(df.loc[idx, "no_show_prob"])
    label, color = risk_label(prob)

    st.markdown(f"**Predicted no-show probability:** {prob:.2f}  ")
    st.markdown(style_badge(label, color), unsafe_allow_html=True)

    # Explain
    if st.button("Explain this prediction"):
        st.info("Computing SHAP reason codes...")
        expl = explain_dataset(model, patient, top_k=3)
        reasons = expl.loc[0, "top_reasons"]

        st.markdown("**Top Reasons (feature, contribution)**")
        for feat, val in reasons:
            sign = "+" if val > 0 else "-"
            st.write(f"{feat}: {sign}{abs(val):.3f}")

        # Try to show a waterfall
        try:
            preproc = model.named_steps["preprocess"]
            X_trans = preproc.transform(patient)
            explainer = shap.TreeExplainer(model.named_steps["clf"])
            shap_values = explainer.shap_values(X_trans)
            # waterfall plot for class 1
            if isinstance(shap_values, list):
                sv = shap_values[1][0]
            else:
                sv = shap_values[0]

            st.pyplot(shap.plots._waterfall.waterfall_legacy(explainer.expected_value[1], sv, feature_names=explainer.feature_names if hasattr(explainer, 'feature_names') else None, show=False))
        except Exception as e:
            st.warning(f"SHAP plotting failed: {e}")

    st.markdown("---")
    st.caption("JijaArogyaCare · A professional no-show risk assistant for hospitals")


if __name__ == "__main__":
    main()
