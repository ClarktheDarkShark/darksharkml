import numpy as np
# Patch np.bool for SHAP compatibility with NumPy >=1.24
if not hasattr(np, 'bool'):
    np.bool = bool

import shap
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64

def fig_to_base64(fig):
    """Convert a matplotlib figure to a base64-encoded PNG image"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64

def calculate_shap_values(pipeline, X):
    """Calculate SHAP values for a sklearn pipeline with preprocessing"""
    X_processed = pipeline.named_steps['pre'].transform(X)
    model = pipeline.named_steps['reg']
    if hasattr(model, 'regressor_'):
        model = model.regressor_
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_processed)
        base_value = explainer.expected_value
        return shap_values, base_value, explainer
    except Exception as e:
        print(f"SHAP calculation failed: {str(e)}")
        # Return dummy values in case of error
        return np.zeros((X_processed.shape[0], X_processed.shape[1])), 0, None

def calculate_shap_explanation(pipeline, X):
    X_processed = pipeline.named_steps['pre'].transform(X)
    model = pipeline.named_steps['reg']
    # Wrap model + preprocessor in a single explainer
    explainer = shap.Explainer(model, X_processed)
    explanation = explainer(X_processed)   # This is an Explanation object
    return explanation

def generate_shap_plots(pipeline, df, features):
    X = df[features]
    # 1) preprocess & explainer
    X_proc = pipeline.named_steps['pre'].transform(X)
    model  = pipeline.named_steps['reg']
    explainer   = shap.Explainer(model, X_proc)
    explanation = explainer(X_proc)

    # 2) Beeswarm (aka summary-dot)
    fig = plt.figure()
    shap.plots.beeswarm(explanation, show=False)
    summary_img = fig_to_base64(fig)

    # 3) Dependence (scatter of top feature colored by 2nd)
    top_idx   = explanation.values.mean(0).argsort()[-1]
    second_idx= explanation.values.mean(0).argsort()[-2]
    fig = plt.figure()
    shap.plots.scatter(
        explanation[:, top_idx],
        color=explanation[:, second_idx],
        show=False
    )
    dependence_img = fig_to_base64(fig)

    # 4) Force (first sample, HTML)
    # note: matplotlib=False gives HTML widget you can .save_html()
    force_html = shap.plots.force(
        explanation[0], matplotlib=False
    ).save_html()

    # 5) Bar (global importance)
    fig = plt.figure()
    shap.plots.bar(explanation, show=False)
    bar_img = fig_to_base64(fig)

    # 6) Decision (first 10)
    fig = plt.figure()
    shap.plots.decision(explanation[:10], show=False)
    decision_img = fig_to_base64(fig)

    return {
        'summary':   summary_img,
        'dependence':dependence_img,
        'force':     force_html,
        'bar':       bar_img,
        'decision':  decision_img
    }
