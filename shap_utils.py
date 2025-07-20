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

def generate_shap_plots(pipeline, df, features):
    """Generate SHAP plots as base64-encoded PNG images and HTML"""
    X = df[features]
    shap_values, base_value, explainer = calculate_shap_values(pipeline, X)

    # 1) Summary Plot (distribution, importance, direction)
    plt.figure()
    shap.summary_plot(shap_values, X, feature_names=features, show=False, plot_type="dot")
    summary_img = fig_to_base64(plt.gcf())

    # 2) Dependence Plot (top feature, colored by second top feature)
    top_features = np.argsort(np.abs(shap_values).mean(0))[-2:]
    dep_feat = features[top_features[-1]]
    color_feat = features[top_features[-2]]
    plt.figure()
    shap.dependence_plot(dep_feat, shap_values, X, interaction_index=color_feat, show=False)
    dependence_img = fig_to_base64(plt.gcf())

    # 3) Force Plot (first sample)
    force_plot_html = shap.force_plot(
        base_value, shap_values[0], X.iloc[0], matplotlib=True, show=False
    ).save_html(None)
    # For dashboard, just pass the HTML string

    # 4) Bar Plot (global importance)
    plt.figure()
    shap.summary_plot(shap_values, X, feature_names=features, show=False, plot_type="bar")
    bar_img = fig_to_base64(plt.gcf())

    # 5) Decision Plot (first 10 samples)
    plt.figure()
    shap.decision_plot(
        base_value, shap_values[:10], X.iloc[:10], feature_names=features, show=False
    )
    decision_img = fig_to_base64(plt.gcf())

    return {
        'summary': summary_img,
        'dependence': dependence_img,
        'force': force_plot_html,
        'bar': bar_img,
        'decision': decision_img
    }
