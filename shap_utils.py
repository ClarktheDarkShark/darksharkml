import numpy as np
# Patch np.bool for SHAP compatibility with NumPy >=1.24
if not hasattr(np, 'bool'):
    np.bool = bool

import shap
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

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
    """Generate SHAP plots as Plotly figures (JSON)"""
    X = df[features]
    shap_values, base_value, explainer = calculate_shap_values(pipeline, X)

    # 1) Summary Plot (distribution, importance, direction)
    summary_fig = shap.summary_plot(
        shap_values, X, feature_names=features, show=False, plot_type="dot"
    )
    summary_plotly = pio.to_json(shap.plots._utils.matplotlib_to_plotly(summary_fig))

    # 2) Dependence Plot (top feature, colored by second top feature)
    top_features = np.argsort(np.abs(shap_values).mean(0))[-2:]
    dep_feat = features[top_features[-1]]
    color_feat = features[top_features[-2]]
    dep_fig = shap.dependence_plot(
        dep_feat, shap_values, X, interaction_index=color_feat, show=False
    )
    dep_plotly = pio.to_json(shap.plots._utils.matplotlib_to_plotly(dep_fig))

    # 3) Force Plot (first sample)
    force_plot_html = shap.force_plot(
        base_value, shap_values[0], X.iloc[0], matplotlib=True, show=False
    ).save_html(None)
    # For dashboard, just pass the HTML string

    # 4) Bar Plot (global importance)
    mean_abs_shap = np.abs(shap_values).mean(0)
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': mean_abs_shap
    }).sort_values('importance', ascending=True)
    bar_fig = go.Figure(go.Bar(
        x=feature_importance['importance'],
        y=feature_importance['feature'],
        orientation='h'
    ))
    bar_fig.update_layout(
        title='SHAP Feature Importance (Bar)',
        xaxis_title='mean(|SHAP value|)',
        template='plotly_dark',
        plot_bgcolor='#1e1e1e',
        paper_bgcolor='#121212',
        font={'color': '#e0e0e0'}
    )
    bar_plotly = bar_fig.to_json()

    # 5) Decision Plot (first 10 samples)
    decision_fig = shap.decision_plot(
        base_value, shap_values[:10], X.iloc[:10], feature_names=features, show=False
    )
    decision_plotly = pio.to_json(shap.plots._utils.matplotlib_to_plotly(decision_fig))

    return {
        'summary': summary_plotly,
        'dependence': dep_plotly,
        'force': force_plot_html,
        'bar': bar_plotly,
        'decision': decision_plotly
    }
