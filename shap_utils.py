import shap
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def calculate_shap_values(pipeline, X):
    """Calculate SHAP values for a sklearn pipeline with preprocessing"""
    # Get the preprocessed features
    X_processed = pipeline.named_steps['pre'].transform(X)
    
    # Get the actual model (handle TransformedTargetRegressor if present)
    model = pipeline.named_steps['reg']
    if hasattr(model, 'regressor_'):
        model = model.regressor_

    # Calculate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_processed)
    
    return shap_values

def generate_shap_plots(pipeline, df, features):
    """Generate four key SHAP plots as Plotly figures"""
    # Calculate SHAP values
    X = df[features]
    shap_values = calculate_shap_values(pipeline, X)

    # 1) Summary Plot (Bar)
    mean_abs_shap = np.abs(shap_values).mean(0)
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': mean_abs_shap
    }).sort_values('importance', ascending=True)

    fig1 = go.Figure(go.Bar(
        x=feature_importance['importance'],
        y=feature_importance['feature'],
        orientation='h'
    ))
    fig1.update_layout(
        title='SHAP Feature Importance',
        xaxis_title='mean(|SHAP value|)',
        template='plotly_dark',
        plot_bgcolor='#1e1e1e',
        paper_bgcolor='#121212',
        font={'color': '#e0e0e0'}
    )

    # 2) Scatter Plot (SHAP vs Feature) for top 3 features
    top_features = feature_importance['feature'].tail(3).tolist()
    fig2 = make_subplots(rows=1, cols=3)
    
    for i, feat in enumerate(top_features, 1):
        idx = features.index(feat)
        fig2.add_trace(
            go.Scatter(
                x=X[feat],
                y=shap_values[:, idx],
                mode='markers',
                name=feat,
                marker=dict(
                    color=X[feat],
                    colorscale='RdBu',
                    showscale=True
                )
            ),
            row=1, col=i
        )
        fig2.update_xaxes(title_text=feat, row=1, col=i)
        fig2.update_yaxes(title_text='SHAP value' if i==1 else '', row=1, col=i)

    fig2.update_layout(
        title='SHAP Dependence Plots',
        template='plotly_dark',
        plot_bgcolor='#1e1e1e',
        paper_bgcolor='#121212',
        font={'color': '#e0e0e0'},
        showlegend=False,
        height=400
    )

    # Convert to JSON for template
    plot_data = {
        'summary': fig1.to_json(),
        'dependence': fig2.to_json(),
    }
    
    return plot_data
