import numpy as np
# Patch np.bool for SHAP compatibility with NumPy >=1.24
if not hasattr(np, 'bool'):
    np.bool = bool

import shap
import matplotlib.pyplot as plt
import io
import base64

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.read()).decode('utf-8')

def generate_shap_plots(pipeline, df, features):
    X = df[features]

    # 1) Make one explainer over pipeline.predict (handles pre + model + inverseTransform)
    explainer   = shap.Explainer(pipeline.predict, X)
    explanation = explainer(X)  # SHAP Explanation object

    # 2) Beeswarm
    fig = plt.figure()
    shap.plots.beeswarm(explanation, show=False)
    summary_img = fig_to_base64(fig)

    # 3) Dependence (scatter)
    top_idx    = explanation.values.mean(0).argsort()[-1]
    second_idx = explanation.values.mean(0).argsort()[-2]
    fig = plt.figure()
    shap.plots.scatter(
        explanation[:, top_idx],
        color=explanation[:, second_idx],
        show=False
    )
    dependence_img = fig_to_base64(fig)

    # 4) Force (HTML)
    force_html = shap.plots.force(
        explanation[0],
        matplotlib=False
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
        'summary':    summary_img,
        'dependence': dependence_img,
        'force':      force_html,
        'bar':        bar_img,
        'decision':   decision_img
    }
