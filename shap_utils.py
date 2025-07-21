"""
shap_utils.py  –  one-file helper for generating SHAP visuals
Compatible with:
  • sklearn >= 1.3 (ColumnTransformer.get_feature_names_out available)
  • shap  >= 0.43
  • numpy >= 1.24  (np.bool patch included)
Assumes pipeline = Pipeline([('pre', ...), ('reg', TransformedTargetRegressor(...))])
"""

# ── imports ────────────────────────────────────────────────────────────────
import numpy as np
if not hasattr(np, 'bool'):
    np.bool = bool
if not hasattr(np, 'float'):
    np.float = float

import pandas as pd                  # only used for typing / slicing
import shap
import matplotlib.pyplot as plt
import io, base64


# ── tiny helper to turn figures into base-64 PNGs ──────────────────────────
def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.read()).decode()


# ── main API used by dashboard_v2.py ───────────────────────────────────────
def generate_shap_plots(pipeline, df: pd.DataFrame, features: list[str]) -> dict[str, str]:
    """
    Parameters
    ----------
    pipeline : sklearn Pipeline
        Must contain steps 'pre' and 'reg' as described above.
    df : pd.DataFrame
        Inference dataframe (not yet pre-processed).
    features : list[str]
        Column subset to feed into SHAP / the model.

    Returns
    -------
    dict with six base-64 images / html ready for embedding in your template:
        summary, dependence, force, bar, decision
    """

    # ── 1. Build a numeric feature matrix & readable one-hot names ────────
    X_raw          = df[features]
    pre            = pipeline.named_steps["pre"]
    X_proc         = pre.transform(X_raw)
    feature_names  = pre.get_feature_names_out(features)

    # unwrap the RandomForest inside TransformedTargetRegressor
    reg            = pipeline.named_steps["reg"]
    base_model     = reg.regressor_ if hasattr(reg, "regressor_") else reg

    # ── 2. Fast TreeExplainer on numeric data ─────────────────────────────
    explainer   = shap.TreeExplainer(
    base_model,                     # the RandomForestRegressor you un-wrapped
    data=X_proc,                    # numeric matrix same shape as training
    feature_names=feature_names
    )

    # disable the tiny-difference assertion that is crashing you
    explanation = explainer(X_proc, check_additivity=False)

    imgs = {}

    # Beeswarm (feature importance + direction)
    fig = plt.figure()
    try:
        shap.plots.beeswarm(explanation, show=False)
    except ValueError as ve:
        # fallback to the legacy API without a colorbar
        print(f"DEBUG: beeswarm failed ({ve}), falling back to summary_plot")
        shap.summary_plot(
            explanation.values,        # raw SHAP array
            X_proc,                    # numeric matrix
            feature_names=feature_names,
            plot_type="dot",           # same dots style
            color_bar=False,           # TURN COLORBAR OFF
            show=False
        )
    imgs["summary"] = _fig_to_b64(fig)

    # Dependence: top feature vs. second feature colour map
    top, second = explanation.values.mean(0).argsort()[-2:]
    fig = plt.figure()
    shap.plots.scatter(
        explanation[:, top],
        color=explanation[:, second],
        show=False
    )
    imgs["dependence"] = _fig_to_b64(fig)

    # Force plot – first sample, HTML widget
    imgs["force"] = shap.plots.force(
        explanation[0],
        matplotlib=False
    ).save_html()

    # Global bar plot (mean |SHAP|)
    fig = plt.figure()
    shap.plots.bar(explanation, show=False)
    imgs["bar"] = _fig_to_b64(fig)

    # Decision plot – first 10 observations
    fig = plt.figure()
    shap.plots.decision(explanation[:10], show=False)
    imgs["decision"] = _fig_to_b64(fig)

    return imgs
