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
if not hasattr(np, 'object'):
    np.object = object

import pandas as pd                  # only used for typing / slicing
import shap
import matplotlib.pyplot as plt
import io, base64
import plotly.express as px
import plotly.graph_objects as go
from sklearn.compose import TransformedTargetRegressor


# ── tiny helper to turn figures into base-64 PNGs ──────────────────────────
def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.read()).decode()


# # ── main API used by dashboard_v2.py ───────────────────────────────────────
# def generate_shap_plots(pipeline, df: pd.DataFrame, features: list[str]) -> dict[str, str]:
#     """
#     Parameters
#     ----------
#     pipeline : sklearn Pipeline
#         Must contain steps 'pre' and 'reg' as described above.
#     df : pd.DataFrame
#         Inference dataframe (not yet pre-processed).
#     features : list[str]
#         Column subset to feed into SHAP / the model.

#     Returns
#     -------
#     dict with six base-64 images / html ready for embedding in your template:
#         summary, dependence, force, bar, decision
#     """

#     # ── 1. Build a numeric feature matrix & readable one-hot names ────────
#     X_raw          = df[features]
#     pre            = pipeline.named_steps["pre"]
#     X_proc         = pre.transform(X_raw)
#     feature_names  = pre.get_feature_names_out(features)

#     # unwrap the RandomForest inside TransformedTargetRegressor
#     reg            = pipeline.named_steps["reg"]
#     base_model     = reg.regressor_ if hasattr(reg, "regressor_") else reg

#     # ── 2. Fast TreeExplainer on numeric data ─────────────────────────────
#     explainer   = shap.TreeExplainer(
#     base_model,                     # the RandomForestRegressor you un-wrapped
#     data=X_proc,                    # numeric matrix same shape as training
#     feature_names=feature_names
#     )

#     explanation = explainer(X_proc, check_additivity=False)

#     imgs = {}

#     # Beeswarm (feature importance + direction)
#     fig = plt.figure()
#     try:
#         shap.plots.beeswarm(explanation, show=False)
#     except ValueError as ve:
#         # fallback to the legacy API without a colorbar
#         print(f"DEBUG: beeswarm failed ({ve}), falling back to summary_plot")
#         shap.summary_plot(
#             explanation.values,        # raw SHAP array
#             X_proc,                    # numeric matrix
#             feature_names=feature_names,
#             plot_type="dot",           # same dots style
#             color_bar=False,           # TURN COLORBAR OFF
#             show=False
#         )
#     imgs["summary"] = _fig_to_b64(fig)

#     # Dependence: top feature vs. second feature colour map
#     top, second = explanation.values.mean(0).argsort()[-2:]
#     fig = plt.figure()
#     shap.plots.scatter(
#         explanation[:, top],
#         color=explanation[:, second],
#         show=False
#     )
#     imgs["dependence"] = _fig_to_b64(fig)

#     # Force plot – first sample, HTML widget
#     # get the Visualizer
#     force_vis = shap.plots.force(
#         explanation[0],
#         matplotlib=False,
#         show=False
#     )

#     # Option A: grab just the snippet
#     force_html = force_vis.html()
#     imgs["force"] = force_html 

#     # Global bar plot (mean |SHAP|)
#     fig = plt.figure()
#     shap.plots.bar(explanation, show=False)
#     imgs["bar"] = _fig_to_b64(fig)

#     # Decision plot – first 10 observations
#     fig = plt.figure()
#     shap.decision_plot(
#         explanation.base_values[0],    # single base value for all samples
#         explanation.values[:10],       # your first 10 rows of SHAP values
#         feature_names,                 # your array of names
#         show=False
#     )
#     imgs["decision"] = _fig_to_b64(fig)

#     print("DEBUG: generate_shap_plots returning images:")
#     for name, content in imgs.items():
#         if content is None:
#             print(f"  {name}: None")
#         else:
#             print(f"  {name}: length={len(content)} chars")
#     return imgs

def df_helper(explanation, feature_names, X_proc):
    shap_df = pd.DataFrame(explanation.values, columns=feature_names)

    # 1b) Raw feature values → DataFrame
    raw_df  = pd.DataFrame(X_proc,     columns=feature_names)

    # 1c) Melt both to long form
    shap_long = shap_df.melt(
        var_name="feature", 
        value_name="shap_value"
    )
    raw_long = raw_df.melt(
        var_name="feature", 
        value_name="feature_value"
    )

    # 1d) Combine them side by side
    plot_df = pd.concat(
        [shap_long, raw_long["feature_value"]], 
        axis=1
    )
    return plot_df

def generate_shap_plots(pipeline, X_raw, features):
    """
    Compute SHAP values for a fitted scikit‑learn pipeline and return HTML
    snippets for the summary (beeswarm), dependence, bar, decision and force
    plots.  The beeswarm is rendered with Plotly to match the dashboard style.
    """
    import shap
    import pandas as pd
    import plotly.express as px
    from sklearn.compose import TransformedTargetRegressor

    # ------------------------------------------------------------------
    # 1. Transform the raw feature frame once
    # ------------------------------------------------------------------
    X_trans = pipeline[:-1].transform(X_raw)

    # ------------------------------------------------------------------
    # 2. Recover or synthesise feature names
    # ------------------------------------------------------------------
    try:
        feature_names = pipeline[:-1].get_feature_names_out(features)
    except Exception:
        feature_names = [f"f{i}" for i in range(X_trans.shape[1])]

    # ------------------------------------------------------------------
    # 3. Fit the SHAP explainer
    # ------------------------------------------------------------------
    reg = pipeline.named_steps["reg"]
    if isinstance(reg, TransformedTargetRegressor):
        reg = reg.regressor_

    explainer   = shap.TreeExplainer(reg, feature_perturbation="interventional")
    shap_values = explainer.shap_values(X_trans)

    # Wrap everything in an Explanation so SHAP is happy with new API
    exp = shap.Explanation(
        values       = shap_values,
        base_values  = explainer.expected_value,
        data         = X_trans,
        feature_names= feature_names,
    )

    # ------------------------------------------------------------------
    # 4a. ― Beeswarm (Plotly)
    # ------------------------------------------------------------------
    # Turn the Explanation into a tidy DataFrame
    shap_df   = pd.DataFrame(exp.values,  columns=feature_names)
    raw_df    = pd.DataFrame(X_trans,     columns=feature_names)

    long_df   = (
        shap_df.melt(var_name="feature", value_name="shap_value")
        .join(
            raw_df.melt(var_name="feature", value_name="feature_value")[
                ["feature_value"]
            ]
        )
    )

    # keep top‑N absolute contributors (otherwise huge frames kill Plotly)
    top_feats = (
        shap_df.abs().mean().sort_values(ascending=False).head(20).index.tolist()
    )
    plot_df_filt = long_df[long_df["feature"].isin(top_feats)]

    beeswarm_fig = px.strip(
        plot_df_filt,
        x="shap_value",
        y="feature",
        color="feature_value",
        stripmode="overlay",
        orientation="h",
        title="SHAP Summary (Beeswarm)",
    )
    beeswarm_fig.update_traces(marker=dict(size=6))

    # ------------------------------------------------------------------
    # 4b. ― Other plots (use SHAP’s built‑ins)
    # ------------------------------------------------------------------
    out = {
        "summary":  beeswarm_fig.to_html(full_html=False),
        "dependence": shap.plots.scatter(exp, show=False).html(),
        "bar":        shap.plots.bar(exp,     show=False).html(),
        "decision":   shap.plots.decision(exp, show=False).html(),
        "force":      shap.force_plot(
            explainer.expected_value, exp.values, exp.data,
            feature_names=feature_names, matplotlib=False
        ).html(),
    }

    return out

