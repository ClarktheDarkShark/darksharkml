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

def generate_shap_plots(pipeline, df: pd.DataFrame, features: list[str]) -> dict[str, str]:
    X_raw         = df[features]
    pre           = pipeline.named_steps["pre"]
    X_proc        = pre.transform(X_raw)
    feature_names = pre.get_feature_names_out(features)
    reg           = pipeline.named_steps["reg"].regressor_

    explainer   = shap.TreeExplainer(reg, data=X_proc, feature_names=feature_names)
    explanation = explainer(X_proc, check_additivity=False)

    out: dict[str, str] = {}
    out["js"] = shap.getjs()

    # ── 1. Interactive feature‐importance bar (mean |SHAP|)
    mean_abs = np.abs(explanation.values).mean(axis=0)
    df_bar = pd.DataFrame({
        "feature": feature_names,
        "mean_abs": mean_abs
    })
    fig = px.bar(
        df_bar,
        x="mean_abs",
        y="feature",
        orientation="h",
        labels={"mean_abs":"mean |SHAP value|","feature":"feature"},
        title="Global Feature Importance"
    )

    # 4. Sort categories so largest mean_abs is at the top
    fig.update_yaxes(categoryorder="total descending")

    out["bar"] = fig.to_html(full_html=False)

    # ── 2. Interactive dependence plot for top 2 features
    top2 = np.argsort(mean_abs)[-2:]
    i, j = top2
    fig = px.scatter(
        x=X_proc[:, i],
        y=explanation.values[:, i],
        color=explanation.values[:, j],
        labels={
          "x": feature_names[i],
          "y": f"SHAP({feature_names[i]})",
          "color": feature_names[j]
        },
        title=f"Dependence: {feature_names[i]} colored by {feature_names[j]}"
    )
    out["dependence"] = fig.to_html(full_html=False)

    # ── 3. SHAP Force plot (still use SHAP’s JS widget)
    force_vis = shap.plots.force(
      explanation[0], matplotlib=False, show=False
    )
    out["force"] = force_vis.html()

    # ── 4. Interactive decision plot ────────────────────────────────
    base      = explainer.expected_value        # scalar
    shap_vals = explanation.values[:10]         # (#samples × #features)

    dec_obj = shap.plots.decision(
        base, 
        shap_vals,
        features=X_raw.iloc[:10],
        feature_names=feature_names,
        show=False,
        return_objects=True
    )
    
    # dec_fig will be None when SHAP draws the chart inline (Plotly backend)
    if dec_fig is not None and hasattr(dec_fig, "to_html"):
        out["decision"] = dec_fig.to_html(full_html=False)
    else:
        # nothing to embed – SHAP already inserted a <script> into the page
        out["decision"] = (
            "<p><em>Decision plot rendered inline by SHAP.</em></p>"
        )

    plot_df = df_helper(explanation, feature_names, X_proc)
    beeswarm_fig = px.strip(
        plot_df,
        x="shap_value",
        y="feature",
        color="feature_value",
        stripmode="overlay",    # dots overlap to show density
        orientation="h",        # feature names on y‐axis
        title="SHAP Summary (Beeswarm)"
    )
    # Optional styling: reduce marker size if you like
    beeswarm_fig.update_traces(marker=dict(size=6))
    out["summary"] = beeswarm_fig.to_html(full_html=False)


    return out