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


import numpy as np
import pandas as pd  
import shap
import matplotlib.pyplot as plt
import io, base64
import plotly.express as px
import plotly.graph_objects as go
from sklearn.compose import TransformedTargetRegressor, ColumnTransformer
from sklearn.pipeline import Pipeline, FunctionTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder

from pipeline import join_raw_tags


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


# ----------------------------------------------------------------------
def generate_shap_plots(pipeline, df: pd.DataFrame, features: list[str]) -> dict[str, str]:
    """
    Compute SHAP values on a full bag‑of‑words tag matrix (no SVD),
    so each tag token appears by name. Returns HTML snippets.
    """
    df = df.sample(min(1000, len(df)), random_state=0)

    # ────────────────────────────────────────────────────────────────
    # 1) Build an “explain” preprocessor WITHOUT the SVD step
    # ────────────────────────────────────────────────────────────────
    num_cols = [c for c in df[features].select_dtypes(include="number")]
    cat_cols = [c for c in df[features].select_dtypes(include="object")
                if c not in ("raw_tags", "day_of_week")]

    explain_pre = ColumnTransformer(
        transformers=[
            ("num", MinMaxScaler(), num_cols),
            ("tags", Pipeline([
                ("join",      FunctionTransformer(join_raw_tags, validate=False)),
                ("vectorize", CountVectorizer(max_features=200))
            ]), ["raw_tags"]),
            ("dow", OrdinalEncoder(
                categories=[['Monday','Tuesday','Wednesday','Thursday',
                             'Friday','Saturday','Sunday']],
                handle_unknown="use_encoded_value",
                unknown_value=-1
            ), ["day_of_week"]),
            ("cat", OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1
            ), cat_cols),
        ],
        remainder="drop"
    )

    # ────────────────────────────────────────────────────────────────
    # 2) Fit & transform to get a sparse bag‑of‑words matrix
    # ────────────────────────────────────────────────────────────────
    X_raw = df[features]
    X_bow = explain_pre.fit_transform(X_raw)

    # ─── convert to dense if sparse
    if hasattr(X_bow, "toarray"):
        X_bow = X_bow.astype(np.float32).toarray()

    # ────────────────────────────────────────────────────────────────
    # 3) Manually reconstruct the token‑level feature names
    # ────────────────────────────────────────────────────────────────
    fnames: list[str] = []
    for name, trans, cols in explain_pre.transformers_:
        if name == "num":
            fnames.extend(cols)
        elif name == "tags":
            tokens = trans.named_steps["vectorize"].get_feature_names_out()
            fnames.extend([f"{name}__{t}" for t in tokens])
        elif name in ("dow", "cat"):
            fnames.extend(cols)

    # ────────────────────────────────────────────────────────────────
    # 4) Run SHAP on that dense bag‑of‑words data
    # ────────────────────────────────────────────────────────────────
    reg = pipeline.named_steps["reg"]
    if isinstance(reg, TransformedTargetRegressor):
        reg = reg.regressor_

    explainer = shap.TreeExplainer(reg, data=X_bow, feature_names=fnames)
    shap_exp  = explainer(X_bow, check_additivity=False)

    out: dict[str, str] = {"js": shap.getjs()}

    # ────────────────────────────────────────────────────────────────
    # 5) Build a long DataFrame for Plotly (beeswarm, etc.)
    # ────────────────────────────────────────────────────────────────
    plot_df = df_helper(shap_exp, fnames, X_bow)

    # ────────────────────────────────────────────────────────────────
    # 6) Summary Beeswarm (top 20 tokens)
    # ────────────────────────────────────────────────────────────────
    top_feats = (
        plot_df
        .assign(abs_shap=lambda d: d["shap_value"].abs())
        .groupby("feature")["abs_shap"]
        .mean()
        .sort_values(ascending=False)
        .head(20)
        .index
    )
    beeswarm_fig = px.strip(
        plot_df[plot_df["feature"].isin(top_feats)],
        x="shap_value",
        y="feature",
        color="feature_value",
        stripmode="overlay",
        orientation="h",
        title="SHAP Summary (Beeswarm)"
    )
    beeswarm_fig.update_traces(marker=dict(size=6))
    out["summary"] = beeswarm_fig.to_html(full_html=False)

    # ────────────────────────────────────────────────────────────────
    # 7) Dependence Plot (top‑2 tokens)
    # ────────────────────────────────────────────────────────────────
    mean_abs = np.abs(shap_exp.values).mean(axis=0)
    i, j = np.argsort(mean_abs)[-2:]
    dep_fig = px.scatter(
        x=X_bow[:, i],
        y=shap_exp.values[:, i],
        color=shap_exp.values[:, j],
        labels={
            "x": fnames[i],
            "y": f"SHAP({fnames[i]})",
            "color": fnames[j]
        },
        title=f"Dependence: {fnames[i]} colored by {fnames[j]}"
    )
    out["dependence"] = dep_fig.to_html(full_html=False)

    # ────────────────────────────────────────────────────────────────
    # 8) Bar Plot (mean |SHAP|)
    # ────────────────────────────────────────────────────────────────
    df_bar = pd.DataFrame({"feature": fnames, "mean_abs": mean_abs})
    bar_fig = px.bar(
        df_bar.sort_values("mean_abs", ascending=True),
        x="mean_abs",
        y="feature",
        orientation="h",
        title="Global Feature Importance"
    )
    bar_fig.update_yaxes(categoryorder="total descending")
    out["bar"] = bar_fig.to_html(full_html=False)

    # ────────────────────────────────────────────────────────────────
    # 9) Decision Plot (first 10 samples)
    # ────────────────────────────────────────────────────────────────
    base_val   = shap_exp.base_values[0] if hasattr(shap_exp, "base_values") else explainer.expected_value
    shap_vals  = shap_exp.values[:10]
    X_df       = pd.DataFrame(X_bow[:10], columns=fnames)
    dec_obj    = shap.plots.decision(
        base_val,
        shap_vals,
        features=X_df,
        feature_names=fnames,
        feature_display_range=slice(0, 20),
        show=False,
        return_objects=True
    )
    # re‑build as Plotly
    order = dec_obj.feature_idx
    vals  = dec_obj.shap_values[:, order]
    cums  = np.c_[np.full((vals.shape[0], 1), dec_obj.base_value),
                  dec_obj.base_value + np.cumsum(vals, axis=1)]
    fig_dec = go.Figure()
    for row in cums:
        fig_dec.add_trace(go.Scatter(x=row, y=np.arange(len(row)),
                                     mode="lines", line=dict(width=1),
                                     showlegend=False))
    fig_dec.update_layout(
        title="Decision Plot",
        xaxis=dict(range=dec_obj.xlim),
        yaxis=dict(tickmode="array",
                   tickvals=np.arange(1, len(order)+1),
                   ticktext=[fnames[idx] for idx in order],
                   autorange="reversed"),
        margin=dict(l=200, r=20, t=50, b=20)
    )
    out["decision"] = fig_dec.to_html(full_html=False)

    # ────────────────────────────────────────────────────────────────
    # 10) Force Plot (first sample)
    # ────────────────────────────────────────────────────────────────
    force_vis    = shap.plots.force(shap_exp[0], matplotlib=False, show=False)
    out["force"] = force_vis.html()

    return out