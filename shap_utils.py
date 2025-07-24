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

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
from sklearn.compose import TransformedTargetRegressor

# ----------------------------------------------------------------------
def generate_shap_plots(pipeline, df: pd.DataFrame, features: list[str]) -> dict[str, str]:
    """
    Parameters
    ----------
    pipeline  : fitted `Pipeline([('pre', …), ('reg', TransformedTargetRegressor|Reg)])`
    df        : *raw* dataframe containing all feature columns *including* 'raw_tags'
    features  : list of column names the model expects (matches training)

    Returns
    -------
    dict[str, str]   HTML blobs keyed by plot name
    """

    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline, FunctionTransformer
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder


    num_cols = [c for c in df[features].select_dtypes(include="number").columns
                if c not in ["start_time_hour"]]  # etc
    cat_cols = [c for c in df[features].select_dtypes(include="object").columns
                if c not in ["raw_tags","day_of_week"]]
    explain_pre = ColumnTransformer([
        ("num",  MinMaxScaler(), num_cols),
        ("tags", Pipeline([
            ("join",      FunctionTransformer(join_raw_tags, validate=False)),
            ("vectorize", CountVectorizer(max_features=200))
        ]), ["raw_tags"]),
        ("dow",  OrdinalEncoder(categories=[['Monday','Tuesday','Wednesday',
                                             'Thursday','Friday','Saturday','Sunday']],
                                handle_unknown="use_encoded_value",
                                unknown_value=-1), ["day_of_week"]),
        ("cat",  OrdinalEncoder(handle_unknown="use_encoded_value",
                                unknown_value=-1),
                 cat_cols),
    ], remainder="drop")

    # 1) transform once through the pre‑processor
    X_raw  = df[features]
    X_bow  = explain_pre.fit_transform(X_raw)  # or fit on your training set once
    fnames = explain_pre.get_feature_names_out(features)

    reg = pipeline.named_steps["reg"]
    expl = shap.TreeExplainer(
        reg.regressor_ if hasattr(reg, "regressor_") else reg,
        data=X_bow,
        feature_names=fnames
    )
    shap_exp = expl(X_bow, check_additivity=False)

    pre    = pipeline.named_steps["pre"]
    X_proc = pre.transform(X_raw)

    # 2) feature names — new tag pipeline may not implement get_feature_names_out
    try:
        # the “official” way; will work if every sub‐transformer implements get_feature_names_out
        feature_names = pre.get_feature_names_out(features)
    except Exception:
        # manual fallback: pull names out of each transformer in order
        feature_names = []
        for name, trans, cols in pre.transformers_:
            if name == "num":
                # numeric columns come through as-is
                feature_names.extend(cols)
            elif name == "tags":
                # tag_pipeline = Pipeline([…, ('svd', TruncatedSVD(n_components=N))])
                n_comp = trans.named_steps["svd"].n_components
                feature_names.extend([f"tags__svd_{i}" for i in range(n_comp)])
            elif name == "bool":
                feature_names.extend(cols)
            elif name == "dow":
                feature_names.extend(cols)        # day_of_week encoded as integer
            elif name == "cat":
                feature_names.extend(cols)        # other categorical cols
            # remainder is 'drop', so nothing else to add

    # 3) unwrap regressor if it’s inside a TransformedTargetRegressor
    reg = pipeline.named_steps["reg"]
    if isinstance(reg, TransformedTargetRegressor):
        reg = reg.regressor_

    # 4) SHAP explanation
    explainer   = shap.TreeExplainer(reg, data=X_proc, feature_names=feature_names)
    explanation = explainer(X_proc, check_additivity=False)

    out: dict[str, str] = {}
    out["js"] = shap.getjs()                       # include once per page

    # ------------------------------------------------------------------
    # helper → long dataframe for Plotly beeswarm etc.
    # ------------------------------------------------------------------
    shap_df = pd.DataFrame(explanation.values, columns=feature_names)
    raw_df  = pd.DataFrame(X_proc,               columns=feature_names)
    plot_df = (
        shap_df.melt(var_name="feature", value_name="shap_value")
        .join(
            raw_df.melt(var_name="feature", value_name="feature_value")[["feature_value"]]
        )
    )

    # ── 1. Global Importance Bar ───────────────────────────────────────
    mean_abs = np.abs(explanation.values).mean(axis=0)
    df_bar = pd.DataFrame({
        "feature": feature_names,
        "mean_abs": mean_abs
    })
    fig_bar = px.bar(
        df_bar,
        x="mean_abs",
        y="feature",
        orientation="h",
        labels={"mean_abs": "mean |SHAP value|", "feature": "feature"},
        title="Global Feature Importance"
    )
    fig_bar.update_yaxes(categoryorder="total descending")
    out["bar"] = fig_bar.to_html(full_html=False)

    # ── 2. Dependence Plot (top‑2 features) ────────────────────────────
    top2 = np.argsort(mean_abs)[-2:]
    i, j = top2
    fig_dep = px.scatter(
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
    out["dependence"] = fig_dep.to_html(full_html=False)

    # ── 3. SHAP Force plot (first sample) ──────────────────────────────
    force_vis = shap.plots.force(
        explanation[0], matplotlib=False, show=False
    )
    out["force"] = force_vis.html()



    # ── 4. Decision Plot (first 10 samples) ────────────────────────────
    base      = explainer.expected_value
    shap_vals = explanation.values[:10]                 # (10, n_features)

    # build a DataFrame that matches the SHAP column space
    X_proc_df = pd.DataFrame(X_proc[:10], columns=feature_names)

    dec_obj = shap.plots.decision(
        base,
        shap_vals,
        features=X_proc_df,               # ← was X_raw.iloc[:10]
        feature_names=feature_names,
        show=False,
        return_objects=True
    )

    order = dec_obj.feature_idx
    vals  = dec_obj.shap_values[:, order]

    # cumulative sums for each sample
    cums = np.c_[np.full((vals.shape[0], 1), dec_obj.base_value),
                 dec_obj.base_value + np.cumsum(vals, axis=1)]

    fig_dec = go.Figure()
    for row in cums:
        fig_dec.add_trace(go.Scatter(
            x=row,
            y=np.arange(len(row)),
            mode="lines",
            line=dict(width=1),
            showlegend=False
        ))

    fig_dec.update_layout(
        title="Decision Plot",
        xaxis=dict(range=dec_obj.xlim),
        yaxis=dict(
            tickmode="array",
            tickvals=np.arange(1, len(order) + 1),
            ticktext=[feature_names[idx] for idx in order],
            autorange="reversed"
        ),
        margin=dict(l=200, r=20, t=50, b=20)
    )
    out["decision"] = fig_dec.to_html(full_html=False)



    # ── 5. Beeswarm (Plotly strip) ─────────────────────────────────────
    top_feats = (
        plot_df
        .assign(abs_shap=lambda d: d["shap_value"].abs())
        .groupby("feature")["abs_shap"]
        .mean()
        .sort_values(ascending=False)
        .head(20).index
    )
    plot_df_filt = plot_df[plot_df["feature"].isin(top_feats)]
    beeswarm_fig = px.strip(
        plot_df_filt,
        x="shap_value",
        y="feature",
        color="feature_value",
        stripmode="overlay",
        orientation="h",
        title="SHAP Summary (Beeswarm)"
    )
    beeswarm_fig.update_traces(marker=dict(size=6))
    out["summary"] = beeswarm_fig.to_html(full_html=False)

    return out


