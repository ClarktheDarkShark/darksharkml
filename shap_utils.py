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
if not hasattr(np, "bool"):          # SHAP < 0.44 expects np.bool
    np.bool = bool

import pandas as pd                  # only used for typing / slicing
import shap
import matplotlib.pyplot as plt
import io, base64

# Save original
_orig_colorbar = plt.colorbar

def _debug_colorbar(mappable=None, cax=None, ax=None, *args, **kwargs):
    print(">>> DEBUG colorbar invoked!")
    print("   mappable:", repr(mappable))
    print("   ax param:", ax)
    print("   cax param:", cax)
    print("   plt.gca():", plt.gca())
    fig = plt.gcf()
    print("   fig.axes:", fig.axes)
    for i, a in enumerate(fig.axes):
        print(f"     axes[{i}] children types:", [type(ch) for ch in a.get_children()][:5], "…")
    # call through so we see the real error (or not)
    return _orig_colorbar(mappable, cax=cax, ax=ax, *args, **kwargs)

# Override
plt.colorbar = _debug_colorbar

# ── tiny helper to turn figures into base-64 PNGs ──────────────────────────
def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.read()).decode()


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
    X_raw         = df[features]
    print("DEBUG: X_raw dtypes:", X_raw.dtypes.to_dict())
    pre           = pipeline.named_steps["pre"]
    X_proc        = pre.transform(X_raw)
    print("DEBUG: X_proc shape:", X_proc.shape)
    feature_names = pre.get_feature_names_out(features)
    print("DEBUG: feature_names:", feature_names[:5], "...")

    # unwrap the RandomForest inside TransformedTargetRegressor
    reg         = pipeline.named_steps["reg"]
    base_model  = reg.regressor_ if hasattr(reg, "regressor_") else reg
    print("DEBUG: base_model type:", type(base_model))

    # ── 2. Fast TreeExplainer on numeric data ─────────────────────────────
    explainer = shap.TreeExplainer(
        base_model,
        data=X_proc,
        feature_names=feature_names
    )
    print("DEBUG: Explainer type:", type(explainer))

    # disable the tiny-difference assertion that was crashing you
    try:
        explanation = explainer(X_proc, check_additivity=False)
    except Exception as e:
        print("DEBUG: explainer(...) failed with:", repr(e))
        raise
    print("DEBUG: explanation.values.shape:", explanation.values.shape)
    print("DEBUG: explanation.base_values (first 5):", explanation.base_values[:5])

    imgs = {}

    # ── Beeswarm (feature importance + direction) ────────────────────────
    fig, ax = plt.subplots()
    print("DEBUG: fig.axes BEFORE beeswarm:", fig.axes)

    try:
        # capture whatever beeswarm returns (often a PathCollection or QuadMesh)
        mappable = shap.plots.beeswarm(explanation, show=False)
        print("DEBUG: beeswarm returned mappable:", type(mappable))
    except Exception as e:
        print("DEBUG: beeswarm ERROR:", repr(e))
        # inspect the Axes list
        axes = fig.get_axes()
        print(f"DEBUG: fig has {len(axes)} axes:")
        for idx, a in enumerate(axes):
            print(f"  axes[{idx}] = {a}")
            print("    children count:", len(a.get_children()))
            print("    xlim, ylim:", a.get_xlim(), a.get_ylim())
        # inspect figure children too
        print("DEBUG: fig.get_children():", fig.get_children()[:5], "… (total", len(fig.get_children()), ")")
        raise

    print("DEBUG: fig.axes AFTER beeswarm:", fig.axes)
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
