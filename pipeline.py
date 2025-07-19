# pipeline.py

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import FunctionTransformer


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE BUILD
# ─────────────────────────────────────────────────────────────────────────────
def _build_pipeline(X: pd.DataFrame):
    bool_cols        = X.select_dtypes(include=['bool']).columns.tolist()
    numeric_cols_all = X.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols     = [c for c in numeric_cols_all if c not in bool_cols]
    categorical_cols = [c for c in X.select_dtypes(include=['object','category']).columns
                        if c!='day_of_week']

    ordered_days = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    preprocessor = ColumnTransformer(transformers=[
        ('num', MinMaxScaler(), numeric_cols),
        ('bool','passthrough', bool_cols),
        ('dow', OrdinalEncoder(
                    categories=[ordered_days],
                    dtype=int,
                    handle_unknown='use_encoded_value',
                    unknown_value=-1),
         ['day_of_week']),
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),
         categorical_cols),
    ])

    # ─────────────────────────────────────────────────────────────────────────────
    # MODELS
    # ─────────────────────────────────────────────────────────────────────────────

    rf = RandomForestRegressor(
        n_estimators=200, 
        max_depth=5, 
        max_features=0.8,
        bootstrap=True, 
        random_state=42, 
        n_jobs=-1
    )
    rf  = TransformedTargetRegressor(rf,
                  transformer=FunctionTransformer(np.log1p, np.expm1))

    hgb = HistGradientBoostingRegressor(
        early_stopping=False,
        validation_fraction=0.2,
        random_state=42,
        max_iter=200
    )
    # transformer = FunctionTransformer(np.log1p, np.expm1, check_inverse=False)

    # 3) wrap HGB in TransformedTargetRegressor
    # hgb_ttr = TransformedTargetRegressor(regressor=hgb, transformer=transformer)

    # from sklearn.ensemble import BaggingRegressor
    # hgb = BaggingRegressor(
    #     estimator=hgb_ttr,
    #     n_estimators=10,       # enough bags to estimate σ
    #     bootstrap=True,
    #     random_state=42,
    #     n_jobs=-1
    # )

    return Pipeline([('pre', preprocessor), ('reg', rf)]), 'rf'