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
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition         import TruncatedSVD


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE BUILD
# ─────────────────────────────────────────────────────────────────────────────
def join_raw_tags(df: pd.DataFrame) -> pd.Series:
    """
    Take the df['raw_tags'] column (a List[str]) and 
    turn it into a single whitespace-joined string per row.
    """
    return df['raw_tags'].apply(lambda tags: " ".join(tags))

def _build_pipeline(X: pd.DataFrame):
    bool_cols        = X.select_dtypes(include=['bool']).columns.tolist()
    numeric_cols_all = X.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols     = [c for c in numeric_cols_all
                        if c not in bool_cols + ['start_time_hour']]

    categorical_cols = [c for c in X.select_dtypes(include=['object','category']).columns
                        if c!='day_of_week' and c!='raw_tags'] 
    
    print(X)
    
    # 2) Build a tag‑vectorizing pipeline:
    tag_pipeline = Pipeline([
        ('join',       FunctionTransformer(join_raw_tags, validate=False)),
        ('vectorize',  CountVectorizer(
                           max_features=200,
                           token_pattern=r"(?u)\b\w+\b"
                       )),
        ('svd',        TruncatedSVD(n_components=16, random_state=42)),
    ])

    ordered_days = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    preprocessor = ColumnTransformer(transformers=[
        ('num', MinMaxScaler(), numeric_cols),
        ('tags', tag_pipeline,        ['raw_tags']),
        ('bool','passthrough', bool_cols),
        ('dow', OrdinalEncoder(
                    categories=[ordered_days],
                    dtype=int,
                    handle_unknown='use_encoded_value',
                    unknown_value=-1),
         ['day_of_week']),
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),
         categorical_cols),
    ], remainder='passthrough')

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