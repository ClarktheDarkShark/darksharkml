# pipeline.py

import warnings

# Ignore all UserWarnings whose message starts with "n_quantiles"
warnings.filterwarnings(
    "ignore",
    message=r"n_quantiles .* greater than the total number of samples",
    category=UserWarning
)

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition         import TruncatedSVD
from sklearn.svm import SVR
from sklearn.preprocessing import QuantileTransformer

from sklearn.base import TransformerMixin, BaseEstimator

class ShiftedLogTransformer(BaseEstimator, TransformerMixin):
    """
    On fit: compute shift = max(−y.min()+ε, ε)
    On transform:   log1p(y + shift)
    On inverse:  expm1(y_trans) − shift
    Ensures y+shift ≥ 0 for Poisson/Tweedie.
    """
    def __init__(self, epsilon: float = 1e-6):
        self.epsilon = epsilon
        self.shift_ = None

    def fit(self, y, **fit_params):
        # y comes in as shape (n_samples,) or (n_samples, 1)
        y = np.asarray(y).ravel()
        y_min = y.min()
        # if already ≥ 0, we still want a tiny epsilon to avoid log1p(0)
        self.shift_ = (-y_min + self.epsilon) if y_min < 0 else self.epsilon
        return self

    def transform(self, y):
        y = np.asarray(y)
        return np.log1p(y + self.shift_)

    def inverse_transform(self, y_trans):
        y_trans = np.asarray(y_trans)
        return np.expm1(y_trans) - self.shift_
    

def signed_log1p(y):
    return np.sign(y) * np.log1p(np.abs(y))

def signed_expm1(y):
    return np.sign(y) * (np.expm1(np.abs(y)))


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE BUILD
# ─────────────────────────────────────────────────────────────────────────────
def join_raw_tags(x):
    """
    Accept anything (list, ndarray, Series, DataFrame) that scikit‑learn
    might send from a ColumnTransformer and return a 1‑D Pandas Series
    where each element is a single whitespace‑joined tag string.
    """
    # 1) Normalise to 1‑D iterable of row values
    if isinstance(x, pd.DataFrame):
        x = x.iloc[:, 0]                         # first (and only) column
    elif isinstance(x, pd.Series):
        pass                                     # already 1‑D
    else:
        # covers ndarray, list, tuple …
        x = pd.Series(np.asarray(x).ravel())

    # 2) Robust join
    return x.apply(
        lambda tags: " ".join(tags)              # list/tuple/array → string
        if isinstance(tags, (list, tuple, np.ndarray))
        else ("" if pd.isna(tags) else str(tags))# NaNs / weird values
    )



def _build_pipeline(X: pd.DataFrame):
    bool_cols        = X.select_dtypes(include=['bool']).columns.tolist()
    numeric_cols_all = X.select_dtypes(include=[np.number]).columns.tolist()

    # KEEP all numeric cols now; no special exclusion
    numeric_cols = [c for c in numeric_cols_all if c not in bool_cols]

    categorical_cols = [c for c in X.select_dtypes(include=['object','category']).columns
                        if c!='day_of_week' and c!='raw_tags'] 
    
    # print(X)
    
    # # 2) Build a tag‑vectorizing pipeline:
    # tag_pipeline = Pipeline([
    #     ('join',       FunctionTransformer(join_raw_tags, validate=False)),
    #     ('vectorize',  CountVectorizer(
    #                        max_features=200,
    #                        token_pattern=r"(?u)\b\w+\b"
    #                    )),
    #     ('svd',        TruncatedSVD(n_components=16, random_state=42)),
    # ])

    tag_pipeline = Pipeline([
        ('join',       FunctionTransformer(join_raw_tags, validate=False)),
        ('vectorize',  CountVectorizer(max_features=2000,
                                    ngram_range=(1,2),
                                    token_pattern=r"(?u)\b\w+\b"))
        # no SVD
    ])

    ordered_days = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numeric_cols),
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
    ], 
    remainder='drop',
    sparse_threshold=0.0)

    
    
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
                  transformer=FunctionTransformer(signed_log1p, signed_expm1))

    hgb = HistGradientBoostingRegressor(
        early_stopping=False,
        validation_fraction=0.2,
        random_state=42,
        max_iter=200
    )
    # transformer = FunctionTransformer(signed_log1p, signed_expm1, check_inverse=False)
    transformer = ShiftedLogTransformer(epsilon=1e-6)

    # 3) wrap HGB in TransformedTargetRegressor
    hgb_ttr = TransformedTargetRegressor(regressor=hgb, transformer=transformer)

    # from sklearn.ensemble import BaggingRegressor
    # hgb = BaggingRegressor(
    #     estimator=hgb_ttr,
    #     n_estimators=10,       # enough bags to estimate σ
    #     bootstrap=True,
    #     random_state=42,
    #     n_jobs=-1
    # )

    # svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)

    # # ── 2) Wrap in a log‑transform TTR just like you did for RF ───────────────
    # svr = TransformedTargetRegressor(
    #     regressor=svr,
    #     transformer=FunctionTransformer(signed_log1p, signed_expm1),
    #     check_inverse=False
    # )

    qt = FunctionTransformer(
        QuantileTransformer(output_distribution='normal', random_state=42).fit_transform,
        inverse_func=None,   # let the regressor predict in gaussian space
        validate=False)

    svr = TransformedTargetRegressor(
            regressor=SVR(),
            transformer=qt,
            check_inverse=False)

    # return Pipeline([('pre', preprocessor), ('reg', rf)]), 'rf'
    return Pipeline([('pre', preprocessor), ('reg', hgb_ttr)]), 'hgb'
    # return Pipeline([('pre', preprocessor), ('reg', svr)]), 'svr'