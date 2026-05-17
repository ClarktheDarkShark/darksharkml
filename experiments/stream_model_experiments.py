from __future__ import annotations

import argparse
import json
import os
import re
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from feature_engineering import _prepare_training_frame

from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler


TARGET_STREAM_DEFAULT = "thelegendyagami"
DEFAULT_TARGETS = ("total_subscriptions", "net_follower_change")
BASELINE_MODELS = {"zero", "target_last", "target_roll7", "train_mean"}
NON_NEGATIVE_TARGETS = {"total_subscriptions", "avg_concurrent_viewers", "positive_follower_change"}


def signed_log1p(y):
    y = np.asarray(y)
    return np.sign(y) * np.log1p(np.abs(y))


def signed_expm1(y):
    y = np.asarray(y)
    return np.sign(y) * np.expm1(np.abs(y))


class PositiveHurdleRegressor(BaseEstimator, RegressorMixin):
    """Estimate E[y] as P(y > threshold) * E[y | y > threshold]."""

    def __init__(
        self,
        classifier=None,
        regressor=None,
        threshold: float = 0.0,
        log_amount: bool = True,
    ):
        self.classifier = classifier
        self.regressor = regressor
        self.threshold = threshold
        self.log_amount = log_amount

    def fit(self, X, y):
        y = np.asarray(y, dtype=float).ravel()
        positive = y > self.threshold
        self.positive_rate_ = float(np.mean(positive)) if len(y) else 0.0

        if positive.any() and (~positive).any():
            classifier = self.classifier or HistGradientBoostingClassifier(random_state=42)
            self.classifier_ = clone(classifier)
            self.classifier_.fit(X, positive.astype(int))
        else:
            self.classifier_ = None

        if positive.any():
            regressor = self.regressor or HistGradientBoostingRegressor(random_state=42)
            self.regressor_ = clone(regressor)
            amount = y[positive]
            if self.log_amount:
                amount = np.log1p(amount)
            self.regressor_.fit(X[positive], amount)
            self.amount_fallback_ = float(np.mean(amount))
        else:
            self.regressor_ = None
            self.amount_fallback_ = 0.0
        return self

    def predict(self, X):
        if self.classifier_ is None:
            probability = np.full(X.shape[0], self.positive_rate_, dtype=float)
        else:
            probability = self.classifier_.predict_proba(X)[:, 1]

        if self.regressor_ is None:
            amount = np.full(X.shape[0], self.amount_fallback_, dtype=float)
        else:
            amount = self.regressor_.predict(X)
        if self.log_amount:
            amount = np.expm1(amount)
        amount = np.maximum(amount, 0.0)
        return probability * amount


def join_raw_tags(x):
    if isinstance(x, pd.DataFrame):
        x = x.iloc[:, 0]
    elif not isinstance(x, pd.Series):
        x = pd.Series(np.asarray(x).ravel())

    def _join(tags):
        if isinstance(tags, (list, tuple, np.ndarray)):
            return " ".join(str(t) for t in tags)
        if pd.isna(tags):
            return ""
        return str(tags)

    return x.apply(_join)


def normalize_tags(value):
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, tuple):
        return [str(v) for v in value]
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, list):
                return [str(v) for v in parsed]
        except json.JSONDecodeError:
            pass
        return [stripped]
    return []


def load_daily_stats(database_url: str | None = None) -> pd.DataFrame:
    load_dotenv(ROOT / ".env")
    db_url = database_url or os.environ.get("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL is not configured.")
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)
    engine = create_engine(db_url)
    return pd.read_sql("SELECT * FROM daily_stats", con=engine)


def prepare_model_frame(df_daily: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df_daily = df_daily.copy()
    df_daily["raw_tags"] = df_daily.get("tags", pd.Series(index=df_daily.index)).apply(normalize_tags)
    df_clean, features, _ = _prepare_training_frame(df_daily)
    df_clean = df_clean.copy()
    df_clean["stream_name_lc"] = df_clean["stream_name"].astype(str).str.lower()
    df_clean["stream_date"] = pd.to_datetime(df_clean["stream_date"]).dt.normalize()
    df_clean["game_category"] = df_clean["game_category"].astype(str).str.lower()
    if "net_follower_change" in df_clean.columns:
        df_clean["positive_follower_change"] = np.maximum(df_clean["net_follower_change"].astype(float), 0.0)
    df_clean = df_clean.sort_values(["stream_name_lc", "stream_date", "stream_start_time"]).reset_index(drop=True)
    return df_clean, list(features)


def select_feature_columns(base_features: list[str], feature_set: str) -> list[str]:
    features = list(base_features)
    if feature_set == "planner":
        return features
    if feature_set == "planner_no_tags":
        return [f for f in features if f != "raw_tags"]
    if feature_set == "planner_with_streamer_id":
        return ["stream_name"] + features
    if feature_set == "controllable_only":
        keep = {
            "day_of_week",
            "start_time_hour",
            "is_weekend",
            "days_since_previous_stream",
            "game_category",
            "stream_duration",
            "raw_tags",
        }
        return [f for f in features if f in keep]
    raise ValueError(f"Unknown feature set: {feature_set}")


def make_folds(
    df: pd.DataFrame,
    target_stream: str,
    n_folds: int,
    min_train_frac: float,
) -> list[dict]:
    stream = target_stream.lower()
    target_dates = sorted(df.loc[df["stream_name_lc"] == stream, "stream_date"].dropna().unique())
    if len(target_dates) < 20:
        raise RuntimeError(f"Not enough target-stream dates to backtest: {len(target_dates)}")

    start_idx = int(len(target_dates) * min_train_frac)
    start_idx = max(1, min(start_idx, len(target_dates) - 1))
    boundaries = np.linspace(start_idx, len(target_dates), n_folds + 1).astype(int)
    boundaries = np.unique(boundaries)

    folds = []
    for i in range(len(boundaries) - 1):
        test_start_idx = int(boundaries[i])
        test_end_idx = int(boundaries[i + 1])
        if test_end_idx <= test_start_idx:
            continue
        cutoff_start = pd.Timestamp(target_dates[test_start_idx])
        cutoff_end = pd.Timestamp(target_dates[test_end_idx]) if test_end_idx < len(target_dates) else pd.Timestamp.max
        folds.append(
            {
                "fold": len(folds) + 1,
                "cutoff_start": cutoff_start,
                "cutoff_end": cutoff_end,
                "test_start_idx": test_start_idx,
                "test_end_idx": test_end_idx,
            }
        )
    return folds


def scope_train_mask(
    df: pd.DataFrame,
    cutoff_start: pd.Timestamp,
    target_stream: str,
    scope: str,
    max_peer_ccv: float,
) -> pd.Series:
    stream = target_stream.lower()
    before = df["stream_date"] < cutoff_start
    if scope == "target_only":
        return before & (df["stream_name_lc"] == stream)
    if scope in {"all_streams", "all_streams_yagami_x5"}:
        return before
    if scope in {"no_large_streamers", "no_large_streamers_yagami_x5"}:
        historical = df.loc[before]
        med = historical.groupby("stream_name_lc")["avg_concurrent_viewers"].median()
        allowed = set(med[med <= max_peer_ccv].index)
        allowed.add(stream)
        return before & df["stream_name_lc"].isin(allowed)
    raise ValueError(f"Unknown training scope: {scope}")


def test_mask_for_fold(df: pd.DataFrame, fold: dict, target_stream: str) -> pd.Series:
    stream = target_stream.lower()
    return (
        (df["stream_name_lc"] == stream)
        & (df["stream_date"] >= fold["cutoff_start"])
        & (df["stream_date"] < fold["cutoff_end"])
    )


def upsample_target_rows(train_df: pd.DataFrame, target_stream: str, scope: str) -> pd.DataFrame:
    match = re.search(r"_x(\d+)$", scope)
    if not match:
        return train_df
    factor = int(match.group(1))
    if factor <= 1:
        return train_df
    target_rows = train_df[train_df["stream_name_lc"] == target_stream.lower()]
    if target_rows.empty:
        return train_df
    return pd.concat([train_df] + [target_rows] * (factor - 1), ignore_index=True)


def any_tags_present(series: pd.Series) -> bool:
    for value in series:
        if isinstance(value, (list, tuple, np.ndarray)) and len(value) > 0:
            return True
        if isinstance(value, str) and value.strip():
            return True
    return False


def make_preprocessor(X_train: pd.DataFrame, style: str, max_tag_features: int) -> ColumnTransformer:
    bool_cols = X_train.select_dtypes(include=["bool"]).columns.tolist()
    numeric_cols = [c for c in X_train.select_dtypes(include=[np.number]).columns if c not in bool_cols]
    categorical_cols = [
        c
        for c in X_train.select_dtypes(include=["object", "category"]).columns
        if c != "raw_tags"
    ]

    transformers = []
    if numeric_cols:
        numeric_transformer = StandardScaler() if style in {"linear", "mlp"} else "passthrough"
        transformers.append(("num", numeric_transformer, numeric_cols))
    if bool_cols:
        transformers.append(("bool", "passthrough", bool_cols))
    if "raw_tags" in X_train.columns and any_tags_present(X_train["raw_tags"]):
        tag_pipeline = Pipeline(
            [
                ("join", FunctionTransformer(join_raw_tags, validate=False)),
                (
                    "vectorize",
                    CountVectorizer(
                        max_features=max_tag_features,
                        ngram_range=(1, 2),
                        token_pattern=r"(?u)\b\w+\b",
                    ),
                ),
            ]
        )
        transformers.append(("tags", tag_pipeline, ["raw_tags"]))
    if categorical_cols:
        if style in {"linear", "mlp"}:
            cat_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        else:
            cat_transformer = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        transformers.append(("cat", cat_transformer, categorical_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.0)


def make_model(model_name: str, target: str, random_state: int):
    if model_name == "dummy_mean":
        reg = DummyRegressor(strategy="mean")
        style = "linear"
        transform_target = False
    elif model_name == "ridge":
        reg = Ridge(alpha=10.0, random_state=random_state)
        style = "linear"
        transform_target = True
    elif model_name == "hgb":
        reg = HistGradientBoostingRegressor(
            loss="absolute_error",
            learning_rate=0.05,
            max_iter=250,
            max_leaf_nodes=15,
            min_samples_leaf=10,
            l2_regularization=1.0,
            random_state=random_state,
        )
        style = "tree"
        transform_target = True
    elif model_name == "hgb_poisson":
        if target not in NON_NEGATIVE_TARGETS:
            return None, None
        reg = HistGradientBoostingRegressor(
            loss="poisson",
            learning_rate=0.05,
            max_iter=250,
            max_leaf_nodes=15,
            min_samples_leaf=10,
            l2_regularization=1.0,
            random_state=random_state,
        )
        style = "tree"
        transform_target = False
    elif model_name == "hurdle_hgb":
        classifier = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=200,
            max_leaf_nodes=15,
            min_samples_leaf=10,
            l2_regularization=1.0,
            random_state=random_state,
        )
        regressor = HistGradientBoostingRegressor(
            loss="absolute_error",
            learning_rate=0.05,
            max_iter=200,
            max_leaf_nodes=15,
            min_samples_leaf=5,
            l2_regularization=1.0,
            random_state=random_state,
        )
        reg = PositiveHurdleRegressor(classifier=classifier, regressor=regressor, threshold=0.0)
        style = "tree"
        transform_target = False
    elif model_name == "rf":
        reg = RandomForestRegressor(
            n_estimators=350,
            max_depth=8,
            min_samples_leaf=3,
            max_features=0.75,
            random_state=random_state,
            n_jobs=-1,
        )
        style = "tree"
        transform_target = True
    elif model_name == "extra_trees":
        reg = ExtraTreesRegressor(
            n_estimators=400,
            max_depth=9,
            min_samples_leaf=3,
            max_features=0.75,
            random_state=random_state,
            n_jobs=-1,
        )
        style = "tree"
        transform_target = True
    elif model_name == "mlp":
        reg = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            alpha=0.01,
            learning_rate_init=0.001,
            early_stopping=True,
            max_iter=600,
            random_state=random_state,
        )
        style = "mlp"
        transform_target = True
    else:
        raise ValueError(f"Unknown model: {model_name}")

    if transform_target:
        reg = TransformedTargetRegressor(
            regressor=reg,
            func=signed_log1p,
            inverse_func=signed_expm1,
            check_inverse=False,
        )
    return reg, style


def clip_predictions(y_pred: np.ndarray, target: str) -> np.ndarray:
    y_pred = np.asarray(y_pred, dtype=float)
    if target in NON_NEGATIVE_TARGETS:
        y_pred = np.maximum(y_pred, 0.0)
    return y_pred


def baseline_predictions(
    model_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str,
    target_stream: str,
) -> np.ndarray:
    target_history = train_df[train_df["stream_name_lc"] == target_stream.lower()].sort_values(
        ["stream_date", "stream_start_time"]
    )
    if model_name == "zero":
        value = 0.0
    elif model_name == "target_last":
        value = target_history[target].iloc[-1] if len(target_history) else train_df[target].mean()
    elif model_name == "target_roll7":
        value = target_history[target].tail(7).mean() if len(target_history) else train_df[target].mean()
    elif model_name == "train_mean":
        value = train_df[target].mean()
    else:
        raise ValueError(f"Unknown baseline: {model_name}")
    return np.full(len(test_df), value, dtype=float)


def summarize_predictions(preds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["target", "scope", "feature_set", "model"]
    for keys, group in preds.groupby(group_cols, dropna=False):
        y_true = group["y_true"].to_numpy(dtype=float)
        y_pred = group["y_pred"].to_numpy(dtype=float)
        if len(y_true) == 0:
            continue
        spearman = pd.Series(y_true).corr(pd.Series(y_pred), method="spearman") if len(y_true) > 1 else np.nan
        rows.append(
            {
                **dict(zip(group_cols, keys)),
                "rows": int(len(group)),
                "folds": int(group["fold"].nunique()),
                "mae": float(mean_absolute_error(y_true, y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan,
                "spearman": float(spearman) if pd.notna(spearman) else np.nan,
                "pred_mean": float(np.mean(y_pred)),
                "actual_mean": float(np.mean(y_true)),
            }
        )
    summary = pd.DataFrame(rows)
    return summary.sort_values(["target", "mae", "rmse"]).reset_index(drop=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Backtest stream recommendation models against future thelegendyagami rows."
    )
    parser.add_argument("--stream", default=TARGET_STREAM_DEFAULT)
    parser.add_argument("--targets", default=",".join(DEFAULT_TARGETS))
    parser.add_argument(
        "--models",
        default="zero,target_last,target_roll7,train_mean,hgb,hgb_poisson,hurdle_hgb,rf,extra_trees,ridge,mlp",
    )
    parser.add_argument(
        "--scopes",
        default="target_only,all_streams,all_streams_yagami_x5,no_large_streamers,no_large_streamers_yagami_x5",
    )
    parser.add_argument("--feature-sets", default="planner")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--min-train-frac", type=float, default=0.58)
    parser.add_argument("--max-peer-ccv", type=float, default=1000.0)
    parser.add_argument("--max-tag-features", type=int, default=750)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--out-dir", default=str(ROOT / "experiments" / "runs"))
    parser.add_argument("--export-best", action="store_true", help="Persist the best fitted model per target.")
    return parser


def split_csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def run_experiments(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    raw = load_daily_stats()
    df, base_features = prepare_model_frame(raw)
    target_stream = args.stream.lower()
    folds = make_folds(df, target_stream, args.folds, args.min_train_frac)
    targets = split_csv(args.targets)
    scopes = split_csv(args.scopes)
    feature_sets = split_csv(args.feature_sets)
    models = split_csv(args.models)

    run_dir = Path(args.out_dir) / datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stream": args.stream,
        "targets": targets,
        "scopes": scopes,
        "feature_sets": feature_sets,
        "models": models,
        "folds": [
            {
                "fold": f["fold"],
                "cutoff_start": str(f["cutoff_start"].date()),
                "cutoff_end": "end" if f["cutoff_end"] is pd.Timestamp.max else str(f["cutoff_end"].date()),
            }
            for f in folds
        ],
        "source_rows": int(len(raw)),
        "prepared_rows": int(len(df)),
        "target_stream_rows": int((df["stream_name_lc"] == target_stream).sum()),
        "base_feature_count": int(len(base_features)),
        "max_peer_ccv": args.max_peer_ccv,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    prediction_records = []
    best_fits: dict[str, tuple[float, Pipeline, dict]] = {}

    for target in targets:
        if target not in df.columns:
            raise ValueError(f"Target {target!r} is not present in the prepared frame.")

        for feature_set in feature_sets:
            features = select_feature_columns(base_features, feature_set)
            missing = [f for f in features if f not in df.columns]
            if missing:
                raise ValueError(f"Feature set {feature_set!r} references missing columns: {missing}")

            for scope in scopes:
                for fold in folds:
                    train_mask = scope_train_mask(df, fold["cutoff_start"], target_stream, scope, args.max_peer_ccv)
                    test_mask = test_mask_for_fold(df, fold, target_stream)
                    train_df = df.loc[train_mask].dropna(subset=[target]).copy()
                    test_df = df.loc[test_mask].dropna(subset=[target]).copy()
                    if train_df.empty or test_df.empty:
                        continue

                    train_df = upsample_target_rows(train_df, target_stream, scope)
                    X_train = train_df[features]
                    y_train = train_df[target].astype(float).to_numpy()
                    X_test = test_df[features]
                    y_test = test_df[target].astype(float).to_numpy()

                    for model_name in models:
                        try:
                            if model_name in BASELINE_MODELS:
                                y_pred = baseline_predictions(model_name, train_df, test_df, target, target_stream)
                                fit_meta = {"type": "baseline"}
                            else:
                                reg, style = make_model(model_name, target, args.random_state)
                                if reg is None:
                                    continue
                                pre = make_preprocessor(X_train, style, args.max_tag_features)
                                model = Pipeline([("pre", pre), ("reg", clone(reg))])
                                with warnings.catch_warnings():
                                    warnings.simplefilter("ignore")
                                    model.fit(X_train, y_train)
                                y_pred = model.predict(X_test)
                                fit_meta = {
                                    "type": "learned",
                                    "style": style,
                                    "train_rows": int(len(train_df)),
                                }
                            y_pred = clip_predictions(y_pred, target)
                        except Exception as exc:
                            prediction_records.append(
                                {
                                    "target": target,
                                    "scope": scope,
                                    "feature_set": feature_set,
                                    "model": model_name,
                                    "fold": fold["fold"],
                                    "error": repr(exc),
                                }
                            )
                            continue

                        for row_idx, truth, pred in zip(test_df.index, y_test, y_pred):
                            row = test_df.loc[row_idx]
                            prediction_records.append(
                                {
                                    "target": target,
                                    "scope": scope,
                                    "feature_set": feature_set,
                                    "model": model_name,
                                    "fold": fold["fold"],
                                    "cutoff_start": str(fold["cutoff_start"].date()),
                                    "stream_date": str(pd.Timestamp(row["stream_date"]).date()),
                                    "stream_name": row["stream_name"],
                                    "game_category": row["game_category"],
                                    "start_time_hour": row["start_time_hour"],
                                    "stream_duration": row["stream_duration"],
                                    "train_rows": int(len(train_df)),
                                    "train_target_rows": int((train_df["stream_name_lc"] == target_stream).sum()),
                                    "test_rows_in_fold": int(len(test_df)),
                                    "y_true": float(truth),
                                    "y_pred": float(pred),
                                    "abs_error": float(abs(truth - pred)),
                                    **fit_meta,
                                }
                            )

    preds = pd.DataFrame(prediction_records)
    errors = preds[preds["error"].notna()] if "error" in preds.columns else pd.DataFrame()
    scored = preds[preds["y_true"].notna()].copy() if "y_true" in preds.columns else pd.DataFrame()
    summary = summarize_predictions(scored) if not scored.empty else pd.DataFrame()

    preds.to_csv(run_dir / "predictions.csv", index=False)
    summary.to_csv(run_dir / "summary.csv", index=False)
    if not errors.empty:
        errors.to_csv(run_dir / "errors.csv", index=False)

    if args.export_best and not summary.empty:
        export_best_models(df, base_features, summary, targets, feature_sets, scopes, args, run_dir)

    return summary, preds, run_dir


def export_best_models(
    df: pd.DataFrame,
    base_features: list[str],
    summary: pd.DataFrame,
    targets: Iterable[str],
    feature_sets: Iterable[str],
    scopes: Iterable[str],
    args: argparse.Namespace,
    run_dir: Path,
) -> None:
    target_stream = args.stream.lower()
    artifacts = {}
    for target in targets:
        eligible = summary[(summary["target"] == target) & (~summary["model"].isin(BASELINE_MODELS))]
        if eligible.empty:
            continue
        best = eligible.sort_values(["mae", "rmse"]).iloc[0].to_dict()
        feature_set = str(best["feature_set"])
        scope = str(best["scope"])
        model_name = str(best["model"])
        features = select_feature_columns(base_features, feature_set)
        cutoff_start = df.loc[df["stream_name_lc"] == target_stream, "stream_date"].max() + pd.Timedelta(days=1)
        train_mask = scope_train_mask(df, cutoff_start, target_stream, scope, args.max_peer_ccv)
        train_df = upsample_target_rows(df.loc[train_mask].dropna(subset=[target]).copy(), target_stream, scope)
        X_train = train_df[features]
        y_train = train_df[target].astype(float).to_numpy()
        reg, style = make_model(model_name, target, args.random_state)
        if reg is None:
            continue
        model = Pipeline([("pre", make_preprocessor(X_train, style, args.max_tag_features)), ("reg", reg)])
        model.fit(X_train, y_train)
        artifacts[target] = {
            "model": model,
            "features": features,
            "best_summary": best,
            "training_scope": scope,
            "feature_set": feature_set,
            "model_name": model_name,
        }
    if artifacts:
        joblib.dump(artifacts, run_dir / "best_models.joblib")


def main() -> int:
    args = build_arg_parser().parse_args()
    summary, _, run_dir = run_experiments(args)
    print(f"Run directory: {run_dir}")
    if summary.empty:
        print("No scored predictions were produced.")
        return 1
    display_cols = ["target", "scope", "feature_set", "model", "rows", "folds", "mae", "rmse", "r2", "spearman"]
    for target, group in summary.groupby("target"):
        print(f"\nTop results for {target}:")
        print(group[display_cols].head(12).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
