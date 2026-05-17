from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

from feature_engineering import _prepare_training_frame

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder


TARGET_STREAM = "thelegendyagami"
DEFAULT_START_TIMES = list(range(24))
DEFAULT_DURATIONS = [120, 180, 240, 300, 360, 420]
DEFAULT_MAX_PEER_CCV = 1000.0

TARGET_CONFIGS = {
    "total_subscriptions": {
        "label": "Expected subscriptions",
        "scope": "no_large_streamers",
        "feature_set": "planner",
        "promoted": True,
        "validation": {
            "mae": 1.225655,
            "zero_baseline_mae": 1.414894,
            "rolling_7_baseline_mae": 2.460486,
            "spearman": 0.414232,
        },
    },
    "net_follower_change": {
        "label": "Expected follower change",
        "scope": "no_large_streamers_yagami_x5",
        "feature_set": "planner_with_streamer_id",
        "promoted": False,
        "validation": {
            "mae": 0.402704,
            "zero_baseline_mae": 0.404255,
            "rolling_7_baseline_mae": 0.443769,
            "spearman": 0.302954,
        },
    },
}


def signed_log1p(y):
    y = np.asarray(y)
    return np.sign(y) * np.log1p(np.abs(y))


def signed_expm1(y):
    y = np.asarray(y)
    return np.sign(y) * np.expm1(np.abs(y))


def normalize_tags(value) -> list[str]:
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


def tags_to_text(value) -> str:
    return " ".join(normalize_tags(value))


def load_daily_stats(database_url: str | None = None) -> pd.DataFrame:
    load_dotenv()
    db_url = database_url or os.environ.get("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL is required to train the recommender.")
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)
    engine = create_engine(db_url)
    return pd.read_sql("SELECT * FROM daily_stats", con=engine)


def prepare_model_frame(df_daily: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df_daily = df_daily.copy()
    df_daily["raw_tags"] = df_daily.get("tags", pd.Series(index=df_daily.index)).apply(normalize_tags)
    df_clean, features, _ = _prepare_training_frame(df_daily)
    df_clean = df_clean.copy()
    df_clean["stream_name"] = df_clean["stream_name"].astype(str)
    df_clean["stream_name_lc"] = df_clean["stream_name"].str.lower()
    df_clean["stream_date"] = pd.to_datetime(df_clean["stream_date"]).dt.normalize()
    df_clean["game_category"] = df_clean["game_category"].astype(str).str.lower()
    df_clean["raw_tags"] = df_clean["raw_tags"].apply(normalize_tags)
    df_clean["raw_tags_text"] = df_clean["raw_tags"].apply(tags_to_text)

    features = ["raw_tags_text" if f == "raw_tags" else f for f in features]
    features = list(dict.fromkeys(features))
    return df_clean.sort_values(["stream_name_lc", "stream_date", "stream_start_time"]).reset_index(drop=True), features


def select_feature_columns(base_features: list[str], feature_set: str) -> list[str]:
    if feature_set == "planner":
        return list(base_features)
    if feature_set == "planner_with_streamer_id":
        return ["stream_name"] + list(base_features)
    if feature_set == "planner_no_tags":
        return [f for f in base_features if f != "raw_tags_text"]
    raise ValueError(f"Unknown feature_set={feature_set!r}")


def scope_train_frame(
    df: pd.DataFrame,
    target_stream: str,
    scope: str,
    max_peer_ccv: float = DEFAULT_MAX_PEER_CCV,
) -> pd.DataFrame:
    stream = target_stream.lower()
    if scope == "target_only":
        train = df[df["stream_name_lc"] == stream].copy()
    elif scope in {"all_streams", "all_streams_yagami_x5"}:
        train = df.copy()
    elif scope in {"no_large_streamers", "no_large_streamers_yagami_x5"}:
        med = df.groupby("stream_name_lc")["avg_concurrent_viewers"].median()
        allowed = set(med[med <= max_peer_ccv].index)
        allowed.add(stream)
        train = df[df["stream_name_lc"].isin(allowed)].copy()
    else:
        raise ValueError(f"Unknown scope={scope!r}")

    if scope.endswith("_x5"):
        target_rows = train[train["stream_name_lc"] == stream]
        train = pd.concat([train] + [target_rows] * 4, ignore_index=True)
    return train


def build_preprocessor(X_train: pd.DataFrame, max_tag_features: int = 750) -> ColumnTransformer:
    bool_cols = X_train.select_dtypes(include=["bool"]).columns.tolist()
    numeric_cols = [c for c in X_train.select_dtypes(include=[np.number]).columns if c not in bool_cols]
    categorical_cols = [
        c
        for c in X_train.select_dtypes(include=["object", "category"]).columns
        if c != "raw_tags_text"
    ]

    transformers = []
    if numeric_cols:
        transformers.append(("num", "passthrough", numeric_cols))
    if bool_cols:
        transformers.append(("bool", "passthrough", bool_cols))
    if "raw_tags_text" in X_train.columns and X_train["raw_tags_text"].str.len().sum() > 0:
        transformers.append(
            (
                "tags",
                CountVectorizer(
                    max_features=max_tag_features,
                    ngram_range=(1, 2),
                    token_pattern=r"(?u)\b\w+\b",
                ),
                "raw_tags_text",
            )
        )
    if categorical_cols:
        transformers.append(
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                categorical_cols,
            )
        )

    return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.0)


def build_hgb_pipeline(X_train: pd.DataFrame) -> Pipeline:
    regressor = HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.05,
        max_iter=250,
        max_leaf_nodes=15,
        min_samples_leaf=10,
        l2_regularization=1.0,
        random_state=42,
    )
    target_regressor = TransformedTargetRegressor(
        regressor=regressor,
        func=signed_log1p,
        inverse_func=signed_expm1,
        check_inverse=False,
    )
    return Pipeline(
        [
            ("pre", build_preprocessor(X_train)),
            ("reg", target_regressor),
        ]
    )


def train_recommender_artifact(
    *,
    target_stream: str = TARGET_STREAM,
    artifact_path: str | Path = "recommender_artifacts.joblib",
    database_url: str | None = None,
) -> dict:
    raw = load_daily_stats(database_url)
    df, base_features = prepare_model_frame(raw)
    stream_lc = target_stream.lower()

    models = {}
    all_features = set()
    for target, config in TARGET_CONFIGS.items():
        features = select_feature_columns(base_features, config["feature_set"])
        train_df = scope_train_frame(df, target_stream, config["scope"]).dropna(subset=[target])
        X_train = train_df[features]
        y_train = train_df[target].astype(float).to_numpy()
        pipeline = build_hgb_pipeline(X_train)
        pipeline.fit(X_train, y_train)
        models[target] = {
            "pipeline": pipeline,
            "features": features,
            "scope": config["scope"],
            "feature_set": config["feature_set"],
            "promoted": config["promoted"],
            "label": config["label"],
            "validation": config["validation"],
            "train_rows": int(len(train_df)),
            "target_stream_train_rows": int((train_df["stream_name_lc"] == stream_lc).sum()),
        }
        all_features.update(features)

    stream_rows = df[df["stream_name_lc"] == stream_lc]
    category_options = (
        stream_rows["game_category"]
        .dropna()
        .astype(str)
        .value_counts()
        .index.tolist()
    )
    tag_options = sorted({tag for tags in stream_rows["raw_tags"] for tag in normalize_tags(tags)})

    inf_cols = [
        "stream_name",
        "stream_name_lc",
        "stream_date",
        "stream_start_time",
        "raw_tags",
    ] + sorted(all_features)
    inf_cols = list(dict.fromkeys([c for c in inf_cols if c in df.columns]))

    artifact = {
        "version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_stream": target_stream,
        "models": models,
        "df_for_inf": df[inf_cols].copy(),
        "category_options": category_options,
        "tag_options": tag_options,
        "start_times": DEFAULT_START_TIMES,
        "durations": DEFAULT_DURATIONS,
        "notes": {
            "subscription_model": "Promoted candidate: beats zero and rolling baselines on future thelegendyagami rows.",
            "follower_model": "Experimental only: does not materially beat the zero baseline on future thelegendyagami rows.",
        },
    }
    joblib.dump(artifact, artifact_path, compress=3)
    return artifact
