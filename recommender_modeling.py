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
DEFAULT_DURATIONS = [180, 210, 240]
DEFAULT_MAX_PEER_CCV = 1000.0
BLOCKED_CATEGORIES = {
    "just chatting",
    "games + demos",
    "the game awards",
    "retro",
    "marvel rivals",
}
FIGHTING_CATEGORY_CANDIDATES = [
    "2xko",
    "street fighter 6",
    "tekken 8",
    "guilty gear -strive-",
    "blazblue: cross tag battle",
    "super smash bros. ultimate",
    "mortal kombat 1",
    "mortal kombat 11",
    "dragon ball fighterz",
    "granblue fantasy versus: rising",
    "under night in-birth ii sys:celes",
    "the king of fighters xv",
    "fatal fury: city of the wolves",
    "marvel vs. capcom fighting collection: arcade classics",
    "killer instinct",
    "soulcalibur vi",
    "skullgirls",
    "rivals of aether ii",
    "brawlhalla",
    "virtua fighter 5 r.e.v.o.",
]
CURATED_CATEGORY_CANDIDATES = FIGHTING_CATEGORY_CANDIDATES + [
    "dark souls ii: scholar of the first sin",
    "dark souls: remastered",
    "dark souls iii",
    "elden ring",
    "hades",
    "hades ii",
    "slay the spire ii",
    "animal well",
    "blue prince",
    "ball x pit",
    "nine sols",
    "ufo 50",
    "the legend of zelda",
    "the legend of zelda: a link to the past",
    "the legend of zelda: ocarina of time",
    "the legend of zelda: majora's mask",
    "the legend of zelda: the wind waker",
    "the legend of zelda: twilight princess",
    "resident evil 4",
    "resident evil 7 biohazard",
    "resident evil village",
    "resident evil: requiem",
    "mega man 11",
    "mega man zero",
    "mega man zero 2",
    "mega man zero 3",
]
FIGHTING_KEYWORDS = (
    "2xko",
    "street fighter",
    "tekken",
    "guilty gear",
    "blazblue",
    "smash bros",
    "mortal kombat",
    "dragon ball fighterz",
    "granblue fantasy versus",
    "under night",
    "king of fighters",
    "fatal fury",
    "marvel vs. capcom",
    "killer instinct",
    "soulcalibur",
    "skullgirls",
    "rivals of aether",
    "brawlhalla",
    "virtua fighter",
)

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


def is_fighting_category(category: str) -> bool:
    cat = str(category).strip().lower()
    if not cat or cat in BLOCKED_CATEGORIES:
        return False
    return cat in FIGHTING_CATEGORY_CANDIDATES or any(keyword in cat for keyword in FIGHTING_KEYWORDS)


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


def _safe_div(num, den):
    den = np.maximum(np.asarray(den, dtype=float), 1.0)
    return np.asarray(num, dtype=float) / den


def _clipped_mean(values: pd.Series, fallback: float) -> float:
    values = pd.Series(values).dropna().astype(float)
    if values.empty:
        return float(fallback)
    upper = values.quantile(0.95)
    return float(values.clip(upper=upper).mean())


def _smoothed_rate(mean_value: float, n: int, prior: float, weight: int = 12) -> float:
    return float((n * mean_value + weight * prior) / max(n + weight, 1))


def _build_candidate_categories(metrics: pd.DataFrame, target_stream: str) -> list[str]:
    stream_lc = target_stream.lower()
    category_summary = metrics.groupby("game_category").agg(
        rows=("game_category", "size"),
        streamers=("stream_name_lc", "nunique"),
        target_rows=("stream_name_lc", lambda values: int((values == stream_lc).sum())),
        med_ccv=("avg_concurrent_viewers", "median"),
    )

    candidates = set(CURATED_CATEGORY_CANDIDATES)
    for category, row in category_summary.iterrows():
        if category in BLOCKED_CATEGORIES:
            continue
        has_target_history = int(row["target_rows"]) > 0
        has_peer_signal = int(row["rows"]) >= 5 and int(row["streamers"]) >= 2 and float(row["med_ccv"]) <= 1000
        if has_target_history or has_peer_signal:
            candidates.add(str(category))

    return sorted(category for category in candidates if category and category not in BLOCKED_CATEGORIES)


def _build_data_sufficiency(metrics: pd.DataFrame, candidates: list[str], target_stream: str) -> dict:
    stream_lc = target_stream.lower()
    target_rows = metrics[metrics["stream_name_lc"] == stream_lc]
    target_candidates = target_rows[target_rows["game_category"].isin(candidates)]
    peer_candidates = metrics[
        metrics["game_category"].isin(candidates) & ~metrics["stream_name_lc"].eq(stream_lc)
    ]
    saturday = target_rows[target_rows["day_of_week"].eq("Saturday")]
    duration_hours = target_rows["stream_duration"].astype(float) / 60.0
    category_counts = target_candidates["game_category"].value_counts()
    thin_categories = int((category_counts < 5).sum())

    median_duration = float(duration_hours.median()) if not duration_hours.empty else 0.0
    p75_duration = float(duration_hours.quantile(0.75)) if not duration_hours.empty else 0.0
    p90_duration = float(duration_hours.quantile(0.90)) if not duration_hours.empty else 0.0
    coverage_level = "good"
    if len(target_rows) < 300 or len(peer_candidates) < 500 or thin_categories > len(category_counts) / 2:
        coverage_level = "thin"
    if len(target_rows) < 150 or len(peer_candidates) < 250:
        coverage_level = "very thin"

    return {
        "coverage_level": coverage_level,
        "target_rows": int(len(target_rows)),
        "candidate_categories": int(len(candidates)),
        "target_candidate_categories": int(target_candidates["game_category"].nunique()),
        "thin_target_categories": thin_categories,
        "peer_candidate_rows": int(len(peer_candidates)),
        "peer_streamers": int(peer_candidates["stream_name_lc"].nunique()),
        "saturday_rows": int(len(saturday)),
        "saturday_fighting_rows": int(saturday["is_fighting_category"].sum()) if not saturday.empty else 0,
        "median_duration_hours": round(median_duration, 2),
        "p75_duration_hours": round(p75_duration, 2),
        "p90_duration_hours": round(p90_duration, 2),
        "recommendation": (
            "Collect 300-500 more comparable streams across Yagami-like categories, with at least 30-50 rows "
            "per test category and fresh Saturday fighting-game peer rows."
        ),
    }


def _chatters_per_100_viewers(rows: pd.DataFrame) -> float:
    if rows.empty or "total_chatters" not in rows or "avg_concurrent_viewers" not in rows:
        return 0.0
    return float((100.0 * _safe_div(rows["total_chatters"], rows["avg_concurrent_viewers"])).clip(0, 300).mean())


def _build_category_pulse(
    metrics: pd.DataFrame,
    candidates: list[str],
    category_stats: dict,
    global_sub_rate: float,
) -> dict:
    max_date = pd.to_datetime(metrics["stream_date"]).max()
    if pd.isna(max_date):
        max_date = pd.Timestamp.now()
    recent_cutoff = max_date - pd.Timedelta(days=45)

    pulse = {}
    for category in candidates:
        rows = metrics[metrics["game_category"] == category].copy()
        stats = category_stats.get(category, {})
        if rows.empty:
            pulse[category] = {
                "status": "Explore",
                "days_since_seen": None,
                "recent_rows": 0,
                "latest_date": "",
                "pulse_score": 24,
                "chatters_per_100_viewers": 0.0,
                "reason": "No direct rows yet; use only as a controlled experiment.",
            }
            continue

        latest = pd.to_datetime(rows["stream_date"]).max()
        days_since_seen = int(max((max_date - latest).days, 0))
        recent_rows = int((pd.to_datetime(rows["stream_date"]) >= recent_cutoff).sum())
        confidence = float(stats.get("confidence", 0.0))
        sub_rate = float(stats.get("subs_per_100_viewers", global_sub_rate))
        freshness = max(0.0, 1.0 - (days_since_seen / 90.0))
        relative_sub = min(sub_rate / max(global_sub_rate, 1e-6), 2.0)
        pulse_score = round(100.0 * (0.45 * confidence + 0.35 * freshness + 0.20 * (relative_sub / 2.0)))

        if confidence >= 0.65 and days_since_seen <= 21:
            status = "Fresh"
        elif days_since_seen <= 60:
            status = "Usable"
        else:
            status = "Stale"

        pulse[category] = {
            "status": status,
            "days_since_seen": days_since_seen,
            "recent_rows": recent_rows,
            "latest_date": latest.strftime("%Y-%m-%d"),
            "pulse_score": int(pulse_score),
            "chatters_per_100_viewers": round(_chatters_per_100_viewers(rows), 1),
            "reason": f"{recent_rows} candidate-pool rows in the last 45 days.",
        }
    return pulse


def _build_community_targets(metrics: pd.DataFrame, target_stream: str, candidates: list[str]) -> list[dict]:
    stream_lc = target_stream.lower()
    relevant = metrics[metrics["game_category"].isin(candidates)].copy()
    if relevant.empty:
        return []

    max_date = pd.to_datetime(relevant["stream_date"]).max()
    recent_cutoff = max_date - pd.Timedelta(days=90)
    target_rows = relevant[relevant["stream_name_lc"] == stream_lc]
    target_categories = set(target_rows["game_category"].dropna().astype(str))
    target_ccv = float(target_rows.tail(30)["avg_concurrent_viewers"].median())
    if not np.isfinite(target_ccv) or target_ccv <= 0:
        target_ccv = 10.0

    targets = []
    for stream_name, rows in relevant[relevant["stream_name_lc"] != stream_lc].groupby("stream_name_lc"):
        categories = sorted(set(rows["game_category"].dropna().astype(str)))
        overlap = sorted(set(categories).intersection(target_categories))
        med_ccv = float(rows["avg_concurrent_viewers"].median())
        latest = pd.to_datetime(rows["stream_date"]).max()
        days_since_seen = int(max((max_date - latest).days, 0))
        recent_rows = int((pd.to_datetime(rows["stream_date"]) >= recent_cutoff).sum())
        category_fit = len(overlap) * 3.0 + min(len(categories), 3)
        size_ratio = med_ccv / max(target_ccv, 1.0)
        size_fit = max(0.0, 3.0 - abs(np.log2(max(size_ratio, 0.25))))
        freshness_fit = max(0.0, 2.0 - days_since_seen / 45.0)
        fit_score = category_fit + size_fit + freshness_fit + min(len(rows), 6) / 3.0
        if size_ratio <= 3:
            target_type = "Peer collab"
        elif size_ratio <= 15:
            target_type = "Reach target"
        else:
            target_type = "Trend watch"

        targets.append(
            {
                "stream_name": str(rows["stream_name"].iloc[-1]),
                "type": target_type,
                "fit_score": round(float(fit_score), 2),
                "median_ccv": round(med_ccv, 1),
                "rows": int(len(rows)),
                "recent_rows": recent_rows,
                "days_since_seen": days_since_seen,
                "categories": ", ".join(overlap or categories[:3]),
                "reason": (
                    f"{len(overlap)} shared {'category' if len(overlap) == 1 else 'categories'}; "
                    f"{recent_rows} rows in last 90 days."
                ),
            }
        )

    return sorted(targets, key=lambda item: item["fit_score"], reverse=True)[:5]


def build_opportunity_stats(df: pd.DataFrame, target_stream: str) -> dict:
    metrics = df.copy()
    metrics["positive_followers"] = np.maximum(metrics["net_follower_change"].astype(float), 0.0)
    metrics["subs_per_100_viewers"] = 100.0 * _safe_div(
        metrics["total_subscriptions"], metrics["avg_concurrent_viewers"]
    )
    metrics["followers_per_100_viewers"] = 100.0 * _safe_div(
        metrics["positive_followers"], metrics["avg_concurrent_viewers"]
    )
    metrics["subs_per_100_viewers"] = metrics["subs_per_100_viewers"].clip(lower=0.0, upper=150.0)
    metrics["followers_per_100_viewers"] = metrics["followers_per_100_viewers"].clip(lower=0.0, upper=150.0)
    metrics["duration_bucket"] = (
        (metrics["stream_duration"].astype(float) / 60.0).round().clip(2, 7) * 60
    ).astype(int)
    metrics["is_fighting_category"] = metrics["game_category"].map(is_fighting_category)

    candidates = _build_candidate_categories(metrics, target_stream)
    candidate_frame = metrics[metrics["game_category"].isin(candidates)].copy()
    prior_frame = candidate_frame if not candidate_frame.empty else metrics
    global_sub_rate = _clipped_mean(prior_frame["subs_per_100_viewers"], 0.0)
    global_follow_rate = _clipped_mean(prior_frame["followers_per_100_viewers"], 0.0)
    all_sub_rate = _clipped_mean(metrics["subs_per_100_viewers"], global_sub_rate)
    all_follow_rate = _clipped_mean(metrics["followers_per_100_viewers"], global_follow_rate)
    global_sub_rate = global_sub_rate or all_sub_rate or 1.0
    global_follow_rate = global_follow_rate or all_follow_rate or 0.25

    category_stats = {}
    stream_lc = target_stream.lower()
    for category in candidates:
        rows = metrics[metrics["game_category"] == category]
        n = int(len(rows))
        streamers = int(rows["stream_name_lc"].nunique()) if n else 0
        target_rows = int((rows["stream_name_lc"] == stream_lc).sum()) if n else 0
        if n:
            sub_mean = _clipped_mean(rows["subs_per_100_viewers"], global_sub_rate)
            follow_mean = _clipped_mean(rows["followers_per_100_viewers"], global_follow_rate)
            source = "target" if target_rows else "peer"
        else:
            # No observed rows means no evidence. Keep curated games visible as
            # exploration ideas, but do not let them outrank categories with data.
            sub_mean = global_sub_rate * 0.65
            follow_mean = global_follow_rate * 0.65
            source = "curated"

        confidence = min(1.0, np.log1p(n) / np.log1p(35)) * min(1.0, max(streamers, 1) / 3.0)
        if target_rows:
            confidence = min(1.0, confidence + min(0.35, np.log1p(target_rows) / np.log1p(20) * 0.35))
        if source == "curated":
            confidence = 0.05
            smoothed_sub_rate = sub_mean
            smoothed_follow_rate = follow_mean
        else:
            smoothed_sub_rate = _smoothed_rate(sub_mean, n, global_sub_rate)
            smoothed_follow_rate = _smoothed_rate(follow_mean, n, global_follow_rate)

        category_stats[category] = {
            "rows": n,
            "streamers": streamers,
            "target_rows": target_rows,
            "source": source,
            "is_fighting": bool(is_fighting_category(category)),
            "confidence": round(float(confidence), 3),
            "subs_per_100_viewers": round(smoothed_sub_rate, 4),
            "followers_per_100_viewers": round(smoothed_follow_rate, 4),
        }

    def factor_table(group_col: str) -> dict:
        out = {}
        for key, group in metrics.groupby(group_col):
            n = int(len(group))
            sub_mean = _clipped_mean(group["subs_per_100_viewers"], all_sub_rate)
            follow_mean = _clipped_mean(group["followers_per_100_viewers"], all_follow_rate)
            out[int(key)] = {
                "subs": round(_smoothed_rate(sub_mean, n, all_sub_rate) / max(all_sub_rate, 1e-6), 4),
                "followers": round(
                    _smoothed_rate(follow_mean, n, all_follow_rate) / max(all_follow_rate, 1e-6), 4
                ),
                "rows": n,
            }
        return out

    stream_rows = metrics[metrics["stream_name_lc"] == target_stream.lower()].sort_values(
        ["stream_date", "stream_start_time"]
    )
    recent = stream_rows.tail(30)
    profile = {
        "recent_avg_ccv": round(float(recent["avg_concurrent_viewers"].mean()), 3),
        "recent_subs_per_stream": round(float(recent["total_subscriptions"].mean()), 3),
        "recent_positive_followers_per_stream": round(float(recent["positive_followers"].mean()), 3),
        "recent_rows": int(len(recent)),
    }
    category_pulse = _build_category_pulse(metrics, candidates, category_stats, global_sub_rate)
    saturday_candidates = [category for category in candidates if is_fighting_category(category)]
    community_targets = _build_community_targets(metrics, target_stream, candidates)
    saturday_community_targets = _build_community_targets(metrics, target_stream, saturday_candidates)
    data_sufficiency = _build_data_sufficiency(metrics, candidates, target_stream)

    return {
        "candidate_categories": candidates,
        "saturday_candidate_categories": saturday_candidates,
        "blocked_categories": sorted(BLOCKED_CATEGORIES),
        "category_stats": category_stats,
        "category_pulse": category_pulse,
        "community_targets": community_targets,
        "saturday_community_targets": saturday_community_targets,
        "data_sufficiency": data_sufficiency,
        "hour_factors": factor_table("start_time_hour"),
        "duration_factors": factor_table("duration_bucket"),
        "global_rates": {
            "candidate_subs_per_100_viewers": round(float(global_sub_rate), 4),
            "candidate_followers_per_100_viewers": round(float(global_follow_rate), 4),
            "fighting_subs_per_100_viewers": round(float(global_sub_rate), 4),
            "fighting_followers_per_100_viewers": round(float(global_follow_rate), 4),
            "all_subs_per_100_viewers": round(float(all_sub_rate), 4),
            "all_followers_per_100_viewers": round(float(all_follow_rate), 4),
        },
        "stream_profile": profile,
        "method": (
            "Scores combine the validated Yagami-scale raw-sub model with smoothed category/hour/duration "
            "opportunity rates learned from the broader streamer sample. Saturdays use Fight Night filtering; "
            "other days use the broader Yagami-compatible category pool."
        ),
    }


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

    opportunity_stats = build_opportunity_stats(df, target_stream)
    stream_rows = df[df["stream_name_lc"] == stream_lc]
    category_options = opportunity_stats["candidate_categories"]
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
        "opportunity_stats": opportunity_stats,
        "start_times": DEFAULT_START_TIMES,
        "durations": DEFAULT_DURATIONS,
        "notes": {
            "subscription_model": "Growth score uses Yagami-scale validation plus broader streamer opportunity rates.",
            "follower_model": "Followers are still experimental; current target-stream validation does not materially beat zero.",
            "candidate_pool": "Recommendations use Yagami-compatible categories, exclude known off-brand categories, and switch Saturdays into Fight Night mode.",
        },
    }
    joblib.dump(artifact, artifact_path, compress=3)
    return artifact
