import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" 
# from models import DailyStats
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import pandas as pd
import numpy as np


load_dotenv()  # reads .env in working directory
db_url = os.getenv("DATABASE_URL")
db_url = "postgres://u76gs2mfn3l2um:p1577d0e7a90ab74ce0a8dcbd5dab809f4240ab92836929df08e328b6c82c5a13@c2hbg00ac72j9d.cluster-czrs8kj4isg7.us-east-1.rds.amazonaws.com:5432/d35v2smh8r48et"

# Heroku uses postgres:// but SQLAlchemy expects postgresql://
if db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)

engine = create_engine(db_url)

def winsorize_series(s, lo=1.0, hi=99.0):
    lo_v, hi_v = np.percentile(s, [lo, hi])
    return np.clip(s, lo_v, hi_v)

def data_loader():
    # — read & prep
    df_daily = pd.read_sql("SELECT * FROM daily_stats;", con=engine)
    df_daily['raw_tags'] = df_daily['tags'].apply(lambda x: x if isinstance(x, list) else [])

    df_clean, features, hist_cols = _prepare_training_frame(df_daily)

    # match the cleaning in your "full pipeline" example
    # df_clean = df_clean.dropna()
    df_clean = drop_outliers(
        df_clean,
        # cols=['total_subscriptions', 'avg_concurrent_viewers', 'net_follower_change'],
        cols=['total_subscriptions'],
        # cols=['avg_concurrent_viewers'],
        method='iqr',
        factor=2.0
    )
    # df_clean['total_subscriptions'] = winsorize_series(df_clean['total_subscriptions'], lo=1, hi=99)
    # df_clean['net_follower_change'] = winsorize_series(df_clean['net_follower_change'], lo=1, hi=99)

    # stable ordering (not required for the split, but nice to keep)
    df_clean = df_clean.sort_values(['stream_name', 'stream_date']).reset_index(drop=True)

    # build pipeline and grab ONLY the preprocessor
    full_pipe, model_type = _build_pipeline(df_clean[features])
    pre: pd.DataFrame = full_pipe.named_steps['pre']
    # ─────────────────────────────────────────────────────────────────────────────
    # 1) Preprocess (fit on train only, per-stream 80% split by time)
    # ─────────────────────────────────────────────────────────────────────────────
    df_sorted = df_clean.sort_values(['stream_name','stream_date']).reset_index(drop=True)

    # per-stream split point (80%)
    split_points = {name: int(0.8*len(g)) for name, g in df_sorted.groupby('stream_name', sort=False)}

    row_is_train = np.zeros(len(df_sorted), dtype=bool)
    for name, g in df_sorted.groupby('stream_name', sort=False):
        ntr = split_points[name]
        row_is_train[g.index[:ntr]] = True

    # fit preprocessor on TRAIN rows only
    pre = full_pipe.named_steps['pre']
    pre.fit(df_sorted.loc[row_is_train, features])

    # transform ALL rows with train-fitted preprocessor
    def _to_dense(X):
        return X.toarray() if hasattr(X, "toarray") else np.asarray(X)

    # —— Guarantee a test set for sequence windows ——
    def _make_split_points_with_test(df_sorted, timesteps, start_ratio=0.80, min_ratio=0.60, step=0.05):
        """
        Returns split_points[name] = ntr (row count for train per stream) such that
        at least one test window can be formed across the dataset, if feasible.
        """
        # Try progressively smaller train ratios until at least one test window exists
        ratio = start_ratio
        while ratio >= min_ratio:
            split_points = {}
            total_test_windows = 0

            for name, g in df_sorted.groupby('stream_name', sort=False):
                n = len(g)

                if n < timesteps:
                    # Not enough rows for any window; all train, no test here
                    split_points[name] = n
                    continue

                # Base split from ratio
                ntr = int(ratio * n)

                # Ensure at least ONE test window: need ntr <= (n - timesteps)
                ntr = min(ntr, n - timesteps)

                # If the stream is long enough to have at least one train window too,
                # ensure ntr >= timesteps
                if n >= 2 * timesteps:
                    ntr = max(ntr, timesteps)

                # Clamp
                ntr = max(0, min(n, ntr))
                split_points[name] = ntr

                # Count how many test windows this split would yield
                last_start_test = n - timesteps
                if last_start_test >= ntr:
                    total_test_windows += (last_start_test - ntr + 1)

            if total_test_windows > 0:
                return split_points  # success with this ratio

            ratio -= step

        # Last resort: force a split that guarantees test windows if n >= timesteps
        # (may sacrifice train windows on very short streams)
        split_points = {}
        for name, g in df_sorted.groupby('stream_name', sort=False):
            n = len(g)
            split_points[name] = max(0, n - timesteps)  # ensures last_start_test >= ntr
        return split_points

    timesteps = 1


    # —— 1) Build robust split points (guarantee at least one test window overall) ——
    split_points = _make_split_points_with_test(df_sorted, timesteps=timesteps)

    # Your existing transforms
    X_all = _to_dense(pre.transform(df_sorted[features])).astype(np.float32)
    y_all = df_sorted['total_subscriptions'].values.astype(np.float32)
    # y_all = df_sorted['avg_concurrent_viewers'].values.astype(np.float32)
    # y_all = df_sorted['net_follower_change'].values.astype(np.float32)

    # from sklearn.preprocessing import StandardScaler

    # scaler_y = StandardScaler()
    # y_all = scaler_y.fit_transform(y_all.reshape(-1, 1)).astype(np.float32)

    # —— 2) Sequence builder (unchanged) ——


    X_train_seq, y_train_seq = [], []
    X_test_seq,  y_test_seq  = [], []

    for name, g in df_sorted.groupby('stream_name', sort=False):
        idx = g.index.to_numpy()
        n   = len(idx)
        ntr = split_points[name]

        # TRAIN windows: start in [0, ntr - timesteps]
        last_start_train = ntr - timesteps
        if last_start_train >= 0:
            for start in range(0, last_start_train + 1):
                sl = idx[start:start+timesteps]
                X_train_seq.append(X_all[sl, :])
                y_train_seq.append(y_all[sl[-1]])

        # TEST windows: start in [ntr, n - timesteps]
        last_start_test = n - timesteps
        if last_start_test >= ntr:
            for start in range(ntr, last_start_test + 1):
                sl = idx[start:start+timesteps]
                X_test_seq.append(X_all[sl, :])
                y_test_seq.append(y_all[sl[-1]])

    # stack to arrays
    X_train = np.stack(X_train_seq) if X_train_seq else np.empty((0, timesteps, X_all.shape[1]), dtype=np.float32)
    y_train = np.asarray(y_train_seq, dtype=np.float32)

    X_test  = np.stack(X_test_seq)  if X_test_seq  else np.empty((0, timesteps, X_all.shape[1]), dtype=np.float32)
    y_test  = np.asarray(y_test_seq, dtype=np.float32)


    print('X_train shape', X_train.shape)
    print('X_test shape', X_test.shape)

    # (optional) assert we actually have a test set now
    if len(X_test) == 0:
        raise RuntimeError("No test windows could be formed. Consider lowering timesteps or widening the test split.")

    timesteps     = X_train.shape[1]   # e.g. 10
    n_features    = X_train.shape[2]