# train_and_export.py
import joblib
from main import create_app
from predictor import train_predictor, get_predictor_artifacts

if __name__ == "__main__":
    # 1) Spin up your Flask app context so DB can load
    app = create_app()
    # 2) Train once
    train_predictor(app)
    # 3) Grab everything out of memory
    pipeline, df_for_inf, features, cat_opts, start_times, dur_opts, metrics = get_predictor_artifacts()

    # 4) Persist to disk (choose your filename)
    joblib.dump({
        "pipeline": pipeline,
        "df_for_inf": df_for_inf,
        "features": features,
        "stream_category_options_inf": cat_opts,
        "optional_start_times": start_times,
        "stream_duration_opts": dur_opts,
        "metrics": metrics,
    }, "predictor_artifacts.joblib")

    print("Artifacts saved to predictor_artifacts.joblib")
