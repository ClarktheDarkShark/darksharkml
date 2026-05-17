from __future__ import annotations

from recommender_modeling import train_recommender_artifact


if __name__ == "__main__":
    artifact = train_recommender_artifact()
    print("Saved recommender_artifacts.joblib")
    for target, info in artifact["models"].items():
        validation = info["validation"]
        print(
            f"{target}: scope={info['scope']} feature_set={info['feature_set']} "
            f"mae={validation['mae']:.3f} promoted={info['promoted']}"
        )
