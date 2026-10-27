from __future__ import annotations

from typing import Literal


TaskType = Literal["classification", "regression"]


def make_traditional_model(name: str, task_type: TaskType, seed: int = 42, n_jobs: int = -1):
    name = name.lower()
    if name == "rf":
        if task_type == "classification":
            from sklearn.ensemble import RandomForestClassifier

            return RandomForestClassifier(
                n_estimators=500,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=seed,
                n_jobs=n_jobs,
                class_weight="balanced_subsample",
            )
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(
            n_estimators=500,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=n_jobs,
        )

    if name == "xgboost":
        try:
            from xgboost import XGBClassifier, XGBRegressor
        except ImportError as exc:
            raise ImportError("Install xgboost to use the XGBoost baseline.") from exc
        common = {
            "n_estimators": 1000,
            "max_depth": 6,
            "learning_rate": 0.01,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": seed,
            "n_jobs": n_jobs,
        }
        if task_type == "classification":
            return XGBClassifier(
                **common,
                objective="binary:logistic",
                eval_metric="logloss",
            )
        return XGBRegressor(**common, objective="reg:squarederror")

    if name == "lightgbm":
        try:
            from lightgbm import LGBMClassifier, LGBMRegressor
        except ImportError as exc:
            raise ImportError("Install lightgbm to use the LightGBM baseline.") from exc
        common = {
            "n_estimators": 1000,
            "max_depth": 6,
            "learning_rate": 0.01,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": seed,
            "n_jobs": n_jobs,
            "verbose": -1,
        }
        if task_type == "classification":
            return LGBMClassifier(**common, objective="binary")
        return LGBMRegressor(**common, objective="regression")

    raise ValueError(f"Unknown traditional model: {name}")


def predict_scores(model, x, task_type: TaskType):
    if task_type == "classification" and hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)
        if proba.ndim == 2 and proba.shape[1] > 1:
            return proba[:, 1]
        return proba.reshape(-1)
    return model.predict(x)
