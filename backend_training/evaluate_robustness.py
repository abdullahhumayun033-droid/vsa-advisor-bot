"""Robustness, sensitivity and comparative evaluation for the VSA Advisor Bot.

This script evaluates whether the project's VSA + machine-learning results
remain reasonably stable when the feature set, time period, transaction costs
and model family are varied. It is an offline evaluation component and does not
take part in live trading or change the deployed model during inference.

The evaluation compares the compact core VSA feature set with the full model
feature set, performs walk-forward testing across sequential future folds, and
uses bootstrap resampling to provide uncertainty intervals for selected test
metrics. It also measures sensitivity to increasing spread costs using the
already-selected deployed model and stored recommendation threshold.

Logistic Regression and Random Forest remain the project's required candidate
models. Decision Tree and Histogram Gradient Boosting are included only as
additional contextual benchmarks so their behaviour can be compared without
expanding the formal LR/RF model-selection search. The script also records the
project's rationale for retaining or excluding different model families.

All generated outputs are written to the results directory as evaluation
evidence, including feature ablation, walk-forward performance, bootstrap
intervals, transaction-cost sensitivity and additional model benchmarks.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from modules.ml_pipeline import MODEL_FEATURES, RESULTS_DIR, purged_chronological_split


CORE_VSA_FEATURES = [
    "scenario_id", "direction_num", "relative_spread", "relative_volume",
    "close_location", "body_ratio", "cab_relative_volume", "bars_after_cab",
    "had_opposite_sweep", "breakout_strength",
]


def make_lr(seed: int = 42):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)),
    ])


def make_rf(seed: int = 42):
    return RandomForestClassifier(
        n_estimators=80, max_depth=8, min_samples_leaf=80,
        class_weight="balanced_subsample", n_jobs=-1, random_state=seed,
    )


def positive_probability(model, frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    return model.predict_proba(frame[features])[:, list(model.classes_).index(1)]


def select_threshold(model, validation: pd.DataFrame, features: list[str]) -> float:
    y = validation["target"].astype(int).to_numpy()
    proba = positive_probability(model, validation, features)
    rows = []
    for threshold in np.round(np.arange(0.30, 0.801, 0.01), 2):
        pred = (proba >= threshold).astype(int)
        rows.append((f1_score(y, pred, zero_division=0), balanced_accuracy_score(y, pred), -threshold, threshold))
    return float(max(rows)[3])


def score(model, frame: pd.DataFrame, features: list[str], threshold: float) -> dict:
    y = frame["target"].astype(int).to_numpy()
    proba = positive_probability(model, frame, features)
    pred = (proba >= threshold).astype(int)
    accepted = frame.loc[pred == 1]
    return {
        "threshold": threshold, "rows": len(frame), "accepted_signals": int(pred.sum()),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "roc_auc": roc_auc_score(y, proba),
        "gross_total_r": float(accepted["result_r"].sum()),
        "gross_expectancy_r": float(accepted["result_r"].mean()) if len(accepted) else np.nan,
        "net_total_r": float((accepted["result_r"] - accepted["spread_cost_r"]).sum()),
        "net_expectancy_r": float((accepted["result_r"] - accepted["spread_cost_r"]).mean()) if len(accepted) else np.nan,
    }


def feature_ablation(data: pd.DataFrame) -> pd.DataFrame:
    train, validation, test, _ = purged_chronological_split(data, 36)
    feature_sets = {"core_vsa_10": CORE_VSA_FEATURES, "full_vsa_context_24": MODEL_FEATURES}
    rows = []
    for set_name, features in feature_sets.items():
        model = make_lr()
        model.fit(train[features], train["target"].astype(int))
        threshold = select_threshold(model, validation, features)
        row = score(model, test, features, threshold)
        row.update({"feature_set": set_name, "feature_count": len(features), "model": "LR"})
        rows.append(row)
    return pd.DataFrame(rows)


def feature_set_walk_forward(data: pd.DataFrame) -> pd.DataFrame:
    """Compare the 10-feature core with the 24-feature set across four future folds."""
    ordered = data.sort_values("time").reset_index(drop=True)
    n = len(ordered)
    initial = int(n * 0.50)
    block = max(int(n * 0.125), 1)
    feature_sets = {"core_vsa_10": CORE_VSA_FEATURES, "full_vsa_context_24": MODEL_FEATURES}
    rows = []
    for fold in range(4):
        test_start = initial + fold * block
        test_end = n if fold == 3 else min(test_start + block, n)
        history = ordered.iloc[:test_start].copy()
        inner_cut = int(len(history) * 0.80)
        validation_start_time = pd.to_datetime(history.iloc[inner_cut]["time"])
        train = history.iloc[:inner_cut].copy()
        train = train[pd.to_datetime(train["time"]) < validation_start_time - pd.Timedelta(minutes=180)]
        validation = history.iloc[inner_cut:].copy()
        test = ordered.iloc[test_start:test_end].copy()
        for set_name, features in feature_sets.items():
            model = make_lr(142 + fold)
            model.fit(train[features], train["target"].astype(int))
            threshold = select_threshold(model, validation, features)
            row = score(model, test, features, threshold)
            row.update({
                "fold": fold + 1,
                "feature_set": set_name,
                "feature_count": len(features),
                "train_rows": len(train),
                "validation_rows": len(validation),
                "test_start": str(test.iloc[0]["time"]),
                "test_end": str(test.iloc[-1]["time"]),
            })
            rows.append(row)
    return pd.DataFrame(rows)


def feature_ablation_bootstrap(data: pd.DataFrame, repeats: int = 1_000, seed: int = 42) -> pd.DataFrame:
    """Add uncertainty intervals for the fixed unseen test-period feature comparison."""
    train, validation, test, _ = purged_chronological_split(data, 36)
    feature_sets = {"core_vsa_10": CORE_VSA_FEATURES, "full_vsa_context_24": MODEL_FEATURES}
    fitted = {}
    for set_name, features in feature_sets.items():
        model = make_lr()
        model.fit(train[features], train["target"].astype(int))
        threshold = select_threshold(model, validation, features)
        fitted[set_name] = (features, model, threshold)

    rng = np.random.default_rng(seed)
    indices = [rng.integers(0, len(test), size=len(test)) for _ in range(repeats)]
    rows = []
    for set_name, (features, model, threshold) in fitted.items():
        y = test["target"].astype(int).to_numpy()
        proba = positive_probability(model, test, features)
        pred = (proba >= threshold).astype(int)
        net_r = (test["result_r"] - test["spread_cost_r"]).to_numpy()
        samples = {"f1": [], "balanced_accuracy": [], "net_expectancy_r": []}
        for sample in indices:
            sample_pred = pred[sample]
            sample_y = y[sample]
            samples["f1"].append(f1_score(sample_y, sample_pred, zero_division=0))
            samples["balanced_accuracy"].append(balanced_accuracy_score(sample_y, sample_pred))
            accepted = net_r[sample][sample_pred == 1]
            samples["net_expectancy_r"].append(float(accepted.mean()) if len(accepted) else np.nan)
        point = score(model, test, features, threshold)
        for metric in samples:
            values = np.asarray(samples[metric], dtype=float)
            rows.append({
                "feature_set": set_name,
                "feature_count": len(features),
                "metric": metric,
                "point_estimate": point[metric],
                "bootstrap_mean": float(np.nanmean(values)),
                "ci_2_5_percent": float(np.nanpercentile(values, 2.5)),
                "ci_97_5_percent": float(np.nanpercentile(values, 97.5)),
                "bootstrap_repeats": repeats,
                "test_rows": len(test),
            })
    return pd.DataFrame(rows)


def additional_model_benchmarks(data: pd.DataFrame) -> pd.DataFrame:
    """Benchmark other model families without changing the required LR/RF selection."""
    train, validation, test, _ = purged_chronological_split(data, 36)
    models = {
        "LR_VSA_BOT": make_lr(),
        "RF_VSA_BOT": make_rf(),
        "DECISION_TREE_BENCHMARK": DecisionTreeClassifier(
            max_depth=4, min_samples_leaf=20, class_weight="balanced", random_state=42
        ),
        "HIST_GRADIENT_BOOSTING_BENCHMARK": HistGradientBoostingClassifier(
            max_depth=3, max_iter=100, l2_regularization=1.0, class_weight="balanced", random_state=42
        ),
    }
    rows = []
    for model_name, model in models.items():
        model.fit(train[MODEL_FEATURES], train["target"].astype(int))
        threshold = select_threshold(model, validation, MODEL_FEATURES)
        for split_name, split in (("validation", validation), ("test", test)):
            row = score(model, split, MODEL_FEATURES, threshold)
            row.update({"model": model_name, "split": split_name, "selection_role": "required_candidate" if model_name in {"LR_VSA_BOT", "RF_VSA_BOT"} else "context_benchmark"})
            rows.append(row)
    return pd.DataFrame(rows)


def model_family_justification() -> pd.DataFrame:
    return pd.DataFrame([
        ["Logistic Regression", "Required candidate and deployed model", "Calibrated probability and exact coefficient-based explanations", "Linear decision boundary", "Retained: best validation F1 among the required LR/RF candidates and strongest transparency fit"],
        ["Random Forest", "Required candidate", "Non-linear interactions and robust tabular baseline", "Individual predictions are harder to explain faithfully to novice users", "Retained as separately trained comparison; not deployed"],
        ["Decision Tree", "Additional context benchmark", "Human-readable non-linear rules", "Unstable and prone to overfitting on 431 setup examples", "Benchmarked only; not added to selection search"],
        ["Histogram Gradient Boosting", "Additional context benchmark", "Strong non-linear tabular learner", "Lower transparency and greater tuning burden for the small dataset", "Benchmarked only; not added to selection search"],
        ["Neural Network / Reinforcement Learning", "Literature and future-work comparator", "Can learn complex sequential policies", "Data-hungry, harder to validate and explain; unnecessary for this bounded prototype", "Excluded to avoid unjustified complexity and weak reproducibility"],
    ], columns=["model_family", "role", "strength", "limitation", "project_decision"])


def walk_forward(data: pd.DataFrame) -> pd.DataFrame:
    ordered = data.sort_values("time").reset_index(drop=True)
    n = len(ordered)
    initial = int(n * 0.50)
    block = max(int(n * 0.125), 1)
    rows = []
    for fold in range(4):
        test_start = initial + fold * block
        test_end = n if fold == 3 else min(test_start + block, n)
        history = ordered.iloc[:test_start].copy()
        inner_cut = int(len(history) * 0.80)
        validation_start_time = pd.to_datetime(history.iloc[inner_cut]["time"])
        embargo = pd.Timedelta(minutes=180)
        train = history.iloc[:inner_cut].copy()
        train = train[pd.to_datetime(train["time"]) < validation_start_time - embargo]
        validation = history.iloc[inner_cut:].copy()
        test = ordered.iloc[test_start:test_end].copy()
        for model_name, model in [("LR", make_lr(42 + fold)), ("RF", make_rf(42 + fold))]:
            model.fit(train[MODEL_FEATURES], train["target"].astype(int))
            threshold = select_threshold(model, validation, MODEL_FEATURES)
            row = score(model, test, MODEL_FEATURES, threshold)
            row.update({
                "fold": fold + 1, "model": model_name,
                "train_rows": len(train), "validation_rows": len(validation),
                "test_start": str(test.iloc[0]["time"]), "test_end": str(test.iloc[-1]["time"]),
            })
            rows.append(row)
    return pd.DataFrame(rows)


def cost_sensitivity(data: pd.DataFrame) -> pd.DataFrame:
    _, _, test, _ = purged_chronological_split(data, 36)
    meta = __import__("json").loads(Path("models/model_metadata.json").read_text())
    model = joblib.load("models/selected_model.joblib")
    threshold = float(meta["recommendation_threshold"])
    accepted = test.loc[positive_probability(model, test, MODEL_FEATURES) >= threshold].copy()
    rows = []
    for multiplier in [0.0, 1.0, 1.5, 2.0]:
        net = accepted["result_r"] - accepted["spread_cost_r"] * multiplier
        rows.append({
            "spread_multiplier": multiplier, "accepted_signals": len(accepted),
            "total_r": float(net.sum()), "expectancy_r": float(net.mean()),
            "profitable": bool(net.sum() > 0),
        })
    return pd.DataFrame(rows)


def main() -> None:
    data = pd.read_csv(RESULTS_DIR / "vsa_setup_training_dataset.csv", parse_dates=["time", "cab_time"])
    ablation = feature_ablation(data)
    walk = walk_forward(data)
    costs = cost_sensitivity(data)
    feature_walk = feature_set_walk_forward(data)
    feature_bootstrap = feature_ablation_bootstrap(data)
    extra_models = additional_model_benchmarks(data)
    ablation.to_csv(RESULTS_DIR / "feature_ablation.csv", index=False)
    walk.to_csv(RESULTS_DIR / "walk_forward_evaluation.csv", index=False)
    costs.to_csv(RESULTS_DIR / "transaction_cost_sensitivity.csv", index=False)
    feature_walk.to_csv(RESULTS_DIR / "feature_set_walk_forward.csv", index=False)
    feature_bootstrap.to_csv(RESULTS_DIR / "feature_ablation_bootstrap.csv", index=False)
    extra_models.to_csv(RESULTS_DIR / "additional_model_benchmarks.csv", index=False)
    model_family_justification().to_csv(RESULTS_DIR / "model_family_justification.csv", index=False)
    print("Feature ablation\n", ablation.to_string(index=False))
    print("\nWalk-forward summary\n", walk.groupby("model")[["f1", "balanced_accuracy", "net_expectancy_r"]].mean().to_string())
    print("\nTransaction-cost sensitivity\n", costs.to_string(index=False))
    print("\nFeature-set walk-forward summary\n", feature_walk.groupby("feature_set")[["f1", "balanced_accuracy", "net_expectancy_r"]].mean().to_string())
    print("\nAdditional model-family benchmarks\n", extra_models.to_string(index=False))


if __name__ == "__main__":
    main()
