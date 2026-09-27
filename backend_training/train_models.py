"""Offline model training, selection and evaluation for the VSA Advisor Bot.

This script trains the project's supervised machine-learning models on labelled
examples produced by the formal VSA Scenario 1, 2 and 3 strategy. The models do
not learn directly from arbitrary raw candles; the VSA pipeline first identifies
candidate setups and supplies the market, scenario and outcome features used for
training.

Logistic Regression (LR) and Random Forest (RF) are fitted separately using a
purged chronological train/validation/test split so future-labelled examples do
not leak across the temporal boundaries. Recommendation thresholds are tuned on
the validation period, and the two models are compared using classification,
probability and trading-oriented evidence together with majority-class and
VSA-rule-only baselines.

Model selection is performed from validation results rather than the untouched
test period. The script saves both fitted candidate models, their evaluation
artifacts, the selected fitted model, recommendation threshold and runtime
metadata for later inference. Training is therefore an offline backend process;
the website and live MT5 advisor load an already-fitted model rather than
retraining whenever the interface starts or new market bars arrive.

The generated evidence includes prediction files, confusion matrices, LR
coefficients, RF feature importance, threshold sweeps and managed-R trading
summaries so the final model choice and its behaviour can be reviewed separately
from the live advisory workflow.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    brier_score_loss,
    average_precision_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from modules.ml_pipeline import (
    MODEL_FEATURES,
    MODELS_DIR,
    OUTPUTS_DIR,
    RESULTS_DIR,
    build_vsa_setup_dataset,
    purged_chronological_split,
    ensure_dirs,
    latest_vsa_setup_prediction,
    save_ml_signal_csv,
)


def log(msg: str, log_lines: list[str]) -> None:
    stamp = time.strftime("%H:%M:%S")
    line = f"[{stamp}] {msg}"
    print(line, flush=True)
    log_lines.append(line)


def probabilities(model, df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(df[MODEL_FEATURES])[:, list(model.classes_).index(1)]


def metrics_for(model, df: pd.DataFrame, split_name: str, model_name: str, threshold: float) -> dict:
    X = df[MODEL_FEATURES]
    y = df["target"].astype(int)
    proba = probabilities(model, df)
    pred = (proba >= threshold).astype(int)
    out = {
        "model": model_name,
        "split": split_name,
        "threshold": float(threshold),
        "rows": int(len(df)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, proba)) if len(set(y)) == 2 else np.nan,
        "pr_auc": float(average_precision_score(y, proba)) if len(set(y)) == 2 else np.nan,
        "brier": float(brier_score_loss(y, proba)),
        "predicted_valid_trades": int((pred == 1).sum()),
        "actual_valid_trades": int((y == 1).sum()),
    }
    return out


def threshold_sweep(model, validation: pd.DataFrame, model_name: str) -> pd.DataFrame:
    y = validation["target"].astype(int).to_numpy()
    proba = probabilities(model, validation)
    rows = []
    for threshold in np.round(np.arange(0.30, 0.801, 0.01), 2):
        pred = (proba >= threshold).astype(int)
        rows.append({
            "model": model_name,
            "threshold": float(threshold),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "accepted_signals": int(pred.sum()),
            "acceptance_rate": float(pred.mean()),
        })
    return pd.DataFrame(rows)


def majority_baseline(df: pd.DataFrame, split_name: str) -> dict:
    y = df["target"].astype(int).to_numpy()
    pred = np.zeros(len(y), dtype=int)
    prevalence = float(y.mean())
    return {
        "model": "MAJORITY_AVOID_BASELINE", "split": split_name, "threshold": np.nan,
        "rows": int(len(y)), "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision": 0.0, "recall": 0.0, "f1": 0.0, "roc_auc": 0.5,
        "pr_auc": prevalence, "brier": float(np.mean((y - prevalence) ** 2)),
        "predicted_valid_trades": 0, "actual_valid_trades": int(y.sum()),
    }


def vsa_rule_baseline(df: pd.DataFrame, split_name: str) -> dict:
    y = df["target"].astype(int).to_numpy()
    pred = np.ones(len(y), dtype=int)
    prevalence = float(y.mean())
    return {
        "model": "VSA_RULE_ONLY_BASELINE", "split": split_name, "threshold": np.nan,
        "rows": int(len(y)), "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)), "recall": 1.0,
        "f1": float(f1_score(y, pred, zero_division=0)), "roc_auc": 0.5,
        "pr_auc": prevalence, "brier": float(np.mean((y - prevalence) ** 2)),
        "predicted_valid_trades": int(len(y)), "actual_valid_trades": int(y.sum()),
    }


def trading_evaluation(model, df: pd.DataFrame, model_name: str, threshold: float, reward_risk: float) -> dict:
    accepted = df.loc[probabilities(model, df) >= threshold].copy()
    return trading_summary(accepted, len(df), model_name, threshold)


def trading_summary(accepted: pd.DataFrame, total_rows: int, model_name: str, threshold: float = np.nan) -> dict:
    if accepted.empty:
        return {"model": model_name, "threshold": threshold, "accepted_signals": 0}
    if "result_r" not in accepted:
        raise ValueError("Managed-exit result_r is required for trading evaluation.")
    equity = accepted["result_r"].cumsum()
    drawdown = equity - equity.cummax().clip(lower=0)
    gross_profit = float(accepted.loc[accepted["result_r"] > 0, "result_r"].sum())
    gross_loss = float(-accepted.loc[accepted["result_r"] < 0, "result_r"].sum())
    cost_r = accepted.get("spread_cost_r", pd.Series(0.0, index=accepted.index)).astype(float)
    net_result_r = accepted["result_r"].astype(float) - cost_r
    net_equity = net_result_r.cumsum()
    net_drawdown = net_equity - net_equity.cummax().clip(lower=0)
    return {
        "model": model_name, "threshold": float(threshold),
        "accepted_signals": int(len(accepted)),
        "coverage": float(len(accepted) / total_rows),
        "wins": int((accepted["result_r"] > 0).sum()),
        "losses": int((accepted["result_r"] < 0).sum()),
        "expired": int((accepted["result_r"] == 0).sum()),
        "win_rate": float((accepted["result_r"] > 0).mean()),
        "expectancy_r": float(accepted["result_r"].mean()),
        "total_r": float(accepted["result_r"].sum()),
        "profit_factor": float(gross_profit / gross_loss) if gross_loss else np.nan,
        "max_drawdown_r": float(drawdown.min()),
        "estimated_spread_cost_r": float(cost_r.sum()),
        "net_expectancy_r": float(net_result_r.mean()),
        "net_total_r": float(net_result_r.sum()),
        "net_max_drawdown_r": float(net_drawdown.min()),
    }


def save_predictions(model, df: pd.DataFrame, out_path: Path, threshold: float) -> None:
    X = df[MODEL_FEATURES]
    y = df["target"].astype(int)
    proba = probabilities(model, df)
    pred = (proba >= threshold).astype(int)
    cols = ["time", "direction", "scenario_id", "risk_percent", "entry_price", "stop_price", "target_price", "runner_target_price", "outcome", "result_r", "target"]
    p = df[cols].copy()
    p["predicted_target"] = pred
    p["probability_valid_trade"] = proba
    p.to_csv(out_path, index=False)


def save_confusion(model, df: pd.DataFrame, out_path: Path, threshold: float) -> None:
    y = df["target"].astype(int)
    pred = (probabilities(model, df) >= threshold).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    out = pd.DataFrame(cm, index=["actual_avoid_or_failed", "actual_valid_trade"], columns=["pred_avoid_or_failed", "pred_valid_trade"])
    out.to_csv(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LR and RF on VSA strategy setup outcomes.")
    parser.add_argument("--data", default="data/bars.csv")
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--max-bars-after-cab", type=int, default=36)
    parser.add_argument("--outcome-horizon", type=int, default=36, help="Future M5 candles used to judge TP/SL outcome.")
    parser.add_argument("--rr", type=float, default=1.2, help="Reward:risk used for labelling VSA setup outcome.")
    parser.add_argument("--rf-trees", type=int, default=80)
    parser.add_argument("--rf-depth", type=int, default=8)
    parser.add_argument("--rf-leaf", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ensure_dirs()
    log_lines: list[str] = []
    start = time.time()
    log("VSA Advisor Bot backend training started.", log_lines)
    log("The website does not train models; this backend script trains the advisor models for evidence and deployment.", log_lines)
    log(f"Loading 2-year XAUUSD M5 dataset from: {args.data}", log_lines)

    setup_df, meta = build_vsa_setup_dataset(
        data_path=args.data,
        lookback=args.lookback,
        max_bars_after_cab=args.max_bars_after_cab,
        outcome_horizon=args.outcome_horizon,
        reward_risk=args.rr,
    )
    setup_df.to_csv(RESULTS_DIR / "vsa_setup_training_dataset.csv", index=False)

    log(f"Raw bars loaded: {meta['rows_raw']:,}", log_lines)
    log(f"VSA setup examples created: {meta['vsa_setup_rows']:,}", log_lines)
    log(f"Label definition: {meta['label_definition']}", log_lines)
    log(f"Wins/valid setups: {meta['wins']:,}; avoided/failed/expired setups: {meta['non_wins']:,}", log_lines)

    train, val, test, split_audit = purged_chronological_split(setup_df, args.outcome_horizon)
    meta.update(split_audit)
    meta.update({"train_rows": len(train), "validation_rows": len(val), "test_rows": len(test)})
    log(f"Purged chronological split: train={len(train):,}, validation={len(val):,}, test={len(test):,}", log_lines)
    log(f"Embargo: {split_audit['embargo_minutes']} minutes; purged rows: train={split_audit['purged_train_rows']}, validation={split_audit['purged_validation_rows']}", log_lines)
    log(f"Train period: {meta['train_start']} to {meta['train_end']}", log_lines)
    log(f"Validation period: {meta['validation_start']} to {meta['validation_end']}", log_lines)
    log(f"Test period: {meta['test_start']} to {meta['test_end']}", log_lines)

    X_train, y_train = train[MODEL_FEATURES], train["target"].astype(int)

    log("Training LR bot: Logistic Regression + VSA strategy outcomes...", log_lines)
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=args.seed)),
    ])
    lr.fit(X_train, y_train)
    joblib.dump(lr, MODELS_DIR / "lr_vsa_bot.joblib")
    log("Saved LR bot to models/lr_vsa_bot.joblib", log_lines)

    log("Training RF bot: Random Forest + VSA strategy outcomes...", log_lines)
    rf = RandomForestClassifier(
        n_estimators=args.rf_trees,
        max_depth=args.rf_depth,
        min_samples_leaf=args.rf_leaf,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=args.seed,
    )
    rf.fit(X_train, y_train)
    joblib.dump(rf, MODELS_DIR / "rf_vsa_bot.joblib")
    log("Saved RF bot to models/rf_vsa_bot.joblib", log_lines)

    model_pairs = [("LR_VSA_BOT", lr), ("RF_VSA_BOT", rf)]
    sweeps = []
    thresholds = {}
    for model_name, model in model_pairs:
        sweep = threshold_sweep(model, val, model_name)
        sweeps.append(sweep)
        best = sweep.sort_values(["f1", "balanced_accuracy", "threshold"], ascending=[False, False, True]).iloc[0]
        thresholds[model_name] = float(best["threshold"])
    threshold_df = pd.concat(sweeps, ignore_index=True)
    threshold_df.to_csv(RESULTS_DIR / "validation_threshold_sweep.csv", index=False)

    all_metrics = [
        majority_baseline(val, "validation"), majority_baseline(test, "test"),
        vsa_rule_baseline(val, "validation"), vsa_rule_baseline(test, "test"),
    ]
    for model_name, model in model_pairs:
        for split_name, split_df in [("validation", val), ("test", test)]:
            all_metrics.append(metrics_for(model, split_df, split_name, model_name, thresholds[model_name]))
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    metrics_df[metrics_df["model"] == "LR_VSA_BOT"].to_csv(RESULTS_DIR / "lr_metrics.csv", index=False)
    metrics_df[metrics_df["model"] == "RF_VSA_BOT"].to_csv(RESULTS_DIR / "rf_metrics.csv", index=False)

    save_predictions(lr, val, RESULTS_DIR / "lr_validation_predictions.csv", thresholds["LR_VSA_BOT"])
    save_predictions(lr, test, RESULTS_DIR / "lr_test_predictions.csv", thresholds["LR_VSA_BOT"])
    save_predictions(rf, val, RESULTS_DIR / "rf_validation_predictions.csv", thresholds["RF_VSA_BOT"])
    save_predictions(rf, test, RESULTS_DIR / "rf_test_predictions.csv", thresholds["RF_VSA_BOT"])
    save_confusion(lr, val, RESULTS_DIR / "lr_validation_confusion_matrix.csv", thresholds["LR_VSA_BOT"])
    save_confusion(lr, test, RESULTS_DIR / "lr_test_confusion_matrix.csv", thresholds["LR_VSA_BOT"])
    save_confusion(rf, val, RESULTS_DIR / "rf_validation_confusion_matrix.csv", thresholds["RF_VSA_BOT"])
    save_confusion(rf, test, RESULTS_DIR / "rf_test_confusion_matrix.csv", thresholds["RF_VSA_BOT"])

    # Explanatory artifacts
    coef = lr.named_steps["model"].coef_[0]
    pd.DataFrame({"feature": MODEL_FEATURES, "coefficient_valid_trade": coef}).sort_values(
        "coefficient_valid_trade", key=lambda s: s.abs(), ascending=False
    ).to_csv(RESULTS_DIR / "lr_coefficients.csv", index=False)
    pd.DataFrame({"feature": MODEL_FEATURES, "importance": rf.feature_importances_}).sort_values(
        "importance", ascending=False
    ).to_csv(RESULTS_DIR / "rf_feature_importance.csv", index=False)

    val_rows = metrics_df[(metrics_df["split"] == "validation") & metrics_df["model"].isin(["LR_VSA_BOT", "RF_VSA_BOT"])].copy()
    winner = val_rows.sort_values(["f1", "balanced_accuracy"], ascending=False).iloc[0]
    selected_name = str(winner["model"])
    selected_model = lr if selected_name == "LR_VSA_BOT" else rf
    selected_threshold = thresholds[selected_name]
    selection_reason = (
        f"Selected between the teacher-required LR and RF candidates using the highest validation F1 at a "
        f"validation-tuned threshold ({selected_threshold:.2f}); the untouched test period and later contextual "
        "model-family benchmarks were not used for selection."
    )

    trading_rows = [trading_summary(test.copy(), len(test), "VSA_RULE_ONLY_BASELINE")]
    trading_rows += [trading_evaluation(model, test, name, thresholds[name], args.rr) for name, model in model_pairs]
    pd.DataFrame(trading_rows).to_csv(RESULTS_DIR / "test_trading_evaluation.csv", index=False)

    joblib.dump(selected_model, MODELS_DIR / "selected_vsa_ml_bot.joblib")
    joblib.dump(selected_model, MODELS_DIR / "selected_model.joblib")
    meta.update({
        "selected_model": selected_name,
        "selection_reason": selection_reason,
        "recommendation_threshold": float(selected_threshold),
        "model_thresholds": thresholds,
        "primary_selection_metric": "validation_f1",
        "feature_columns": MODEL_FEATURES,
        "training_script": "backend_training/train_models.py",
        "runtime_versions": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
        "training_completed_seconds": round(time.time() - start, 2),
    })
    (MODELS_DIR / "model_metadata.json").write_text(json.dumps(meta, indent=2))

    log("LR and RF model comparison saved to results/model_comparison.csv", log_lines)
    for _, row in val_rows.iterrows():
        log(f"{row['model']} validation F1={row['f1']:.3f}, balanced accuracy={row['balanced_accuracy']:.3f}, threshold={row['threshold']:.2f}", log_lines)
    log(f"Selected final advisor model: {selected_name}", log_lines)
    log(f"Selection reason: {selection_reason}", log_lines)

    signal = latest_vsa_setup_prediction(args.data)
    save_ml_signal_csv(signal, OUTPUTS_DIR / "ml_signal.csv")
    log("Latest selected-model signal exported to outputs/ml_signal.csv", log_lines)
    log(f"Latest signal: {signal['direction']} {signal['advisor_output']} probability={signal['ml_probability_valid']:.3f}", log_lines)

    log(f"Training finished in {time.time() - start:.2f} seconds.", log_lines)
    (RESULTS_DIR / "training_log.txt").write_text("\n".join(log_lines) + "\n")


if __name__ == "__main__":
    main()
