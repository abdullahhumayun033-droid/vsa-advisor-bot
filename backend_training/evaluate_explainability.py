"""Explainability-faithfulness evaluation for the VSA Advisor Bot.

This offline evaluation script verifies that the Logistic Regression
explanations used by the project faithfully reproduce the calculations of the
fitted LR pipeline. It loads the labelled VSA setup dataset, recreates the
purged chronological split, and performs the explanation audit on the held-out
test portion using the saved LR model.

The audit reconstructs each prediction from the fitted preprocessing, Logistic
Regression coefficients and intercept, then compares the reconstructed
probability with the model's own predict_proba output. This checks explanation
faithfulness to the deployed model calculation; it does not claim that the
individual features have a causal effect on market outcomes.

Summary and row-level audit evidence are written to the results directory. The
script raises an error if the reconstructed probabilities fall outside the
defined numerical tolerance, making the explainability check an explicit and
testable part of the project's evaluation evidence.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd

from modules.ml_pipeline import RESULTS_DIR, audit_lr_explanations, purged_chronological_split


def main() -> None:
    """Create formal XAI evidence for the deployed Logistic Regression pipeline."""
    data = pd.read_csv(RESULTS_DIR / "vsa_setup_training_dataset.csv", parse_dates=["time", "cab_time"])
    _train, _validation, test, _audit = purged_chronological_split(data, 36)
    model = joblib.load("models/lr_vsa_bot.joblib")
    summary, local_audit = audit_lr_explanations(model, test)

    pd.DataFrame([summary]).to_csv(RESULTS_DIR / "xai_faithfulness_summary.csv", index=False)
    local_audit.to_csv(RESULTS_DIR / "xai_local_probability_audit.csv", index=False)
    (RESULTS_DIR / "xai_faithfulness_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print("XAI faithfulness audit")
    print(pd.DataFrame([summary]).to_string(index=False))
    if not summary["faithful_within_tolerance"]:
        raise AssertionError("LR contribution explanations did not reconstruct the model probability within tolerance.")


if __name__ == "__main__":
    main()
