"""
models/evaluator.py
--------------------
Evaluation framework with:
  1. Constituency-level accuracy (the competition metric)
  2. Vote share MAE per party
  3. Calibration of confidence scores
  4. Backtesting harness (walk-forward validation)
  5. Per-state accuracy breakdown
  6. Confusion matrix for winner prediction
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for server use
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    mean_absolute_error, classification_report
)
from src.utils.helpers import get_logger, ensure_dir

logger = get_logger(__name__)


class ElectionEvaluator:
    """
    Comprehensive evaluation of election predictions vs actuals.

    Usage
    -----
    evaluator = ElectionEvaluator()
    metrics = evaluator.evaluate(predictions_df, actuals_df)
    evaluator.backtest(df_clean, target_year=2021)
    """

    def __init__(self, report_dir: str = "outputs/reports/"):
        self.report_dir = report_dir
        ensure_dir(report_dir)

    # ── Main evaluation ───────────────────────────────────────────────────

    def evaluate(
        self,
        predictions: pd.DataFrame,
        actuals: pd.DataFrame,
        label: str = "eval",
    ) -> Dict[str, float]:
        """
        Parameters
        ----------
        predictions : pd.DataFrame
            Must contain: state, constituency_name, party, predicted_winner, predicted_vs
        actuals : pd.DataFrame
            Must contain: state, constituency_name, party, is_winner, vote_share

        Returns
        -------
        dict with all key metrics
        """
        logger.info("Running evaluation: %s", label)

        # Merge predictions with actuals
        pred_winners = predictions[predictions["predicted_winner"] == 1][
            ["state", "constituency_name", "party", "predicted_vs",
             "win_probability", "confidence_level"]
        ].rename(columns={"party": "pred_party"})

        actual_winners = actuals[actuals["is_winner"] == 1][
            ["state", "constituency_name", "party", "vote_share"]
        ].rename(columns={"party": "actual_party", "vote_share": "actual_vs"})

        merged = pred_winners.merge(
            actual_winners, on=["state", "constituency_name"], how="inner"
        )

        if merged.empty:
            logger.error("Merge produced 0 rows — check constituency name matching")
            return {}

        # ── Core metrics ────────────────────────────────────────────────

        metrics = {}

        # 1. Overall winner accuracy (competition metric)
        merged["correct"] = (merged["pred_party"] == merged["actual_party"]).astype(int)
        metrics["overall_accuracy"] = merged["correct"].mean()
        metrics["correct_count"]    = merged["correct"].sum()
        metrics["total_count"]      = len(merged)

        logger.info(
            "Overall accuracy: %.1f%% (%d / %d)",
            metrics["overall_accuracy"] * 100,
            metrics["correct_count"],
            metrics["total_count"]
        )

        # 2. Per-state accuracy
        state_acc = (
            merged.groupby("state")["correct"]
            .agg(["mean", "sum", "count"])
            .rename(columns={"mean": "accuracy", "sum": "correct", "count": "total"})
        )
        metrics["state_accuracy"] = state_acc.to_dict("index")
        logger.info("Per-state accuracy:\n%s", state_acc.to_string())

        # 3. Vote share MAE (vs regression predictions)
        vs_merged = predictions.merge(
            actuals[["state", "constituency_name", "party", "vote_share"]],
            on=["state", "constituency_name", "party"],
            how="inner",
        )
        if not vs_merged.empty and "predicted_vs" in vs_merged.columns:
            metrics["vs_mae"] = mean_absolute_error(
                vs_merged["vote_share"], vs_merged["predicted_vs"]
            )
            logger.info("Vote share MAE: %.3f pp", metrics["vs_mae"])

        # 4. Accuracy by confidence level
        if "confidence_level" in merged.columns:
            conf_acc = (
                merged.groupby("confidence_level")["correct"]
                .agg(["mean", "count"])
                .rename(columns={"mean": "accuracy", "count": "n"})
            )
            metrics["accuracy_by_confidence"] = conf_acc.to_dict("index")
            logger.info("Accuracy by confidence:\n%s", conf_acc.to_string())

        # 5. Accuracy by seat category (safe / swing / tossup)
        if "seat_category" in predictions.columns:
            cat_merged = merged.merge(
                predictions[predictions["predicted_winner"] == 1][
                    ["state", "constituency_name", "seat_category"]
                ].drop_duplicates(),
                on=["state", "constituency_name"],
                how="left",
            )
            cat_acc = (
                cat_merged.groupby("seat_category")["correct"]
                .agg(["mean", "count"])
            )
            metrics["accuracy_by_category"] = cat_acc.to_dict("index")
            logger.info("Accuracy by seat category:\n%s", cat_acc.to_string())

        # 6. Calibration score (is confidence well-calibrated?)
        if "win_probability" in merged.columns:
            metrics["calibration_score"] = self._calibration_score(merged)

        # Save report
        self._save_text_report(metrics, label)

        return metrics

    # ── Backtesting ───────────────────────────────────────────────────────

    def backtest(
        self,
        df_clean: pd.DataFrame,
        target_year: int,
        states: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Full walk-forward backtest:
          Train on all years < target_year
          Validate on target_year actual results.
        
        This simulates the exact competition setup.
        """
        from src.features.base_features import build_features
        from src.models.lgbm_model import ElectionLGBM
        from src.rules.rule_engine import RuleEngine
        from src.features.alliance_inference import build_alliance_features

        logger.info("Backtesting for year %d …", target_year)

        if states:
            df_clean = df_clean[df_clean["state"].isin(states)].copy()

        # Build features
        features = build_features(df_clean, cutoff_year=target_year)
        features = build_alliance_features(df_clean, features, target_year)

        # Train model
        model = ElectionLGBM()
        model.train(features, df_clean, target_year)

        # Predict
        predictions = model.predict(features)

        # Apply rules
        engine = RuleEngine()
        predictions = engine.apply(predictions, features)

        # Evaluate against actuals
        actuals = df_clean[df_clean["year"] == target_year].copy()
        metrics = self.evaluate(predictions, actuals, label=f"backtest_{target_year}")

        return metrics

    # ── Calibration ───────────────────────────────────────────────────────

    def _calibration_score(self, merged: pd.DataFrame) -> float:
        """
        Compute Expected Calibration Error (ECE).
        Lower = better calibration.
        """
        probs = merged["win_probability"].clip(0, 1)
        actuals = merged["correct"]

        bins = np.linspace(0, 1, 11)
        ece = 0.0
        n = len(merged)

        for i in range(len(bins) - 1):
            mask = (probs >= bins[i]) & (probs < bins[i + 1])
            if mask.sum() == 0:
                continue
            avg_prob = probs[mask].mean()
            avg_acc  = actuals[mask].mean()
            ece += (mask.sum() / n) * abs(avg_prob - avg_acc)

        logger.info("Expected Calibration Error (ECE): %.4f", ece)
        return ece

    # ── Report saving ─────────────────────────────────────────────────────

    def _save_text_report(self, metrics: Dict, label: str):
        lines = [
            f"Election Prediction Evaluation — {label}",
            "=" * 50,
            f"Overall Accuracy:  {metrics.get('overall_accuracy', 0)*100:.1f}%",
            f"Correct / Total:   {metrics.get('correct_count', 0)} / {metrics.get('total_count', 0)}",
            f"Vote Share MAE:    {metrics.get('vs_mae', 'N/A'):.3f} pp" if isinstance(metrics.get('vs_mae'), float) else "Vote Share MAE: N/A",
            f"Calibration ECE:   {metrics.get('calibration_score', 'N/A')}",
            "",
            "Per-state Accuracy:",
        ]
        for state, sa in metrics.get("state_accuracy", {}).items():
            lines.append(f"  {state}: {sa['accuracy']*100:.1f}% ({sa['correct']}/{sa['total']})")

        path = os.path.join(self.report_dir, f"eval_{label}.txt")
        with open(path, "w") as f:
            f.write("\n".join(lines))
        logger.info("Report saved to %s", path)


import os  # needed for report saving
