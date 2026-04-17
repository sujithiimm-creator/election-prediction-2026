"""
run_pipeline.py
----------------
Master end-to-end pipeline runner.

Usage
-----
  # Full run (train + predict + submit)
  python run_pipeline.py --mode full --states "Tamil Nadu" "Kerala"

  # Backtest only (check accuracy on 2021 data)
  python run_pipeline.py --mode backtest --target_year 2021

  # Predict only (model already trained)
  python run_pipeline.py --mode predict

  # Generate submission from existing predictions
  python run_pipeline.py --mode submit

Pipeline Stages
---------------
  1. INGEST   → load raw ECI files from data/raw/
  2. CLEAN    → standardise, compute vote share, margin
  3. FEATURES → build constituency × party feature matrix
  4. ALLIANCE → infer/apply alliance vote transfer
  5. TRAIN    → LightGBM vote share regression
  6. PREDICT  → generate predictions for 2026
  7. RULES    → apply rule-based overrides
  8. REVIEW   → HITL — export swing seats, merge corrections
  9. OUTPUT   → generate submission Excel + methodology note
"""

import argparse
import logging
import sys
import os
from datetime import datetime

# ── Ensure src is importable ─────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline.ingestion import ingest_data
from src.pipeline.cleaning import clean_data
from src.features.base_features import build_features
from src.features.alliance_inference import build_alliance_features, TN_2026_ALLIANCES
from src.models.lgbm_model import ElectionLGBM
from src.models.evaluator import ElectionEvaluator
from src.rules.rule_engine import RuleEngine
from src.human_loop.review_interface import HumanReviewInterface
from src.pipeline.output_generator import SubmissionGenerator, generate_methodology_note
from src.utils.helpers import get_logger, load_config, ensure_dir

logger = get_logger("run_pipeline", logging.INFO)

# ── Config ────────────────────────────────────────────────────────────────────

TARGET_YEAR      = 2026
RAW_DATA_DIR     = "data/raw/"
PROCESSED_DIR    = "data/processed/"
OUTPUTS_DIR      = "outputs/"
REVIEW_FILE      = "outputs/review/swing_seats_review.xlsx"

# Tamil Nadu manual overrides — update with your TN domain knowledge
# Format: {(state, constituency_name_lowercase): "PARTY"}
TN_MANUAL_OVERRIDES = {
    # Example entries — fill these with your expert predictions:
    # ("Tamil Nadu", "ariyalur"):      "DMK",
    # ("Tamil Nadu", "tanjore"):       "DMK",
    # ("Tamil Nadu", "coimbatore south"): "BJP",
}


def run_full_pipeline(
    states: list = None,
    target_year: int = TARGET_YEAR,
    tune_hyperparams: bool = False,
    run_backtest: bool = True,
):
    """
    Complete end-to-end pipeline run.
    """
    start_time = datetime.now()
    logger.info("=" * 60)
    logger.info("ELECTION PREDICTION PIPELINE — %s", start_time.strftime("%Y-%m-%d %H:%M"))
    logger.info("Target year: %d", target_year)
    logger.info("=" * 60)

    ensure_dir(PROCESSED_DIR)

    # ── Stage 1: Ingest ───────────────────────────────────────────────────
    logger.info("\n[Stage 1/9] Ingesting raw data …")
    df_raw = ingest_data(RAW_DATA_DIR)

    if df_raw.empty:
        logger.error(
            "No data loaded from %s\n"
            "Place ECI Excel files in data/raw/ following naming convention:\n"
            "  TamilNadu_2021_DetailedResults.xlsx\n"
            "  Kerala_2016_DetailedResults.xlsx\n  etc.",
            RAW_DATA_DIR
        )
        sys.exit(1)

    if states:
        df_raw = df_raw[df_raw["state"].isin(states)]
        logger.info("Filtered to states: %s → %d rows", states, len(df_raw))

    # ── Stage 2: Clean ────────────────────────────────────────────────────
    logger.info("\n[Stage 2/9] Cleaning data …")
    df_clean = clean_data(df_raw)
    df_clean.to_parquet(os.path.join(PROCESSED_DIR, "clean_data.parquet"), index=False)
    logger.info("Clean data saved: %d rows", len(df_clean))

    # ── Stage 3: Backtest (optional) ──────────────────────────────────────
    if run_backtest:
        logger.info("\n[Stage 3/9] Running backtest on most recent election …")
        _run_backtest(df_clean, target_year)
    else:
        logger.info("\n[Stage 3/9] Skipping backtest (run_backtest=False)")

    # ── Stage 4: Build features for 2026 ─────────────────────────────────
    logger.info("\n[Stage 4/9] Building features for %d prediction …", target_year)
    features = build_features(df_clean, cutoff_year=target_year)
    logger.info("Feature matrix: %d rows × %d cols", len(features), len(features.columns))

    # ── Stage 5: Alliance features ────────────────────────────────────────
    logger.info("\n[Stage 5/9] Applying alliance features …")
    features = build_alliance_features(
        df_clean, features, target_year, manual_alliances=TN_2026_ALLIANCES
    )

    features.to_parquet(
        os.path.join(PROCESSED_DIR, f"features_{target_year}.parquet"), index=False
    )

    # ── Stage 6: Train model ──────────────────────────────────────────────
    logger.info("\n[Stage 6/9] Training LightGBM model …")
    model = ElectionLGBM()
    cv_metrics = model.train(
        features, df_clean, target_year,
        tune_hyperparams=tune_hyperparams
    )
    logger.info("Training CV metrics: %s", cv_metrics)

    # SHAP feature importance
    try:
        importance = model.explain(features)
        logger.info("Top 5 features by SHAP:\n%s", importance.head().to_string())
    except Exception as e:
        logger.warning("SHAP explanation failed: %s", e)

    # ── Stage 7: Predict ──────────────────────────────────────────────────
    logger.info("\n[Stage 7/9] Generating predictions …")
    predictions = model.predict(features)
    n_consts = predictions["constituency_name"].nunique()
    logger.info("Predictions generated: %d constituencies", n_consts)

    # ── Stage 8: Rule engine ──────────────────────────────────────────────
    logger.info("\n[Stage 8/9] Applying rule-based overrides …")
    engine = RuleEngine()
    predictions = engine.apply(predictions, features, manual_overrides=TN_MANUAL_OVERRIDES)
    rule_summary = engine.get_rule_summary()
    logger.info("Rule summary:\n%s", rule_summary.to_string())

    # ── Stage 8b: Human review ────────────────────────────────────────────
    logger.info("\n[Stage 8b/9] Human-in-the-loop review …")
    hitl = HumanReviewInterface(output_dir="outputs/review/")
    predictions = hitl.run_review_cycle(predictions, review_file=REVIEW_FILE)

    # ── Stage 9: Output ───────────────────────────────────────────────────
    logger.info("\n[Stage 9/9] Generating submission files …")
    generator = SubmissionGenerator()
    submission_path = generator.generate_submission(predictions)
    note_path = generate_methodology_note(cv_metrics)

    elapsed = (datetime.now() - start_time).seconds
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE in %ds", elapsed)
    logger.info("Submission file : %s", submission_path)
    logger.info("Methodology note: %s", note_path)
    logger.info("=" * 60)

    return predictions, submission_path


def _run_backtest(df_clean, target_year: int):
    """Run walk-forward backtest on the last available election."""
    available_years = sorted(df_clean["year"].dropna().unique())
    if len(available_years) < 2:
        logger.warning("Not enough historical years for backtesting")
        return

    val_year = [y for y in available_years if y < target_year]
    if not val_year:
        return
    val_year = max(val_year)

    logger.info("Backtesting: training on < %d, validating on %d", val_year, val_year)
    evaluator = ElectionEvaluator()
    try:
        metrics = evaluator.backtest(df_clean, target_year=val_year)
        logger.info(
            "Backtest accuracy: %.1f%% (%d/%d)",
            metrics.get("overall_accuracy", 0) * 100,
            metrics.get("correct_count", 0),
            metrics.get("total_count", 0),
        )
    except Exception as e:
        logger.warning("Backtest failed: %s", e)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Election Prediction Pipeline 2026")
    parser.add_argument(
        "--mode",
        choices=["full", "backtest", "predict", "submit"],
        default="full",
        help="Pipeline mode",
    )
    parser.add_argument(
        "--states", nargs="+",
        default=None,
        help="Filter to specific states, e.g. --states 'Tamil Nadu' 'Kerala'",
    )
    parser.add_argument(
        "--target_year", type=int, default=TARGET_YEAR,
        help="Election year to predict",
    )
    parser.add_argument(
        "--tune", action="store_true",
        help="Run Optuna hyperparameter tuning (slower, ~50 trials)",
    )
    parser.add_argument(
        "--no_backtest", action="store_true",
        help="Skip backtesting step",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "full":
        run_full_pipeline(
            states=args.states,
            target_year=args.target_year,
            tune_hyperparams=args.tune,
            run_backtest=not args.no_backtest,
        )

    elif args.mode == "backtest":
        df_raw   = ingest_data(RAW_DATA_DIR)
        df_clean = clean_data(df_raw)
        evaluator = ElectionEvaluator()
        evaluator.backtest(df_clean, target_year=args.target_year, states=args.states)

    elif args.mode == "predict":
        df_clean = pd.read_parquet(os.path.join(PROCESSED_DIR, "clean_data.parquet"))
        features = pd.read_parquet(
            os.path.join(PROCESSED_DIR, f"features_{args.target_year}.parquet")
        )
        model = ElectionLGBM()
        model.load_model()
        predictions = model.predict(features)
        engine = RuleEngine()
        predictions = engine.apply(predictions, features)
        generator = SubmissionGenerator()
        generator.generate_submission(predictions)

    elif args.mode == "submit":
        import pandas as pd
        predictions = pd.read_parquet(
            os.path.join(PROCESSED_DIR, f"predictions_{args.target_year}.parquet")
        )
        generator = SubmissionGenerator()
        generator.generate_submission(predictions)
        generate_methodology_note()
