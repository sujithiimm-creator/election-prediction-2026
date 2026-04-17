"""
models/lgbm_model.py
---------------------
LightGBM-based election prediction model with:
  1. Vote share regression per party (primary signal)
  2. Winner classification derived from vote share predictions
  3. Time-aware cross-validation (train on t-2/t-1, validate on t)
  4. SHAP-based explainability
  5. Optuna hyperparameter optimisation (optional)
  6. Model persistence (save/load)
"""

import os
import joblib
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
from typing import Dict, List, Optional, Tuple
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score
from src.utils.helpers import get_logger, ensure_dir, load_config

warnings.filterwarnings("ignore", category=UserWarning)
logger = get_logger(__name__)


# ── Feature columns used by the model ────────────────────────────────────────
# Exclude IDs and targets — anything else numeric is fair game.

ID_COLS    = ["state", "constituency_name", "party", "const_state_key",
              "const_key", "_source_file", "inferred_allies", "seat_category"]
TARGET_VS  = "vs_t1"           # vote share to regress (shifted to actual during training)
TARGET_WIN = "is_winner"       # binary classification target


class ElectionLGBM:
    """
    Dual-head LightGBM model:
      - Head 1: Regress vote share per (constituency, party)
      - Head 2: Derive winner by argmax of predicted vote shares

    Parameters
    ----------
    config_path : str
        Path to config/config.yaml
    model_dir : str
        Where to save trained models
    """

    def __init__(
        self,
        config_path: str = "config/config.yaml",
        model_dir: str = "outputs/models/",
    ):
        cfg = load_config(config_path)
        self.lgbm_params = cfg["model"]["lgbm_params"]
        self.model_dir = model_dir
        ensure_dir(model_dir)

        self.vs_model: Optional[lgb.LGBMRegressor] = None
        self.feature_cols: List[str] = []
        self.shap_explainer = None

    # ── Training ──────────────────────────────────────────────────────────

    def train(
        self,
        df_features: pd.DataFrame,
        df_actuals: pd.DataFrame,
        target_year: int,
        tune_hyperparams: bool = False,
        n_optuna_trials: int = 50,
    ) -> Dict[str, float]:
        """
        Train on historical elections, validate on the most recent past election.

        Parameters
        ----------
        df_features : pd.DataFrame
            Feature matrix (one row per constituency × party).
            Built by BaseFeatureBuilder for years < target_year.
        df_actuals : pd.DataFrame
            Cleaned actual results — used to build the regression target.
        target_year : int
            The year being predicted (used to define train/val split).
        tune_hyperparams : bool
            If True, run Optuna HPO before final training.
        """
        logger.info("Preparing training dataset for target year %d …", target_year)

        # Build training dataset: for each historical election year,
        # create a training row where the TARGET is the actual vote share
        # and FEATURES are computed from data BEFORE that year.
        train_df = self._build_training_set(df_features, df_actuals, target_year)

        if train_df.empty:
            raise ValueError("Training set is empty — check your data and years.")

        self.feature_cols = self._select_feature_cols(train_df)
        X = train_df[self.feature_cols]
        y = train_df["target_vs"]           # actual vote share (regression target)
        groups = train_df["constituency_name"]  # for GroupKFold

        logger.info(
            "Training set: %d rows × %d features", len(X), len(self.feature_cols)
        )

        if tune_hyperparams:
            self.lgbm_params = self._tune_hyperparams(X, y, groups, n_optuna_trials)

        # Final model training
        self.vs_model = lgb.LGBMRegressor(**self.lgbm_params)

        # Time-aware CV for reporting (not for fitting — we fit on all data)
        cv_scores = self._time_cv(X, y, groups, target_year, train_df)

        # Fit on full training data
        self.vs_model.fit(
            X, y,
            eval_set=[(X, y)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
        )

        # Build SHAP explainer
        self.shap_explainer = shap.TreeExplainer(self.vs_model)
        logger.info("Training complete. CV MAE: %.3f", cv_scores["mae"])

        self._save_model()
        return cv_scores

    # ── Prediction ────────────────────────────────────────────────────────

    def predict(self, df_features: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions for all (constituency × party) rows.

        Returns
        -------
        pd.DataFrame with columns:
          state, constituency_name, party,
          predicted_vs        : predicted vote share %
          predicted_winner    : 1 for the party with highest pred_vs in constituency
          win_probability     : normalised probability proxy
          confidence_level    : 'high' | 'medium' | 'low'
          flag_for_review     : bool
        """
        if self.vs_model is None:
            raise RuntimeError("Model not trained. Call train() first or load_model().")

        # Align features
        missing = [c for c in self.feature_cols if c not in df_features.columns]
        if missing:
            logger.warning("Missing feature cols (filling with 0): %s", missing)
            for col in missing:
                df_features[col] = 0.0

        X = df_features[self.feature_cols]
        pred_vs = self.vs_model.predict(X)
        pred_vs = np.clip(pred_vs, 0, 100)

        result = df_features[["state", "constituency_name", "party"]].copy()
        result["predicted_vs"] = pred_vs

        # Winner = party with highest predicted vote share per constituency
        result = self._derive_winner(result)

        # Confidence scoring
        result = self._add_confidence(result, df_features)

        return result

    # ── SHAP Explainability ───────────────────────────────────────────────

    def explain(
        self,
        df_features: pd.DataFrame,
        top_n: int = 15,
    ) -> pd.DataFrame:
        """
        Return top_n most important features with SHAP values.
        """
        if self.shap_explainer is None:
            raise RuntimeError("SHAP explainer not available. Train model first.")

        X = df_features[self.feature_cols].head(500)   # limit for speed
        shap_values = self.shap_explainer.shap_values(X)
        
        importance = pd.DataFrame({
            "feature": self.feature_cols,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        }).sort_values("mean_abs_shap", ascending=False).head(top_n)

        logger.info("Top features:\n%s", importance.to_string(index=False))
        return importance

    # ── Internal: Build training set ──────────────────────────────────────

    def _build_training_set(
        self,
        df_features: pd.DataFrame,
        df_actuals: pd.DataFrame,
        target_year: int,
    ) -> pd.DataFrame:
        """
        For each historical election year y < target_year:
          - Compute features using data from years < y  (no leakage)
          - Join with actual vote share in year y  (this is the target)
        
        Note: Since BaseFeatureBuilder already computed features using data
        before target_year, we use a simpler approach here — join features
        to actuals for the most recent historical year as validation proxy.
        """
        from src.features.base_features import BaseFeatureBuilder

        all_years = sorted(df_actuals["year"].unique())
        historical_years = [y for y in all_years if y < target_year]

        if not historical_years:
            logger.error("No historical years available before %d", target_year)
            return pd.DataFrame()

        frames = []
        for val_year in historical_years[1:]:   # skip earliest (need t-2 history)
            # Build features for predicting val_year (using years < val_year)
            try:
                feat = BaseFeatureBuilder(
                    df_actuals[df_actuals["year"] < val_year],
                    cutoff_year=val_year,
                ).build()
            except Exception as e:
                logger.warning("Feature build failed for val_year=%d: %s", val_year, e)
                continue

            # Get actual vote shares for val_year
            actuals_yr = (
                df_actuals[df_actuals["year"] == val_year]
                [["state", "constituency_name", "party", "vote_share", "is_winner"]]
                .rename(columns={"vote_share": "target_vs", "is_winner": "target_win"})
            )

            # Join
            merged = feat.merge(
                actuals_yr,
                on=["state", "constituency_name", "party"],
                how="inner",
            )
            merged["val_year"] = val_year
            frames.append(merged)
            logger.info("  val_year=%d → %d training rows", val_year, len(merged))

        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    # ── Internal: Time-aware CV ───────────────────────────────────────────

    def _time_cv(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        groups: pd.Series,
        target_year: int,
        train_df: pd.DataFrame,
    ) -> Dict[str, float]:
        """
        Walk-forward CV: train on older elections, validate on most recent.
        """
        val_year = train_df["val_year"].max()
        is_val = train_df["val_year"] == val_year

        X_train, X_val = X[~is_val], X[is_val]
        y_train, y_val = y[~is_val], y[is_val]

        if len(X_train) == 0 or len(X_val) == 0:
            return {"mae": 999.0, "r2": 0.0, "winner_acc": 0.0}

        cv_model = lgb.LGBMRegressor(**self.lgbm_params)
        cv_model.fit(X_train, y_train)

        preds = cv_model.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        r2  = r2_score(y_val, preds)

        # Winner accuracy from val year
        val_df = train_df[is_val].copy()
        val_df["pred_vs"] = preds
        winner_acc = self._compute_winner_accuracy(val_df)

        logger.info(
            "CV (train→%d, val=%d): MAE=%.3f | R²=%.3f | Winner Acc=%.1f%%",
            val_year - 1, val_year, mae, r2, winner_acc * 100
        )
        return {"mae": mae, "r2": r2, "winner_acc": winner_acc}

    # ── Internal: Winner derivation ───────────────────────────────────────

    def _derive_winner(self, result: pd.DataFrame) -> pd.DataFrame:
        """Assign winner = party with max predicted_vs per constituency."""
        idx_winners = (
            result.groupby(["state", "constituency_name"])["predicted_vs"]
            .idxmax()
        )
        result["predicted_winner"] = 0
        result.loc[idx_winners.values, "predicted_winner"] = 1

        # Win probability proxy: softmax-like normalisation per constituency
        result["win_probability"] = result.groupby(
            ["state", "constituency_name"]
        )["predicted_vs"].transform(
            lambda x: x / x.sum() if x.sum() > 0 else x
        )
        return result

    # ── Internal: Confidence ──────────────────────────────────────────────

    def _add_confidence(
        self, result: pd.DataFrame, features: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Flag low-confidence predictions for human review.
        Confidence is based on margin between top-2 predicted vote shares.
        """
        top2 = (
            result.sort_values(
                ["state", "constituency_name", "predicted_vs"], ascending=False
            )
            .groupby(["state", "constituency_name"])
            .head(2)
            .groupby(["state", "constituency_name"])["predicted_vs"]
            .agg(["first", "last"])
            .reset_index()
            .rename(columns={"first": "vs_1st", "last": "vs_2nd"})
        )
        top2["pred_margin"] = top2["vs_1st"] - top2["vs_2nd"]

        result = result.merge(
            top2[["state", "constituency_name", "pred_margin"]],
            on=["state", "constituency_name"],
            how="left",
        )

        result["confidence_level"] = pd.cut(
            result["pred_margin"],
            bins=[-np.inf, 3, 8, np.inf],
            labels=["low", "medium", "high"],
        ).astype(str)

        result["flag_for_review"] = result["confidence_level"] == "low"
        return result

    # ── Internal: Winner accuracy helper ─────────────────────────────────

    @staticmethod
    def _compute_winner_accuracy(df: pd.DataFrame) -> float:
        pred_winners = (
            df.loc[df.groupby(["state", "constituency_name"])["pred_vs"].idxmax()]
            [["state", "constituency_name", "party"]]
            .rename(columns={"party": "pred_party"})
        )
        actual_winners = (
            df[df["target_win"] == 1]
            [["state", "constituency_name", "party"]]
            .rename(columns={"party": "actual_party"})
        )
        merged = pred_winners.merge(
            actual_winners, on=["state", "constituency_name"], how="inner"
        )
        if merged.empty:
            return 0.0
        return (merged["pred_party"] == merged["actual_party"]).mean()

    # ── Internal: Feature column selection ───────────────────────────────

    def _select_feature_cols(self, df: pd.DataFrame) -> List[str]:
        exclude = set(ID_COLS + ["target_vs", "target_win", "val_year",
                                  "is_winner", "votes", "vote_share",
                                  "winner_votes", "runner_up_votes",
                                  "margin_votes", "margin_pct",
                                  "candidate_name", "candidate_sex",
                                  "candidate_type", "position",
                                  "total_votes_polled", "total_electors"])
        cols = [c for c in df.columns if c not in exclude
                and df[c].dtype in [np.float64, np.float32, np.int64, np.int32]]
        logger.info("Selected %d feature columns", len(cols))
        return cols

    # ── Hyperparameter tuning ─────────────────────────────────────────────

    def _tune_hyperparams(
        self, X: pd.DataFrame, y: pd.Series, groups: pd.Series, n_trials: int
    ) -> Dict:
        """Optuna-based HPO — only runs if tune_hyperparams=True."""
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)

            def objective(trial):
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 200, 800),
                    "learning_rate": trial.suggest_float("lr", 0.02, 0.1, log=True),
                    "num_leaves": trial.suggest_int("num_leaves", 31, 127),
                    "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
                    "feature_fraction": trial.suggest_float("ff", 0.5, 1.0),
                    "bagging_fraction": trial.suggest_float("bf", 0.5, 1.0),
                    "bagging_freq": 5,
                    "verbose": -1,
                }
                gkf = GroupKFold(n_splits=3)
                maes = []
                for tr_idx, val_idx in gkf.split(X, y, groups):
                    m = lgb.LGBMRegressor(**params)
                    m.fit(X.iloc[tr_idx], y.iloc[tr_idx])
                    preds = m.predict(X.iloc[val_idx])
                    maes.append(mean_absolute_error(y.iloc[val_idx], preds))
                return np.mean(maes)

            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
            logger.info("Best HPO params: %s", study.best_params)
            best = {**self.lgbm_params, **study.best_params}
            return best

        except ImportError:
            logger.warning("Optuna not available — skipping HPO")
            return self.lgbm_params

    # ── Save / Load ───────────────────────────────────────────────────────

    def _save_model(self):
        path = os.path.join(self.model_dir, "lgbm_vs_model.pkl")
        joblib.dump({
            "model": self.vs_model,
            "feature_cols": self.feature_cols,
        }, path)
        logger.info("Model saved to %s", path)

    def load_model(self):
        path = os.path.join(self.model_dir, "lgbm_vs_model.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"No saved model at {path}")
        obj = joblib.load(path)
        self.vs_model = obj["model"]
        self.feature_cols = obj["feature_cols"]
        self.shap_explainer = shap.TreeExplainer(self.vs_model)
        logger.info("Model loaded from %s", path)
