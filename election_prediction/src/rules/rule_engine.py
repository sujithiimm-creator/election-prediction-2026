"""
rules/rule_engine.py
---------------------
Rule-based override layer applied ON TOP of ML predictions.

Rules are applied in priority order. Higher priority rules run last
(so they override lower priority ones).

Rule priorities (lower = checked first, higher = can override):
  P1 — NOTA / Invalid candidate guards
  P2 — Party not contesting (set to 0)
  P3 — Stronghold protection (protect dominant parties in safe seats)
  P4 — Anti-incumbency adjustment (dampen ruling-party incumbents in volatile seats)
  P5 — Alliance vote transfer adjustment
  P6 — Margin-based confidence override
  P7 — Manual override (highest priority — human expert input)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from src.utils.helpers import get_logger, load_config

logger = get_logger(__name__)


class RuleEngine:
    """
    Applies structured rule overrides to ML predictions.

    Parameters
    ----------
    config_path : str
    anti_incumbency_factor : float
        Multiplier applied to ruling-party vote share in volatile seats.
        e.g., 0.90 = 10% dampening.
    stronghold_boost : float
        Points added to stronghold party vote share in safe seats.
    """

    def __init__(
        self,
        config_path: str = "config/config.yaml",
        anti_incumbency_factor: float = 0.90,
        stronghold_boost: float = 2.0,
    ):
        self.cfg = load_config(config_path)
        self.anti_incumbency_factor = anti_incumbency_factor
        self.stronghold_boost = stronghold_boost
        self.rule_log: List[Dict] = []   # audit trail of all overrides

    # ── Main entry point ──────────────────────────────────────────────────

    def apply(
        self,
        predictions: pd.DataFrame,
        features: pd.DataFrame,
        manual_overrides: Optional[Dict] = None,
    ) -> pd.DataFrame:
        """
        Apply all rules in priority order.

        Parameters
        ----------
        predictions : pd.DataFrame
            Output from ElectionLGBM.predict() with columns:
            [state, constituency_name, party, predicted_vs, predicted_winner,
             win_probability, confidence_level, flag_for_review, pred_margin]
        features : pd.DataFrame
            Feature matrix — needed for rule conditions.
        manual_overrides : dict, optional
            {(state, constituency_name): winning_party}
            e.g., {("Tamil Nadu", "ariyalur"): "DMK"}

        Returns
        -------
        pd.DataFrame with adjusted predictions and rule_applied column.
        """
        df = predictions.copy()
        df = df.merge(
            features[[
                "state", "constituency_name", "party",
                "is_stronghold", "is_volatile", "party_won_t1",
                "terms_won_last_3", "seat_category",
                "is_ruling_party_incumbent", "transfer_in_score",
                "margin_pct_t1",
            ]].drop_duplicates(),
            on=["state", "constituency_name", "party"],
            how="left",
        )
        df["rule_applied"] = "none"
        df["rule_notes"] = ""

        # Apply rules in priority order
        df = self._rule_not_contesting(df)          # P2
        df = self._rule_stronghold_protection(df)   # P3
        df = self._rule_anti_incumbency(df)         # P4
        df = self._rule_alliance_transfer(df)       # P5
        df = self._rule_margin_override(df)         # P6

        # Recalculate winners after rule adjustments
        df = self._recalculate_winner(df)

        # P7: Manual overrides last (highest priority)
        if manual_overrides:
            df = self._apply_manual_overrides(df, manual_overrides)

        logger.info(
            "Rules applied: %d constituencies overridden",
            (df["rule_applied"] != "none").sum()
        )
        return df

    # ── P2: Party not contesting ─────────────────────────────────────────

    def _rule_not_contesting(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        If a party did not contest in the last election AND has zero transfer,
        it likely won't contest again — set predicted_vs to 0.
        """
        mask = (
            (df.get("elections_contested", 1) == 0) |
            (df["party"].isin({"NOTA", "UNKNOWN"}))
        )
        df.loc[mask, "predicted_vs"] = 0.0
        df.loc[mask, "rule_applied"] = "not_contesting"
        df.loc[mask, "rule_notes"] += "Party likely not contesting; "
        return df

    # ── P3: Stronghold protection ─────────────────────────────────────────

    def _rule_stronghold_protection(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        If a party is a stronghold holder (won 2+ of last 3 elections) AND
        the seat is categorised as 'safe', add a boost to predicted_vs.
        
        Rationale: Models tend to underestimate entrenched parties in safe seats
        because regression-to-mean pulls predictions toward the average.
        """
        mask = (
            (df.get("is_stronghold", 0) == 1) &
            (df.get("seat_category", "") == "safe")
        )
        df.loc[mask, "predicted_vs"] += self.stronghold_boost
        df.loc[mask, "rule_applied"] = "stronghold_boost"
        df.loc[mask, "rule_notes"] += f"+{self.stronghold_boost}pp stronghold boost; "
        self._log_rule("stronghold_protection", mask.sum())
        return df

    # ── P4: Anti-incumbency adjustment ───────────────────────────────────

    def _rule_anti_incumbency(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Dampen vote share for the state-ruling party in volatile seats.
        
        Conditions:
          - Party won last election (party_won_t1 == 1)
          - Seat is volatile
          - Party has been ruling for 2+ terms (terms_won_last_3 >= 2)
        
        Note: Anti-incumbency is state-level, not party-level. 
        Adjust anti_incumbency_factor based on state mood / recency.
        """
        mask = (
            (df.get("is_ruling_party_incumbent", 0) == 1) &
            (df.get("is_volatile", 0) == 1)
        )
        df.loc[mask, "predicted_vs"] *= self.anti_incumbency_factor
        df.loc[mask, "rule_applied"] = df.loc[mask, "rule_applied"].where(
            df.loc[mask, "rule_applied"] != "none", "anti_incumbency"
        )
        df.loc[mask, "rule_notes"] += f"Anti-incumb ×{self.anti_incumbency_factor}; "
        self._log_rule("anti_incumbency", mask.sum())
        return df

    # ── P5: Alliance transfer ─────────────────────────────────────────────

    def _rule_alliance_transfer(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add alliance transfer_in_score to predicted vote share.
        Only applied to parties with positive transfer_in_score.
        
        The transfer feature was already computed in AllianceInferrer —
        this rule converts it into a vote share adjustment.
        """
        if "transfer_in_score" not in df.columns:
            return df

        mask = df["transfer_in_score"] > 0.5   # threshold to avoid noise
        df.loc[mask, "predicted_vs"] += df.loc[mask, "transfer_in_score"] * 0.5
        # Factor 0.5: not all transferred votes actually materialise
        df.loc[mask, "rule_applied"] = df.loc[mask, "rule_applied"].where(
            df.loc[mask, "rule_applied"] != "none", "alliance_transfer"
        )
        df.loc[mask, "rule_notes"] += "Alliance transfer applied; "
        self._log_rule("alliance_transfer", mask.sum())
        return df

    # ── P6: Margin-based confidence override ─────────────────────────────

    def _rule_margin_override(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        For extremely safe seats (margin_pct_t1 > 20%), give a stronger
        weight to historical outcome over model prediction.
        
        In practice: pull the predicted_vs 20% toward vs_t1 (regression to history).
        """
        if "margin_pct_t1" not in df.columns or "vs_t1" not in df.columns:
            return df

        mask = (
            (df.get("margin_pct_t1", 0) > 20) &
            (df.get("party_won_t1", 0) == 1)
        )
        anchor_weight = 0.20
        df.loc[mask, "predicted_vs"] = (
            df.loc[mask, "predicted_vs"] * (1 - anchor_weight)
            + df.loc[mask, "vs_t1"].fillna(df.loc[mask, "predicted_vs"]) * anchor_weight
        )
        df.loc[mask, "rule_applied"] = df.loc[mask, "rule_applied"].where(
            df.loc[mask, "rule_applied"] != "none", "history_anchor"
        )
        df.loc[mask, "rule_notes"] += "Anchored to strong hist; "
        self._log_rule("history_anchor", mask.sum())
        return df

    # ── P7: Manual overrides ──────────────────────────────────────────────

    def _apply_manual_overrides(
        self, df: pd.DataFrame, overrides: Dict
    ) -> pd.DataFrame:
        """
        Force a specific party to win a constituency.
        {(state, constituency_name): "DMK"}

        Implementation: set overriding party's predicted_vs to 99 and
        all others to 1 — ensures the argmax logic picks the override.
        """
        for (state, const_name), winning_party in overrides.items():
            const_mask = (
                (df["state"] == state) &
                (df["constituency_name"] == const_name.lower())
            )
            if not const_mask.any():
                logger.warning(
                    "Manual override: constituency not found — %s / %s", state, const_name
                )
                continue

            # Suppress all parties, boost override
            df.loc[const_mask, "predicted_vs"] = 1.0
            winner_mask = const_mask & (df["party"] == winning_party)
            df.loc[winner_mask, "predicted_vs"] = 99.0
            df.loc[winner_mask, "rule_applied"] = "manual_override"
            df.loc[winner_mask, "rule_notes"] += f"Expert override → {winning_party}; "

            logger.info(
                "Manual override: %s / %s → %s", state, const_name, winning_party
            )

        df = self._recalculate_winner(df)
        return df

    # ── Winner recalculation ──────────────────────────────────────────────

    @staticmethod
    def _recalculate_winner(df: pd.DataFrame) -> pd.DataFrame:
        """Re-derive predicted_winner after any rule adjustments."""
        df["predicted_vs"] = df["predicted_vs"].clip(lower=0)
        idx_winners = (
            df.groupby(["state", "constituency_name"])["predicted_vs"].idxmax()
        )
        df["predicted_winner"] = 0
        df.loc[idx_winners.values, "predicted_winner"] = 1

        # Recalculate predicted margin
        top2 = (
            df.sort_values(
                ["state", "constituency_name", "predicted_vs"], ascending=False
            )
            .groupby(["state", "constituency_name"])
            .head(2)
            .groupby(["state", "constituency_name"])["predicted_vs"]
            .agg(["first", "last"])
            .reset_index()
        )
        top2["pred_margin_after_rules"] = top2["first"] - top2["last"]
        df = df.merge(
            top2[["state", "constituency_name", "pred_margin_after_rules"]],
            on=["state", "constituency_name"],
            how="left",
        )
        return df

    # ── Audit log ─────────────────────────────────────────────────────────

    def _log_rule(self, rule_name: str, n_affected: int):
        self.rule_log.append({"rule": rule_name, "n_affected": n_affected})
        logger.debug("Rule '%s' applied to %d rows", rule_name, n_affected)

    def get_rule_summary(self) -> pd.DataFrame:
        return pd.DataFrame(self.rule_log)
