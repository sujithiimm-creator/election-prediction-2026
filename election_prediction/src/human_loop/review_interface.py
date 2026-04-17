"""
human_loop/review_interface.py
--------------------------------
Human-in-the-Loop (HITL) layer.

Workflow:
  1. Identify low-confidence constituencies from model + rules output
  2. Export a review CSV / Excel for expert annotation
  3. Read back manual corrections
  4. Merge corrections into final predictions

This is the most important layer for maximising accuracy on swing seats —
the 20–30% of seats where your TN political knowledge matters most.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from src.utils.helpers import get_logger, ensure_dir

logger = get_logger(__name__)


class HumanReviewInterface:
    """
    Manages the HITL workflow: flag → export → review → merge.

    Parameters
    ----------
    output_dir : str
        Where to write review files.
    margin_threshold : float
        Constituencies with predicted margin < this are flagged.
    confidence_flag : str
        Flag constituencies with this confidence level ('low', 'medium').
    """

    def __init__(
        self,
        output_dir: str = "outputs/review/",
        margin_threshold: float = 6.0,
        confidence_flag: str = "low",
    ):
        self.output_dir = output_dir
        self.margin_threshold = margin_threshold
        self.confidence_flag = confidence_flag
        ensure_dir(output_dir)

    # ── Step 1: Flag swing seats ──────────────────────────────────────────

    def flag_for_review(self, predictions: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split predictions into:
          - flagged: needs human review
          - safe: model is confident
        
        Returns (safe_df, flagged_df)
        """
        # Identify the winner row per constituency for margin check
        margin_col = "pred_margin_after_rules" if "pred_margin_after_rules" in predictions.columns \
                     else "pred_margin"

        winners = predictions[predictions["predicted_winner"] == 1].copy()
        
        flag_mask = (
            (winners.get(margin_col, pd.Series(99, index=winners.index)) < self.margin_threshold) |
            (winners.get("confidence_level", pd.Series("high", index=winners.index)) == self.confidence_flag) |
            (winners.get("flag_for_review", pd.Series(False, index=winners.index)) == True)
        )

        flagged_consts = winners[flag_mask][["state", "constituency_name"]].drop_duplicates()
        flagged_key = set(
            zip(flagged_consts["state"], flagged_consts["constituency_name"])
        )

        is_flagged = predictions.apply(
            lambda r: (r["state"], r["constituency_name"]) in flagged_key, axis=1
        )

        safe_df    = predictions[~is_flagged].copy()
        flagged_df = predictions[is_flagged].copy()

        logger.info(
            "Flagged %d constituencies for review (%.1f%% of total). Safe: %d.",
            len(flagged_key),
            len(flagged_key) / predictions["constituency_name"].nunique() * 100,
            predictions["constituency_name"].nunique() - len(flagged_key),
        )
        return safe_df, flagged_df

    # ── Step 2: Export review sheet ───────────────────────────────────────

    def export_review_sheet(
        self,
        flagged_df: pd.DataFrame,
        filename: str = "swing_seats_review.xlsx",
    ) -> str:
        """
        Write a human-readable Excel file with:
          - Current model prediction (top 3 parties per constituency)
          - Historical vote shares for context
          - Empty column for expert to fill in winner
          - Notes column for reasoning
        """
        # Build summary: one row per constituency with top 3 parties
        records = []
        for (state, const), grp in flagged_df.groupby(["state", "constituency_name"]):
            top3 = grp.sort_values("predicted_vs", ascending=False).head(3)
            row = {
                "state": state,
                "constituency_name": const,
                "model_predicted_winner": top3.iloc[0]["party"] if len(top3) > 0 else "",
                "model_pred_vs_1st": round(top3.iloc[0]["predicted_vs"], 1) if len(top3) > 0 else "",
                "party_2nd": top3.iloc[1]["party"] if len(top3) > 1 else "",
                "model_pred_vs_2nd": round(top3.iloc[1]["predicted_vs"], 1) if len(top3) > 1 else "",
                "party_3rd": top3.iloc[2]["party"] if len(top3) > 2 else "",
                "model_pred_vs_3rd": round(top3.iloc[2]["predicted_vs"], 1) if len(top3) > 2 else "",
                # Context columns
                "winner_2021": top3.iloc[0].get("party_won_t1", ""),
                "margin_2021_pct": round(top3.iloc[0].get("margin_pct_t1", 0), 1),
                "seat_category": top3.iloc[0].get("seat_category", ""),
                "is_volatile": top3.iloc[0].get("is_volatile", ""),
                "confidence": top3.iloc[0].get("confidence_level", ""),
                # Human input columns (empty)
                "EXPERT_OVERRIDE_WINNER": "",
                "EXPERT_NOTES": "",
            }
            records.append(row)

        review_df = pd.DataFrame(records)
        filepath = os.path.join(self.output_dir, filename)

        with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
            review_df.to_excel(writer, sheet_name="SwingSeats", index=False)

            # Format the sheet for readability
            ws = writer.sheets["SwingSeats"]
            ws.column_dimensions["B"].width = 25
            ws.column_dimensions["C"].width = 20
            ws.column_dimensions["O"].width = 25   # EXPERT_OVERRIDE
            ws.column_dimensions["P"].width = 40   # EXPERT_NOTES

        logger.info(
            "Review sheet exported: %s (%d constituencies)", filepath, len(review_df)
        )
        return filepath

    # ── Step 3: Read corrections ──────────────────────────────────────────

    def read_corrections(
        self, review_filepath: str
    ) -> Dict[Tuple[str, str], str]:
        """
        Read back the expert-annotated review file.
        
        Returns
        -------
        dict: {(state, constituency_name): expert_winner_party}
        """
        if not os.path.exists(review_filepath):
            logger.warning("Review file not found: %s", review_filepath)
            return {}

        df = pd.read_excel(review_filepath, sheet_name="SwingSeats")
        overrides = {}

        for _, row in df.iterrows():
            expert_pick = row.get("EXPERT_OVERRIDE_WINNER", "")
            if pd.notna(expert_pick) and str(expert_pick).strip():
                key = (str(row["state"]).strip(), str(row["constituency_name"]).strip())
                overrides[key] = str(expert_pick).strip()

        logger.info(
            "Read %d expert overrides from %s", len(overrides), review_filepath
        )
        return overrides

    # ── Step 4: Merge corrections ─────────────────────────────────────────

    def merge_corrections(
        self,
        predictions: pd.DataFrame,
        corrections: Dict[Tuple[str, str], str],
    ) -> pd.DataFrame:
        """
        Apply expert overrides to predictions DataFrame.
        Overriding works by forcing predicted_winner=1 for the expert pick.
        """
        df = predictions.copy()

        for (state, const_name), winning_party in corrections.items():
            const_mask = (df["state"] == state) & (df["constituency_name"] == const_name)
            
            if not const_mask.any():
                logger.warning("Override: constituency not found %s / %s", state, const_name)
                continue

            # Reset all winners in this constituency
            df.loc[const_mask, "predicted_winner"] = 0
            df.loc[const_mask, "rule_applied"] = df.loc[const_mask, "rule_applied"].fillna("none")
            
            # Set override
            winner_mask = const_mask & (df["party"] == winning_party)
            if not winner_mask.any():
                # Party wasn't in predictions — add a synthetic row
                logger.warning(
                    "Party '%s' not in predictions for %s/%s — skipping override",
                    winning_party, state, const_name
                )
                continue

            df.loc[winner_mask, "predicted_winner"] = 1
            df.loc[winner_mask, "predicted_vs"] = max(
                df.loc[winner_mask, "predicted_vs"].values[0], 35.0
            )
            df.loc[winner_mask, "rule_applied"] = "human_override"
            df.loc[winner_mask, "rule_notes"] = (
                df.loc[winner_mask, "rule_notes"].fillna("") + "Human expert override; "
            )

        logger.info("Merged %d expert corrections", len(corrections))
        return df

    # ── Convenience: Full pipeline ────────────────────────────────────────

    def run_review_cycle(
        self,
        predictions: pd.DataFrame,
        review_file: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Auto-runs the full HITL cycle if a review file exists.
        If not, exports the review sheet and returns predictions unchanged.
        """
        _, flagged_df = self.flag_for_review(predictions)

        review_path = review_file or os.path.join(
            self.output_dir, "swing_seats_review.xlsx"
        )

        if os.path.exists(review_path):
            # Check if expert has filled in any overrides
            corrections = self.read_corrections(review_path)
            if corrections:
                logger.info("Applying %d expert corrections …", len(corrections))
                return self.merge_corrections(predictions, corrections)
            else:
                logger.info(
                    "Review file exists but no corrections filled in yet. "
                    "Open %s and fill EXPERT_OVERRIDE_WINNER column.", review_path
                )
        else:
            # Export for review
            exported = self.export_review_sheet(flagged_df, os.path.basename(review_path))
            logger.info(
                "\n" + "="*60 +
                "\n ACTION NEEDED: Open the review file and fill in expert predictions:\n"
                " %s\n"
                " Then re-run the pipeline.\n" + "="*60,
                exported
            )

        return predictions
