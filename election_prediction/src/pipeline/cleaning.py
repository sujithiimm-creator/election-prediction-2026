"""
pipeline/cleaning.py
--------------------
Takes the raw ingested DataFrame and produces a clean, consistent
constituency-candidate level table ready for feature engineering.

Key operations:
  1. Handle missing vote totals (recompute from candidate rows)
  2. Identify and assign winner position if missing
  3. Remove duplicates and junk rows
  4. Standardise party names (secondary pass)
  5. Classify candidate type (NOTA, IND, Major, Minor)
  6. Compute derived base columns (vote_share, turnout, margin)
"""

import pandas as pd
import numpy as np
from src.utils.helpers import get_logger, report_nulls, compute_vote_share

logger = get_logger(__name__)

# Parties treated as "independent/noise" for modelling purposes
INDEPENDENT_LABELS = {"IND", "INDEPENDENT", "NOTA", "UNKNOWN"}

# Minimum votes to be considered a "real" candidate (filter write-ins / errors)
MIN_VOTES_THRESHOLD = 50


class DataCleaner:
    """
    Cleans the unified raw DataFrame produced by ECIDataIngester.

    Usage
    -----
    cleaner = DataCleaner()
    df_clean = cleaner.clean(df_raw)
    """

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Starting cleaning — input: %d rows", len(df))
        df = df.copy()

        df = self._drop_junk_rows(df)
        df = self._recompute_totals(df)
        df = self._assign_positions(df)
        df = self._add_vote_share(df)
        df = self._add_turnout(df)
        df = self._add_margin(df)
        df = self._classify_candidate(df)
        df = self._add_constituency_key(df)

        report_nulls(df, "post-clean")
        logger.info("Cleaning complete — output: %d rows", len(df))
        return df

    # ── Step 1: Drop junk rows ────────────────────────────────────────────

    def _drop_junk_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        # Drop rows with null constituency or party
        df = df.dropna(subset=["constituency_name", "party"])
        # Drop rows with trivially low votes (likely header leakage)
        df = df[df["votes"] >= MIN_VOTES_THRESHOLD]
        # Drop exact duplicates (same candidate, constituency, year, party)
        key_cols = ["constituency_name", "state", "year", "candidate_name", "party"]
        key_cols = [c for c in key_cols if c in df.columns]
        df = df.drop_duplicates(subset=key_cols, keep="first")
        logger.info("Junk removal: %d → %d rows", before, len(df))
        return df.reset_index(drop=True)

    # ── Step 2: Recompute total_votes_polled per constituency-year ────────

    def _recompute_totals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        If total_votes_polled is missing or inconsistent, 
        recompute it as the sum of all candidate votes in that constituency-year.
        """
        group_keys = ["constituency_name", "state", "year"]

        computed_totals = (
            df.groupby(group_keys)["votes"]
            .sum()
            .reset_index()
            .rename(columns={"votes": "_computed_total"})
        )
        df = df.merge(computed_totals, on=group_keys, how="left")

        if "total_votes_polled" not in df.columns:
            df["total_votes_polled"] = df["_computed_total"]
        else:
            # Use recomputed value where original is null or clearly wrong
            mask_bad = (
                df["total_votes_polled"].isna() |
                (df["total_votes_polled"] < df["_computed_total"] * 0.5)
            )
            df.loc[mask_bad, "total_votes_polled"] = df.loc[mask_bad, "_computed_total"]

        df.drop(columns=["_computed_total"], inplace=True)
        return df

    # ── Step 3: Assign position (winner = 1) ─────────────────────────────

    def _assign_positions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rank candidates within each constituency-year by votes descending.
        Overwrites any existing position column for consistency.
        """
        group_keys = ["constituency_name", "state", "year"]
        df["position"] = (
            df.groupby(group_keys)["votes"]
            .rank(method="first", ascending=False)
            .astype(int)
        )
        df["is_winner"] = (df["position"] == 1).astype(int)
        return df

    # ── Step 4: Vote share ────────────────────────────────────────────────

    def _add_vote_share(self, df: pd.DataFrame) -> pd.DataFrame:
        df["vote_share"] = compute_vote_share(
            df["votes"], df["total_votes_polled"]
        )
        return df

    # ── Step 5: Turnout ───────────────────────────────────────────────────

    def _add_turnout(self, df: pd.DataFrame) -> pd.DataFrame:
        if "total_electors" in df.columns:
            df["turnout_pct"] = np.where(
                df["total_electors"] > 0,
                df["total_votes_polled"] / df["total_electors"] * 100,
                np.nan,
            )
        else:
            df["turnout_pct"] = np.nan
        return df

    # ── Step 6: Margin of victory ─────────────────────────────────────────

    def _add_margin(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        For each constituency-year, compute:
          - winner_votes, runner_up_votes
          - margin_votes, margin_pct
        """
        group_keys = ["constituency_name", "state", "year"]

        top2 = (
            df[df["position"] <= 2]
            .sort_values(group_keys + ["position"])
            .groupby(group_keys)
            .agg(
                winner_votes=("votes", "first"),
                runner_up_votes=("votes", "last"),
                total_votes_polled=("total_votes_polled", "first"),
            )
            .reset_index()
        )
        top2["margin_votes"] = top2["winner_votes"] - top2["runner_up_votes"]
        top2["margin_pct"] = np.where(
            top2["total_votes_polled"] > 0,
            top2["margin_votes"] / top2["total_votes_polled"] * 100,
            0.0,
        )
        top2 = top2[group_keys + ["winner_votes", "runner_up_votes",
                                   "margin_votes", "margin_pct"]]

        df = df.merge(top2, on=group_keys, how="left")
        return df

    # ── Step 7: Candidate classification ─────────────────────────────────

    def _classify_candidate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tag each candidate row with a type:
          NOTA      → None Of The Above
          IND       → Independent
          MAJOR     → top-N parties in the state
          MINOR     → all others
        """
        df["candidate_type"] = "MINOR"
        df.loc[df["party"] == "NOTA", "candidate_type"] = "NOTA"
        df.loc[df["party"].isin({"IND", "INDEPENDENT"}), "candidate_type"] = "IND"

        # Mark major parties: any party that won at least one seat historically
        winners = df[df["is_winner"] == 1]["party"].unique()
        df.loc[df["party"].isin(winners), "candidate_type"] = "MAJOR"

        return df

    # ── Step 8: Constituency composite key ───────────────────────────────

    def _add_constituency_key(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create a stable composite key for joining across feature modules.
        Format: 'tamil_nadu||ariyalur||2021'  (all lowercase)
        """
        df["const_key"] = (
            df["state"].str.lower().str.replace(" ", "_")
            + "||"
            + df["constituency_name"]
            + "||"
            + df["year"].astype(str)
        )
        df["const_state_key"] = (
            df["state"].str.lower().str.replace(" ", "_")
            + "||"
            + df["constituency_name"]
        )
        return df


def clean_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    """One-line entry point."""
    return DataCleaner().clean(df_raw)
