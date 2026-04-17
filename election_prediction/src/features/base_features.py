"""
features/base_features.py
--------------------------
Builds the core constituency × party × year feature matrix.

Every feature here is computed ONLY from data available BEFORE the election
being predicted — enforced through the `cutoff_year` parameter.

Features produced:
  ┌─────────────────────────────────────────────────────────────────┐
  │ INCUMBENCY FEATURES                                             │
  │   party_incumbent, candidate_incumbent, terms_won_last_3       │
  │                                                                 │
  │ VOTE SHARE TREND FEATURES                                       │
  │   vote_share_t1, vote_share_t2, vote_share_t3                  │
  │   vote_share_trend (linear slope), vote_share_ma2              │
  │                                                                 │
  │ MARGIN FEATURES                                                 │
  │   margin_pct_last, margin_pct_t2, margin_volatility            │
  │                                                                 │
  │ STRONGHOLD / VOLATILITY                                         │
  │   is_stronghold, constituency_volatility, seat_category        │
  │                                                                 │
  │ TURNOUT FEATURES                                                │
  │   turnout_last, turnout_change                                  │
  │                                                                 │
  │ PARTY PRESENCE                                                  │
  │   party_contested_last, party_present_t1_t2_t3                 │
  └─────────────────────────────────────────────────────────────────┘
"""

import pandas as pd
import numpy as np
from typing import List, Optional
from scipy import stats as scipy_stats
from src.utils.helpers import get_logger

logger = get_logger(__name__)


class BaseFeatureBuilder:
    """
    Builds the feature matrix for ML model training and inference.

    Parameters
    ----------
    df_clean : pd.DataFrame
        Output of DataCleaner — one row per candidate per constituency per year.
    cutoff_year : int
        The election year being predicted. Only data from years < cutoff_year
        is used to compute features (prevents data leakage).
    target_parties : list, optional
        If provided, only generate rows for these parties per constituency.
        Defaults to all parties that have historically contested.
    """

    def __init__(
        self,
        df_clean: pd.DataFrame,
        cutoff_year: int,
        target_parties: Optional[List[str]] = None,
        stronghold_threshold: int = 2,
        volatility_threshold: float = 0.12,
    ):
        self.df = df_clean.copy()
        self.cutoff_year = cutoff_year
        self.target_parties = target_parties
        self.stronghold_threshold = stronghold_threshold
        self.volatility_threshold = volatility_threshold

        # Historical data only (before prediction year)
        self.hist = self.df[self.df["year"] < cutoff_year].copy()
        self.election_years = sorted(self.hist["year"].unique())

        logger.info(
            "BaseFeatureBuilder init: cutoff=%d, hist years=%s",
            cutoff_year, self.election_years
        )

    # ── Public entry point ────────────────────────────────────────────────

    def build(self) -> pd.DataFrame:
        """
        Returns a DataFrame with one row per (constituency × party) pair,
        containing all features needed for ML training/inference.
        """
        logger.info("Building constituency × party feature matrix …")

        # Step 1: Build the base skeleton
        skeleton = self._build_skeleton()

        # Step 2: Attach historical vote shares (t-1, t-2, t-3)
        skeleton = self._attach_vote_share_history(skeleton)

        # Step 3: Incumbency features
        skeleton = self._attach_incumbency(skeleton)

        # Step 4: Margin features
        skeleton = self._attach_margin_history(skeleton)

        # Step 5: Stronghold & volatility
        skeleton = self._attach_stronghold_volatility(skeleton)

        # Step 6: Turnout features
        skeleton = self._attach_turnout(skeleton)

        # Step 7: Party presence features
        skeleton = self._attach_presence(skeleton)

        # Step 8: Derived / interaction features
        skeleton = self._add_derived_features(skeleton)

        # Step 9: Fill remaining NaNs sensibly
        skeleton = self._fill_nulls(skeleton)

        logger.info(
            "Feature matrix built: %d rows × %d cols",
            len(skeleton), len(skeleton.columns)
        )
        return skeleton

    # ── Step 1: Skeleton (const × party index) ───────────────────────────

    def _build_skeleton(self) -> pd.DataFrame:
        """
        One row per (state, constituency_name, party) combination.
        Only includes parties that have historically contested in that constituency.
        """
        pairs = (
            self.hist[["state", "constituency_name", "party"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )

        if self.target_parties:
            pairs = pairs[pairs["party"].isin(self.target_parties)]

        # Add const_state_key for joining
        pairs["const_state_key"] = (
            pairs["state"].str.lower().str.replace(" ", "_")
            + "||"
            + pairs["constituency_name"]
        )
        logger.info("Skeleton: %d (constituency × party) rows", len(pairs))
        return pairs

    # ── Step 2: Vote share history ────────────────────────────────────────

    def _attach_vote_share_history(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pivot historical vote shares into columns: vs_t1, vs_t2, vs_t3
        where t1 = most recent past election.
        """
        years = self.election_years[-3:]   # up to last 3 elections

        for i, yr in enumerate(reversed(years), start=1):
            col_name = f"vs_t{i}"
            yr_data = (
                self.hist[self.hist["year"] == yr]
                [["state", "constituency_name", "party", "vote_share"]]
                .rename(columns={"vote_share": col_name})
            )
            df = df.merge(
                yr_data,
                on=["state", "constituency_name", "party"],
                how="left",
            )

        # Vote share trend (linear slope over available history)
        vs_cols = [f"vs_t{i}" for i in range(1, len(years) + 1) if f"vs_t{i}" in df.columns]
        df["vs_trend"] = df[vs_cols].apply(self._linear_slope, axis=1)

        # 2-election moving average
        if "vs_t1" in df.columns and "vs_t2" in df.columns:
            df["vs_ma2"] = df[["vs_t1", "vs_t2"]].mean(axis=1, skipna=True)
        else:
            df["vs_ma2"] = df.get("vs_t1", np.nan)

        # Swing: t1 vs t2
        if "vs_t1" in df.columns and "vs_t2" in df.columns:
            df["vs_swing_t1_t2"] = df["vs_t1"] - df["vs_t2"]
        else:
            df["vs_swing_t1_t2"] = 0.0

        return df

    @staticmethod
    def _linear_slope(row: pd.Series) -> float:
        """Compute OLS slope of vote share over elections (most recent last)."""
        vals = row.dropna().values
        if len(vals) < 2:
            return 0.0
        x = np.arange(len(vals))
        slope, _, _, _, _ = scipy_stats.linregress(x, vals)
        return round(slope, 4)

    # ── Step 3: Incumbency ────────────────────────────────────────────────

    def _attach_incumbency(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        party_won_t1       : 1 if this party won in last election
        party_won_t2       : 1 if won in election before last
        terms_won_last_3   : count of wins in last 3 elections
        candidate_incumbent: 1 if last winner is contesting again (needs 2026 data)
        """
        last_yr = self.election_years[-1] if self.election_years else None

        if last_yr:
            winners_t1 = (
                self.hist[(self.hist["year"] == last_yr) & (self.hist["is_winner"] == 1)]
                [["state", "constituency_name", "party"]]
                .assign(party_won_t1=1)
            )
            df = df.merge(winners_t1, on=["state", "constituency_name", "party"], how="left")
            df["party_won_t1"] = df["party_won_t1"].fillna(0).astype(int)

        if len(self.election_years) >= 2:
            yr_t2 = self.election_years[-2]
            winners_t2 = (
                self.hist[(self.hist["year"] == yr_t2) & (self.hist["is_winner"] == 1)]
                [["state", "constituency_name", "party"]]
                .assign(party_won_t2=1)
            )
            df = df.merge(winners_t2, on=["state", "constituency_name", "party"], how="left")
            df["party_won_t2"] = df["party_won_t2"].fillna(0).astype(int)
        else:
            df["party_won_t2"] = 0

        # Wins count over last 3 elections
        win_counts = (
            self.hist[self.hist["is_winner"] == 1]
            .groupby(["state", "constituency_name", "party"])
            .size()
            .reset_index(name="terms_won_last_3")
        )
        df = df.merge(win_counts, on=["state", "constituency_name", "party"], how="left")
        df["terms_won_last_3"] = df["terms_won_last_3"].fillna(0).astype(int)

        # Anti-incumbency flag: ruling party that also won last election
        # (simple proxy — refine with state-level ruling party from config)
        df["is_ruling_party_incumbent"] = (
            (df["party_won_t1"] == 1) & (df["terms_won_last_3"] >= 2)
        ).astype(int)

        return df

    # ── Step 4: Margin history ────────────────────────────────────────────

    def _attach_margin_history(self, df: pd.DataFrame) -> pd.DataFrame:
        last_yr = self.election_years[-1] if self.election_years else None

        if last_yr:
            margins = (
                self.hist[self.hist["year"] == last_yr]
                [["state", "constituency_name", "margin_pct", "margin_votes"]]
                .drop_duplicates(subset=["state", "constituency_name"])
                .rename(columns={
                    "margin_pct": "margin_pct_t1",
                    "margin_votes": "margin_votes_t1",
                })
            )
            df = df.merge(margins, on=["state", "constituency_name"], how="left")
        else:
            df["margin_pct_t1"] = np.nan
            df["margin_votes_t1"] = np.nan

        # Safe / Swing / Toss-up classification
        df["seat_category"] = pd.cut(
            df["margin_pct_t1"].fillna(0),
            bins=[-np.inf, 3, 8, 15, np.inf],
            labels=["tossup", "swing", "leaning", "safe"],
        ).astype(str)

        return df

    # ── Step 5: Stronghold & volatility ───────────────────────────────────

    def _attach_stronghold_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        is_stronghold      : party won >= stronghold_threshold of last 3 elections
        const_volatility   : std of winner vote share over history (constituency-level)
        """
        df["is_stronghold"] = (
            df["terms_won_last_3"] >= self.stronghold_threshold
        ).astype(int)

        # Constituency-level winner vote share volatility
        winner_vs = (
            self.hist[self.hist["is_winner"] == 1]
            .groupby(["state", "constituency_name"])["vote_share"]
            .std()
            .reset_index(name="const_volatility")
        )
        df = df.merge(winner_vs, on=["state", "constituency_name"], how="left")
        df["const_volatility"] = df["const_volatility"].fillna(
            df["const_volatility"].median()
        )
        df["is_volatile"] = (df["const_volatility"] > self.volatility_threshold * 100).astype(int)

        return df

    # ── Step 6: Turnout ────────────────────────────────────────────────────

    def _attach_turnout(self, df: pd.DataFrame) -> pd.DataFrame:
        last_yr = self.election_years[-1] if self.election_years else None

        if last_yr and "turnout_pct" in self.hist.columns:
            turnout = (
                self.hist[self.hist["year"] == last_yr]
                [["state", "constituency_name", "turnout_pct"]]
                .drop_duplicates(subset=["state", "constituency_name"])
                .rename(columns={"turnout_pct": "turnout_t1"})
            )
            df = df.merge(turnout, on=["state", "constituency_name"], how="left")

            if len(self.election_years) >= 2:
                yr_t2 = self.election_years[-2]
                turnout_t2 = (
                    self.hist[self.hist["year"] == yr_t2]
                    [["state", "constituency_name", "turnout_pct"]]
                    .drop_duplicates(subset=["state", "constituency_name"])
                    .rename(columns={"turnout_pct": "turnout_t2"})
                )
                df = df.merge(turnout_t2, on=["state", "constituency_name"], how="left")
                df["turnout_change"] = df["turnout_t1"] - df["turnout_t2"]
            else:
                df["turnout_change"] = 0.0
        else:
            df["turnout_t1"] = np.nan
            df["turnout_change"] = 0.0

        return df

    # ── Step 7: Party presence ────────────────────────────────────────────

    def _attach_presence(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        How consistently has this party contested in this constituency?
        elections_contested : count across all hist years
        """
        presence = (
            self.hist.groupby(["state", "constituency_name", "party"])
            .size()
            .reset_index(name="elections_contested")
        )
        df = df.merge(presence, on=["state", "constituency_name", "party"], how="left")
        df["elections_contested"] = df["elections_contested"].fillna(0).astype(int)

        total_elections = len(self.election_years)
        df["presence_ratio"] = df["elections_contested"] / max(total_elections, 1)

        return df

    # ── Step 8: Derived / interaction features ────────────────────────────

    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Momentum: incumbency + positive swing
        df["momentum_score"] = (
            df.get("party_won_t1", 0) * 2
            + df.get("vs_swing_t1_t2", 0) / 10
            + df.get("terms_won_last_3", 0)
        )

        # Risk score: is_stronghold penalised by volatility
        df["stability_score"] = (
            df.get("is_stronghold", 0) * 10
            - df.get("const_volatility", 5)
        )

        # Regression-to-mean signal: was last win unusually high?
        df["vs_reversion_risk"] = np.where(
            df.get("vs_t1", 0) > df.get("vs_ma2", 0) + 5,
            1, 0
        )

        # Party-strength composite (used in rule engine later)
        df["party_strength_score"] = (
            df.get("vs_ma2", 0) * 0.5
            + df.get("terms_won_last_3", 0) * 5
            + df.get("vs_trend", 0) * 2
        )

        return df

    # ── Step 9: Fill nulls ────────────────────────────────────────────────

    def _fill_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val if not np.isnan(median_val) else 0.0)

        categorical_cols = df.select_dtypes(include=["object", "category"]).columns
        for col in categorical_cols:
            df[col] = df[col].fillna("UNKNOWN")

        return df


def build_features(
    df_clean: pd.DataFrame,
    cutoff_year: int,
    **kwargs,
) -> pd.DataFrame:
    """One-line entry point."""
    return BaseFeatureBuilder(df_clean, cutoff_year, **kwargs).build()
