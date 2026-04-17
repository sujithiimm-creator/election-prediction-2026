"""
features/alliance_inference.py
--------------------------------
Dynamically infers political alliances from historical co-occurrence patterns.

Core Insight
------------
If Party A and Party B NEVER contest the same constituency in the same election,
they are likely in an alliance (only one fields a candidate per seat).
If they ALWAYS overlap, they are opponents.

This module:
  1. Computes a co-occurrence matrix (party A + party B in same constituency)
  2. Infers likely allies vs. opponents
  3. Generates a vote-transfer coefficient for simulating alliance effects
  4. Accepts a manual override dictionary for known 2026 alliances

Alliance Vote Transfer Logic
----------------------------
When Party A withdraws in favour of Party B (alliance), an estimated
`transfer_rate` of A's historical vote share flows to B.
"""

import pandas as pd
import numpy as np
from itertools import combinations
from typing import Dict, List, Optional, Tuple
from src.utils.helpers import get_logger

logger = get_logger(__name__)

# Default vote transfer rate assumptions (can be tuned)
DEFAULT_TRANSFER_RATES = {
    "strong_ally":   0.80,   # tight alliance, high loyalty
    "moderate_ally": 0.65,   # loose alliance, some leakage
    "weak_ally":     0.45,   # nominal alliance, significant leakage
    "opponent":      0.10,   # contested seats, minimal transfer
}


class AllianceInferrer:
    """
    Infers alliance structure from historical data and applies
    vote-transfer adjustments for a target election year.

    Parameters
    ----------
    df_clean : pd.DataFrame
        Cleaned candidate-level data.
    target_year : int
        The election being predicted.
    manual_alliances : dict, optional
        Known alliances for target_year.
        Format: {state: {party: [allied_parties]}}
        Example: {"Tamil Nadu": {"DMK": ["INC", "VCK", "CPI", "CPI(M)"]}}
    """

    def __init__(
        self,
        df_clean: pd.DataFrame,
        target_year: int,
        manual_alliances: Optional[Dict] = None,
    ):
        self.df = df_clean.copy()
        self.target_year = target_year
        self.manual_alliances = manual_alliances or {}
        self.hist = self.df[self.df["year"] < target_year].copy()

    # ── Public API ────────────────────────────────────────────────────────

    def infer_alliances(self) -> Dict[str, Dict[str, str]]:
        """
        Returns a per-state dict mapping each (party_a, party_b) pair
        to relationship type: 'strong_ally' | 'moderate_ally' | 'opponent'
        
        Structure: {state: {(partyA, partyB): relationship_type}}
        """
        results = {}
        for state in self.hist["state"].unique():
            state_data = self.hist[self.hist["state"] == state]
            co_matrix = self._compute_cooccurrence(state_data)
            relationships = self._classify_relationships(co_matrix, state)
            results[state] = relationships
            logger.info(
                "%s: inferred %d party-pair relationships",
                state, len(relationships)
            )
        return results

    def build_alliance_features(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds alliance-related features to the feature matrix:
          - expected_transfer_in  : estimated votes flowing in from allies
          - expected_transfer_out : estimated votes candidate yields to others
          - alliance_strength_score: composite
        """
        alliances = self.infer_alliances()
        records = []

        for state in feature_df["state"].unique():
            state_feat = feature_df[feature_df["state"] == state].copy()
            state_rel = alliances.get(state, {})
            state_manual = self.manual_alliances.get(state, {})

            state_feat = self._apply_transfer_features(
                state_feat, state_rel, state_manual, state
            )
            records.append(state_feat)

        if records:
            return pd.concat(records, ignore_index=True)
        return feature_df

    # ── Co-occurrence matrix ──────────────────────────────────────────────

    def _compute_cooccurrence(self, state_data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute, for each (party_a, party_b) pair:
          - n_contested_same: constituencies where both contested simultaneously
          - n_total_consts: total constituencies in state
          - overlap_rate: n_contested_same / n_total_consts
        """
        # Build presence matrix: constituency × party (binary)
        presence = (
            state_data
            .groupby(["year", "constituency_name", "party"])
            .size()
            .unstack(fill_value=0)
            .clip(upper=1)
        )

        parties = [p for p in presence.columns if p not in {"NOTA", "IND", "UNKNOWN"}]
        records = []

        for yr in state_data["year"].unique():
            yr_presence = presence.loc[yr] if yr in presence.index.get_level_values("year") else pd.DataFrame()
            if yr_presence.empty:
                continue

            total_consts = len(yr_presence)
            for pa, pb in combinations(parties, 2):
                if pa not in yr_presence.columns or pb not in yr_presence.columns:
                    continue
                both = int((yr_presence[pa] & yr_presence[pb]).sum())
                records.append({
                    "year": yr,
                    "party_a": pa,
                    "party_b": pb,
                    "n_contested_same": both,
                    "n_total": total_consts,
                    "overlap_rate": both / total_consts if total_consts > 0 else 0,
                })

        if not records:
            return pd.DataFrame(columns=["party_a", "party_b", "avg_overlap"])

        co_df = pd.DataFrame(records)
        # Aggregate across years: use mean overlap rate
        co_agg = (
            co_df.groupby(["party_a", "party_b"])["overlap_rate"]
            .mean()
            .reset_index(name="avg_overlap")
        )
        return co_agg

    # ── Relationship classification ───────────────────────────────────────

    def _classify_relationships(
        self, co_matrix: pd.DataFrame, state: str
    ) -> Dict[Tuple[str, str], str]:
        """
        Classify each pair based on average overlap rate:
          < 0.10  → strong_ally   (almost never contest same seat)
          0.10–0.30 → moderate_ally
          0.30–0.60 → weak_ally
          > 0.60  → opponent
        """
        relationships = {}
        manual_state = self.manual_alliances.get(state, {})

        for _, row in co_matrix.iterrows():
            pa, pb = row["party_a"], row["party_b"]

            # Manual override takes precedence
            if pa in manual_state and pb in manual_state.get(pa, []):
                rel = "strong_ally"
            elif pb in manual_state and pa in manual_state.get(pb, []):
                rel = "strong_ally"
            else:
                overlap = row["avg_overlap"]
                if overlap < 0.10:
                    rel = "strong_ally"
                elif overlap < 0.30:
                    rel = "moderate_ally"
                elif overlap < 0.60:
                    rel = "weak_ally"
                else:
                    rel = "opponent"

            relationships[(pa, pb)] = rel
            relationships[(pb, pa)] = rel  # symmetric

        return relationships

    # ── Vote transfer feature builder ─────────────────────────────────────

    def _apply_transfer_features(
        self,
        state_feat: pd.DataFrame,
        state_rel: Dict,
        state_manual: Dict,
        state: str,
    ) -> pd.DataFrame:
        """
        For each party in the feature matrix, compute:
          transfer_in_score  : sum(ally_vs_t1 × transfer_rate) for all allies
          alliance_parties   : comma-joined list of inferred allies
        """
        # Build {party: vote_share_t1} lookup per constituency
        vs_lookup = {}
        if "vs_t1" in state_feat.columns:
            for _, row in state_feat.iterrows():
                key = (row["constituency_name"], row["party"])
                vs_lookup[key] = row.get("vs_t1", 0.0)

        transfer_scores = []
        alliance_lists = []

        for _, row in state_feat.iterrows():
            party = row["party"]
            const = row["constituency_name"]
            allies = state_manual.get(party, [])

            # Add inferred allies if not already covered by manual
            for (pa, pb), rel in state_rel.items():
                if pa == party and rel in ("strong_ally", "moderate_ally"):
                    if pb not in allies:
                        allies.append(pb)

            # Compute transfer-in score
            transfer_in = 0.0
            for ally in allies:
                ally_vs = vs_lookup.get((const, ally), 0.0) or 0.0
                rate = DEFAULT_TRANSFER_RATES.get(
                    state_rel.get((party, ally), "weak_ally"), 0.45
                )
                transfer_in += ally_vs * rate

            transfer_scores.append(transfer_in)
            alliance_lists.append("|".join(allies[:5]))  # cap at 5 for readability

        state_feat["transfer_in_score"] = transfer_scores
        state_feat["inferred_allies"] = alliance_lists
        return state_feat


# ── Tamil Nadu 2026 manual alliance config ───────────────────────────────────
# Update this with confirmed 2026 alliance details as they become available.

TN_2026_ALLIANCES = {
    "Tamil Nadu": {
        "DMK":    ["INC", "VCK", "CPI", "CPI(M)", "MDMK", "IUML"],
        "INC":    ["DMK", "VCK", "CPI", "CPI(M)", "MDMK", "IUML"],
        "VCK":    ["DMK", "INC", "CPI", "CPI(M)"],
        "AIADMK": ["PMK"],     # adjust if BJP-AIADMK alliance confirmed
        "BJP":    ["AIADMK"],  # tentative — update as news emerges
    }
}


def build_alliance_features(
    df_clean: pd.DataFrame,
    feature_df: pd.DataFrame,
    target_year: int,
    manual_alliances: Optional[Dict] = None,
) -> pd.DataFrame:
    """One-line entry point."""
    manual = manual_alliances or TN_2026_ALLIANCES
    inferrer = AllianceInferrer(df_clean, target_year, manual)
    return inferrer.build_alliance_features(feature_df)
