"""
simulation/swing_simulator.py
------------------------------
Scenario-based swing simulation engine.

Applies state-level swing adjustments to constituency predictions
under three scenarios: optimistic, neutral, pessimistic (for each party).

Key concepts:
  - State swing: uniform shift in party vote share across all constituencies
  - Differential swing: modulated by local stronghold / volatility
  - Scenario blending: final prediction is a weighted average of scenarios
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from src.utils.helpers import get_logger

logger = get_logger(__name__)


DEFAULT_SWING_SCENARIOS: Dict[str, Dict[str, Dict[str, float]]] = {
    "Tamil Nadu": {
        "neutral": {
            "DMK": 0.0, "AIADMK": 0.0, "BJP": +1.0,
            "INC": 0.0, "VCK": +0.5, "NTK": +1.0, "TVK": +2.0,
        },
        "optimistic_dmk": {
            "DMK": +3.0, "AIADMK": -2.0, "BJP": -1.0,
            "INC": +1.0, "VCK": +1.0,
        },
        "pessimistic_dmk": {
            "DMK": -3.0, "AIADMK": +3.0, "BJP": +1.5,
            "INC": -0.5,
        },
    },
    "Kerala": {
        "neutral":  {"LDF": 0.0, "UDF": 0.0, "NDA": +1.0},
        "ldf_wave": {"LDF": +3.0, "UDF": -2.0, "NDA": 0.0},
        "udf_wave": {"LDF": -3.0, "UDF": +3.0, "NDA": 0.0},
    },
    "West Bengal": {
        "neutral":   {"TMC": 0.0, "BJP": 0.0, "INC": +1.0, "ISF": +0.5},
        "tmc_wave":  {"TMC": +3.0, "BJP": -3.0},
        "bjp_surge": {"TMC": -2.0, "BJP": +3.0, "INC": -1.0},
    },
    "Assam": {
        "neutral":    {"BJP": 0.0, "INC": 0.0, "AIUDF": +1.0},
        "bjp_strong": {"BJP": +3.0, "INC": -2.0, "AIUDF": -1.0},
        "opposition": {"BJP": -3.0, "INC": +2.0, "AIUDF": +1.5},
    },
    "Puducherry": {
        "neutral":  {"INC": 0.0, "AINRC": 0.0, "BJP": +1.0, "DMK": 0.0},
        "inc_wave": {"INC": +4.0, "AINRC": -3.0, "BJP": -1.0},
        "nda_wave": {"INC": -3.0, "AINRC": +2.0, "BJP": +2.0},
    },
}


class SwingSimulator:
    """
    Applies state-level swing scenarios to predictions.

    Parameters
    ----------
    swing_scenarios : dict, optional
    differential_factor : float
        0 = uniform; 1 = fully modulated by local volatility
    """

    def __init__(
        self,
        swing_scenarios: Optional[Dict] = None,
        differential_factor: float = 0.3,
    ):
        self.scenarios = swing_scenarios or DEFAULT_SWING_SCENARIOS
        self.differential_factor = differential_factor

    def apply(
        self,
        predictions: pd.DataFrame,
        features: pd.DataFrame,
        active_scenario: Optional[str] = None,
    ) -> pd.DataFrame:
        df = predictions.copy()

        if "is_volatile" in features.columns:
            vol = features[["state", "constituency_name", "party",
                            "is_volatile", "is_stronghold"]].drop_duplicates()
            df = df.merge(vol, on=["state", "constituency_name", "party"],
                          how="left", suffixes=("", "_feat"))

        results = []
        for state in df["state"].unique():
            state_df = df[df["state"] == state].copy()
            state_scens = self.scenarios.get(state, {"neutral": {}})

            if active_scenario:
                adjusted = self._apply_single(
                    state_df, state_scens.get(active_scenario, {}), active_scenario
                )
            else:
                adjusted = self._apply_blended(state_df, state_scens)

            results.append(adjusted)

        out = pd.concat(results, ignore_index=True)
        return self._recalc_winner(out)

    def scenario_comparison(
        self, predictions: pd.DataFrame, features: pd.DataFrame
    ) -> pd.DataFrame:
        """Return a seat-projection table across all scenarios."""
        records = []
        for state in predictions["state"].unique():
            for scen in self.scenarios.get(state, {}).keys():
                adj = self.apply(predictions, features, active_scenario=scen)
                winners = adj[(adj["predicted_winner"] == 1) & (adj["state"] == state)]
                counts = winners["party"].value_counts().head(5).to_dict()
                records.append({"state": state, "scenario": scen, **counts})
        return pd.DataFrame(records).fillna(0)

    def _apply_single(self, df, swing, label):
        df = df.copy()
        for party, delta in swing.items():
            mask = df["party"] == party
            if not mask.any():
                continue
            if "is_volatile" in df.columns:
                factor = np.where(
                    df.loc[mask, "is_volatile"] == 1,
                    1.0 + self.differential_factor,
                    1.0 - self.differential_factor * 0.5,
                )
            else:
                factor = 1.0
            df.loc[mask, "predicted_vs"] = (
                df.loc[mask, "predicted_vs"] + delta * factor
            ).clip(lower=0)
        df["simulation_scenario"] = label
        return df

    def _apply_blended(self, df, state_scens):
        names = list(state_scens.keys())
        if not names:
            df = df.copy()
            df["simulation_scenario"] = "none"
            return df
        n = len(names)
        w = [0.60] + [0.40 / max(n - 1, 1)] * (n - 1)
        w = [x / sum(w) for x in w[:n]]

        blended = pd.Series(0.0, index=df.index)
        for i, name in enumerate(names):
            s = self._apply_single(df, state_scens[name], name)
            blended += w[i] * s["predicted_vs"]

        df = df.copy()
        df["predicted_vs"] = blended.clip(lower=0)
        df["simulation_scenario"] = "blended"
        return df

    @staticmethod
    def _recalc_winner(df):
        df["predicted_vs"] = df["predicted_vs"].clip(lower=0)
        idx = df.groupby(["state", "constituency_name"])["predicted_vs"].idxmax()
        df["predicted_winner"] = 0
        df.loc[idx.values, "predicted_winner"] = 1
        return df
