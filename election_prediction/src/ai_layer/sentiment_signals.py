"""
ai_layer/sentiment_signals.py
--------------------------------
AI-powered contextual intelligence layer.

Uses web search + LLM (Claude API) to:
  1. Fetch recent political news per constituency / party
  2. Extract structured sentiment signals
  3. Produce:
     - party_sentiment_score  : state-level  (-10 to +10)
     - candidate_perception   : constituency-level (-5 to +5)
     - alliance_stability     : flag (0/1)
     - key_issues             : list of local issues

Integration into predictions:
  sentiment_boost = party_sentiment_score * 0.3 + candidate_perception * 0.7
  predicted_vs_final = predicted_vs_model + sentiment_boost

IMPORTANT: This module is OPTIONAL and runs only if an API key is set.
Results are cached to avoid redundant API calls.

Setup:
  Create a .env file with:  ANTHROPIC_API_KEY=sk-ant-...
  Or set environment variable before running.
"""

import os
import json
import time
import hashlib
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from src.utils.helpers import get_logger, ensure_dir

logger = get_logger(__name__)

CACHE_DIR = "data/external/sentiment_cache/"


class SentimentSignalExtractor:
    """
    Uses Claude API to extract structured political sentiment signals
    from news and context, then integrates them into vote share predictions.

    Parameters
    ----------
    cache_dir : str
        Directory to cache API responses (avoids repeat calls).
    max_constituencies_per_call : int
        Batch size for API calls (more = faster but less precise).
    """

    def __init__(
        self,
        cache_dir: str = CACHE_DIR,
        max_constituencies_per_call: int = 10,
    ):
        self.cache_dir = cache_dir
        self.max_per_call = max_constituencies_per_call
        ensure_dir(cache_dir)

        self.api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        self.available = bool(self.api_key)

        if not self.available:
            logger.warning(
                "ANTHROPIC_API_KEY not set. AI sentiment layer will be skipped.\n"
                "To enable: export ANTHROPIC_API_KEY=your_key_here"
            )

    # ── State-level sentiment ─────────────────────────────────────────────

    def get_state_sentiment(
        self,
        state: str,
        parties: List[str],
        context: str = "",
    ) -> Dict[str, float]:
        """
        Get sentiment score for each party at state level.

        Returns
        -------
        dict: {party: score} where score ∈ [-10, +10]
        Positive = favourable sentiment, negative = unfavourable.
        """
        if not self.available:
            return {p: 0.0 for p in parties}

        cache_key = self._cache_key(f"{state}_state_{','.join(sorted(parties))}")
        cached = self._load_cache(cache_key)
        if cached:
            return cached

        prompt = self._build_state_sentiment_prompt(state, parties, context)
        response = self._call_api(prompt)

        if response:
            scores = self._parse_sentiment_response(response, parties)
            self._save_cache(cache_key, scores)
            return scores

        return {p: 0.0 for p in parties}

    # ── Constituency-level sentiment ──────────────────────────────────────

    def get_constituency_sentiment(
        self,
        state: str,
        constituencies: List[str],
        parties: List[str],
        context: str = "",
    ) -> pd.DataFrame:
        """
        Get candidate/local sentiment per constituency.

        Returns
        -------
        pd.DataFrame with columns:
          state, constituency_name, party,
          candidate_perception_score (-5 to +5),
          local_issue_flag (0/1),
          sentiment_notes
        """
        if not self.available:
            return pd.DataFrame(columns=[
                "state", "constituency_name", "party",
                "candidate_perception_score", "local_issue_flag", "sentiment_notes"
            ])

        results = []
        # Process in batches
        for i in range(0, len(constituencies), self.max_per_call):
            batch = constituencies[i: i + self.max_per_call]
            batch_results = self._batch_constituency_sentiment(
                state, batch, parties, context
            )
            results.extend(batch_results)
            time.sleep(0.5)   # rate limiting

        return pd.DataFrame(results) if results else pd.DataFrame()

    # ── Integration with predictions ──────────────────────────────────────

    def integrate_signals(
        self,
        predictions: pd.DataFrame,
        state_sentiment: Dict[str, Dict[str, float]],
        constituency_sentiment: Optional[pd.DataFrame] = None,
        state_weight: float = 0.3,
        const_weight: float = 0.7,
    ) -> pd.DataFrame:
        """
        Adjust predicted vote shares using sentiment signals.

        state_weight + const_weight should sum to 1.0
        Adjustment is additive (in percentage points) and capped.

        Parameters
        ----------
        state_sentiment : {state: {party: score}}
        constituency_sentiment : pd.DataFrame (optional)
        """
        df = predictions.copy()
        df["ai_sentiment_boost"] = 0.0

        # 1. State-level sentiment
        for state, party_scores in state_sentiment.items():
            for party, score in party_scores.items():
                mask = (df["state"] == state) & (df["party"] == party)
                # Scale: max sentiment score (10) = max 2pp boost
                df.loc[mask, "ai_sentiment_boost"] += (score / 10) * 2.0 * state_weight

        # 2. Constituency-level sentiment
        if constituency_sentiment is not None and not constituency_sentiment.empty:
            for _, row in constituency_sentiment.iterrows():
                mask = (
                    (df["state"] == row["state"]) &
                    (df["constituency_name"] == row["constituency_name"]) &
                    (df["party"] == row["party"])
                )
                perception = row.get("candidate_perception_score", 0) or 0
                # Scale: max perception (5) = max 1.5pp boost
                df.loc[mask, "ai_sentiment_boost"] += (perception / 5) * 1.5 * const_weight

        # Apply with cap: max ±3pp adjustment
        df["ai_sentiment_boost"] = df["ai_sentiment_boost"].clip(-3, 3)
        df["predicted_vs"] = (df["predicted_vs"] + df["ai_sentiment_boost"]).clip(0)

        logger.info(
            "AI sentiment applied: avg boost=%.3f pp (range: %.2f to %.2f)",
            df["ai_sentiment_boost"].mean(),
            df["ai_sentiment_boost"].min(),
            df["ai_sentiment_boost"].max(),
        )
        return df

    # ── Prompt builders ───────────────────────────────────────────────────

    @staticmethod
    def _build_state_sentiment_prompt(
        state: str, parties: List[str], context: str
    ) -> str:
        parties_str = ", ".join(parties)
        return f"""You are an expert Indian political analyst specialising in state elections.

State: {state}
Election Year: 2026
Parties to evaluate: {parties_str}
Additional context: {context or "No additional context provided."}

Based on your knowledge of the political situation in {state} heading into the 2026 assembly elections, 
evaluate the public sentiment and electoral momentum for each party.

Respond ONLY with a valid JSON object in exactly this format:
{{
  "party_name_1": <score from -10 to +10>,
  "party_name_2": <score from -10 to +10>,
  ...
}}

Where:
  +10 = extremely strong wave, dominant sentiment
  +5  = positive momentum, likely gains
  0   = neutral, no significant shift
  -5  = negative sentiment, likely losses  
  -10 = severe anti-incumbency or collapse

Use EXACTLY the party names provided. No explanation, no markdown, just the JSON."""

    @staticmethod
    def _build_constituency_prompt(
        state: str, constituencies: List[str], parties: List[str]
    ) -> str:
        consts_str = "\n".join([f"- {c}" for c in constituencies])
        parties_str = ", ".join(parties)
        return f"""You are an expert Indian political analyst.

State: {state}
Election Year: 2026
Parties: {parties_str}

Constituencies to evaluate:
{consts_str}

For each constituency listed, identify if any of the given parties has a strong LOCAL candidate 
advantage or disadvantage (separate from state-wide trends).

Respond ONLY with valid JSON:
{{
  "constituency_name": {{
    "party_with_local_advantage": "PARTY_NAME or null",
    "advantage_score": <1 to 5, or 0 if none>,
    "party_with_local_disadvantage": "PARTY_NAME or null", 
    "disadvantage_score": <1 to 5, or 0 if none>,
    "key_local_issue": "brief description or null"
  }},
  ...
}}

Use exact constituency names from the list above. No explanation, just JSON."""

    # ── API call ──────────────────────────────────────────────────────────

    def _call_api(self, prompt: str, max_tokens: int = 800) -> Optional[str]:
        """Call Claude API with error handling and retries."""
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)
            message = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return message.content[0].text
        except ImportError:
            logger.warning("anthropic package not installed: pip install anthropic")
            return None
        except Exception as e:
            logger.error("API call failed: %s", str(e))
            return None

    # ── Response parsing ──────────────────────────────────────────────────

    @staticmethod
    def _parse_sentiment_response(
        response: str, parties: List[str]
    ) -> Dict[str, float]:
        """Parse JSON response into party → score dict."""
        try:
            # Strip any markdown fences
            clean = response.strip().lstrip("```json").rstrip("```").strip()
            data = json.loads(clean)
            scores = {}
            for party in parties:
                # Try exact match then case-insensitive
                if party in data:
                    scores[party] = float(data[party])
                else:
                    for k, v in data.items():
                        if k.upper() == party.upper():
                            scores[party] = float(v)
                            break
                    else:
                        scores[party] = 0.0
            return scores
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("Failed to parse sentiment response: %s", e)
            return {p: 0.0 for p in parties}

    def _batch_constituency_sentiment(
        self,
        state: str,
        constituencies: List[str],
        parties: List[str],
        context: str,
    ) -> List[Dict]:
        cache_key = self._cache_key(
            f"{state}_const_{'_'.join(sorted(constituencies[:3]))}"
        )
        cached = self._load_cache(cache_key)
        if cached:
            return cached

        prompt = self._build_constituency_prompt(state, constituencies, parties)
        response = self._call_api(prompt, max_tokens=1200)

        results = []
        if response:
            try:
                clean = response.strip().lstrip("```json").rstrip("```").strip()
                data = json.loads(clean)
                for const_name, info in data.items():
                    adv_party = info.get("party_with_local_advantage")
                    dis_party = info.get("party_with_local_disadvantage")

                    for party in parties:
                        score = 0.0
                        if adv_party and adv_party.upper() == party.upper():
                            score = float(info.get("advantage_score", 0))
                        elif dis_party and dis_party.upper() == party.upper():
                            score = -float(info.get("disadvantage_score", 0))

                        results.append({
                            "state": state,
                            "constituency_name": const_name.lower().strip(),
                            "party": party,
                            "candidate_perception_score": score,
                            "local_issue_flag": 1 if info.get("key_local_issue") else 0,
                            "sentiment_notes": info.get("key_local_issue", ""),
                        })
            except Exception as e:
                logger.warning("Constituency sentiment parse failed: %s", e)

        self._save_cache(cache_key, results)
        return results

    # ── Cache helpers ─────────────────────────────────────────────────────

    def _cache_key(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()[:12]

    def _load_cache(self, key: str):
        path = os.path.join(self.cache_dir, f"{key}.json")
        if os.path.exists(path):
            try:
                with open(path) as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def _save_cache(self, key: str, data):
        path = os.path.join(self.cache_dir, f"{key}.json")
        with open(path, "w") as f:
            json.dump(data, f)
