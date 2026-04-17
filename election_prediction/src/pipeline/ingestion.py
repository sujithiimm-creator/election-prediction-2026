"""
pipeline/ingestion.py
---------------------
Reads raw ECI Excel files (one per state-year), validates schema,
and outputs a single unified raw DataFrame.

ECI files typically come as multi-sheet Excel workbooks.  
This module handles the most common sheet layouts from ECI statistical reports.
"""

import os
import glob
import pandas as pd
import numpy as np
from typing import List, Optional, Dict
from src.utils.helpers import get_logger, normalise_party, normalise_constituency, ensure_dir

logger = get_logger(__name__)


# ── Expected sheet names in ECI workbooks (case-insensitive search) ──────────

CANDIDATE_SHEET_HINTS  = ["detailed results", "candidate", "result", "detail"]
SUMMARY_SHEET_HINTS    = ["constituency", "summary", "const wise"]
ELECTOR_SHEET_HINTS    = ["elector", "annexure", "electors"]


# ── Column mapping: raw ECI names → internal schema ─────────────────────────
#    Adjust these if your specific ECI file uses different headers.

COLUMN_MAP = {
    # Constituency identifiers
    "st_name":              "state",
    "state":                "state",
    "pc_name":              "state",           # sometimes PC is state-level proxy
    "ac_name":              "constituency_name",
    "assembly_constituency": "constituency_name",
    "constituency":         "constituency_name",
    "ac_no":                "constituency_no",
    "assembly_no":          "constituency_no",
    "sno":                  "constituency_no",

    # Candidate fields
    "candidate":            "candidate_name",
    "candidate_name":       "candidate_name",
    "name_of_the_candidate": "candidate_name",
    "party_abbreviation":   "party",
    "party":                "party",
    "partyabbre":           "party",
    "votes":                "votes",
    "total_votes":          "votes",
    "votesgot":             "votes",

    # Electorate / turnout
    "total_electors":       "total_electors",
    "electors":             "total_electors",
    "no_of_electors":       "total_electors",
    "total_votes_polled":   "total_votes_polled",
    "total_valid_votes":    "total_votes_polled",
    "validvotes":           "total_votes_polled",

    # Position (rank within constituency)
    "position":             "position",
    "pos":                  "position",

    # Sex of candidate
    "sex":                  "candidate_sex",
    "gender":               "candidate_sex",
}


class ECIDataIngester:
    """
    Reads one or more ECI election result files and produces a
    unified long-format DataFrame (one row per candidate per constituency).
    
    Usage
    -----
    ingester = ECIDataIngester(raw_data_dir="data/raw/")
    df_raw = ingester.load_all()
    """

    def __init__(self, raw_data_dir: str = "data/raw/"):
        self.raw_data_dir = raw_data_dir
        self.loaded_files: List[str] = []
        self.failed_files: List[str] = []

    # ── Public entry point ───────────────────────────────────────────────────

    def load_all(self) -> pd.DataFrame:
        """
        Scan raw_data_dir for all .xlsx / .xls / .csv files,
        parse each, and return a single concatenated DataFrame.
        """
        files = self._discover_files()
        if not files:
            logger.warning("No data files found in %s", self.raw_data_dir)
            return pd.DataFrame()

        frames: List[pd.DataFrame] = []
        for f in files:
            df = self._load_single_file(f)
            if df is not None and not df.empty:
                frames.append(df)
                self.loaded_files.append(f)
            else:
                self.failed_files.append(f)

        if not frames:
            raise RuntimeError("All files failed to load. Check raw data directory.")

        combined = pd.concat(frames, ignore_index=True)
        logger.info(
            "Ingestion complete: %d files loaded, %d failed → %d total rows",
            len(self.loaded_files), len(self.failed_files), len(combined),
        )
        return combined

    def load_from_dataframe(self, df: pd.DataFrame, state: str, year: int) -> pd.DataFrame:
        """
        Accept an already-loaded DataFrame (e.g., from Jupyter exploration)
        and apply the same normalisation pipeline.
        """
        df = df.copy()
        df["state"] = state
        df["year"] = year
        return self._normalise(df)

    # ── File discovery ───────────────────────────────────────────────────────

    def _discover_files(self) -> List[str]:
        patterns = ["*.xlsx", "*.xls", "*.csv"]
        files = []
        for p in patterns:
            files.extend(glob.glob(os.path.join(self.raw_data_dir, "**", p), recursive=True))
        logger.info("Discovered %d files in %s", len(files), self.raw_data_dir)
        return sorted(files)

    # ── Single file loader ───────────────────────────────────────────────────

    def _load_single_file(self, filepath: str) -> Optional[pd.DataFrame]:
        """
        Load one file.  Infers state + year from the filename if not in data.
        Expected filename convention: TamilNadu_2021_DetailedResults.xlsx
        """
        try:
            state, year = self._infer_state_year_from_path(filepath)
            ext = os.path.splitext(filepath)[1].lower()

            if ext == ".csv":
                df = pd.read_csv(filepath, low_memory=False)
            else:
                df = self._load_excel(filepath)

            if df is None or df.empty:
                return None

            df["_source_file"] = os.path.basename(filepath)
            if "state" not in df.columns:
                df["state"] = state
            if "year" not in df.columns:
                df["year"] = year

            df = self._normalise(df)
            logger.info("  ✓ %s → %d rows (state=%s, year=%s)", 
                        os.path.basename(filepath), len(df), state, year)
            return df

        except Exception as e:
            logger.error("  ✗ Failed to load %s: %s", filepath, str(e))
            return None

    def _load_excel(self, filepath: str) -> Optional[pd.DataFrame]:
        """
        Try to find the 'Detailed Results' sheet; fall back to first sheet.
        """
        try:
            xl = pd.ExcelFile(filepath)
            sheet_name = self._find_best_sheet(xl.sheet_names, CANDIDATE_SHEET_HINTS)
            df = xl.parse(sheet_name, header=0)
            logger.debug("  Using sheet '%s' from %s", sheet_name, os.path.basename(filepath))
            return df
        except Exception as e:
            logger.warning("  ExcelFile failed for %s: %s — trying read_excel", filepath, e)
            return pd.read_excel(filepath, header=0)

    # ── Sheet selector ───────────────────────────────────────────────────────

    @staticmethod
    def _find_best_sheet(sheet_names: List[str], hints: List[str]) -> str:
        """Return the sheet name that best matches any hint keyword."""
        for hint in hints:
            for name in sheet_names:
                if hint.lower() in name.lower():
                    return name
        return sheet_names[0]  # fallback: first sheet

    # ── State/year inference from filename ──────────────────────────────────

    @staticmethod
    def _infer_state_year_from_path(filepath: str):
        """
        Infer state and year from filename.
        E.g. 'TamilNadu_2021_DetailedResults.xlsx' → ('Tamil Nadu', 2021)
        """
        basename = os.path.basename(filepath).replace("_", " ").replace("-", " ")
        
        state_map = {
            "tamilnadu": "Tamil Nadu", "tamil nadu": "Tamil Nadu",
            "kerala": "Kerala",
            "westbengal": "West Bengal", "west bengal": "West Bengal",
            "assam": "Assam",
            "puducherry": "Puducherry", "pondicherry": "Puducherry",
        }
        
        detected_state = "Unknown"
        for key, val in state_map.items():
            if key in basename.lower():
                detected_state = val
                break
        
        detected_year = None
        for token in basename.split():
            if token.isdigit() and 2000 <= int(token) <= 2030:
                detected_year = int(token)
                break
        
        return detected_state, detected_year

    # ── Normalisation ────────────────────────────────────────────────────────

    def _normalise(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply column renaming, type coercion, and basic cleaning.
        """
        df = df.copy()

        # Lowercase all column names for consistent mapping
        df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

        # Apply column map
        rename = {k: v for k, v in COLUMN_MAP.items() if k in df.columns}
        df.rename(columns=rename, inplace=True)

        # Ensure required columns exist
        for col in ["constituency_name", "party", "votes"]:
            if col not in df.columns:
                logger.warning("Column '%s' not found after normalisation", col)
                df[col] = np.nan

        # Type coercions
        if "votes" in df.columns:
            df["votes"] = pd.to_numeric(
                df["votes"].astype(str).str.replace(",", ""), errors="coerce"
            ).fillna(0).astype(int)

        if "total_electors" in df.columns:
            df["total_electors"] = pd.to_numeric(
                df["total_electors"].astype(str).str.replace(",", ""), errors="coerce"
            )

        if "total_votes_polled" in df.columns:
            df["total_votes_polled"] = pd.to_numeric(
                df["total_votes_polled"].astype(str).str.replace(",", ""), errors="coerce"
            )

        if "constituency_no" in df.columns:
            df["constituency_no"] = pd.to_numeric(df["constituency_no"], errors="coerce")

        if "year" in df.columns:
            df["year"] = pd.to_numeric(df["year"], errors="coerce")

        # Normalise text fields
        df["party"] = df["party"].apply(normalise_party)
        df["constituency_name"] = df["constituency_name"].apply(normalise_constituency)

        if "state" in df.columns:
            df["state"] = df["state"].astype(str).str.strip()

        # Drop rows with no votes at all (headers that leaked in)
        df = df[df["votes"] > 0].reset_index(drop=True)

        return df


# ── Convenience function ─────────────────────────────────────────────────────

def ingest_data(raw_dir: str = "data/raw/") -> pd.DataFrame:
    """One-line entry point for the pipeline."""
    ingester = ECIDataIngester(raw_data_dir=raw_dir)
    return ingester.load_all()
