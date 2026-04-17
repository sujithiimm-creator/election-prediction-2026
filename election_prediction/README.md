# Election Prediction 2026 — Complete ML System

**India Predicts 2026 | Data Science Academy**  
824 constituencies · 5 states · Deadline: April 30, 11:59 PM

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ELECTION PREDICTION PIPELINE                      │
│                                                                      │
│  RAW ECI DATA                                                        │
│      │                                                               │
│      ▼                                                               │
│  [1. INGESTION]  ──  Loads all state × year Excel files             │
│      │               Standardises column names                       │
│      ▼                                                               │
│  [2. CLEANING]   ──  Vote share, margin, turnout, position          │
│      │               Removes junk, recomputes totals                 │
│      ▼                                                               │
│  [3. FEATURES]   ──  vs_t1/t2/t3, trend, incumbency                │
│      │               margin category, stronghold, volatility         │
│      ▼                                                               │
│  [4. ALLIANCE]   ──  Co-occurrence inference                        │
│      │               Vote transfer coefficients                      │
│      ▼                                                               │
│  [5. LightGBM]   ──  Regress vote share per party                  │
│      │               Time-aware CV, SHAP explainability              │
│      ▼                                                               │
│  [6. RULES]      ──  Stronghold boost                               │
│      │               Anti-incumbency dampening                       │
│      │               Alliance transfer adjustment                    │
│      ▼                                                               │
│  [7. AI LAYER]   ──  Claude API sentiment scoring (optional)        │
│      │               Party + candidate perception signals            │
│      ▼                                                               │
│  [8. SIMULATION] ──  UNS swing scenarios (optimistic/neutral/pessim)│
│      │               Consensus blending                              │
│      ▼                                                               │
│  [9. HUMAN LOOP] ──  Swing seat flagging                            │
│      │               Expert review Excel export                      │
│      │               Manual override merging                         │
│      ▼                                                               │
│  [10. OUTPUT]    ──  Competition submission Excel                   │
│                      Methodology note (auto-generated)               │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Folder Structure

```
election_prediction/
├── config/
│   └── config.yaml              ← All tunable parameters
│
├── data/
│   ├── raw/                     ← Put your ECI Excel files here
│   ├── processed/               ← Auto-generated clean data
│   └── external/                ← Sentiment cache, polls
│
├── src/
│   ├── pipeline/
│   │   ├── ingestion.py         ← ECI file loader + normaliser
│   │   ├── cleaning.py          ← Data cleaning + derived columns
│   │   └── output_generator.py  ← Submission file builder
│   │
│   ├── features/
│   │   ├── base_features.py     ← Core feature engineering
│   │   └── alliance_inference.py ← Dynamic alliance detection
│   │
│   ├── models/
│   │   ├── lgbm_model.py        ← LightGBM vote share regressor
│   │   └── evaluator.py         ← Accuracy metrics + backtesting
│   │
│   ├── rules/
│   │   └── rule_engine.py       ← Stronghold / anti-incumb / alliance rules
│   │
│   ├── simulation/
│   │   └── swing_simulator.py   ← UNS scenario modelling
│   │
│   ├── ai_layer/
│   │   └── sentiment_signals.py ← Claude API sentiment extraction
│   │
│   ├── human_loop/
│   │   └── review_interface.py  ← HITL swing seat review
│   │
│   └── utils/
│       └── helpers.py           ← Logging, config, shared utils
│
├── notebooks/
│   └── election_prediction_2026.ipynb  ← Interactive exploration
│
├── outputs/
│   ├── predictions/             ← Submission Excel files
│   ├── reports/                 ← Methodology notes, eval reports
│   ├── models/                  ← Saved model files
│   └── review/                  ← HITL swing seat review sheets
│
├── run_pipeline.py              ← Master CLI runner
└── requirements.txt
```

---

## Quick Start (3 Steps)

### Step 1: Setup
```bash
pip install -r requirements.txt
```

### Step 2: Add Data
Place ECI Excel files in `data/raw/` with this naming convention:
```
TamilNadu_2021_DetailedResults.xlsx
TamilNadu_2016_DetailedResults.xlsx
TamilNadu_2011_DetailedResults.xlsx
Kerala_2021_DetailedResults.xlsx
... (repeat for all states and years)
```

**ECI data sources:**
- https://www.eci.gov.in/statistical-reports
- https://myneta.info/

### Step 3: Run
```bash
# Full pipeline for Tamil Nadu only (start here)
python run_pipeline.py --mode full --states "Tamil Nadu"

# Full pipeline for all states
python run_pipeline.py --mode full

# Backtest first to check model accuracy on 2021 data
python run_pipeline.py --mode backtest --target_year 2021

# Or use the Jupyter notebook (recommended for first run)
jupyter notebook notebooks/election_prediction_2026.ipynb
```

---

## Configuration

Edit `config/config.yaml` to tune:

| Parameter | Default | Effect |
|---|---|---|
| `model.lgbm_params.n_estimators` | 500 | More trees = better but slower |
| `features.swing_damping` | 0.7 | How much state swing affects local |
| `confidence.low_threshold` | 0.55 | Below this prob → flag for review |
| `features.stronghold_threshold` | 2 | Wins needed to be a stronghold |
| `features.margin_safe_threshold` | 10 | Margin above this = safe seat |

---

## HITL Workflow (Most Important for Accuracy)

After first run, the system exports a file:
```
outputs/review/swing_seats_review.xlsx
```

Open it. For each flagged constituency:
1. Look at `model_predicted_winner` and supporting vote share columns
2. Use your TN domain knowledge to validate or override
3. Fill `EXPERT_OVERRIDE_WINNER` with the party abbreviation (e.g., `DMK`)
4. Add notes in `EXPERT_NOTES` (optional)
5. Re-run pipeline — corrections are auto-merged

**This single step can lift accuracy by 5–10 percentage points on swing seats.**

---

## AI Sentiment Layer (Optional)

Set your Anthropic API key to enable Claude-powered sentiment scoring:
```bash
export ANTHROPIC_API_KEY=sk-ant-your-key-here
```

Then in `run_pipeline.py`, the AI layer will automatically:
- Score each party's electoral momentum (-10 to +10)
- Flag constituencies with strong candidate advantages
- Adjust vote shares by up to ±3 percentage points

---

## Manual Overrides (TN Expert Knowledge)

In `run_pipeline.py`, edit `TN_MANUAL_OVERRIDES`:
```python
TN_MANUAL_OVERRIDES = {
    ("Tamil Nadu", "coimbatore south"): "BJP",
    ("Tamil Nadu", "tanjore"):          "DMK",
    ("Tamil Nadu", "vellore"):          "DMK",
    # Add as many as your domain knowledge supports
}
```
These are applied LAST and override everything else.

---

## Accuracy Target: 85%+

| Layer | Expected Accuracy Gain |
|---|---|
| Baseline (incumbency only) | ~60–65% |
| + Feature engineering + LightGBM | ~70–75% |
| + Rule engine | +3–5% |
| + Alliance modelling | +2–3% |
| + HITL expert review (swing seats) | +5–10% |
| + AI sentiment signals | +1–2% |
| **Total target** | **~82–88%** |

---

## Submission Checklist

- [ ] Predictions cover all 824 constituencies
- [ ] Excel file uses competition template format
- [ ] Methodology note is 150+ words (auto-generated at `outputs/reports/methodology_note.txt`)
- [ ] Submitted before April 30, 11:59 PM
- [ ] **Submit early drafts** — tiebreaker uses submission timestamp!

---

## Key Dates

| Date | Event |
|---|---|
| April 9 | Kerala, Assam, Puducherry polls |
| April 23–29 | Tamil Nadu, West Bengal polls |
| **April 30** | **Submission deadline 11:59 PM** |
| May 2 | Competition results |
| May 4 | ECI declares official results |

---

*Built for India Predicts 2026 · Data Science Academy*
