# Sakhi

Sakhi is a women's safety triage and rights navigator for India. It accepts
descriptions in Hindi (Devanagari), Hinglish (Roman script), or English and
provides:

- multi-label situation classification;
- single-label urgency triage;
- retrieval of relevant provisions from official legal texts;
- plain-language explanations generated only from retrieved text and
  hand-written templates; and
- routing to the appropriate authority or support channel.

Sakhi is a guidance and triage aid. It is **not an emergency service and not
legal advice**, and it never gives legal verdicts. If a situation may involve
immediate danger, the product must show **Call 112** before all other content.

## Repository layout

```text
sakhi/
  config/                 Label, helpline, spelling, and font configuration
  data/raw/               Team-authored input scenarios (no real victim data)
  data/processed/         Saved preprocessing snapshots and data splits
  data/legal/             Official legal text and section metadata
  notebooks/              Exploratory and training notebooks
  src/                    Data, preprocessing, models, retrieval, and pipeline
  app/                    Streamlit application
  reports/figures/        EDA outputs and evaluation figures
  tests/                  Unit and integration tests
  requirements.txt        Python dependencies
```

## Setup

Sakhi requires Python 3.10 or newer. Create and activate a virtual environment,
then install the dependencies:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Training code is designed to run locally on small samples and in Google Colab
or Kaggle for the full experiments. No paid API or hosted LLM API is required.

The application and scripts will be documented here as they are implemented.
Run tests from the repository root with:

```powershell
python -m pytest
```

## Data and experiment policy

- The team writes all training scenarios. Do not add real victims' data,
  identifying details, or case records.
- The dataset includes a `group_id`; paraphrases or variants of one scenario
  must share that ID and remain in one split to prevent leakage.
- The split script assigns `split` once, saves `frozen_dataset.csv` as the
  source of truth, and refuses to regenerate it. Use the script without
  `--freeze` for previews; only pass `--freeze` after the complete dataset and
  law-student review are finished. The derived train/val/test files must not be
  recreated after experiments begin.
- Permanent freezing is blocked below 400 rows as a guard against accidentally
  freezing an incomplete dataset. Preview runs may use the fictional sample or
  a throwaway copy.
- Every split run writes `split_class_counts.csv`, so reported test metrics can
  be paired with the number of examples for every urgency and situation class.
- The freeze checks for exact and near-identical text across different splits
  and logs every flagged pair for manual review.
- The first real freeze must contain every configured urgency and situation
  label in the test split, with at least 5 examples per urgency class and at
  least 3 examples per situation label. This makes missing-class recall visible
  before evaluation; a freeze that fails this check must be reviewed rather than
  silently accepted.
- To deliberately unfreeze, the team must agree first, then manually delete
  `frozen_dataset.csv` and the generated split files before running a new
  freeze.
- `data/raw/sample_fictional.csv` is a separate pipeline smoke test only and
  must never be used to create production splits.
- Use the fixed random seed defined by the project code and never tune on the
  test split.
- Report macro-F1 and per-class metrics, prioritising recall for
  `immediate_danger` and `urgent`.
- Compare BM25 with multilingual embedding retrieval and report Recall@k and
  MRR on the hand-made query-to-section evaluation set.
- Legal text must come from official sources and retain Act and section
  metadata. An IPC-to-BNS mapping is used only when ingesting older legal text.
- Preprocessing is saved in five stages. Word clouds use a Devanagari-capable
  font configured in `config/preprocess.yaml`; download the open-licensed
  Noto Sans Devanagari font from
  https://fonts.google.com/specimen/Noto+Sans+Devanagari and place the `.ttf`
  at the configured path.

## Safety, privacy, and legal constraints

- The app must not log or store user messages.
- The UI must show a clear disclaimer that Sakhi is not an emergency service
  and not legal advice.
- For `immediate_danger`, **Call 112** is always displayed first.
- Low-confidence predictions use a safe fallback with general support options
  and a request for more information.
- Helpline numbers are kept in one configuration file and marked
  **VERIFY BEFORE RELEASE**. Do not hardcode them elsewhere.
- Legal explanations use retrieved statute text and fixed, hand-written
  templates; they are not generated by a generative model.
- Current entity masking covers phone numbers, email addresses, and long digit
  strings only. Names are deliberately not detected or masked yet; this is a
  known limitation for the current fictional-data prototype and must be
  addressed before handling real user messages.
- Do not publish model outputs as legal conclusions or guarantees.

## Development status

This repository is being built incrementally for the CS459 university project.
The initial scaffold is intentionally free of real legal corpora, real user
messages, and verified production contact details.
