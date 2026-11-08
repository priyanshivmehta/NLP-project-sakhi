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

The app and inference pipeline load trained artifacts from `sakhi/models/` by
default. Set `SAKHI_MODELS_DIR` to use another trained-artifact directory:

```powershell
$env:SAKHI_MODELS_DIR = "C:\path\to\trained\models"
streamlit run sakhi/app/streamlit_app.py
```

Set `SAKHI_DEPLOYED=true` in deployed environments. The app refuses to start
without a trained model in that mode; it never substitutes a keyword demo.

The application and scripts will be documented here as they are implemented.
Run tests from the repository root with:

```powershell
python -m pytest
```

## Baseline training and evaluation

The classical baseline is trained with:

```powershell
python -m sakhi.src.train_baseline
```

It fits word (1-2 gram) and character (2-5 gram) TF-IDF features on
`train.csv` only. Urgency uses balanced logistic regression; situation uses
one-vs-rest balanced logistic regression. Situation thresholds are tuned on
`val.csv` only and saved with the model artifacts under `sakhi/models/`.

Evaluate validation by default:

```powershell
python -m sakhi.src.evaluate
```

Test evaluation requires an explicit one-time flag:

```powershell
python -m sakhi.src.evaluate --final-test
```

For a code-path check using only the fictional sample:

```powershell
python -m sakhi.src.evaluate --smoke
```

Smoke outputs are labelled **SMOKE TEST, NOT A REAL RESULT** and must not be
reported as model performance.

## Legal corpus and retrieval

Place official legal source files only under `sakhi/data/legal/source/`.
Ingestion never creates or paraphrases statute text. It writes one chunk per
parsed section to `sakhi/data/legal/chunks/chunks.csv` and records sections
that need manual repair in `sakhi/data/legal/unparsed_sections.txt`:

```powershell
python -m sakhi.src.legal_ingest
```

BM25 retrieval uses those chunks:

```powershell
python -m sakhi.src.retrieval "your query" --chunks sakhi/data/legal/chunks/chunks.csv
python -m sakhi.src.eval_retrieval
python -m sakhi.src.eval_retrieval --smoke
```

The smoke evaluation uses only a clearly labeled `FAKE LAW TEXT` fixture and
must not be reported as legal-retrieval performance. The IPC-to-BNS mapping
template is intentionally header-only until it is populated and verified by
the project team.

Audit a dataset for duplicate text, repeated phrase patterns, and label leakage:

```powershell
python -m sakhi.src.audit_dataset sakhi/data/raw/dev_dataset_450.csv
```

## Data and experiment policy

- The team writes all training scenarios. Do not add real victims' data,
  identifying details, or case records.
- `data/raw/dev_dataset_450.csv` contains 450 AI-generated rows for training
  only. Evaluation rows may be AI-generated, but they enter validation/test
  only after human verification.
- Every row records `generator` and `human_verified`. The 450 training rows use
  `chatgpt_train` and are never used for validation or test.
- Optional additional training CSVs can be placed in
  `sakhi/data/raw/train_extra/`. Run `python -m sakhi.src.merge_train_extra`
  to validate and merge them into `train_extra.csv`; exact and near duplicates
  are reported in `sakhi/reports/audit/removed_train_extra.csv`.
- Extra merged rows use `source=ai_generated`, `generator=chatgpt_train`, and
  `human_verified=false`.
- The preview readiness report requires at least 30 training rows for every
  situation label and reports the exact shortfall when a label is under that
  threshold.
- `sakhi/data/archive/dev_copy_legacy_ai.csv` is an archived duplicate of the
  450-row training dataset and is not an evaluation source.
- Human-written rows use the `team_written` or `peer_written` source values.
- Validation and test may contain only `human_verified=true` rows whose
  generator does not occur in any training row. Unverified AI rows are
  training-only.
- Generator names are provenance labels, not cryptographic proof of which
  external model or tool produced a row. The project must record the actual
  tool separately when known; differing filename labels alone do not establish
  independent generation.
- The dataset includes a `group_id`; paraphrases or variants of one scenario
  must share that ID and remain in one split to prevent leakage.
- The split script assigns `split` once, saves `frozen_dataset.csv` as the
  source of truth, and refuses to regenerate it. Use the script without
  `--freeze` for previews; only pass `--freeze` after the complete dataset and
  law-student review are finished. The derived train/val/test files must not be
  recreated after experiments begin.
- Permanent freezing requires at least 400 total rows and 150
  human-verified evaluation rows. The human test split must contain at least 30 rows in each language,
  at least 5 rows in each urgency class, and at least 3 rows for each
  situation label.
- Coverage-enabled preview and freeze allocation targets 55% of verified
  evaluation rows for test and 45% for validation, while keeping at least 30
  rows per language in test, 20 per language in validation, and at least 5
  immediate-danger rows in validation. Any unsatisfied rule is reported with
  its exact shortfall.
- AI-generated rows are kept in training only, and rows with TF-IDF cosine
  similarity above 0.85 to a human validation/test row are removed.
- Every split run writes `split_class_counts.csv`, so reported test metrics can
  be paired with the source, generator, verification status, and language
  composition of every urgency and situation class.
- The freeze checks for exact and near-identical text across different splits
  and logs every flagged pair for manual review.
- The first real freeze must contain every configured urgency and situation
  label in the human test split. This makes missing-class recall visible before
  evaluation; a freeze that fails this check must be reviewed rather than
  silently accepted.
- To deliberately unfreeze, the team must agree first, then manually delete
  `frozen_dataset.csv` and the generated split files before running a new
  freeze.
- `data/raw/sample_fictional.csv` is a separate 10-row pipeline smoke test only
  must never be used to create production splits.
- EDA images generated from `sample_fictional.csv` are smoke-test artifacts,
  not report results. After the real dataset is frozen, rerun EDA on that
  frozen dataset and use only those images in reports.
- Use the fixed random seed defined by the project code and never tune on the
  test split.
- Merge fictional evaluation batches and create a review sheet with:

  ```powershell
  python -m sakhi.src.merge_eval_batches
  python -m sakhi.src.import_verified_eval
  python -m sakhi.src.augment_noise
  ```

  Review `sakhi/data/raw/eval_verification_sheet.csv` in a spreadsheet before
  importing it. Only rows marked `ok` or `corrected` become verified; rejected
  rows are dropped. Noise augmentation writes a separate `noisy=true` file,
  and clean/noisy results remain separate.
- Report overall, per-language, per-situation-label, and clean/noisy macro-F1
  with support counts. Verified synthetic evaluation is labelled
  **SYNTHETIC BENCHMARK (AI-generated, human-verified), NOT REAL-WORLD ACCURACY**.
  Report macro-F1 and per-class metrics, prioritising recall for
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
- Configuration-driven safety rules run before model inference. Weapons, acid,
  knives, confinement, kidnapping, and current threats force Call 112 first.
  Minor-related wording adds the child-protection contact, and self-harm
  wording adds crisis support, a caring message, and skips legal explanation.
- Safety keyword patterns live in `config/safety_rules.yaml`; helpline entries
  are loaded from `config/helplines.yaml` and remain marked **VERIFY BEFORE
  RELEASE**.
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
- The bundled Noto Sans Devanagari font is from Google Fonts, copyright The
  Noto Project Authors, and is distributed under the SIL Open Font License
  1.1; see `sakhi/config/fonts/OFL.txt`.

## Development status

This repository is being built incrementally for the CS459 university project.
The initial scaffold is intentionally free of real legal corpora, real user
messages, and verified production contact details.
