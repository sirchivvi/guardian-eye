# GuardianEye

> **Academic proof of concept — not a validated field alert system.**
>
> The public datasets used in this project do not contain real, time-aligned
> poaching incidents.  Fusion evaluation uses **synthetic image–audio pairs**.
> Results must not be presented as ground-truth field performance data.

---

GuardianEye is a multi-modal anti-poaching risk-assessment system that
combines an **image classification branch** with an **audio classification
branch** to produce a categorical risk level (HIGH / MEDIUM / LOW) for each
scene.

---

## Table of Contents

1. [Project overview](#1-project-overview)
2. [Modalities](#2-modalities)
3. [Risk decision rules](#3-risk-decision-rules)
4. [Repository layout](#4-repository-layout)
5. [Installation — local (CPU)](#5-installation--local-cpu)
6. [Configuration and path precedence](#6-configuration-and-path-precedence)
7. [Running tests and lint locally](#7-running-tests-and-lint-locally)
8. [Smoke-test mode](#8-smoke-test-mode)
9. [Data packaging for cloud upload](#9-data-packaging-for-cloud-upload)
10. [Running on Google Colab](#10-running-on-google-colab)
11. [Running on Kaggle](#11-running-on-kaggle)
12. [Resuming from checkpoints](#12-resuming-from-checkpoints)
13. [Phased development](#13-phased-development)
14. [Datasets](#14-datasets)
15. [Academic scope disclaimer](#15-academic-scope-disclaimer)

---

## 1. Project overview

GuardianEye fuses predictions from a vision model (image branch) and an audio
model (audio branch) to assess the risk level of a wildlife monitoring scene:

```
image frame ──► image branch ──► label (human|vehicle|animal|empty) ─┐
                                                                       ├─► risk()  ──► HIGH|MEDIUM|LOW
audio clip  ──► audio branch ──► label (gunshot|background)          ─┘
```

The system is designed for academic research on the fusion of heterogeneous
wildlife-monitoring modalities.

---

## 2. Modalities

| Modality | Input | Labels | Branch |
|---|---|---|---|
| **Image** | Camera-trap frames (or video frames) | `human`, `vehicle`, `animal`, `empty` | `src/image/` |
| **Audio** | Acoustic sensor clips (WAV/FLAC) | `gunshot`, `background` | `src/audio/` |

Video is handled by decoding frames and routing them through the image branch.
There is no separate video branch.

---

## 3. Risk decision rules

| Image label | Audio label | Suspicious | Risk |
|---|---|---:|---|
| `human` | `gunshot` | **1** | **HIGH** |
| `vehicle` | `gunshot` | **1** | **HIGH** |
| `animal` | `gunshot` | 0 | MEDIUM |
| `human` | `background` | 0 | MEDIUM |
| `vehicle` | `background` | 0 | MEDIUM |
| `animal` | `background` | 0 | LOW |
| `empty` | `background` | 0 | LOW |

**Notes**

- HIGH requires *both* a strong image indicator (human or vehicle) *and* a
  gunshot audio signal.  Neither alone is sufficient.
- ESC-50 thunderstorm, wind, and rain clips remain labelled **background**
  with a `noise_tag` of `thunder_wind`.  They are never relabelled as
  `gunshot`.  When paired with an `empty` image they produce a LOW risk score.
- The `noise_tag` field is upstream metadata; the `suspicious()` and `risk()`
  functions each accept exactly two arguments: `image_label` and `audio_label`.

---

## 4. Repository layout

```
guardian-eye/
├── .agents/rules/guardian.md    ← standing rules for all contributors
├── .github/workflows/ci.yml     ← CI: ruff + pytest on Python 3.11
├── data/
│   ├── raw/                     ← dataset directories (never committed)
│   └── processed/
├── docs/
├── notebooks/
│   ├── colab_run.ipynb          ← Colab orchestration
│   └── kaggle_run.ipynb         ← Kaggle orchestration
├── results/                     ← checkpoints and outputs (never committed)
├── scripts/
│   ├── run.py                   ← main CLI entry point
│   └── pack_data.py             ← dataset packaging
├── src/
│   ├── config.py                ← Pydantic v2 config loader
│   ├── decisions.py             ← risk decision rules
│   ├── logging_config.py        ← logging setup + device detection
│   ├── checkpoint.py            ← shared checkpoint utilities
│   ├── image/                   ← Phase 1+
│   ├── audio/                   ← Phase 1+
│   ├── fusion/                  ← Phase 2+
│   ├── evaluation/              ← Phase 2+
│   ├── bda/                     ← Phase 3+
│   └── api/                     ← Phase 4+
├── tests/
│   ├── test_decisions.py
│   ├── test_config.py
│   ├── test_checkpoint.py
│   └── test_smoke.py
├── .env.example
├── .gitignore
├── config.yaml                  ← all experiment settings
├── Makefile
├── pyproject.toml
└── README.md
```

---

## 5. Installation — local (CPU)

```powershell
# Requires Python 3.11
py -3.11 -m pip install -e ".[dev]"
# Install CPU-only torch if you don't have a GPU:
py -3.11 -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

## 6. Configuration and path precedence

All settings live in `config.yaml`.  Paths follow this precedence
(**highest first**):

| Level | How to set |
|---|---|
| 1 — CLI argument | `--data-root /path/to/data` |
| 2 — Environment variable | `DATA_ROOT=/path/to/data` |
| 3 — `config.yaml` | `paths.data_root: data` |

```powershell
# Env-var override
$env:DATA_ROOT = "D:\datasets\guardian-eye"
py -3.11 -m scripts.run

# CLI override (highest precedence)
py -3.11 -m scripts.run --data-root D:\datasets\guardian-eye
```

Copy `.env.example` to `.env` and fill in your paths:

```powershell
Copy-Item .env.example .env
# Edit .env — never commit it
```

---

## 7. Running tests and lint locally

```powershell
# All tests (CPU-only, no CUDA required)
make test
# or directly:
py -3.11 -m pytest tests/ -v

# Linting
make lint
# or directly:
py -3.11 -m ruff check .
```

Tests must pass without a GPU.  CUDA-specific behaviour is covered by
monkeypatching in `test_smoke.py`.

---

## 8. Smoke-test mode

Smoke mode verifies that the environment is correctly configured.
It uses tiny synthetic tensors — **no real datasets or GPU required**.
Target runtime: **< 5 minutes on an ordinary CPU**.

```powershell
make smoke
# or directly:
py -3.11 -m scripts.run --smoke
```

Actual elapsed time is logged.  If the run exceeds the 5-minute budget, a
warning is logged instead of claiming success.

---

## 9. Data packaging for cloud upload

Package each dataset directory into a separate `.tar.gz` archive for upload
to Google Drive or Kaggle Datasets:

```powershell
make pack-data
# or with explicit output directory:
py -3.11 scripts/pack_data.py --output-dir D:\upload\archives
```

- One archive per non-empty dataset directory.
- Missing or empty directories produce a warning; packaging continues.
- Safe to rerun — existing archives are overwritten, source data is untouched.

**Storage notes:** Image dataset archives can be several GB each.  Verify
available disk space before running.  Kaggle Datasets has a 100 GB total
limit; large individual archives may exceed Google Drive's 5 TB per-item limit.

---

## 10. Running on Google Colab

Open `notebooks/colab_run.ipynb`.

### Authentication (private repository)

The GitHub repository is private.  Authenticate **without saving tokens in
notebook files**:

1. Create a GitHub personal access token (PAT) with `repo` scope.
2. In Colab, use **Secrets** (`🔑` icon, left sidebar):
   - Add secret `GITHUB_TOKEN` = your PAT.
3. In the notebook, load it:
   ```python
   from google.colab import userdata
   token = userdata.get("GITHUB_TOKEN")
   ```
4. Clone with the token (the notebook cell already handles this — you only
   supply the secret):
   ```bash
   git clone https://<token>@github.com/<user>/guardian-eye.git
   ```

Never paste the token directly into a notebook cell.

---

## 11. Running on Kaggle

Open `notebooks/kaggle_run.ipynb`.

### Authentication (private repository)

1. Add your GitHub PAT as a **Kaggle Secret** (Account → API → Secrets):
   - Key: `GITHUB_TOKEN`
2. Enable the secret for the notebook session.
3. The notebook reads it from `os.environ["GITHUB_TOKEN"]`.

### Datasets

Upload each archive produced by `make pack-data` as a **Kaggle Dataset**.
The notebook mounts these datasets and sets `DATA_ROOT` accordingly.

---

## 12. Resuming from checkpoints

All training scripts (Phase 1+) save checkpoints after every epoch using
`src.checkpoint.save_checkpoint()`.  Checkpoints are stored under
`RESULTS_ROOT/checkpoints/<branch>/epoch_NNNN.pt`.

To resume a previous run, simply rerun the same `make` command — training
scripts automatically discover and load the latest checkpoint via
`src.checkpoint.find_latest_checkpoint()`.

To resume from an explicit checkpoint:

```python
from src.checkpoint import load_checkpoint
state = load_checkpoint("results/checkpoints/image_branch/epoch_0015.pt")
model.load_state_dict(state["model_state"])
optimizer.load_state_dict(state["optimizer_state"])
start_epoch = state["epoch"] + 1
```

Checkpoint state always includes:
`model_state`, `optimizer_state`, `epoch`, `config`, `seed`,
and optionally `scheduler_state`, `scaler_state`.

---

## 13. Phased development

| Phase | Scope | Status |
|---|---|---|
| **0** | Scaffold, config, decisions, checkpoints, CI | ✅ Complete |
| 1 | Image branch + audio branch training | Not started |
| 2 | Pairing, fusion, evaluation | Not started |
| 3 | Bootstrap difference analysis (BDA) | Not started |
| 4 | REST API for inference | Not started |

Each phase is completed and reviewed before the next begins.

---

## 14. Datasets

| Dataset | Modality | Notes |
|---|---|---|
| BirdsAI | Image | Aerial thermal camera traps |
| CCT (Caltech Camera Traps) | Image | Ground-level camera traps |
| Serengeti | Image | Camera trap survey |
| Belize | Image | Tropical forest camera traps |
| Vietnam | Image | Camera trap survey |
| ESC-50 | Audio | Environmental sound classification |

Datasets are **never downloaded automatically**.  Place them under
`data/raw/<dataset-name>/` and run `make data-check` to verify.

---

## 15. Academic scope disclaimer

GuardianEye is an **academic proof of concept**.

- The public datasets do **not** provide real, time-aligned poaching incidents.
- Fusion evaluation uses **synthetic image–audio pairs** constructed from the
  individual modality datasets.
- This system has **not** been validated in the field.
- Results must not be presented as a real-world alert system's performance.
