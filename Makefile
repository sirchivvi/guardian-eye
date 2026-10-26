# =============================================================================
# GuardianEye Makefile
# =============================================================================
# Usage:
#   make setup         — install Python 3.11 dev dependencies
#   make test          — run pytest (CPU-only, no CUDA required)
#   make lint          — run ruff check
#   make data-check    — verify expected dataset directories exist
#   make manifest      — list all dataset files to stdout
#   make pack-data     — archive each dataset dir for cloud upload
#   make train-image   — (Phase 1) train image branch          [NOT YET]
#   make train-audio   — (Phase 1) train audio branch          [NOT YET]
#   make pairs         — (Phase 2) create image-audio pairs    [NOT YET]
#   make fuse          — (Phase 2) run late-fusion evaluation  [NOT YET]
#   make evaluate      — (Phase 2) run full evaluation harness [NOT YET]
#   make bda           — (Phase 3) run BDA streaming jobs
#   make bda-check     — (Phase 3) run BDA self-check on fixtures
#   make serve         — (Phase 4) start the REST API server   [NOT YET]
#   make smoke         — run --smoke mode (fast CPU self-test)
# =============================================================================

PYTHON   := py -3.11
PIP      := $(PYTHON) -m pip
PYTEST   := $(PYTHON) -m pytest
RUFF     := $(PYTHON) -m ruff

# Allow DATA_ROOT / RESULTS_ROOT to be set externally
DATA_ROOT    ?=
RESULTS_ROOT ?=

# Build optional CLI path flags when env vars are set
_DATA_FLAG    := $(if $(DATA_ROOT),--data-root $(DATA_ROOT),)
_RESULTS_FLAG := $(if $(RESULTS_ROOT),--results-root $(RESULTS_ROOT),)

.PHONY: setup test lint smoke data-check manifest pack-data \
        train-image train-audio pairs pairs-check fuse evaluate bda bda-check serve \
        clean help

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
setup:
	$(PIP) install -e ".[dev]"

# ---------------------------------------------------------------------------
# Test & Lint (Phase 0 — CPU-only)
# ---------------------------------------------------------------------------
test:
	$(PYTEST) tests/ -v --tb=short

lint:
	$(RUFF) check .

smoke:
	$(PYTHON) -m scripts.run --smoke $(_DATA_FLAG) $(_RESULTS_FLAG)

# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------
data-check:
	@$(PYTHON) -c "import pathlib; p = pathlib.Path('data/raw'); print('Checking expected dataset directories...'); [print(f'  [OK]  {p / d}' if (p / d).is_dir() else f'  [!!]  MISSING: {p / d}') for d in ['birdsai', 'cct', 'serengeti', 'belize', 'vietnam', 'esc50']]"

manifest:
	$(PYTHON) scripts/build_manifest.py $(_DATA_FLAG) $(_RESULTS_FLAG)

verify-data:
	$(PYTHON) scripts/verify_data.py $(_DATA_FLAG)

pack-data:
	$(PYTHON) scripts/pack_data.py $(_DATA_FLAG) $(_RESULTS_FLAG)

# ---------------------------------------------------------------------------
# Phase 1+ targets — NOT YET AVAILABLE
# ---------------------------------------------------------------------------
train-image:
	$(PYTHON) scripts/train_image.py $(_DATA_FLAG) $(_RESULTS_FLAG)

train-audio:
	$(PYTHON) scripts/train_audio.py $(_DATA_FLAG) $(_RESULTS_FLAG)

evaluate-unimodal:
	$(PYTHON) scripts/evaluate_unimodal.py $(_DATA_FLAG) $(_RESULTS_FLAG)

pairs:
	$(PYTHON) scripts/run_pairing.py $(_DATA_FLAG) $(_RESULTS_FLAG)

pairs-check:
	$(PYTHON) scripts/run_pairing.py --self-check $(_DATA_FLAG) $(_RESULTS_FLAG)

fuse:
	$(_not_yet)

evaluate:
	$(_not_yet)

bda:
	$(PYTHON) scripts/run_bda.py $(_DATA_FLAG) $(_RESULTS_FLAG)

bda-check:
	$(PYTHON) scripts/run_bda.py --self-check $(_DATA_FLAG) $(_RESULTS_FLAG)

serve:
	$(_not_yet)

# ---------------------------------------------------------------------------
# Housekeeping
# ---------------------------------------------------------------------------
clean:
	@echo Removing Python caches...
	@$(PYTHON) -c "import pathlib, shutil; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('.pytest_cache')]; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('.ruff_cache')]"

help:
	@grep -E '^[a-zA-Z_-]+:' Makefile | \
		awk -F: '{printf "  %-16s\n", $$1}'
