# GuardianEye — Standing Agent Rules

These rules apply to every phase and every agent or contributor working on
the GuardianEye project.  Read them before making any change.

---

## 1. Modalities

GuardianEye has exactly **two modalities**: **image** and **audio**.

- Video is handled by decoding frames and routing them through the **image branch**.
- There is no separate video branch.

---

## 2. Approved technology stack

| Tool | Version constraint |
|---|---|
| Python | 3.11 |
| PyTorch | ≥ 2.3 |
| torchaudio | ≥ 2.3 |
| scikit-learn | ≥ 1.5 |
| pandas | ≥ 2.2 |
| PyYAML | ≥ 6.0 |
| Pydantic | ≥ 2.7 |
| pytest | ≥ 8.2 |
| Ruff | ≥ 0.4 |

**Ask before adding any dependency not listed above.**

---

## 3. Configuration

- All hyperparameters and filesystem paths **must** be read from `config.yaml`.
- Never hard-code absolute paths.
- Path precedence: CLI argument > `DATA_ROOT`/`RESULTS_ROOT` env vars > `config.yaml`.
- Use `src.config.load_config()` and `src.config.resolve_paths()`.

---

## 4. Phased development

- Work **one phase at a time**.
- Stop after each phase and provide a summary of:
  - Files changed or created
  - Commands to run
  - Actual test and lint results (or `NOT YET RUN` with explanation)
- **Never begin the next phase without explicit approval.**

---

## 5. Metrics and results integrity

- **Never invent, fabricate, or estimate metrics.**
- If data or trained models are unavailable, report exactly:
  > `NOT YET RUN — <reason>`
- Do not run evaluations on placeholder or synthetic data and present the
  numbers as if they reflect real dataset performance.

---

## 6. Dataset split discipline

- Assign **train / val / test splits before creating any image–audio pairs**.
- **Never create cross-split pairs** (a training image paired with a test
  audio clip, etc.).
- Split fractions are defined in `config.yaml`:
  `train = 0.70`, `val = 0.15`, `test = 0.15`.

---

## 7. Prohibited synthetic data

- **Never synthesise vehicle images.**  Vehicle imagery is scarce; inventing
  examples would invalidate evaluation results.
- ESC-50 thunderstorm, wind, and rain clips **remain labelled `background`**
  with a `noise_tag` of `thunder_wind`.  They are **never relabelled as
  `gunshot`**.

---

## 8. Dataset access

- **Never download datasets automatically.**
- Read only from the configured `data/raw/` directory.
- If a dataset is missing, log a clear warning and continue or stop — never
  fabricate data.

---

## 9. What must never be committed

| Forbidden | Reason |
|---|---|
| Dataset files | Size and licence |
| Model weights (`.pt`, `.pth`, `.bin`) | Size |
| `.env` files | Secrets |
| API keys, tokens, passwords | Security |
| Generated archives (`*.tar.gz`) | Size |

---

## 10. Code quality

Every public function and class **must** have:
- **Type hints** on all arguments and the return value.
- A **docstring** (Google or NumPy style).
- At least one **unit test** exercising the main behaviour.
- **Logging** (via `logging`) instead of `print()` in application code.

---

## 11. Compute environment

- Local development (Windows laptop, CPU-only) supports tests, linting,
  configuration validation, and smoke tests only.
- All substantive model training and evaluation runs on **Google Colab** or
  **Kaggle**.
- Tests must **never require CUDA**.  Use `monkeypatch` or conditional skips
  for GPU-specific behaviour.

---

## 12. Checkpointing

All training modules **must** use `src.checkpoint`:
- Call `save_checkpoint()` after every epoch.
- Call `find_latest_checkpoint()` on startup to resume automatically.
- Never start training without checking for an existing checkpoint.
- Checkpoint state must include: `model_state`, `optimizer_state`, `epoch`,
  `config`, `seed`, and optionally `scheduler_state` / `scaler_state`.

---

## 13. Destructive or elevated commands

**Always ask before executing** commands that:
- Delete files or directories
- Install system packages (outside pip)
- Modify the Git history

---

## 14. Academic scope disclaimer

GuardianEye is an **academic proof of concept**.  The public datasets used do
not contain real, time-aligned poaching incidents.  Fusion evaluation relies
on **synthetic image–audio pairs**.  Results must not be presented as a
validated field alert system.
