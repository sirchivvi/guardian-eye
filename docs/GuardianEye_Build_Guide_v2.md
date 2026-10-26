# GuardianEye Build Guide v2 (matches CA4 + BDA reports)

This replaces the earlier guide. It follows your latest scope: **two modalities** (video becomes frames inside the image branch), the five datasets from CA2/CA3/CA4, early/intermediate/late fusion, the `NOT YET RUN` honesty policy, the Hadoop-style counting jobs, and a path to GitHub and a global deployment.

Contents
1. What changed since v1
2. What you need
3. Antigravity: how to use it (beginner section)
4. Antigravity vs VS Code
5. GitHub setup, step by step
6. Data download, step by step (the five real datasets)
7. Rules file to paste into Antigravity
8. MASTER PROMPT (paste into Antigravity)
9. DASHBOARD + DEPLOY PROMPT
10. Global deployment, step by step
11. Consistency flags in your reports

---

## 1. What changed since v1

| Item | v1 guide | Now (CA4 + BDA) |
|---|---|---|
| Modalities | Video, image, audio | **Image + audio.** Video is decoded to frames and enters the image branch |
| Image data | Snapshot Serengeti, CCT, LILA, VisDrone, HIT-UAV | **BIRDSAI, Caltech Camera Traps, Snapshot Serengeti** |
| Audio data | UrbanSound8K, FSD50K, AudioSet, RFCx | **Belize gunshot set (Katsis), Vietnam Chu Mom Ray set (Vu), ESC-50 thunder/wind/rain only as a noise tag** |
| Image classes | person, vehicle, animal | **human, vehicle, animal, empty** |
| Audio classes | gunshot, chainsaw, engine, other | **gunshot vs background** (thunder_wind is a tag, not a class) |
| Models | YOLO + PANNs | **Small CNN image classifier + log-mel CNN audio classifier** (pretrained backbones become an optional extra) |
| Fusion | Noisy-OR + logistic regression | **Early, intermediate (4 heads) and late (calibrated rule + weighted), plus image-only and audio-only baselines** |
| Results policy | Generate metrics | **Anything not produced by code is written NOT YET RUN** |
| Big data | Not included | **src/bda** streaming mapper/reducer jobs, optional Spark and Hive, and a Hadoop vs Spark case study |

---

## 2. What you need

### Accounts
| Account | Why |
|---|---|
| Google account | Sign in to Antigravity (Pro), use Colab |
| GitHub | Repo, CI, and Vercel deploys |
| Hugging Face | Host model weights and the API (free Docker Space) |
| Vercel | Dashboard hosting |
| Mendeley / Zenodo | Dataset downloads (free; Mendeley may ask you to sign in) |

### Software on your laptop
- Python 3.11, Git, **GitHub CLI (`gh`)**, Node.js 20+, FFmpeg
- Docker Desktop (for the API image), Vercel CLI (`npm i -g vercel`), Hugging Face CLI (`pip install huggingface_hub`)
- Optional for the BDA module: Java 11 or 17 and `pyspark` (Hadoop itself is not required; your code already falls back to local streaming)

Check everything:
~~~bash
python --version && git --version && gh --version && node --version && ffmpeg -version | head -1 && docker --version
~~~

### Hardware
16 GB RAM is comfortable. Free disk: Belize audio is large in file count (35,000+ small WAVs), plus image subsets; plan for about 30 to 60 GB. Training the small CNNs works on CPU for smoke tests; use a free Colab GPU for full runs.

---

## 3. Antigravity: how to use it (beginner section)

Antigravity is Google's agent-first coding environment, launched in November 2025. It is built on a heavily modified VS Code fork and has two main views: the familiar **Editor** and the **Agent Manager** (a mission-control screen where you give tasks to agents and review results). Details change quickly, so if a menu looks different in your version, trust your screen and the docs at antigravity.google/docs.

### 3.1 Install and first launch
1. Download from antigravity.google/download (Windows .exe, macOS .dmg, Linux .deb/.rpm).
2. At setup, you can **import settings and extensions from VS Code or Cursor**, or start fresh.
3. Sign in with the Google account that holds your Pro plan.
4. Open your project folder (`File > Open Folder`), not a single file. The agent works on the open workspace.

### 3.2 The two views
| View | Use it for |
|---|---|
| Agent Manager | Start tasks, run several agents in parallel, read their plans and results (called Artifacts), approve or comment |
| Editor | Read and edit code yourself, run the terminal, use Git, use extensions |
Use the switch button in the Agent Manager to jump to the Editor and back.

### 3.3 Modes: Fast vs Planning
- **Planning**: the agent writes an implementation plan first, then builds. Use this for every GuardianEye phase.
- **Fast**: quick edits and small fixes (rename a variable, fix a typo, add a test).

### 3.4 Artifacts (how you review the agent)
Agents produce reviewable artifacts: a plan, a task list, a walkthrough of what changed, screenshots and browser recordings. **Read the plan before approving.** If something is wrong, comment on the plan and ask for a revision instead of letting it build the wrong thing.

### 3.5 Safety settings (important, set these first)
Agents can run terminal commands and edit files. In Settings, find the agent permissions / terminal execution options and:
- Keep terminal commands on **ask before running** (review-driven) until you trust the setup.
- Never approve commands that delete folders, reformat disks, or touch paths outside the project.
- Keep the browser agent on an allowlist if the setting is offered.
- Check **Settings > AI Credits** (or the plan page) to see how much of your Pro allowance you use; heavy runs on big models drain it faster. Use Fast mode for small edits.

### 3.6 Rules, workflows, skills (customising the agent)
| Thing | What it is | Where it lives |
|---|---|---|
| Rules | Standing instructions the agent always follows | Global: `~/.gemini/GEMINI.md`. Workspace: `.agents/rules/` (older versions: `.agent/rules/`) |
| Workflows | Saved step-by-step prompts you trigger on demand with `/name` | Workspace: `.agents/workflows/` (older: `.agent/workflows/`) |
| Skills | Folders with a `SKILL.md` the agent loads when relevant | Workspace: `.agents/skills/<name>/SKILL.md` |
| MCP servers | Connections to outside tools | Agent panel `...` menu > MCP Servers > View raw config (`mcp_config.json`) |
Each Rules or Workflows file is limited to **12,000 characters**. Section 7 gives you a ready rules file. Your version may also offer slash commands such as `/plan` and `/browser`; use them if they appear.

### 3.7 Extensions and tools to install
Antigravity runs most VS Code extensions. Search by name in the Extensions panel; if one is missing from its marketplace, skip it or install a `.vsix` file manually.

| Install | Why |
|---|---|
| Python + Pylance | Language support |
| Jupyter | Run notebooks |
| Ruff | Linting and formatting for Python |
| YAML | Edit `config.yaml` safely |
| Docker | Build and inspect the API image |
| ESLint, Prettier, Tailwind CSS IntelliSense | Dashboard code |
| Markdown Preview Mermaid Support | View architecture diagrams |
| GitLens (optional) | Git history |

**Plugins/MCP that are actually worth it:** you do not need any MCP server to start. The agent can run `gh` (GitHub CLI), `git`, `docker`, `vercel` and `hf` in the terminal. If you want one extra, add a documentation lookup server from the MCP store (for example a library-docs one) so the agent reads current PyTorch/FastAPI docs instead of guessing. Skip anything that asks for broad account access.

### 3.8 A good daily workflow
1. Create a branch for the phase (`git switch -c phase-1-manifest`).
2. Agent Manager > new task > paste the phase text > mode **Planning**.
3. Read the plan; comment; approve.
4. Let it build. When it stops, read the walkthrough artifact.
5. In the Editor, run `make test` and `make lint` yourself.
6. Commit, push, open a pull request, merge when CI is green.
7. Start the next phase in a **new conversation** (long chats get confused; the rules file keeps the context).

### 3.9 Common beginner mistakes
- Pasting the entire project prompt and saying "build everything": it will make a mess. Run one phase at a time.
- Approving the plan without reading it.
- Letting the agent download datasets automatically: do the downloads yourself (Section 6) and let the agent only read from `data/raw`.
- Committing data or weights to GitHub.
- Trusting numbers the agent writes in the report. Every number must come from a results file your code produced.

---

## 4. Antigravity vs VS Code

Antigravity is a VS Code fork, so the editor feels the same. The question is really: agent-first workflow, or classic editor with an assistant?

| Criterion | Antigravity (you have Pro) | Plain VS Code (+ Copilot or another assistant) |
|---|---|---|
| Multi-step builds from a prompt | Strong: plans, parallel agents, reviewable artifacts | Good with agent modes, less polished orchestration |
| Testing a web dashboard | Built-in browser agent can open and check the app | Manual, or add Playwright yourself |
| Stability and maturity | Newer; rough edges and limits reported | Very stable |
| Extensions | Most VS Code extensions work; marketplace can differ | Full Microsoft marketplace |
| Cost | Already paid (Pro); watch AI credits | Extra subscription for a comparable assistant |
| Notebooks and remote dev | Works, same engine | Works, plus the widest ecosystem |
| Learning curve | Extra concepts (Manager, artifacts, rules) | Familiar |

**Recommendation:** use **Antigravity as your main tool** since you already have Pro and this project is multi-phase. Keep plain VS Code installed as a fallback for anything that misbehaves (it opens the same folder and the same Git repo). You do not need to choose; the repo is the source of truth.

---

## 5. GitHub setup, step by step

~~~bash
# 1. Create the project and repo
mkdir guardian-eye && cd guardian-eye
git init -b main
gh auth login                      # choose GitHub.com, HTTPS, login via browser
gh repo create guardian-eye --private --source=. --remote=origin

# 2. First files
printf "data/raw/\ndata/processed/\nresults/runs/\n*.pt\n*.pth\n*.ckpt\n.env\n__pycache__/\n.venv/\nnode_modules/\n.next/\nstorage/\n" > .gitignore
echo "# GuardianEye" > README.md
git add . && git commit -m "chore: initial commit"
git push -u origin main
~~~

Rules for the repo:
- **Never commit** datasets, model weights, `.env`, or API keys. Weights go to the Hugging Face Hub; datasets are downloaded by each person with the steps in Section 6.
- Work on branches: `phase-0-scaffold`, `phase-1-manifest`, and so on. Open a pull request per phase: `gh pr create --fill`.
- Add teammates: `gh repo` settings > Collaborators (or `gh api -X PUT repos/<you>/guardian-eye/collaborators/<username>`).
- Protect `main` (require the CI check) once CI exists.
- Keep it **private** until submission; make it public (or add the examiner) only after you have checked dataset licences and removed anything sensitive.

CI file (the agent will create it in Phase 0): `.github/workflows/ci.yml` running `ruff check`, `pytest -q` on Python 3.11 for every push and pull request.

Releases: tag when a phase is done (`git tag v0.1-phase1 && git push --tags`), and tag `v1.0-submission` for the final report.

---

## 6. Data download, step by step (the five datasets)

Create the folders:
~~~bash
mkdir -p data/raw/{birdsai,cct,serengeti,belize,vietnam,esc50} data/processed results
~~~
Do the downloads yourself. The pipeline then reads only from `data/raw/`.

### 6.1 BIRDSAI (thermal aerial, image branch)
1. Open https://lila.science and find **BIRDSAI**.
2. Download the **real** training and test archives (frames or videos plus annotation files). Start with real data; the synthetic part is optional.
3. Unzip into `data/raw/birdsai/`.
4. Annotations are MOT-style rows; per your CA4 mapping, class code **1 = human**, **0 = animal**, and a frame with no box is **empty**. If a frame has both, store it as human and write the manifest note `human_and_animal_boxes`.
5. If you get videos, decode them to frames with FFmpeg (1 frame per second is enough):
~~~bash
mkdir -p data/raw/birdsai/frames
ffmpeg -i video.mp4 -vf fps=1 data/raw/birdsai/frames/video_%05d.jpg
~~~
View label: `aerial_thermal`.

### 6.2 Caltech Camera Traps (ground camera traps)
1. On https://lila.science open **Caltech Camera Traps**. Use the benchmark subset (CCT20); its images are already downsized to 1024 pixels on a side, which saves a lot of space.
2. Download the metadata JSON (COCO Camera Traps format) first; it is small.
3. Download only a sample of images (a few thousand, balanced across empty, animal, any human or vehicle), using the base URL listed on the dataset page for your preferred cloud (GCP, Azure or AWS). The LILA collection is also mirrored as a public bucket on Source Cooperative, which you can browse without an account:
~~~bash
pip install awscli
aws s3 ls --no-sign-request s3://us-west-2.opendata.source.coop/agentmorris/lila-wildlife/
~~~
4. Arrange images in class folders for the pipeline: `data/raw/cct/{human,vehicle,animal,empty}/...`. Many CCT images are animals or empty; humans and vehicles are rare, so keep every one you find.
View label: `camera_trap`.

### 6.3 Snapshot Serengeti (ground camera traps)
1. On https://lila.science open **Snapshot Serengeti**. The full archive is terabyte-scale: **do not download it all.**
2. Download the metadata files, pick a sample (for example 5,000 to 10,000 images, with plenty of blank frames), and download only those image files.
3. Arrange as `data/raw/serengeti/{human,vehicle,animal,empty}/...` (humans are scarce here; some seasons contain a few).
View label: `camera_trap`.

### 6.4 Belize gunshot set (primary audio)
1. Open DOI **10.17632/x48cwz364j** (Mendeley Data: "Tropical forest gunshot classification training audio dataset", version 3).
2. Download the dataset (sign in to Mendeley if asked) and unzip into `data/raw/belize/`.
3. Structure: `Training data/` and `Validation data/`, each with `Gunshot/` and `Background/`. Files are WAV, 4.09 seconds, 8 kHz. The collection has 749 gunshot files and over 35,000 background files.
4. The authors split by time (first 80% of recording period vs last 20%) to avoid leakage. Your CA4 plan re-splits by seed after building the manifest; keep the original folder name in the manifest as a column so you can still compare with the authors' split.

### 6.5 Vietnam Chu Mom Ray set (second site)
1. Open DOI **10.5281/zenodo.13893977** (Zenodo).
2. Download the files listed on the record and unzip into `data/raw/vietnam/`.
3. Inspect what is inside: the BDA report says audio and spectrogram images are provided. Your audio pipeline needs **audio files**; if the record has labelled audio, use those, and ignore the spectrogram images. If any part is only spectrograms, write that in the limitations section instead of guessing.
4. Resample to the same 8 kHz and 4.0 s windows as Belize.
Useful fact for your report: Vu et al. report that models trained on their data generalised poorly to the Belize set (F1 about 0.52), which supports your cross-site test.

### 6.6 ESC-50 (thunder, wind, rain: noise tag only)
~~~bash
cd data/raw/esc50
wget https://github.com/karolpiczak/ESC-50/archive/master.zip
unzip master.zip
~~~
Keep only the `thunderstorm`, `wind` and `rain` categories (use `meta/esc50.csv`). Tag them `noise_tag = thunder_wind`. They are **never relabelled as gunshots**, and Belize/Vietnam background is never relabelled as thunder.

### 6.7 Verify before you train
~~~bash
du -sh data/raw/*
find data/raw -type f | wc -l
python scripts/verify_data.py        # created in Phase 1: counts files per dataset and prints PASS/FAIL
~~~
If a dataset folder is empty, the report cells stay **NOT YET RUN**. That is the correct behaviour; it is exactly what happened in CA4.

### 6.8 Class and tag mapping
| Branch | Label | Sources |
|---|---|---|
| Image | human | BIRDSAI code 1; CCT/Serengeti human folders |
| Image | vehicle | CCT/Serengeti vehicle folders only (never invented) |
| Image | animal | BIRDSAI code 0; CCT/Serengeti animal folders |
| Image | empty | Frames with no box or empty folders |
| Audio | gunshot | Belize Gunshot; Vietnam gunshot |
| Audio | background | Belize Background; Vietnam background; ESC-50 thunder/wind/rain (with noise tag) |

---

## 7. Rules file for Antigravity

Create `.agents/rules/guardian.md` (or `.agent/rules/` if that is what your version shows). Keep it under 12,000 characters. Paste:

~~~markdown
# GuardianEye project rules
- Project: multimodal wildlife poaching detection (image branch + audio branch + fusion). Video is decoded to frames and handled by the image branch. There is no third modality.
- Python 3.11, PyTorch, torchaudio, scikit-learn, pandas, pyyaml, pytest, ruff. FastAPI for the API. Ask before adding any other dependency.
- All hyperparameters live in config.yaml. No hard-coded paths or magic numbers.
- Work one phase at a time. Write a plan first. Stop after each phase and summarise what changed, how to run it, and which tests passed.
- NEVER invent results. Every number in docs or reports must come from a file written by the code under results/. If a dataset or result is missing, write "NOT YET RUN" with the reason, never an estimate.
- Splits are assigned BEFORE pairing. Pairs are built only inside one split. Fixed seeds. Never split audio or images randomly across sites if a site key exists.
- Vehicle images are never synthesised. If the vehicle pool is empty, log built count 0.
- Thunder/wind clips are accepted only when tagged thunder_wind (from ESC-50). Never relabel background as thunder, never relabel thunder as gunshot.
- Do not download datasets. Read only from data/raw. If missing, fail with a clear message naming the dataset and the section of the guide.
- Never commit data, weights, .env or secrets. Weights go to the Hugging Face Hub.
- Type hints, docstrings on public functions, logging not print, small functions, tests for every module.
- Do not try to identify real people in images. "human" is only a class label.
- Ask me before running any command that deletes files or installs system packages.
~~~

---

## 8. MASTER PROMPT (paste into Antigravity, mode: Planning)

````text
ROLE
You are a senior ML engineer. Build "GuardianEye", an academic proof of concept that fuses an image branch and an audio branch to flag possible wildlife poaching. Follow the repo rules file. Work in the phases below, one at a time. For each phase: write a plan, implement, run tests, then STOP and summarise. Do not start the next phase until I say "proceed".

SCOPE (fixed)
- Two modalities only. Video is decoded to frames (1 fps default) and goes into the image branch. No temporal model.
- Image classes: human, vehicle, animal, empty. Image "view" column: aerial_thermal (BIRDSAI) or camera_trap (Caltech Camera Traps, Snapshot Serengeti).
- Audio classes: background, gunshot. Extra column noise_tag: thunder_wind (from ESC-50 thunderstorm/wind/rain only) or not_thunder_wind. Thunder/wind clips are background, held out of audio training, but still scored so fusion can use them.
- Suspicious = 1 only when image label in {human, vehicle} AND audio label = gunshot. Risk tiers: HIGH = both strong indicators; MEDIUM = exactly one strong indicator (human or vehicle with background; animal with gunshot); LOW = neither (animal+background, empty+background, empty+thunder_wind).
- Public data has no real poaching incidents and no time-aligned audio+image pairs, so fusion is evaluated on SYNTHETIC pairs. State this in every report and in the API docs.

DATA (already downloaded by me; paths in config.yaml; do not download anything)
data/raw/birdsai (MOT-style annotations: class code 1 = human, 0 = animal; frame with no box = empty; both = human with note "human_and_animal_boxes")
data/raw/cct and data/raw/serengeti (class folders human/vehicle/animal/empty)
data/raw/belize (Training data / Validation data, each with Gunshot and Background; WAV, 4.09 s, 8 kHz)
data/raw/vietnam (Chu Mom Ray recordings; use audio files only)
data/raw/esc50 (thunderstorm, wind, rain only, as noise tag)

CONFIG (config.yaml, all values read from here)
seeds: [27, 2701, 26009]; split train/val/test 0.7/0.15/0.15
image: classes [human, vehicle, animal, empty]; size 128; epochs 8; lr 0.001; embedding 128; augment random crop, brightness scale, Gaussian blur (training split only)
audio: classes [background, gunshot]; sample_rate 8000; clip_seconds 4.0; n_mels 64; n_fft 512; hop 128; n_mfcc 20; epochs 8; lr 0.001; 2 time masks width 16; 2 frequency masks width 8 (unimodal training only)
pairing: max_pairs_per_scenario 200; generator seed = experiment seed + 1 (train), +2 (val), +3 (test)
early_fusion: image_pool 8; dropout_prob 0.15; logistic C 1.0
late_fusion: weight_step 0.05; threshold_step 0.05; conservative_tau_delta 0.15; tau_cap 0.95
robustness: noise_snr_db 5; blur_radius 2.0; brightness_factor 0.3
bootstrap_resamples 1000
latency: events 10; repeats 3; target_seconds 5

REPO LAYOUT
guardian-eye/
  README.md  pyproject.toml  Makefile  config.yaml  .env.example  Dockerfile
  .github/workflows/ci.yml
  scripts/ (verify_data.py, build_manifest.py, run_all.py)
  src/
    manifest.py splits.py decisions.py models.py
    image/ (dataset.py, train.py, evaluate.py)
    audio/ (features.py, dataset.py, train.py, evaluate.py)
    pairing.py
    fusion/ (early.py, intermediate.py, late.py, baselines.py, calibrate.py)
    evaluation/ (metrics.py, robustness.py, latency.py, bootstrap.py, failures.py)
    bda/ (mapper_class.py, reducer_sum.py, mapper_scenario.py, mapper_metric.py, reducer_metric.py, stream.py, spark_jobs.py, schema.hql)
    api/ (main.py, routes.py, schemas.py, storage.py)
  tests/  results/  docs/  notebooks/ (thin Colab wrappers)

PHASE 0: scaffold
Create the layout, pyproject, Makefile (setup, data-check, manifest, train-image, train-audio, pairs, fuse, evaluate, bda, serve, test, lint), config.yaml with the values above, a config loader (pydantic), logging, and the CI workflow (ruff + pytest on Python 3.11). Implement src/decisions.py: suspicious(image_label, audio_label) and risk(image_label, audio_label) exactly as in SCOPE, with unit tests for all seven scenarios below.
Seven scenarios (image, audio, noise_tag -> suspicious, risk): human_gunshot (human, gunshot, any -> 1, high); vehicle_gunshot (vehicle, gunshot, any -> 1, high); animal_gunshot (animal, gunshot, any -> 0, medium); human_background (human, background, not_thunder_wind -> 0, medium); animal_background (animal, background, not_thunder_wind -> 0, low); empty_background (empty, background, not_thunder_wind -> 0, low); empty_thunder (empty, background, thunder_wind -> 0, low). Vehicle with ordinary background is also medium.
Acceptance: make test and make lint pass; CI green.

PHASE 1: manifest and splits
build_manifest.py scans data/raw and writes one manifest (path, modality, source, label, view, site_or_device if known, noise_tag, original_split, split, notes). Write results/runs/seed_<n>/manifest/manifest_status.json with status, image count, audio count, rows, unmapped count. If no files are found, status = "NOT YET RUN" with the reason; do not crash and do not fabricate rows.
Stratify splits by modality, source, label and noise_tag using the seed and fractions. Provide scripts/verify_data.py (counts and opens 20 random files per dataset; PASS/FAIL table).
Acceptance: tests with a tiny fixture folder; no row appears in two splits; empty-data case produces the NOT YET RUN status.

PHASE 2: unimodal models
ImageEncoder: four conv blocks, linear head over 4 classes, 128-d penultimate embedding. Class weights = inverse frequency inside cross-entropy; a class with no training rows gets weight 0. Audio: resample, crop/pad to 4.0 s, log-mel with torchaudio, AudioEncoder (conv net), class-weighted cross-entropy; apply time/frequency masking in unimodal training only. Train the in-domain audio model on pooled Belize+Vietnam train split excluding thunder_wind clips. Also run cross-site checks: train Belize / test Vietnam and the reverse (separate checkpoints, not used for fusion embeddings).
Save logits, probabilities and penultimate embeddings for EVERY scanned file (including held-out thunder clips) to results/runs/seed_<n>/outputs/.
Metrics: image per-class precision/recall/F1 and per-view (aerial_thermal vs camera_trap) reported separately; audio precision, recall, F1, FPR = FP/(FP+TN) with gunshot as positive, ROC-AUC. A metric with a zero denominator is left undefined, never stored as 0. A view or class with no test files is written NOT YET RUN.
Optional flag (off by default): swap in a pretrained backbone (ResNet18 for images, PANNs/YAMNet embeddings for audio) for comparison.
Acceptance: smoke training on fixture data runs in under 5 minutes on CPU; outputs saved; NOT YET RUN handling tested.

PHASE 3: synthetic pairs
pairing.py builds scenarios only inside a split (pair_split keeps rows whose split matches). Cap each scenario at max_pairs_per_scenario. Files may be reused inside a split, but the exact (image, audio, scenario) triple is not repeated. If the vehicle pool is empty, vehicle_gunshot is written with built count 0; never invent vehicle images. Compute suspicious and risk with decisions.py. Write results/protocol/pairing_self_check.json (not_a_dataset_evaluation true, cross_split_pairs 0, per-split scenario counts) from a tiny in-memory fixture.
Acceptance: test that no pair crosses splits; test the empty vehicle pool; deterministic with seeds.

PHASE 4: fusion
Early: resize and average each image to an image_pool x image_pool spatial pool; audio contributes mean and std of log-mel bands and of the MFCC sequence; concatenate with image_present and audio_present bits; StandardScaler fit on clean train pairs; logistic regression (balanced class weights) fit on clean pairs plus a modality-dropout copy; missing modality = zeros with presence bit 0.
Intermediate: four heads on the same pairs: intermediate_joint_concat (MLP on two embeddings + presence bits, fine-tunes both encoders), intermediate_joint_gated (project each embedding, learned vector for missing modality, mask missing with a large negative score, softmax over two tokens, fine-tunes encoders), intermediate_frozen_concat and intermediate_frozen_gated (train only the joint layer on saved embeddings). Choose checkpoints by validation binary cross-entropy; choose the threshold by grid search on validation.
Late: temperature scaling per branch on validation logits (temperature = exp(parameter), positive); if validation lacks both classes, record NOT YET RUN. Grid search audio weight and visual weight (sum to 1, step 0.05) and a threshold, maximise validation F1 then prefer lower FPR. Rule head: HIGH if P(human or vehicle) and P(gunshot) both clear validation thresholds; MEDIUM if exactly one; LOW otherwise; binary prediction positive only on HIGH. Missing modality: weighted head uses the remaining probability with threshold raised by conservative_tau_delta (capped at tau_cap); the rule head cannot reach HIGH with one sensor, so its binary prediction stays negative.
Baselines on the same pairs: image-only (calibrated P(human)+P(vehicle)) and audio-only (calibrated P(gunshot)); if their modality is dropped they predict negative and never borrow the other sensor.
Acceptance: unit tests for each missing-modality policy; each method returns scores for clean and dropped conditions.

PHASE 5: evaluation
Score every method on: clean, audio dropped, image dropped, waveform noise at 5 dB SNR, image blur radius 2.0, brightness factor 0.3. Do NOT refit thresholds or temperatures on corrupted inputs. CPU latency: time file read + forward pass for 10 test events x 3 repeats after weights are loaded; write within_target (5 s) to latency.csv. Bootstrap 95% intervals (1000 resamples of test pairs within one seed; metrics with undefined denominators are skipped). Across seeds, a reducer writes count, mean and sample standard deviation (divisor n-1; empty std cell if fewer than 2 values). Write fusion_metrics.csv, failures.csv, per-scenario results, and figures. The key table: fused vs image-only vs audio-only, with FPR beside F1.
Acceptance: make evaluate produces all files; any missing input produces NOT YET RUN cells, not zeros.

PHASE 6: BDA module (src/bda)
Hadoop-streaming style jobs: class-count job (modality, source, label), scenario job (split, scenario, suspicious, risk), metric job (method, condition, metric, value) with a reducer that emits n, mean, sample std. stream.py uses the Hadoop streaming jar if HADOOP_HOME and hadoop are present; otherwise it sorts mapper output locally and runs the same reducer; log the engine name (e.g. python_streaming_local). Provide spark_jobs.py (manifest ingest to Parquet, class weights w_c = N/(k*n_c), audio log-mel features via pandas UDF) that runs only if pyspark imports, otherwise report "Spark: NOT YET RUN". Provide schema.hql (Hive DDL) but do not execute it. Include fixtures and write results/bda/self_check/*.csv.
Acceptance: self-check outputs match the fixture counts; engine name logged; Spark absence handled.

PHASE 7: API + dashboard + deploy (see the separate prompt)
FastAPI routes: GET /api/health; POST /api/analyze (optional image file(s) or video file decoded to frames, optional audio file, sensor_id, timestamp) returning image label+probs, audio label+probs, method scores, risk tier, and a "synthetic evidence only" disclaimer; GET /api/results (reads results/*.csv; returns NOT YET RUN when absent); GET /api/demo. Load weights from the Hugging Face Hub at startup via env var; start in demo mode if weights are missing. Dockerfile exposes port 7860.
Acceptance: TestClient tests for each route; docker build succeeds.

GLOBAL RULES: follow the rules file. No invented metrics. Ask before adding dependencies. Start with Phase 0 now: show the file tree, then implement.
````

---

## 9. DASHBOARD + DEPLOY PROMPT (second Antigravity task, in `dashboard/`)

````text
Build a Next.js 14 (App Router, TypeScript, Tailwind, shadcn/ui, Recharts) dashboard "GuardianEye Console" in the folder dashboard/. It calls the FastAPI backend at NEXT_PUBLIC_API_BASE_URL (default http://localhost:8000) and works in DEMO MODE with built-in mock data if the API is unreachable (show a "Demo data" badge).

PAGES
1. / Analyze: upload an image (or a short video) and/or an audio clip, enter a sensor id; call POST /api/analyze; show image label and probabilities, audio label and probabilities, each fusion method's score (early, intermediate, late, image-only, audio-only), the risk tier (HIGH red, MEDIUM amber, LOW blue, with icon and text, not colour alone), and a permanent banner: "Proof of concept evaluated on synthetic pairs. Not a confirmed poaching alert."
2. /results: read GET /api/results. Show the comparison table (fused vs image-only vs audio-only: precision, recall, F1, FPR), robustness table (clean, audio dropped, image dropped, noise, blur, low light), latency table, and bar charts. Cells the backend returns as "NOT YET RUN" must render exactly as "NOT YET RUN" with the reason on hover; never show zeros or placeholders as results.
3. /data: dataset cards (BIRDSAI, Caltech Camera Traps, Snapshot Serengeti, Belize, Vietnam, ESC-50) with source, licence note, role, and counts from the manifest status.
4. /architecture: the pipeline diagram (image branch, audio branch, fusion, risk) and the big-data layer (HDFS, Kafka, Spark, Hive) as a static diagram.
5. /about: limitations (synthetic pairs, rule-derived label, no field validation).

QUALITY: mobile responsive, accessible, loading skeletons, empty and error states, dark mode, no secrets in the client, .env.example, README with local run and Vercel deploy steps.
````

---

## 10. Global deployment, step by step

Architecture: **Vercel** (dashboard, served from a global CDN) calls a **FastAPI container** (Hugging Face Space or Cloud Run). Weights live in a Hugging Face model repo.

1. **Prepare**: all tests pass, `main` is green on GitHub, no secrets in the repo.
2. **Upload weights** (only the small final checkpoints, never data):
~~~bash
pip install huggingface_hub && hf auth login
hf repo create guardian-eye-weights --type model
hf upload <you>/guardian-eye-weights results/final_weights .
~~~
3. **Deploy the API on a Hugging Face Space** (simplest, free CPU tier, public URL): create a new Space, choose **Docker**, connect or push your repo, make sure the Dockerfile exposes port **7860**, add Space secrets `HF_WEIGHTS_REPO` and `ALLOWED_ORIGINS`. Free Spaces can go to sleep after inactivity, so the first request may be slow.
   - Alternative for steadier production use: **Google Cloud Run** (`gcloud run deploy --source .`), which scales to zero and gives an HTTPS URL; or **Fly.io** if you later want multiple regions.
4. **Deploy the dashboard on Vercel**: `vercel login`, then import the GitHub repo in the Vercel dashboard, set **Root Directory = dashboard**, add env var `NEXT_PUBLIC_API_BASE_URL` = your Space or Cloud Run URL, deploy. Every push to `main` redeploys.
5. **CORS**: set `ALLOWED_ORIGINS` on the API to your Vercel domain (and `http://localhost:3000` for dev).
6. **Hardening**: limit upload size (for example 10 MB image, 5 MB audio), add a simple rate limit, accept only image/audio MIME types, and do not log uploaded files. Do not expose dataset files, real sensor coordinates, or anything from restricted data.
7. **Smoke test the live site**: `/api/health` returns OK; upload a sample image + clip; open `/results` and confirm NOT YET RUN cells appear exactly where results are missing.
8. **Optional**: custom domain on Vercel; put the API behind Cloudflare for caching and DDoS protection; uptime check on `/api/health`.
9. **Tag the release**: `git tag v1.0-submission && git push --tags`.

"Global" here means the dashboard is served worldwide from Vercel's CDN and the API is reachable over HTTPS from anywhere. A single API region is fine for a college project; multi-region is only needed for real production traffic.

---

## 11. Consistency flags in your reports (fix before submitting)

1. **Detector mismatch.** The BDA report describes a YOLOv8-class image detector; the CA4 report implements a four-block CNN classifier. Align the wording (for example: "CA4 uses a CNN classifier; a YOLO-style detector is future work").
2. **Names and PRN.** The reports list "Akhil Chilvkula" and "Sukhada Dhingra" with PRN 23070126505. Earlier in this chat the details were "Akhil Chivukula" and "Sukhada Dinghra" with PRN 24070126506. Confirm the correct spellings and numbers and use the same ones in every document.
3. **Datasets in earlier assignments.** Assignments 1 and 2 (Word docs) mention VisDrone, HIT-UAV, UrbanSound8K, FSD50K and AudioSet. Your CA4 scope uses BIRDSAI, CCT, Serengeti, Belize and Vietnam. Update Assignment 2 if it will be submitted alongside CA4.
4. **Modalities.** Assignments 1 and 2 describe three modalities (video, image, audio). CA2 onward uses two.
5. **Status honesty.** CA4 says dataset evaluation is NOT YET RUN. Once data is on disk and the pipeline runs, regenerate every table; do not edit numbers by hand.
6. **Dataset sizes.** Belize: 749 gunshot and over 35,000 background WAVs, 4.09 s at 8 kHz (matches your 2.1% imbalance calculation). Re-check Caltech and Serengeti sizes against the exact release you download.
