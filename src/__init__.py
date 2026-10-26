"""
GuardianEye — anti-poaching multi-modal risk-assessment system.

Package structure
-----------------
src/
├── config.py         — Pydantic configuration loader
├── decisions.py      — Risk decision rules (image + audio → HIGH/MEDIUM/LOW)
├── logging_config.py — Structured logging and device detection
├── checkpoint.py     — Shared checkpoint save/load/discovery utilities
├── image/            — Image branch (Phase 1+)
├── audio/            — Audio branch (Phase 1+)
├── fusion/           — Late-fusion module (Phase 2+)
├── evaluation/       — Evaluation harness (Phase 2+)
├── bda/              — Bootstrap difference analysis (Phase 3+)
└── api/              — REST API (Phase 4+)
"""
