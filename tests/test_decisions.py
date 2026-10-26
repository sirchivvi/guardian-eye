"""Unit tests for src/decisions.py.

Covers:
- All 7 required scenario pairs (image × audio → suspicious, risk)
- vehicle + background  → suspicious=0, risk=MEDIUM  (explicit requirement)
- All four image labels paired with each audio label
- Invalid image labels
- Invalid audio labels
- Invalid both labels
"""

import pytest

from src.decisions import risk, suspicious

# ---------------------------------------------------------------------------
# Required scenario table
# ---------------------------------------------------------------------------
#
# image     | audio      | suspicious | risk
# ----------+------------+------------+--------
# human     | gunshot    | 1          | HIGH
# vehicle   | gunshot    | 1          | HIGH
# animal    | gunshot    | 0          | MEDIUM
# human     | background | 0          | MEDIUM
# animal    | background | 0          | LOW
# empty     | background | 0          | LOW
# empty     | background | 0          | LOW   (thunder_wind noise_tag — same API)

REQUIRED_SCENARIOS: list[tuple[str, str, int, str]] = [
    ("human", "gunshot", 1, "HIGH"),
    ("vehicle", "gunshot", 1, "HIGH"),
    ("animal", "gunshot", 0, "MEDIUM"),
    ("human", "background", 0, "MEDIUM"),
    ("animal", "background", 0, "LOW"),
    ("empty", "background", 0, "LOW"),
    # Scenario 7: empty image paired with a thunder_wind ESC-50 clip.
    # The audio label remains "background" — the noise_tag is upstream metadata
    # that does NOT change the function signature.
    ("empty", "background", 0, "LOW"),
]


@pytest.mark.parametrize(
    "image_label, audio_label, expected_suspicious, expected_risk",
    REQUIRED_SCENARIOS,
    ids=[
        "human_gunshot",
        "vehicle_gunshot",
        "animal_gunshot",
        "human_background",
        "animal_background",
        "empty_background",
        "empty_background_thunder_wind",
    ],
)
def test_required_scenarios(
    image_label: str,
    audio_label: str,
    expected_suspicious: int,
    expected_risk: str,
) -> None:
    """Verify all 7 required scenario pairs produce the correct outputs."""
    assert suspicious(image_label, audio_label) == expected_suspicious
    assert risk(image_label, audio_label) == expected_risk


# ---------------------------------------------------------------------------
# Explicit additional requirements
# ---------------------------------------------------------------------------


def test_vehicle_background_is_medium_not_high() -> None:
    """vehicle + background must be MEDIUM, not HIGH (build guide requirement)."""
    assert suspicious("vehicle", "background") == 0
    assert risk("vehicle", "background") == "MEDIUM"


def test_vehicle_background_not_suspicious() -> None:
    """vehicle without gunshot is not suspicious."""
    assert suspicious("vehicle", "background") == 0


def test_high_implies_suspicious() -> None:
    """Every HIGH risk pair must also be suspicious."""
    high_pairs = [
        ("human", "gunshot"),
        ("vehicle", "gunshot"),
    ]
    for img, aud in high_pairs:
        assert suspicious(img, aud) == 1, f"Expected suspicious for ({img}, {aud})"
        assert risk(img, aud) == "HIGH"


def test_medium_not_suspicious() -> None:
    """MEDIUM risk pairs are never suspicious."""
    medium_pairs = [
        ("animal", "gunshot"),
        ("human", "background"),
        ("vehicle", "background"),
    ]
    for img, aud in medium_pairs:
        assert suspicious(img, aud) == 0, f"Expected not suspicious for ({img}, {aud})"
        assert risk(img, aud) == "MEDIUM"


def test_low_not_suspicious() -> None:
    """LOW risk pairs are never suspicious."""
    low_pairs = [
        ("animal", "background"),
        ("empty", "background"),
    ]
    for img, aud in low_pairs:
        assert suspicious(img, aud) == 0, f"Expected not suspicious for ({img}, {aud})"
        assert risk(img, aud) == "LOW"


# ---------------------------------------------------------------------------
# Return type checks
# ---------------------------------------------------------------------------


def test_suspicious_returns_int() -> None:
    """suspicious() must return a plain int (not bool)."""
    result = suspicious("human", "gunshot")
    assert isinstance(result, int)
    assert type(result) is int


def test_risk_returns_str() -> None:
    """risk() must return a plain str."""
    result = risk("empty", "background")
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Invalid label handling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "image_label",
    ["dog", "Human", "HUMAN", "", "unknown", "none", "poacher"],
)
def test_invalid_image_label_suspicious(image_label: str) -> None:
    """suspicious() raises ValueError for unknown image labels."""
    with pytest.raises(ValueError, match="image_label"):
        suspicious(image_label, "background")


@pytest.mark.parametrize(
    "image_label",
    ["dog", "Human", "HUMAN", "", "unknown", "none", "poacher"],
)
def test_invalid_image_label_risk(image_label: str) -> None:
    """risk() raises ValueError for unknown image labels."""
    with pytest.raises(ValueError, match="image_label"):
        risk(image_label, "background")


@pytest.mark.parametrize(
    "audio_label",
    ["shot", "Gunshot", "GUNSHOT", "", "noise", "thunder_wind", "wind"],
)
def test_invalid_audio_label_suspicious(audio_label: str) -> None:
    """suspicious() raises ValueError for unknown audio labels."""
    with pytest.raises(ValueError, match="audio_label"):
        suspicious("human", audio_label)


@pytest.mark.parametrize(
    "audio_label",
    ["shot", "Gunshot", "GUNSHOT", "", "noise", "thunder_wind", "wind"],
)
def test_invalid_audio_label_risk(audio_label: str) -> None:
    """risk() raises ValueError for unknown audio labels."""
    with pytest.raises(ValueError, match="audio_label"):
        risk("human", audio_label)


def test_both_invalid_labels_raises() -> None:
    """When both labels are invalid, ValueError is raised (for image first)."""
    with pytest.raises(ValueError):
        suspicious("dog", "shot")

    with pytest.raises(ValueError):
        risk("dog", "shot")


# ---------------------------------------------------------------------------
# Exhaustive label coverage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("image_label", ["human", "vehicle", "animal", "empty"])
@pytest.mark.parametrize("audio_label", ["gunshot", "background"])
def test_all_valid_combinations_do_not_raise(
    image_label: str, audio_label: str
) -> None:
    """Every valid (image, audio) pair must return without exception."""
    suspicious(image_label, audio_label)
    risk(image_label, audio_label)


@pytest.mark.parametrize("image_label", ["human", "vehicle", "animal", "empty"])
@pytest.mark.parametrize("audio_label", ["gunshot", "background"])
def test_all_valid_risk_values_are_canonical(
    image_label: str, audio_label: str
) -> None:
    """risk() must always return exactly HIGH, MEDIUM, or LOW."""
    result = risk(image_label, audio_label)
    assert result in {"HIGH", "MEDIUM", "LOW"}


@pytest.mark.parametrize("image_label", ["human", "vehicle", "animal", "empty"])
@pytest.mark.parametrize("audio_label", ["gunshot", "background"])
def test_all_valid_suspicious_values_are_binary(
    image_label: str, audio_label: str
) -> None:
    """suspicious() must always return 0 or 1."""
    result = suspicious(image_label, audio_label)
    assert result in {0, 1}
