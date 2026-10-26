"""
GuardianEye risk decision rules.

This module contains two pure functions that map (image_label, audio_label)
pairs to a binary suspicious flag and a categorical risk level.

Labels
------
Valid image labels  : ``"human"``, ``"vehicle"``, ``"animal"``, ``"empty"``
Valid audio labels  : ``"gunshot"``, ``"background"``

Risk rules
----------
HIGH   : image ∈ {human, vehicle}  AND  audio == gunshot  → suspicious = 1
MEDIUM : exactly one strong indicator present:
         - image ∈ {human, vehicle}  AND  audio == background
         - image == animal           AND  audio == gunshot
         → suspicious = 0
LOW    : image ∈ {animal, empty}    AND  audio == background
         → suspicious = 0
         (Includes ESC-50 thunderstorm/wind/rain clips that are labelled as
         *background* with a ``noise_tag`` of ``"thunder_wind"``.  Because
         these clips carry no gunshot audio, they are correctly classified LOW.
         The ``noise_tag`` metadata is **not** a positional argument here; it
         is handled upstream in the data pipeline and does not change the
         risk output.)

Vehicle + background policy
----------------------------
A vehicle without gunshot audio is classified MEDIUM, not HIGH.  The presence
of a vehicle alone is a single strong indicator and insufficient evidence for
an immediate alert.

Noise-tag note
--------------
ESC-50 clips tagged as ``thunder_wind`` are kept as ``"background"`` throughout
the pipeline (they are *never* relabelled as ``"gunshot"``).  When paired with
an ``"empty"`` image they produce a LOW risk score, exactly matching the last
row of the required scenario table.  Tests for this scenario pass
``image_label="empty"`` and ``audio_label="background"``; no third argument is
needed.
"""

from __future__ import annotations

_VALID_IMAGE_LABELS: frozenset[str] = frozenset({"human", "vehicle", "animal", "empty"})
_VALID_AUDIO_LABELS: frozenset[str] = frozenset({"gunshot", "background"})

_STRONG_IMAGE: frozenset[str] = frozenset({"human", "vehicle"})


def _validate(image_label: str, audio_label: str) -> None:
    """
    Raise ``ValueError`` when either label is unknown.

    Parameters
    ----------
    image_label:
        Predicted class from the image branch.
    audio_label:
        Predicted class from the audio branch.

    Raises
    ------
    ValueError
        If ``image_label`` or ``audio_label`` is not a recognised value.

    """
    if image_label not in _VALID_IMAGE_LABELS:
        raise ValueError(
            f"Unknown image_label {image_label!r}. "
            f"Must be one of {sorted(_VALID_IMAGE_LABELS)}."
        )
    if audio_label not in _VALID_AUDIO_LABELS:
        raise ValueError(
            f"Unknown audio_label {audio_label!r}. "
            f"Must be one of {sorted(_VALID_AUDIO_LABELS)}."
        )


def suspicious(image_label: str, audio_label: str) -> int:
    """
    Return 1 when the scene is classified as suspicious, else 0.

    A scene is suspicious only when **both** a strong image indicator
    (human or vehicle) and a gunshot audio signal are present simultaneously.

    Parameters
    ----------
    image_label:
        Predicted label from the image branch.
        One of ``"human"``, ``"vehicle"``, ``"animal"``, ``"empty"``.
    audio_label:
        Predicted label from the audio branch.
        One of ``"gunshot"``, ``"background"``.

    Returns
    -------
    int
        ``1`` if suspicious, ``0`` otherwise.

    Raises
    ------
    ValueError
        If either label is not recognised.

    Examples
    --------
    >>> suspicious("human", "gunshot")
    1
    >>> suspicious("animal", "gunshot")
    0
    >>> suspicious("vehicle", "background")
    0

    """
    _validate(image_label, audio_label)
    return int(image_label in _STRONG_IMAGE and audio_label == "gunshot")


def risk(image_label: str, audio_label: str) -> str:
    """
    Return a categorical risk level for the given label pair.

    Parameters
    ----------
    image_label:
        Predicted label from the image branch.
        One of ``"human"``, ``"vehicle"``, ``"animal"``, ``"empty"``.
    audio_label:
        Predicted label from the audio branch.
        One of ``"gunshot"``, ``"background"``.

    Returns
    -------
    str
        One of ``"HIGH"``, ``"MEDIUM"``, or ``"LOW"``.

    Raises
    ------
    ValueError
        If either label is not recognised.

    Examples
    --------
    >>> risk("human", "gunshot")
    'HIGH'
    >>> risk("animal", "gunshot")
    'MEDIUM'
    >>> risk("empty", "background")
    'LOW'

    """
    _validate(image_label, audio_label)

    # HIGH — both a strong image indicator and gunshot audio
    if image_label in _STRONG_IMAGE and audio_label == "gunshot":
        return "HIGH"

    # MEDIUM — exactly one strong indicator
    if image_label in _STRONG_IMAGE and audio_label == "background":
        return "MEDIUM"
    if image_label == "animal" and audio_label == "gunshot":
        return "MEDIUM"

    # LOW — animal or empty with background (covers thunder_wind tagged clips)
    return "LOW"
