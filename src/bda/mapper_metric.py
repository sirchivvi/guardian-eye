"""
Metric mapper for Hadoop-streaming-style BDA jobs.

Reads evaluation metrics JSON entries and emits
``(method, condition, metric_name) → value`` pairs.

Metrics with undefined values (None / null) are **not emitted**, so the
downstream reducer never counts them.

Can be used as a standalone script via ``python -m src.bda.mapper_metric``
which reads JSON lines from stdin and writes to stdout.
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Iterator

logger = logging.getLogger(__name__)

_REQUIRED_KEYS = frozenset({"method", "condition", "metric", "value"})


class MetricMapper:
    r"""Mapper that emits ``(method\tcondition\tmetric) \t value`` per metric entry."""

    def __init__(self) -> None:
        self._rows_processed = 0
        self._rows_skipped = 0

    def map(self, line: str) -> Iterator[tuple[str, str]]:
        r"""
        Parse a JSON metric entry and emit a key-value pair.

        Parameters
        ----------
        line:
            A single JSON object string representing one metric record.

        Yields
        ------
        tuple[str, str]
            ``(key, value)`` where key is ``"method\tcondition\tmetric"``
            and value is a string representation of the numeric metric value.

        """
        stripped = line.strip()
        if not stripped:
            return

        try:
            record = json.loads(stripped)
        except json.JSONDecodeError:
            self._rows_skipped += 1
            logger.debug("Skipping non-JSON line: %s", stripped[:100])
            return

        if not isinstance(record, dict):
            self._rows_skipped += 1
            return

        missing = _REQUIRED_KEYS - set(record.keys())
        if missing:
            self._rows_skipped += 1
            logger.debug("Metric record missing keys: %s", sorted(missing))
            return

        # Skip undefined metrics (zero-denominator honesty)
        value = record["value"]
        if value is None:
            self._rows_skipped += 1
            return

        method = str(record["method"])
        condition = str(record["condition"])
        metric_name = str(record["metric"])

        self._rows_processed += 1
        key = f"{method}\t{condition}\t{metric_name}"
        yield (key, str(value))

    @property
    def rows_processed(self) -> int:
        """Return the number of rows successfully processed."""
        return self._rows_processed

    @property
    def rows_skipped(self) -> int:
        """Return the number of rows skipped due to errors."""
        return self._rows_skipped


def map_from_json_array(json_path: str) -> Iterator[tuple[str, str]]:
    """
    Read a JSON array file and yield all mapper outputs.

    Parameters
    ----------
    json_path:
        Path to a JSON file containing an array of metric records.

    Yields
    ------
    tuple[str, str]
        Key-value pairs from each record.

    """
    mapper = MetricMapper()
    with open(json_path) as fh:
        records = json.load(fh)

    if not isinstance(records, list):
        records = [records]

    for record in records:
        line = json.dumps(record)
        yield from mapper.map(line)


def main() -> None:
    """Run as a Hadoop Streaming mapper: read JSON lines from stdin, write to stdout."""
    mapper = MetricMapper()
    for line in sys.stdin:
        for key, value in mapper.map(line):
            sys.stdout.write(f"{key}\t{value}\n")


if __name__ == "__main__":
    main()
