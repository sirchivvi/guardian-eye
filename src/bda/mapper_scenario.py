"""
Scenario mapper for Hadoop-streaming-style BDA jobs.

Reads scenario/pairing CSV lines and emits
``(split, scenario, suspicious, risk) → 1`` pairs.

Can be used as a standalone script via ``python -m src.bda.mapper_scenario``
which reads from stdin and writes to stdout.
"""

from __future__ import annotations

import csv
import io
import logging
import sys
from collections.abc import Iterator

logger = logging.getLogger(__name__)

_COL_SPLIT = "split"
_COL_SCENARIO = "scenario"
_COL_SUSPICIOUS = "suspicious"
_COL_RISK = "risk"

_REQUIRED_COLUMNS = frozenset({_COL_SPLIT, _COL_SCENARIO, _COL_SUSPICIOUS, _COL_RISK})


class ScenarioMapper:
    r"""Mapper that emits ``(split\tscenario\tsuspicious\trisk) \t 1`` per scenario row."""

    def __init__(self) -> None:
        self._header: list[str] | None = None
        self._col_indices: dict[str, int] = {}
        self._rows_processed = 0
        self._rows_skipped = 0

    def set_header(self, header_line: str) -> None:
        """
        Parse the CSV header and cache column indices.

        Parameters
        ----------
        header_line:
            The first line of the scenario CSV.

        """
        reader = csv.reader(io.StringIO(header_line.strip()))
        self._header = next(reader, None)
        if self._header is None:
            return
        self._col_indices = {
            col: idx for idx, col in enumerate(self._header)
        }
        missing = _REQUIRED_COLUMNS - set(self._col_indices.keys())
        if missing:
            logger.warning("Scenario header missing columns: %s", sorted(missing))

    def map(self, line: str) -> Iterator[tuple[str, str]]:
        r"""
        Parse a scenario CSV line and emit a key-value pair.

        Parameters
        ----------
        line:
            A single CSV data row (not the header).

        Yields
        ------
        tuple[str, str]
            ``(key, value)`` where key is ``"split\tscenario\tsuspicious\trisk"``
            and value is ``"1"``.

        """
        stripped = line.strip()
        if not stripped:
            return

        reader = csv.reader(io.StringIO(stripped))
        fields = next(reader, None)
        if fields is None:
            self._rows_skipped += 1
            return

        try:
            split = fields[self._col_indices[_COL_SPLIT]]
            scenario = fields[self._col_indices[_COL_SCENARIO]]
            suspicious = fields[self._col_indices[_COL_SUSPICIOUS]]
            risk_level = fields[self._col_indices[_COL_RISK]]
        except (KeyError, IndexError):
            self._rows_skipped += 1
            logger.debug("Skipping malformed scenario row: %s", stripped)
            return

        self._rows_processed += 1
        key = f"{split}\t{scenario}\t{suspicious}\t{risk_level}"
        yield (key, "1")

    @property
    def rows_processed(self) -> int:
        """Return the number of rows successfully processed."""
        return self._rows_processed

    @property
    def rows_skipped(self) -> int:
        """Return the number of rows skipped due to errors."""
        return self._rows_skipped


def main() -> None:
    """Run as a Hadoop Streaming mapper: read stdin, write to stdout."""
    mapper = ScenarioMapper()
    first_line = True
    for line in sys.stdin:
        if first_line:
            mapper.set_header(line)
            first_line = False
            continue
        for key, value in mapper.map(line):
            sys.stdout.write(f"{key}\t{value}\n")


if __name__ == "__main__":
    main()
