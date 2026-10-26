"""
Class-count mapper for Hadoop-streaming-style BDA jobs.

Reads manifest CSV lines and emits ``(modality, source, label) → 1`` pairs.
Can be used as a standalone script via ``python -m src.bda.mapper_class``
which reads from stdin and writes to stdout, exactly as Hadoop Streaming
would invoke it.
"""

from __future__ import annotations

import csv
import io
import logging
import sys
from collections.abc import Iterator

logger = logging.getLogger(__name__)

# Manifest columns we need — index positions in the CSV header
_COL_MODALITY = "modality"
_COL_SOURCE = "source"
_COL_LABEL = "label"

_REQUIRED_COLUMNS = frozenset({_COL_MODALITY, _COL_SOURCE, _COL_LABEL})


class ClassCountMapper:
    r"""Mapper that emits ``(modality\tsource\tlabel) \t 1`` per manifest row."""

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
            The first line of the manifest CSV.

        """
        reader = csv.reader(io.StringIO(header_line.strip()))
        self._header = next(reader, None)
        if self._header is None:
            logger.warning("Empty header line in class-count mapper")
            return
        self._col_indices = {
            col: idx for idx, col in enumerate(self._header)
        }
        missing = _REQUIRED_COLUMNS - set(self._col_indices.keys())
        if missing:
            logger.warning("Manifest header missing columns: %s", sorted(missing))

    def map(self, line: str) -> Iterator[tuple[str, str]]:
        r"""
        Parse a manifest CSV line and emit a key-value pair.

        Parameters
        ----------
        line:
            A single CSV data row (not the header).

        Yields
        ------
        tuple[str, str]
            ``(key, value)`` where key is ``"modality\tsource\tlabel"``
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
            modality = fields[self._col_indices[_COL_MODALITY]]
            source = fields[self._col_indices[_COL_SOURCE]]
            label = fields[self._col_indices[_COL_LABEL]]
        except (KeyError, IndexError):
            self._rows_skipped += 1
            logger.debug("Skipping malformed row: %s", stripped)
            return

        if not modality or not source or not label:
            self._rows_skipped += 1
            logger.debug("Skipping row with empty fields: %s", stripped)
            return

        self._rows_processed += 1
        key = f"{modality}\t{source}\t{label}"
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
    mapper = ClassCountMapper()
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
