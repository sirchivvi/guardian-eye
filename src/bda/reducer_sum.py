r"""
Summation reducer for Hadoop-streaming-style BDA jobs.

Groups by key and sums integer values.  Used by the class-count and scenario
counting pipelines.

Can be used as a standalone script via ``python -m src.bda.reducer_sum``
which reads tab-separated ``key \t value`` pairs from stdin (pre-sorted by
key) and writes ``key \t total`` to stdout.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterator

logger = logging.getLogger(__name__)


class SumReducer:
    """Reducer that sums integer values for each key."""

    def reduce(self, key: str, values: Iterator[str]) -> Iterator[str]:
        r"""
        Sum all values for a given key.

        Parameters
        ----------
        key:
            The grouping key (tab-separated composite key).
        values:
            Iterator of string-encoded integer values.

        Yields
        ------
        str
            ``"key\ttotal"`` line.

        """
        total = 0
        for v in values:
            try:
                total += int(v.strip())
            except ValueError:
                logger.debug("Non-integer value for key %r: %r", key, v)
        yield f"{key}\t{total}"


def main() -> None:
    """Run as a Hadoop Streaming reducer: read sorted stdin, write to stdout."""
    reducer = SumReducer()
    current_key: str | None = None
    current_values: list[str] = []

    for line in sys.stdin:
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.rsplit("\t", maxsplit=1)
        if len(parts) != 2:
            continue
        key, value = parts

        if current_key is not None and key != current_key:
            for output in reducer.reduce(current_key, iter(current_values)):
                sys.stdout.write(output + "\n")
            current_values = []

        current_key = key
        current_values.append(value)

    # Flush last group
    if current_key is not None and current_values:
        for output in reducer.reduce(current_key, iter(current_values)):
            sys.stdout.write(output + "\n")


if __name__ == "__main__":
    main()
