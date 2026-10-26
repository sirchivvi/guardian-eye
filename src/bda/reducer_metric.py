"""
Metric reducer for Hadoop-streaming-style BDA jobs.

For each ``(method, condition, metric_name)`` key, computes:

- **n**: count of values
- **mean**: arithmetic mean
- **sample_std**: ``sqrt(sum((x - mean)^2) / (n - 1))`` when n ≥ 2;
  empty string when n < 2

Can be used as a standalone script via ``python -m src.bda.reducer_metric``
which reads sorted tab-separated input from stdin and writes to stdout.
"""

from __future__ import annotations

import logging
import math
import sys
from collections.abc import Iterator

logger = logging.getLogger(__name__)


class MetricReducer:
    """Reducer that computes n, mean, and sample standard deviation for each key."""

    def reduce(self, key: str, values: Iterator[str]) -> Iterator[str]:
        r"""
        Compute summary statistics for a group of metric values.

        Parameters
        ----------
        key:
            The grouping key (``"method\tcondition\tmetric"``).
        values:
            Iterator of string-encoded float values.

        Yields
        ------
        str
            ``"key\tn\tmean\tsample_std"`` line.  ``sample_std`` is empty
            when n < 2.

        """
        nums: list[float] = []
        for v in values:
            try:
                nums.append(float(v.strip()))
            except ValueError:
                logger.debug("Non-numeric value for key %r: %r", key, v)

        n = len(nums)
        if n == 0:
            return

        mean = sum(nums) / n

        if n >= 2:
            variance = sum((x - mean) ** 2 for x in nums) / (n - 1)
            sample_std = str(math.sqrt(variance))
        else:
            sample_std = ""

        yield f"{key}\t{n}\t{mean}\t{sample_std}"


def main() -> None:
    """Run as a Hadoop Streaming reducer: read sorted stdin, write to stdout."""
    reducer = MetricReducer()
    current_key: str | None = None
    current_values: list[str] = []

    for line in sys.stdin:
        stripped = line.strip()
        if not stripped:
            continue
        # The key has 3 tab-separated fields; value is the 4th
        parts = stripped.split("\t")
        if len(parts) < 4:
            continue
        key = "\t".join(parts[:3])
        value = parts[3]

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
