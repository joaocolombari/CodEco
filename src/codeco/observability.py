"""Shared terminal logging helpers for pipeline stages."""

from __future__ import annotations

import logging

from tqdm import tqdm


class TqdmLoggingHandler(logging.StreamHandler):
    """Write log records without corrupting active progress bars."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.write(self.format(record), file=self.stream)
        except Exception:
            self.handleError(record)
