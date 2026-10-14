#!/usr/bin/env python3
"""Remove a known number of leading bytes from a binary partial download."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) not in {3, 4}:
        raise SystemExit("usage: trim_binary_prefix.py FILE BYTE_COUNT [ORIGINAL_LENGTH]")
    path = Path(sys.argv[1])
    byte_count = int(sys.argv[2])
    original_length = int(sys.argv[3]) if len(sys.argv) == 4 else path.stat().st_size
    temporary = path.with_suffix(path.suffix + ".trimmed")
    with path.open("rb") as input_stream, temporary.open("wb") as output_stream:
        input_stream.seek(byte_count)
        remaining = original_length - byte_count
        while remaining:
            block = input_stream.read(min(8 * 1024 * 1024, remaining))
            if not block:
                raise EOFError("Partial file is shorter than the requested original length")
            output_stream.write(block)
            remaining -= len(block)
    os.replace(temporary, path)


if __name__ == "__main__":
    main()
