#!/usr/bin/env python3
"""Append one binary file to another without loading it into memory."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: append_binary_file.py SOURCE DESTINATION")
    source = Path(sys.argv[1])
    destination = Path(sys.argv[2])
    with source.open("rb") as input_stream, destination.open("ab") as output_stream:
        shutil.copyfileobj(input_stream, output_stream, length=8 * 1024 * 1024)


if __name__ == "__main__":
    main()
