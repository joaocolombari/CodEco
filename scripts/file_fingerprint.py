#!/usr/bin/env python3
"""Portable file-size and digest helper for the acquisition shell scripts."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("size", "md5", "sha256"))
    parser.add_argument("path", type=Path)
    args = parser.parse_args()

    if args.operation == "size":
        print(args.path.stat().st_size)
        return

    digest = hashlib.new(args.operation)
    with args.path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    print(digest.hexdigest())


if __name__ == "__main__":
    main()
