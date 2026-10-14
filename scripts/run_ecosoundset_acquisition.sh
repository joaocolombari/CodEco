#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  printf 'Usage: %s DATA_ROOT SELECTION_MANIFEST\n' "$0" >&2
  exit 2
fi

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly DATA_ROOT="$1"
readonly SELECTION_MANIFEST="$2"

bash "${SCRIPT_DIR}/download_ecosoundset_archives.sh" "${DATA_ROOT}"
python3 "${SCRIPT_DIR}/extract_verify_ecosoundset.py" \
  --data-root "${DATA_ROOT}" \
  --selection "${SELECTION_MANIFEST}"
