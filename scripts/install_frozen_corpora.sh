#!/usr/bin/env bash
set -euo pipefail

readonly ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly TARGET="${1:-all}"

case "${TARGET}" in
  metadata|all|ecosoundset|pteroset) ;;
  *)
    printf 'Usage: %s [metadata|all|ecosoundset|pteroset]\n' "$0" >&2
    exit 2
    ;;
esac

for command_name in git python3 curl; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    printf 'Missing required command: %s\n' "${command_name}" >&2
    exit 1
  fi
done

python3 - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit("Python 3.10 or newer is required")
print(f"Python {sys.version_info.major}.{sys.version_info.minor} detected")
PY

python3 "${ROOT_DIR}/scripts/checkout_frozen_sources.py"
python3 "${ROOT_DIR}/scripts/build_manifests.py"
python3 "${ROOT_DIR}/scripts/verify_frozen_install.py" --stage metadata

if [[ "${TARGET}" == "metadata" ]]; then
  printf 'Metadata-only freeze installation complete.\n'
  exit 0
fi

required_gib=55
if [[ "${TARGET}" == "ecosoundset" ]]; then required_gib=38; fi
if [[ "${TARGET}" == "pteroset" ]]; then required_gib=17; fi
python3 - "${ROOT_DIR}" "${required_gib}" <<'PY'
import shutil, sys
root, required = sys.argv[1], int(sys.argv[2])
free = shutil.disk_usage(root).free
required_bytes = required * 1024**3
if free < required_bytes:
    raise SystemExit(
        f"At least {required} GiB free are required; only {free / 1024**3:.1f} GiB are available"
    )
print(f"Disk preflight passed: {free / 1024**3:.1f} GiB free")
PY

if [[ "${TARGET}" == "all" || "${TARGET}" == "pteroset" ]]; then
  bash "${ROOT_DIR}/scripts/download_pteroset_ppa1.sh"
  python3 "${ROOT_DIR}/scripts/verify_pteroset_ppa1.py"
fi

if [[ "${TARGET}" == "all" || "${TARGET}" == "ecosoundset" ]]; then
  bash "${ROOT_DIR}/scripts/run_ecosoundset_acquisition.sh" \
    "${ROOT_DIR}/data" \
    "${ROOT_DIR}/data/manifests/ecosoundset_download_manifest_v1.csv"
fi

python3 "${ROOT_DIR}/scripts/verify_frozen_install.py" \
  --stage audio \
  --corpus "${TARGET}"

printf 'Corpus freeze 1.3.0 installation complete.\n'
