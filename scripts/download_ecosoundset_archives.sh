#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 1 || "$#" -gt 2 ]]; then
  printf 'Usage: %s DATA_ROOT [all|split1|split2]\n' "$0" >&2
  exit 2
fi

readonly DATA_ROOT="$1"
readonly TARGET="${2:-all}"
readonly DOWNLOAD_DIR="${DATA_ROOT}/downloads/ecosoundset"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly APPEND_TOOL="${SCRIPT_DIR}/append_binary_file.py"
readonly FILE_TOOL="${SCRIPT_DIR}/file_fingerprint.py"
readonly PART_COUNT=32
readonly MAX_PARALLEL=32
mkdir -p "${DOWNLOAD_DIR}"

download_one() {
  local name="$1"
  local expected_bytes="$2"
  local expected_md5="$3"
  local url="$4"
  local final="${DOWNLOAD_DIR}/${name}"
  local partial="${final}.partial"
  local part_dir="${final}.parts"

  if [[ -f "${final}" ]] \
    && [[ "$(python3 "${FILE_TOOL}" size "${final}")" -eq "${expected_bytes}" ]] \
    && [[ "$(python3 "${FILE_TOOL}" md5 "${final}")" == "${expected_md5}" ]]; then
    printf 'Already verified: %s\n' "${name}"
    return
  fi

  mkdir -p "${part_dir}"
  local chunk_bytes=$(( (expected_bytes + PART_COUNT - 1) / PART_COUNT ))
  if [[ -f "${partial}" ]] && [[ ! -e "${part_dir}/part-00" ]] \
    && [[ ! -e "${part_dir}/part-00.tmp" ]]; then
    partial_size="$(python3 "${FILE_TOOL}" size "${partial}")"
    if [[ "${partial_size}" -lt "${chunk_bytes}" ]]; then
      mv "${partial}" "${part_dir}/part-00.tmp"
    else
      rm "${partial}"
    fi
  fi

  printf 'Downloading %s in %s verified ranges\n' "${name}" "${PART_COUNT}"
  local pids=()
  for ((index = 0; index < PART_COUNT; index++)); do
    start=$(( index * chunk_bytes ))
    end=$(( start + chunk_bytes - 1 ))
    if (( end >= expected_bytes )); then
      end=$(( expected_bytes - 1 ))
    fi
    expected=$(( end - start + 1 ))
    part="${part_dir}/part-$(printf '%02d' "${index}")"
    if [[ -f "${part}" ]] && [[ "$(python3 "${FILE_TOOL}" size "${part}")" -eq "${expected}" ]]; then
      continue
    fi
    (
      tmp="${part}.tmp"
      touch "${tmp}"
      observed="$(python3 "${FILE_TOOL}" size "${tmp}")"
      if [[ "${observed}" -gt "${expected}" ]]; then
        printf 'Part %02d exceeds expected size.\n' "${index}" >&2
        exit 1
      fi
      while [[ "${observed}" -lt "${expected}" ]]; do
        range_start=$(( start + observed ))
        remaining=$(( expected - observed ))
        transfer="${part}.transfer"
        rm -f "${transfer}"
        curl --location --fail --silent --show-error \
          --retry 3 --retry-delay 5 --retry-all-errors \
          --speed-time 120 --speed-limit 1024 \
          --range "${range_start}-${end}" --output "${transfer}" "${url}" || true
        received=0
        if [[ -f "${transfer}" ]]; then
          received="$(python3 "${FILE_TOOL}" size "${transfer}")"
        fi
        if [[ "${received}" -eq 0 ]]; then
          sleep 10
          continue
        fi
        if [[ "${received}" -gt "${remaining}" ]]; then
          printf 'Server ignored range for part %02d; discarding oversized response.\n' "${index}" >&2
          rm "${transfer}"
          sleep 10
          continue
        fi
        python3 "${APPEND_TOOL}" "${transfer}" "${tmp}"
        rm "${transfer}"
        observed="$(python3 "${FILE_TOOL}" size "${tmp}")"
      done
      mv "${tmp}" "${part}"
      printf 'Completed %s part %02d/%02d\n' "${name}" "$((index + 1))" "${PART_COUNT}"
    ) &
    pids+=("$!")
    if [[ "${#pids[@]}" -ge "${MAX_PARALLEL}" ]]; then
      for pid in "${pids[@]}"; do wait "${pid}"; done
      pids=()
    fi
  done
  if [[ "${#pids[@]}" -gt 0 ]]; then
    for pid in "${pids[@]}"; do wait "${pid}"; done
  fi

  assembling="${final}.assembling"
  mv "${part_dir}/part-00" "${assembling}"
  for ((index = 1; index < PART_COUNT; index++)); do
    part="${part_dir}/part-$(printf '%02d' "${index}")"
    python3 "${APPEND_TOOL}" "${part}" "${assembling}"
    rm "${part}"
  done
  rmdir "${part_dir}"

  observed_bytes="$(python3 "${FILE_TOOL}" size "${assembling}")"
  observed_md5="$(python3 "${FILE_TOOL}" md5 "${assembling}")"
  if [[ "${observed_bytes}" -ne "${expected_bytes}" ]] \
    || [[ "${observed_md5}" != "${expected_md5}" ]]; then
    printf 'Verification failed for %s: bytes=%s md5=%s\n' \
      "${name}" "${observed_bytes}" "${observed_md5}" >&2
    exit 1
  fi
  mv "${assembling}" "${final}"
  printf 'Verified %s: bytes=%s md5=%s\n' "${name}" "${observed_bytes}" "${observed_md5}"
}

if [[ "${TARGET}" == "all" || "${TARGET}" == "split1" ]]; then
  download_one \
    'Split recordings1.tar.gz' \
    13785342839 \
    c61086491aad58583ea2d3fba66b8d42 \
    'https://zenodo.org/api/records/18636037/files/Split%20recordings1.tar.gz/content'
fi

if [[ "${TARGET}" == "all" || "${TARGET}" == "split2" ]]; then
  download_one \
    'Split recordings2.tar.gz' \
    9084639756 \
    4069994f8aa6602b4ac9e62b99746eb6 \
    'https://zenodo.org/api/records/18636037/files/Split%20recordings2.tar.gz/content'
fi

if [[ "${TARGET}" != "all" && "${TARGET}" != "split1" && "${TARGET}" != "split2" ]]; then
  printf 'Unknown target: %s\n' "${TARGET}" >&2
  exit 2
fi
