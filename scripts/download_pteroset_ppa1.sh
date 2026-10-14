#!/usr/bin/env bash
set -euo pipefail

readonly ARCHIVE_BYTES=15257785797
readonly ARCHIVE_MD5=d5fce3e9d743bad4988131b57aa1c9bc
readonly PART_COUNT=32
readonly MAX_PARALLEL=16
readonly URL=https://zenodo.org/api/records/21829388/files/PPA1.zip/content

readonly ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly APPEND_TOOL="${ROOT_DIR}/scripts/append_binary_file.py"
readonly FILE_TOOL="${ROOT_DIR}/scripts/file_fingerprint.py"
readonly OUTPUT_DIR="${ROOT_DIR}/data/downloads/pteroset"
readonly ARCHIVE="${OUTPUT_DIR}/PPA1.zip"
readonly PART_DIR="${OUTPUT_DIR}/PPA1.parts"

mkdir -p "${OUTPUT_DIR}" "${PART_DIR}"

if [[ -f "${ARCHIVE}" ]] && [[ "$(python3 "${FILE_TOOL}" size "${ARCHIVE}")" -eq "${ARCHIVE_BYTES}" ]]; then
  observed_md5="$(python3 "${FILE_TOOL}" md5 "${ARCHIVE}")"
  if [[ "${observed_md5}" == "${ARCHIVE_MD5}" ]]; then
    printf 'PPA1.zip already downloaded and verified.\n'
    exit 0
  fi
fi

if [[ -f "${ARCHIVE}" ]]; then
  mv "${ARCHIVE}" "${ARCHIVE}.single-connection.partial"
fi

chunk_bytes=$(( (ARCHIVE_BYTES + PART_COUNT - 1) / PART_COUNT ))
pids=()

for ((index = 0; index < PART_COUNT; index++)); do
  start=$(( index * chunk_bytes ))
  end=$(( start + chunk_bytes - 1 ))
  if (( end >= ARCHIVE_BYTES )); then
    end=$(( ARCHIVE_BYTES - 1 ))
  fi
  expected=$(( end - start + 1 ))
  part="${PART_DIR}/part-$(printf '%02d' "${index}")"

  if [[ -f "${part}" ]] && [[ "$(python3 "${FILE_TOOL}" size "${part}")" -eq "${expected}" ]]; then
    continue
  fi

  (
    tmp="${part}.tmp"
    touch "${tmp}"
    observed="$(python3 "${FILE_TOOL}" size "${tmp}")"
    if [[ "${observed}" -gt "${expected}" ]]; then
      printf 'Part %02d exceeds its expected size.\n' "${index}" >&2
      exit 1
    fi

    while [[ "${observed}" -lt "${expected}" ]]; do
      range_start=$(( start + observed ))
      remaining=$(( expected - observed ))
      transfer="${part}.transfer"
      rm -f "${transfer}"
      curl --location --fail --silent --show-error \
        --retry 2 --retry-delay 5 --retry-all-errors \
        --speed-time 120 --speed-limit 1024 \
        --range "${range_start}-${end}" --output "${transfer}" "${URL}" || true
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
      if [[ "${observed}" -gt "${expected}" ]]; then
        printf 'Part %02d exceeded its expected size after resume.\n' "${index}" >&2
        exit 1
      fi
    done
    mv "${tmp}" "${part}"
  ) &
  pids+=("$!")
  if [[ "${#pids[@]}" -ge "${MAX_PARALLEL}" ]]; then
    for pid in "${pids[@]}"; do
      wait "${pid}"
    done
    pids=()
  fi
done

if [[ "${#pids[@]}" -gt 0 ]]; then
  for pid in "${pids[@]}"; do
    wait "${pid}"
  done
fi

assembling="${OUTPUT_DIR}/PPA1.zip.assembling"
mv "${PART_DIR}/part-00" "${assembling}"
for ((index = 1; index < PART_COUNT; index++)); do
  part="${PART_DIR}/part-$(printf '%02d' "${index}")"
  python3 "${APPEND_TOOL}" "${part}" "${assembling}"
  rm "${part}"
done
rmdir "${PART_DIR}"
mv "${assembling}" "${ARCHIVE}"

observed_bytes="$(python3 "${FILE_TOOL}" size "${ARCHIVE}")"
observed_md5="$(python3 "${FILE_TOOL}" md5 "${ARCHIVE}")"
if [[ "${observed_bytes}" -ne "${ARCHIVE_BYTES}" ]] || [[ "${observed_md5}" != "${ARCHIVE_MD5}" ]]; then
  printf 'Archive verification failed: bytes=%s md5=%s\n' "${observed_bytes}" "${observed_md5}" >&2
  exit 1
fi

rm -f "${ARCHIVE}.single-connection.partial"
printf 'Verified PPA1.zip: bytes=%s md5=%s\n' "${observed_bytes}" "${observed_md5}"
