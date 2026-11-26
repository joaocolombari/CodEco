#!/usr/bin/env bash
set -euo pipefail

# Build pinned FFmpeg with the official Google liblc3 implementation. The
# binary is local to the repository and is used only for LC3 experiments.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FFMPEG_VERSION="9.0.2"
FFMPEG_SHA256="8c3850283eb25fa026482078a04051e0be17347b09ef81a0849bec15a96e002e"
INSTALL_DIR="${ROOT_DIR}/.tools/ffmpeg-lc3/${FFMPEG_VERSION}"
FFMPEG_BIN="${INSTALL_DIR}/bin/ffmpeg"

verify_lc3() {
    local binary="$1"
    local check_dir
    check_dir="$(mktemp -d)"
    if ! "${binary}" -hide_banner -encoders 2>/dev/null | grep -q liblc3 || \
       ! "${binary}" -hide_banner -decoders 2>/dev/null | grep -q liblc3 || \
       ! "${binary}" -v error -f lavfi -i "sine=frequency=1000:sample_rate=48000:duration=0.1" \
           -ac 1 -c:a liblc3 -b:a 32k -frame_duration 10 -high_resolution 0 \
           -y "${check_dir}/check.lc3" || \
       ! "${binary}" -v error -i "${check_dir}/check.lc3" -ac 1 -ar 48000 \
           -y "${check_dir}/check.wav" || \
       [[ ! -s "${check_dir}/check.wav" ]]; then
        rm -rf "${check_dir}"
        return 1
    fi
    rm -rf "${check_dir}"
}

if [[ -x "${FFMPEG_BIN}" ]] && verify_lc3 "${FFMPEG_BIN}"; then
    echo "LC3 round-trip verified at ${FFMPEG_BIN}"
    exit 0
fi

for command in curl make pkg-config shasum; do
    command -v "${command}" >/dev/null || { echo "Missing command: ${command}" >&2; exit 1; }
done

case "$(uname -s)" in
    Darwin)
        command -v brew >/dev/null || { echo "Homebrew is required on macOS." >&2; exit 1; }
        brew list liblc3 >/dev/null 2>&1 || brew install liblc3
        export PKG_CONFIG_PATH="$(brew --prefix liblc3)/lib/pkgconfig${PKG_CONFIG_PATH:+:${PKG_CONFIG_PATH}}"
        ;;
    Linux)
        if ! pkg-config --exists lc3; then
            echo "liblc3 development files are required (pkg-config name: lc3)." >&2
            echo "Install liblc3 from https://github.com/google/liblc3 and rerun." >&2
            exit 1
        fi
        ;;
    *)
        echo "Unsupported operating system: $(uname -s)" >&2
        exit 1
        ;;
esac

BUILD_DIR="$(mktemp -d)"
trap 'rm -rf "${BUILD_DIR}"' EXIT
ARCHIVE="${BUILD_DIR}/ffmpeg-${FFMPEG_VERSION}.tar.xz"

curl --fail --location --retry 3 \
    "https://ffmpeg.org/releases/ffmpeg-${FFMPEG_VERSION}.tar.xz" \
    --output "${ARCHIVE}"

ACTUAL_SHA256="$(shasum -a 256 "${ARCHIVE}" | awk '{print $1}')"
if [[ "${ACTUAL_SHA256}" != "${FFMPEG_SHA256}" ]]; then
    echo "FFmpeg archive checksum mismatch." >&2
    exit 1
fi

tar -xf "${ARCHIVE}" -C "${BUILD_DIR}"
cd "${BUILD_DIR}/ffmpeg-${FFMPEG_VERSION}"
./configure \
    --prefix="${INSTALL_DIR}" \
    --disable-doc \
    --disable-debug \
    --disable-ffplay \
    --enable-version3 \
    --enable-liblc3
make -j "$(sysctl -n hw.logicalcpu 2>/dev/null || nproc 2>/dev/null || echo 2)"
make install

verify_lc3 "${FFMPEG_BIN}"
echo "LC3 installed and round-trip verified at ${FFMPEG_BIN}"
