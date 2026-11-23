"""Compatibility configuration for executable experiment scripts.

Authoritative corpus membership remains in the frozen manifests. Signal and
model-input rules come from ``src/codeco/protocol.py`` so scripts and
notebooks cannot silently diverge.
"""
from pathlib import Path
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from codeco.protocol import PROTOCOL

SEED = 42
OUT_DIR = ROOT / "data" / "derived" / "experiment"
CACHE_DIR = OUT_DIR / "cache"

# O FFmpeg distribuido pelo Homebrew nao habilita liblc3. O instalador do
# projeto cria somente o binario LC3 local; os demais codecs continuam usando
# o FFmpeg do sistema, que inclui libmp3lame e libopus.
SYSTEM_FFMPEG = Path(shutil.which("ffmpeg") or "ffmpeg")
LC3_FFMPEG = ROOT / ".tools" / "ffmpeg-lc3" / "9.0.2" / "bin" / "ffmpeg"

# ---------------- Audio ----------------
SR = PROTOCOL.sample_rate_hz
BIT_DEPTH = PROTOCOL.bit_depth
CLIP_SECONDS = PROTOCOL.window_seconds
CLIP_SAMPLES = PROTOCOL.window_samples
SEGMENT_HOP_SECONDS = PROTOCOL.segment_hop_seconds
SEGMENT_HOP_SAMPLES = PROTOCOL.segment_hop_samples

CLASS_SELECTION_K_SIGMA = 1.0
CLASS_SELECTION_MIN_GROUPS = 3
CLASS_SELECTION_SEED = SEED

# ---------------- Mel-espectrograma ----------------
N_FFT = 2048
HOP_RATIO = 0.25            # hop = 25% de N_FFT (=> 75% de sobreposicao)
HOP_LENGTH = int(N_FFT * HOP_RATIO)
N_MELS = 128
FMIN = 0
FMAX = SR // 2
MEL_DB_FLOOR = -100.0       # piso em dB (ref=1.0, amplitude de ponto flutuante em [-1,1])

# ---------------- Indices acusticos (scikit-maad) ----------------
MAAD_NPERSEG = 1024
MAAD_NOVERLAP = 512
MAAD_FLIM_LOW = (0, 1500)
MAAD_FLIM_MID = (1500, 8000)
MAAD_FLIM_HI = (8000, 20000)
PLOT_INDICES = [
    "idx_ACI", "idx_ADI", "idx_AEI", "idx_H", "idx_Ht", "idx_Hf",
    "idx_BIO_2_24", "idx_BIO_2_8", "idx_NDSI_CLASSIC", "idx_NDSI_EXTENDED",
]

# ---------------- Datasets ----------------
DATASETS = {
    "ecosoundset": dict(manifest=ROOT / "data/manifests/ecosoundset_manifest_v1.csv"),
    "pteroset_ppa1": dict(manifest=ROOT / "data/manifests/pteroset_ppa1_manifest_v1.csv"),
}

# ---------------- Variantes e Niveis de Quantizacao do Autoencoder ----------------
AE_VARIANTS = ("wave", "wave_lite", "stft")
AE_QUANT_LEVELS = 256                 # niveis no treino (8 bits/elemento)
AE_EVAL_LEVELS = (16, 64, 256)        # niveis na avaliacao (4, 6 e 8 bits -> gera curva R-D)

# ---------------- Codecs e configuracoes (bitrates em kbps, mono) ----------------
EXPERIMENT_CONFIG = json.loads(
    (ROOT / "configs/experiment-v1.json").read_text(encoding="utf-8")
)
_CODEC_SPEC = EXPERIMENT_CONFIG["codecs"]
CODECS = {
    "flac": [dict(compression_level=level) for level in _CODEC_SPEC["flac"]["compression_levels"]],
    "mp3": [dict(bitrate_kbps=b) for b in _CODEC_SPEC["mp3"]["bitrates_kbps"]],
    "aac": [dict(bitrate_kbps=b) for b in _CODEC_SPEC["aac"]["bitrates_kbps"]],
    "opus": [dict(bitrate_kbps=b) for b in _CODEC_SPEC["opus"]["bitrates_kbps"]],
    "lc3": [dict(bitrate_kbps=b) for b in _CODEC_SPEC["lc3"]["bitrates_kbps"]],
    "encodec": [
        dict(bandwidth_kbps=b, model_name=_CODEC_SPEC["encodec"]["model"])
        for b in _CODEC_SPEC["encodec"]["bitrates_kbps"]
    ],
}

AUTOENCODERS_ENABLED_BY_DEFAULT = False
DEFERRED_AUTOENCODER_CODECS = [
    dict(variant=variant, levels=levels)
    for variant in AE_VARIANTS
    for levels in AE_EVAL_LEVELS
]

BIRDNET_VERSION = "2.4"
PERCH_MODEL_HANDLE = (
    "https://www.kaggle.com/models/google/bird-vocalization-classifier/"
    "TensorFlow2/perch_v2/2"
)

# ---------------- Autoencoder ----------------
AE_WAVE_STRIDES = (4, 4, 5, 3)   # Produto = 240 -> 1000 frames latentes em 5s
AE_LITE_STRIDES = (8, 5, 3, 2)

# --- variante stft ---
STFT_N_FFT = 1024
STFT_HOP = 512                   # 50% de sobreposicao (Hann satisfaz COLA)
STFT_HIDDEN = 128
STFT_DILATIONS = (1, 3, 9)
STFT_LATENT_CH = 16
STFT_COMPRESS_POWER = 0.3
AE_BASE_CH = 16
AE_LATENT_CH = 8
AE_EPOCHS = 50
AE_BATCH = 16
AE_LR = 3e-4
AE_STFT_SIZES = (512, 1024, 2048)

# ---------------- Avaliacao ----------------
EVAL_MAX_CLIPS_PER_DATASET = 200   # None = todos
ALIGN_MAX_LAG = 4096               # amostras (compensa atraso de codecs)
