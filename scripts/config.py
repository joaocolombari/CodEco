"""Todos os parametros ajustaveis do experimento ficam aqui."""
from pathlib import Path

SEED = 42
OUT_DIR = Path("outputs")
CACHE_DIR = OUT_DIR / "cache"

# ---------------- Audio ----------------
SR = 48_000                 # taxa alvo (Hz)
BIT_DEPTH = 16              # profundidade alvo (bits)
CLIP_SECONDS = 5.0          # tamanho dos cortes (5s)
CLIP_SAMPLES = int(SR * CLIP_SECONDS)  # 240.000 amostras exatas
MIN_OVERLAP_SECONDS = 0.5   # sobreposicao minima anotacao x janela p/ atribuir o rotulo
TOP_N_CLASSES = 10          # top-N classes por frequencia de anotacoes (por dataset)
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)   # train/val/test
KEEP_EMPTY_WINDOWS = False  # False = mantem apenas janelas com >=1 vocalizacao das Top-10 (acelera muito)

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
PLOT_INDICES = ["idx_ACI", "idx_ADI", "idx_AEI", "idx_BI", "idx_NDSI", "idx_Hf", "idx_Ht"]

# ---------------- Datasets ----------------
DATASETS = {
    "pteroset": dict(
        audio_dir="data/pteroset/audio",
        annotations_json="data/pteroset/annotations.json",
    ),
    "ecosoundset": dict(
        audio_dir="data/ecosoundset/audio",
        annotations_csv="data/ecosoundset/annotations.csv",
        cols=dict(
            file="audio_segment_file_name",
            start="audio_segment_initial_time",
            end="audio_segment_final_time",
            label="label"
        ),
    ),
}

# ---------------- Variantes e Niveis de Quantizacao do Autoencoder ----------------
AE_VARIANTS = ("wave", "wave_lite", "stft")
AE_QUANT_LEVELS = 256                 # niveis no treino (8 bits/elemento)
AE_EVAL_LEVELS = (16, 64, 256)        # niveis na avaliacao (4, 6 e 8 bits -> gera curva R-D)

# ---------------- Codecs e configuracoes (bitrates em kbps, mono) ----------------
CODECS = {
    "flac":    [dict(compression_level=5)],
    "mp3":     [dict(bitrate_kbps=b) for b in (32, 64, 128)],
    "opus":    [dict(bitrate_kbps=b) for b in (12, 24, 48, 96)],
    "lc3":     [dict(bitrate_kbps=b) for b in (32, 64, 96)],
    "encodec": [dict(bandwidth_kbps=b, model_name="facebook/encodec_48khz") for b in (6, 12, 24)],
    "ae":      [dict(variant=v, levels=lv) for v in AE_VARIANTS for lv in AE_EVAL_LEVELS],
}

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