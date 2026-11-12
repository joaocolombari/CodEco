"""Orquestrador completo do experimento de compressao em soundscape ecology.
Executa todas as etapas em ordem com suporte a retomada (skip-existing) e logs de tempo.

Uso padrao (roda tudo):
    python run_pipeline.py

Uso parcial (exemplos):
    python run_pipeline.py --datasets pteroset
    python run_pipeline.py --skip-existing
    python run_pipeline.py --only eval classify analyze
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import config as C
from autoencoder import ckpt_path


def run_step(desc: str, cmd: list[str]):
    print(f"\n{'=' * 72}")
    print(f"[ETAPA] {desc}")
    print(f"[CMD]   {' '.join(cmd)}")
    print(f"{'=' * 72}")
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True)
    dt = time.perf_counter() - t0
    print(f"[OK] {desc} concluida em {dt / 60:.2f} min ({dt:.1f} s).")


def main():
    ap = argparse.ArgumentParser(description="Pipeline completo de avaliacao de codecs para Bioacustica IoT")
    ap.add_argument("--datasets", nargs="+", default=list(C.DATASETS.keys()),
                    choices=list(C.DATASETS.keys()), help="Datasets a processar")
    ap.add_argument("--models", nargs="+", default=["birdnet", "perch"],
                    choices=["birdnet", "perch"], help="Modelos fundacionais para classificacao")
    ap.add_argument("--head", default="mlp", choices=["linear", "mlp"],
                    help="Cabeca classificadora sobre os embeddings congelados")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Pula treino de autoencoders cujo checkpoint .pt ja exista")
    ap.add_argument("--only", nargs="+",
                    choices=["data", "train_ae", "flops", "eval", "classify", "analyze"],
                    default=["data", "train_ae", "flops", "eval", "classify", "analyze"],
                    help="Filtra quais etapas da pipeline executar")
    ap.add_argument("--perf", action="store_true",
                    help="Passa --perf para flops.py (contagem de instrucoes de CPU no Linux)")
    args = ap.parse_args()

    py = sys.executable
    C.OUT_DIR.mkdir(exist_ok=True)
    t_total = time.perf_counter()

    # 1. Preparacao e fatiamento dos datasets (48 kHz / 16-bit, Top-10 classes, site-disjoint split)
    if "data" in args.only:
        run_step("1/6 Processamento dos datasets (data.py)", [py, "data.py"])

    # 2. Treinamento das variantes do Autoencoder (wave, wave_lite, stft)
    if "train_ae" in args.only:
        for ds in args.datasets:
            for variant in C.AE_VARIANTS:
                ckpt = ckpt_path(ds, variant)
                if args.skip_existing and Path(ckpt).exists():
                    print(f"\n[PULANDO] Checkpoint ja existe: {ckpt}")
                    continue
                run_step(
                    f"2/6 Treino Autoencoder [{ds} | {variant}]",
                    [py, "autoencoder.py", "--dataset", ds, "--variant", variant],
                )

    # 3. Benchmark de FLOPs (redes neurais) e custo de CPU (codecs C)
    if "flops" in args.only:
        cmd_flops = [py, "flops.py"]
        if args.perf:
            cmd_flops.append("--perf")
        run_step("3/6 Benchmark de custo computacional / FLOPs (flops.py)", cmd_flops)

    # 4. Avaliacao de reconstrucao (SNR, Mel-SSIM) e indices acusticos (scikit-maad)
    if "eval" in args.only:
        run_step("4/6 Avaliacao de codecs e indices acusticos (run_codec_eval.py)", [py, "run_codec_eval.py"])

    # 5. Extracao de embeddings e classificacao (BirdNET v2.4 e Google Perch)
    if "classify" in args.only:
        for ds in args.datasets:
            for model in args.models:
                run_step(
                    f"5/6 Classificacao com embeddings [{ds} | {model} | head={args.head}]",
                    [py, "classify_embeddings.py", "--dataset", ds, "--model", model, "--head", args.head],
                )

    # 6. Analise estatistica (correlacoes de indices, scatter plots e curvas taxa-distorcao)
    if "analyze" in args.only:
        run_step("6/6 Geracao de graficos e correlacoes finais (analyze.py)", [py, "analyze.py"])

    elapsed = time.perf_counter() - t_total
    print(f"\n{'=' * 72}")
    print(f"PIPELINE COMPLETA CONCLUIDA COM SUCESSO em {elapsed / 60:.2f} minutos!")
    print(f"Todos os resultados e graficos estao em: {C.OUT_DIR.resolve()}")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()