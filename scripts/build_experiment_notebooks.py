#!/usr/bin/env python3
"""Generate the nine auditable experiment notebooks from versioned text."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "experimental"


def markdown(source: str) -> dict:
    source = source.strip() + "\n"
    return {
        "cell_type": "markdown",
        "id": hashlib.sha1(("markdown:" + source).encode()).hexdigest()[:8],
        "metadata": {},
        "source": source,
    }


def code(source: str) -> dict:
    source = source.strip() + "\n"
    return {
        "cell_type": "code",
        "id": hashlib.sha1(("code:" + source).encode()).hexdigest()[:8],
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


SETUP = r"""
from pathlib import Path
import sys

ROOT = Path.cwd().resolve()
while ROOT != ROOT.parent and not (ROOT / "data" / "freeze-spec-v1.json").is_file():
    ROOT = ROOT.parent
if not (ROOT / "data" / "freeze-spec-v1.json").is_file():
    raise RuntimeError("Execute este notebook dentro do repositório CodEco.")

for location in (ROOT / "src", ROOT / "scripts"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

REPORTS = ROOT / "reports"
DERIVED = ROOT / "data" / "derived"
REPORTS.mkdir(exist_ok=True)
DERIVED.mkdir(exist_ok=True)
print(f"Repositório: {ROOT}")
"""


def notebook(title: str, purpose: str, cells: list[dict]) -> dict:
    return {
        "cells": [
            markdown(f"# {title}\n\n{purpose}\n\n> Execute as células de cima para baixo. Parâmetros editáveis aparecem em uma célula própria e todo artefato gerado informa seu destino."),
            code(SETUP),
            *cells,
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3 (CodEco)", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
            "codeco": {"generated_by": "scripts/build_experiment_notebooks.py", "auditable": True},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


NOTEBOOKS = {
    "01_corpus_audit.ipynb": notebook(
        "01 — Auditoria dos corpora congelados",
        "Confere proveniência, dimensões, splits, grupos e distribuição natural de classes sem alterar os manifests.",
        [
            markdown("## Ambiente\n\nA captura abaixo registra commit, Python, bibliotecas e disponibilidade do FFmpeg."),
            code(r"""
from codeco.notebook_support import environment_snapshot, write_json
environment = environment_snapshot(ROOT)
write_json(environment, REPORTS / "experiment_environment.json")
environment
"""),
            markdown("## Resumo dos manifests\n\nOs manifests versionados, e não as anotações brutas, definem as unidades permitidas no estudo."),
            code(r"""
import pandas as pd
from codeco.corpora import CORPUS_MANIFESTS, corpus_summary, load_frozen_manifest

manifests = {name: load_frozen_manifest(name, ROOT) for name in CORPUS_MANIFESTS}
summaries = pd.DataFrame([corpus_summary(df) for df in manifests.values()])
summaries.to_csv(REPORTS / "corpus_summary.csv", index=False)
summaries
"""),
            markdown("## Distribuição natural\n\nAs barras contam janelas por espécie antes de qualquer seleção. O eixo logarítmico evidencia a cauda longa."),
            code(r"""
import matplotlib.pyplot as plt
from codeco.corpora import class_distribution

fig, axes = plt.subplots(len(manifests), 1, figsize=(12, 5 * len(manifests)))
if len(manifests) == 1:
    axes = [axes]
for ax, (corpus, df) in zip(axes, manifests.items()):
    dist = class_distribution(df)
    ax.bar(range(len(dist)), dist["windows"], color="#3b7d5b")
    ax.set_yscale("log")
    ax.set_title(f"{corpus}: {len(dist)} classes")
    ax.set_ylabel("janelas (log)")
    ax.set_xlabel("classes ordenadas por frequência")
fig.tight_layout()
fig.savefig(REPORTS / "natural_class_distributions.png", dpi=180, bbox_inches="tight")
plt.show()
"""),
            code(r"""
distribution_rows = []
for corpus, df in manifests.items():
    dist = class_distribution(df)
    dist.insert(0, "corpus", corpus)
    distribution_rows.append(dist)
natural_distribution = pd.concat(distribution_rows, ignore_index=True)
natural_distribution.to_csv(REPORTS / "natural_class_distributions.csv", index=False)
natural_distribution.groupby("corpus").head(15)
"""),
        ],
    ),
    "02_class_selection.ipynb": notebook(
        "02 — Seleção e balanceamento das classes",
        "Torna visível o efeito do hiperparâmetro k, escolhe classes usando apenas o treino e congela o recorte reproduzível de cada corpus.",
        [
            markdown("## Parâmetros editáveis\n\n`k_sigma` maior inclui mais classes e reduz o suporte mínimo. Validação e teste preservam a distribuição natural; somente o treino recebe subamostragem determinística."),
            code(r"""
K_SIGMA = {"ecosoundset": 1.0, "pteroset_ppa1": 1.0}
MINIMUM_GROUPS = {"ecosoundset": 3, "pteroset_ppa1": 3}
SEED = 42
SENSITIVITY_K = [0.0, 0.5, 1.0, 1.5, 2.0]
K_SIGMA, MINIMUM_GROUPS, SEED
"""),
            markdown("## Sensibilidade do recorte\n\nEsta tabela deve ser discutida antes de congelar `k_sigma`. Nenhum resultado de teste é usado para escolhê-lo."),
            code(r"""
import pandas as pd
from codeco.corpora import (
    CORPUS_MANIFESTS, SelectionRule, deterministic_balanced_training_sample,
    freeze_selection, load_frozen_manifest, select_classes,
)

manifests = {name: load_frozen_manifest(name, ROOT) for name in CORPUS_MANIFESTS}
sensitivity = []
for corpus, df in manifests.items():
    for k in SENSITIVITY_K:
        _, meta = select_classes(df, SelectionRule(k_sigma=k, minimum_groups=MINIMUM_GROUPS[corpus], seed=SEED))
        sensitivity.append({"corpus": corpus, "k_sigma": k, "classes": meta["selected_class_count"], "cutoff": meta["count_cutoff"]})
sensitivity = pd.DataFrame(sensitivity)
sensitivity
"""),
            code(r"""
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for corpus, group in sensitivity.groupby("corpus"):
    axes[0].plot(group.k_sigma, group.classes, "o-", label=corpus)
    axes[1].plot(group.k_sigma, group.cutoff, "o-", label=corpus)
axes[0].set(xlabel="k σ", ylabel="classes selecionadas", title="Número de classes")
axes[1].set(xlabel="k σ", ylabel="janelas mínimas no treino", title="Limiar de suporte")
axes[1].set_yscale("log")
for ax in axes: ax.legend()
fig.tight_layout()
plt.show()
"""),
            markdown("## Recorte escolhido por corpus\n\nA linha tracejada é o limiar derivado de `μ − kσ` em `log1p(contagem)`. A tabela exibe explicitamente cada classe incluída ou excluída."),
            code(r"""
selection_outputs = {}
selection_tables = {}
for corpus, df in manifests.items():
    rule = SelectionRule(k_sigma=K_SIGMA[corpus], minimum_groups=MINIMUM_GROUPS[corpus], seed=SEED)
    table, meta = select_classes(df, rule)
    selected = meta["selected_classes"]
    quota = meta["count_cutoff"]
    sampled, balance = deterministic_balanced_training_sample(df, selected, quota=quota, seed=SEED)
    paths = freeze_selection(corpus, table, meta, sampled, balance, DERIVED / "selections")
    selection_outputs[corpus] = {key: str(path.relative_to(ROOT)) for key, path in paths.items()}
    selection_tables[corpus] = table
pd.DataFrame([
    {"corpus": corpus, **paths} for corpus, paths in selection_outputs.items()
])
"""),
            code(r"""
fig, axes = plt.subplots(len(selection_tables), 1, figsize=(12, 5 * len(selection_tables)))
if len(selection_tables) == 1: axes = [axes]
for ax, (corpus, table) in zip(axes, selection_tables.items()):
    colors = table.selected.map({True: "#2b8c5a", False: "#b8b8b8"})
    ax.bar(range(len(table)), table.windows, color=colors)
    ax.axhline(table.count_cutoff.iloc[0], color="#a33", linestyle="--", label="limiar")
    ax.set_yscale("log")
    ax.set_title(f"{corpus}: {int(table.selected.sum())} classes selecionadas")
    ax.set_ylabel("janelas de treino (log)")
    ax.legend()
fig.tight_layout()
fig.savefig(REPORTS / "selected_class_distributions.png", dpi=180, bbox_inches="tight")
plt.show()
"""),
            code(r"""
for corpus, table in selection_tables.items():
    print(f"\n{corpus}")
    display(table[["label", "windows", "groups", "z_score", "count_cutoff", "selected", "selection_reason"]])
"""),
        ],
    ),
    "03_preprocessing.ipynb": notebook(
        "03 — Protocolo e pré-processamento",
        "Demonstra a unidade canônica de 4 s, cria as tabelas de execução e audita os adaptadores específicos de BirdNET e Perch.",
        [
            markdown("## Protocolo comum\n\nPayload, codecs, índices e similaridade usam sempre quatro segundos. Para novas segmentações de áudio contínuo, o hop é 1 s e é diferente do hop interno de uma STFT. Os manifests congelados atuais não são resegmentados silenciosamente: ECOSoundSet já é distribuído em clipes de 4 s e o freeze PPA1 conserva `[0,4)` e `[4,8)`. Aplicar o hop de 1 s a esses corpora exigirá um novo freeze auditado."),
            code(r"""
import pandas as pd
from codeco.protocol import PROTOCOL, adapter_audit

pd.Series(PROTOCOL.as_dict(), name="valor").to_frame()
"""),
            code(r"""
from codeco.protocol import sliding_window_starts
pd.DataFrame({"inícios_em_10_s": sliding_window_starts(10.0)})
"""),
            markdown("## Entradas dos modelos fundacionais\n\nBirdNET recebe o centro de 3 s. Perch recebe o sinal de 4 s reamostrado para 32 kHz e 0,5 s de silêncio em cada lado. O padding acontece depois da reconstrução e não entra no payload."),
            code(r"""
audit = pd.Series(adapter_audit(), name="valor").to_frame()
audit
"""),
            code(r"""
import matplotlib.pyplot as plt
import numpy as np
from codeco.protocol import adapt_for_perch

x = np.ones(PROTOCOL.window_samples, dtype=np.float32)
perch = adapt_for_perch(x)
t = np.arange(len(perch)) / PROTOCOL.perch_sample_rate_hz
plt.figure(figsize=(11, 2.8))
plt.plot(t, perch, color="#356a8a")
plt.axvspan(0, 0.5, color="#ccc", alpha=.5, label="padding")
plt.axvspan(4.5, 5.0, color="#ccc", alpha=.5)
plt.xlabel("tempo (s)"); plt.ylabel("amplitude"); plt.title("Adaptador Perch: 4 s + padding simétrico")
plt.legend(); plt.tight_layout(); plt.show()
"""),
            markdown("## Tabelas de execução\n\nEstas tabelas apontam para o áudio instalado, mas não modificam os manifests. Se o áudio não estiver presente nesta máquina, a auditoria continua válida e a execução pesada fica para o EASNCloud."),
            code(r"""
from data import prepare_all_clip_tables
clip_paths = prepare_all_clip_tables()
clip_tables = {name: pd.read_csv(path) for name, path in clip_paths.items()}
pd.DataFrame([
    {"corpus": name, "janelas": len(df), "audio_disponivel": int(df.audio_available.sum()), "arquivo": str(clip_paths[name].relative_to(ROOT))}
    for name, df in clip_tables.items()
])
"""),
        ],
    ),
    "04_codec_sanity_check.ipynb": notebook(
        "04 — Teste funcional dos codecs",
        "Executa um round-trip sintético curto em todos os codecs e, opcionalmente, repete o piloto em janelas reais dos corpora.",
        [
            markdown("## Controles\n\nO teste sintético vem ativado: ele confirma que encode, payload e decode realmente funcionam. O piloto nos corpora fica separado porque exige os WAVs instalados. O EnCodec faz parte deste notebook, mas seu primeiro uso baixa o modelo pré-treinado."),
            code(r"""
RUN_SMOKE_TEST = True
RUN_CORPUS_PILOT = False
MAX_CLIPS_PER_CORPUS = 1
PILOT_CODECS = ["flac", "mp3", "aac", "opus", "lc3", "encodec"]
"""),
            code(r"""
import pandas as pd
import config as C

grid = []
for codec, settings in C.CODECS.items():
    for setting in settings:
        description = ", ".join(f"{key}={value}" for key, value in setting.items())
        grid.append({"codec": codec, "configuração": description,
                     "incluído_no_teste": codec in PILOT_CODECS})
codec_grid = pd.DataFrame(grid)
display(codec_grid)
"""),
            code(r"""
import importlib.util
import subprocess
from codecs_lib import make_codec

checks = [("flac", {}, "flac"),
          ("mp3", {"bitrate_kbps": 64}, "libmp3lame"),
          ("aac", {"bitrate_kbps": 64}, "aac"),
          ("opus", {"bitrate_kbps": 24}, "libopus"),
          ("lc3", {"bitrate_kbps": 32}, "liblc3")]
for codec_name, params, encoder_name in checks:
    try:
        codec = make_codec(codec_name, params)
        encoders = subprocess.check_output(
            [str(codec.ffmpeg), "-hide_banner", "-encoders"],
            text=True, stderr=subprocess.STDOUT,
        )
        available = encoder_name in encoders
        print(f"{codec_name:6s} {'disponível' if available else 'ausente':10s} {codec.ffmpeg}")
    except (RuntimeError, FileNotFoundError, subprocess.SubprocessError) as exc:
        print(f"{codec_name:6s} ausente    {exc}")

neural_dependencies = all(importlib.util.find_spec(name) is not None
                          for name in ("torch", "transformers"))
print(f"encodec {'dependências disponíveis' if neural_dependencies else 'bloqueado: instale scripts/requirements.txt'}")
"""),
            markdown("## Round-trip sintético\n\nEsta etapa não produz resultado científico. Ela usa exatamente 4 s em 48 kHz apenas para provar, em cada configuração, que o encoder gera um payload decodificável com o comprimento esperado. Sucessos e falhas aparecem em tabelas separadas; células vazias não são usadas como resultado."),
            code(r"""
import gc
import numpy as np

from codecs_lib import align_to_reference, make_codec
from metrics import mel_similarity, snr_db

def synthetic_smoke_signal():
    t = np.arange(C.CLIP_SAMPLES, dtype=np.float32) / C.SR
    tone = 0.28 * np.sin(2 * np.pi * 440 * t) + 0.14 * np.sin(2 * np.pi * 4200 * t)
    chirp = 0.10 * np.sin(2 * np.pi * (900 * t + 1400 * t**2))
    return (tone + chirp).astype(np.float32)

smoke_rows, smoke_failures = [], []
if RUN_SMOKE_TEST:
    x = synthetic_smoke_signal()
    for codec_name in PILOT_CODECS:
        for params in C.CODECS.get(codec_name, []):
            try:
                codec = make_codec(codec_name, params)
                encoded = codec.encode(x)
                decoded = codec.decode(encoded)
                decoded_samples = len(decoded)
                y, lag = align_to_reference(x, decoded)
                smoke_rows.append({
                    "codec": codec_name, "configuração": codec.setting,
                    "payload_bytes": encoded.nbytes,
                    "bitrate_efetivo_kbps": encoded.nbytes * 8 / C.CLIP_SECONDS / 1000,
                    "amostras_decodificadas": decoded_samples, "lag": lag,
                    "snr_db": snr_db(x, y), **mel_similarity(x, y),
                })
                del codec
                gc.collect()
            except Exception as exc:
                smoke_failures.append({"codec": codec_name,
                                       "configuração": ", ".join(f"{k}={v}" for k, v in params.items()),
                                       "erro": f"{type(exc).__name__}: {exc}"})

smoke_results = pd.DataFrame(smoke_rows)
smoke_errors = pd.DataFrame(smoke_failures)
display(smoke_results if not smoke_results.empty else pd.DataFrame({"estado": ["nenhum teste executado"]}))
if not smoke_errors.empty:
    display(smoke_errors)
"""),
            markdown("## Piloto com áudio dos corpora\n\nAtive somente depois de instalar os corpora. Ele repete as mesmas condições em uma janela real por corpus; a avaliação completa e retomável continua no notebook 05."),
            code(r"""
if RUN_CORPUS_PILOT:
    from codecs_lib import align_to_reference, make_codec
    from data import load_clip, prepare_all_clip_tables
    from metrics import mel_similarity, snr_db

    # audio_available e path sao propriedades da instalacao atual, nao do
    # freeze. Reconstroi as tabelas para evitar reutilizar caminhos absolutos
    # ou flags de disponibilidade gerados em outra maquina.
    clip_paths = prepare_all_clip_tables()
    availability_rows = []
    for corpus, clip_path in clip_paths.items():
        local_clips = pd.read_csv(clip_path)
        availability_rows.append({"corpus": corpus, "janelas_selecionadas": len(local_clips),
                                  "áudios_disponíveis": int(local_clips.audio_available.astype(bool).sum()),
                                  "tabela": str(clip_path)})
    availability = pd.DataFrame(availability_rows)
    display(availability)
    if availability["áudios_disponíveis"].sum() == 0:
        raise FileNotFoundError(
            "Nenhum WAV do recorte selecionado foi encontrado nesta máquina. "
            "Instale os corpora com ./scripts/install_frozen_corpora.sh all "
            "ou execute este piloto no EASNCloud, dentro do clone que contém data/audio/."
        )

    corpus_rows, corpus_failures = [], []
    for corpus in C.DATASETS:
        clips = pd.read_csv(clip_paths[corpus])
        clips = clips[clips.audio_available.astype(bool)].head(MAX_CLIPS_PER_CORPUS)
        if clips.empty:
            print(f"{corpus}: sem áudio disponível no recorte; corpus ignorado.")
            continue
        for _, clip in clips.iterrows():
            x = load_clip(clip.path, clip.start)
            for codec_name in PILOT_CODECS:
                for params in C.CODECS.get(codec_name, []):
                    setting_label = ", ".join(f"{key}={value}" for key, value in params.items())
                    print(f"Executando {corpus} | {clip.clip_id} | {codec_name} | {setting_label}")
                    try:
                        codec = make_codec(codec_name, params, dataset=corpus)
                        encoded = codec.encode(x)
                        y, lag = align_to_reference(x, codec.decode(encoded))
                        corpus_rows.append({"corpus": corpus, "clip_id": clip.clip_id, "codec": codec_name,
                                            "configuração": codec.setting, "payload_bytes": encoded.nbytes,
                                            "lag": lag, "snr_db": snr_db(x, y), **mel_similarity(x, y)})
                    except Exception as exc:
                        corpus_failures.append({"corpus": corpus, "clip_id": clip.clip_id,
                                                "codec": codec_name, "configuração": str(params),
                                                "erro": f"{type(exc).__name__}: {exc}"})
    corpus_pilot = pd.DataFrame(corpus_rows)
    corpus_errors = pd.DataFrame(corpus_failures)
    corpus_pilot.to_csv(DERIVED / "codec_pilot.csv", index=False)
    display(corpus_pilot)
    if not corpus_errors.empty:
        display(corpus_errors)
else:
    print("Piloto dos corpora não executado. Defina RUN_CORPUS_PILOT=True após instalar os WAVs.")
"""),
        ],
    ),
    "05_reconstruction_metrics.ipynb": notebook(
        "05 — Payload e qualidade da reconstrução",
        "Executa ou lê a avaliação pareada de bytes, compressão, SNR e similaridade log-Mel para exatamente as mesmas janelas.",
        [
            code("RUN_FULL_EVALUATION = False  # altere somente no EASNCloud após validar o notebook 04"),
            code(r"""
import pandas as pd
import config as C

if RUN_FULL_EVALUATION:
    from run_codec_eval import main as run_evaluation
    run_evaluation()

results_path = C.OUT_DIR / "results.csv"
if results_path.is_file():
    results = pd.read_csv(results_path)
    display(results.head())
else:
    results = pd.DataFrame()
    print(f"Ainda não há resultados em {results_path.relative_to(ROOT)}")
"""),
            code(r"""
if not results.empty:
    summary = (results[results.codec != "baseline"]
               .groupby(["dataset", "codec", "setting"], as_index=False)
               .agg(bytes_mean=("nbytes", "mean"), bitrate_kbps=("bitrate_kbps", "mean"),
                    snr_db=("snr_db", "mean"), mel_rmse_db=("mel_rmse_db", "mean"),
                    mel_ssim=("mel_ssim", "mean")))
    summary.to_csv(REPORTS / "reconstruction_summary.csv", index=False)
    display(summary)
"""),
            code(r"""
if not results.empty:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    for (dataset, codec), group in summary.groupby(["dataset", "codec"]):
        ax.scatter(group.bitrate_kbps, group.mel_ssim, label=f"{dataset} — {codec}")
    ax.set_xscale("log"); ax.set(xlabel="bitrate efetivo (kbit/s)", ylabel="SSIM log-Mel")
    ax.legend(fontsize=7); fig.tight_layout(); plt.show()
"""),
        ],
    ),
    "06_acoustic_indices.ipynb": notebook(
        "06 — Preservação dos índices ecoacústicos",
        "Apresenta a configuração congelada e calcula erros e correlações pareadas por índice, corpus, codec e bitrate.",
        [
            code(r"""
from pathlib import Path
print((ROOT / "configs" / "acoustic-indices-v1.yaml").read_text())
"""),
            code(r"""
import pandas as pd
import config as C
results_path = C.OUT_DIR / "results.csv"
results = pd.read_csv(results_path) if results_path.is_file() else pd.DataFrame()
index_columns = [column for column in results.columns if column.startswith("idx_")]
print("Índices disponíveis:", index_columns)
"""),
            code(r"""
if not results.empty and index_columns:
    from analyze import ccc
    from scipy.stats import pearsonr, spearmanr
    import numpy as np

    baseline = results[results.codec == "baseline"][["dataset", "clip_id", *index_columns]]
    paired = results[results.codec != "baseline"].merge(baseline, on=["dataset", "clip_id"], suffixes=("", "_reference"))
    rows = []
    for keys, group in paired.groupby(["dataset", "codec", "setting"]):
        for index in index_columns:
            a = group[f"{index}_reference"].to_numpy(float); b = group[index].to_numpy(float)
            ok = np.isfinite(a) & np.isfinite(b)
            if ok.sum() > 3 and a[ok].std() > 0 and b[ok].std() > 0:
                rows.append({"dataset": keys[0], "codec": keys[1], "setting": keys[2], "index": index,
                             "signed_error": float(np.mean(b[ok] - a[ok])),
                             "absolute_error": float(np.mean(np.abs(b[ok] - a[ok]))),
                             "pearson": pearsonr(a[ok], b[ok]).statistic,
                             "spearman": spearmanr(a[ok], b[ok]).statistic,
                             "ccc": ccc(a[ok], b[ok])})
    index_summary = pd.DataFrame(rows)
    index_summary.to_csv(REPORTS / "index_preservation.csv", index=False)
    display(index_summary)
else:
    print("Execute primeiro a avaliação do notebook 05.")
"""),
        ],
    ),
    "07_foundation_models.ipynb": notebook(
        "07 — BirdNET, Perch e utilidade preditiva",
        "Audita as entradas dos modelos e reúne macro-F1, acurácia balanceada, matrizes de confusão e distâncias entre embeddings.",
        [
            code(r"""
RUN_CLASSIFICATION = False
MODELS = ["birdnet", "perch"]
HEAD = "linear"
"""),
            code(r"""
from codeco.protocol import adapter_audit
import pandas as pd
pd.Series(adapter_audit(), name="valor").to_frame()
"""),
            code(r"""
if RUN_CLASSIFICATION:
    from classify_embeddings import run_classification
    import config as C
    for corpus in C.DATASETS:
        for model in MODELS:
            run_classification(corpus, model, HEAD)
else:
    print("Classificação não executada. O padding do Perch já foi validado acima.")
"""),
            code(r"""
import config as C
tables = []
for corpus in C.DATASETS:
    for model in MODELS:
        path = C.OUT_DIR / f"classification_{corpus}_{model}_{HEAD}.csv"
        if path.is_file():
            table = pd.read_csv(path); table["source_file"] = path.name; tables.append(table)
classification = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
if classification.empty:
    print("Nenhum resultado de classificação disponível.")
else:
    classification.to_csv(REPORTS / "foundation_model_summary.csv", index=False)
    display(classification)
"""),
        ],
    ),
    "08_compute_cost.ipynb": notebook(
        "08 — Custo computacional",
        "Separa FLOPs de redes neurais de tempo/instruções dos codecs tradicionais e mantém a validação embarcada como etapa secundária.",
        [
            code(r"""
RUN_COMPUTE_BENCHMARK = False
USE_LINUX_PERF = False
"""),
            markdown("## Estado dos autoencoders\n\nAs variantes do Gustavo permanecem no repositório para revisão, mas estão desativadas por padrão. A correção de comprimento e a serialização real do payload foram deliberadamente adiadas para uma contribuição separada do Filipe."),
            code(r"""
if RUN_COMPUTE_BENCHMARK:
    from flops import run
    run(use_perf=USE_LINUX_PERF)
"""),
            code(r"""
import pandas as pd
import config as C
path = C.OUT_DIR / "flops.csv"
compute = pd.read_csv(path) if path.is_file() else pd.DataFrame()
if compute.empty:
    print("Benchmark ainda não executado.")
else:
    compute.to_csv(REPORTS / "compute_cost_summary.csv", index=False)
    display(compute)
"""),
            markdown("## Validação secundária no nó\n\nSomente os finalistas serão avaliados em ESP/nRF quanto a RAM, flash, tempo e energia. Os limites de 256 KiB de RAM e 1 MiB de flash não excluem métodos nesta primeira fase."),
        ],
    ),
    "09_summary.ipynb": notebook(
        "09 — Síntese e fronteiras de compromisso",
        "Reúne os artefatos produzidos nos notebooks anteriores sem recalcular os experimentos e explicita quais evidências ainda faltam.",
        [
            code(r"""
import pandas as pd

artifacts = {
    "corpora": REPORTS / "corpus_summary.csv",
    "reconstruction": REPORTS / "reconstruction_summary.csv",
    "indices": REPORTS / "index_preservation.csv",
    "classification": REPORTS / "foundation_model_summary.csv",
    "compute": REPORTS / "compute_cost_summary.csv",
}
status = pd.DataFrame([
    {"etapa": name, "arquivo": str(path.relative_to(ROOT)), "disponível": path.is_file()}
    for name, path in artifacts.items()
])
status
"""),
            code(r"""
loaded = {name: pd.read_csv(path) for name, path in artifacts.items() if path.is_file()}
for name, table in loaded.items():
    print(f"\n{name}: {len(table)} linhas")
    display(table.head())
"""),
            markdown("## Critério de conclusão\n\nA síntese final deverá cruzar payload efetivo, preservação dos índices, macro-F1/acurácia balanceada, distância dos embeddings e custo computacional. Autoencoders só entram nessa fronteira depois da correção e da serialização auditável do código latente."),
        ],
    ),
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for filename, payload in NOTEBOOKS.items():
        path = OUT / filename
        path.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
