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
        """Este notebook define quais classes têm suporte suficiente para participar do experimento sem consultar validação ou teste.

Para cada classe $i$, contamos somente suas janelas de treino, $c_i$, e trabalhamos com $z_i=\\log(1+c_i)$. A transformação reduz a assimetria da distribuição: uma classe com milhares de exemplos deixa de dominar numericamente outra com dezenas. Calculamos então a média $\\mu_z$ e o desvio-padrão $\\sigma_z$ dessas contagens transformadas e definimos o corte

$$T_z = \\mu_z-k\\sigma_z, \\qquad c_{\\min}=\\lceil e^{T_z}-1\\rceil.$$

Aqui, **$k$ é o número de desvios-padrão que deslocamos o corte para baixo da média**. Ele não é o número de classes, não é aprendido pelo classificador e não tem relação com k-NN ou k-means. Com $k=0$, o corte fica na média das contagens em escala logarítmica. Com $k=1$, aceitamos classes até um desvio-padrão abaixo dela. Aumentar $k$ reduz $c_{\\min}$ e inclui classes mais raras; diminuir $k$ torna o recorte mais conservador. Independentemente de $k$, uma classe ainda precisa aparecer no número mínimo de grupos de gravação, evitando selecionar uma espécie sustentada por muitas janelas de uma única origem.

O notebook mostra a análise de sensibilidade para vários valores de $k$, aplica o valor escolhido usando apenas o treino e congela o recorte. Depois, o treino é subamostrado de forma determinística para uma cota comum; validação e teste mantêm sua distribuição natural. A média e o desvio-padrão são usados como uma regra operacional interpretável na escala $\\log(1+c)$ — não como afirmação de que as contagens originais seguem perfeitamente uma Gaussiana.""",
        [
            markdown("""## Parâmetros editáveis

- `K_SIGMA` controla quão abaixo da média logarítmica colocamos o corte. **Quanto maior, mais inclusivo**: entram classes menos frequentes e a cota comum do treino diminui.
- `MINIMUM_GROUPS` é uma proteção independente contra pseudorreplicação: uma classe precisa ocorrer em pelo menos esse número de grupos de gravação.
- `SENSITIVITY_K` contém apenas os valores comparados na análise de sensibilidade; ele não executa uma busca orientada pelo teste.
- `SEED` torna a subamostragem do treino reproduzível.

O valor final de `K_SIGMA` deve ser decidido observando suporte e diversidade no **treino**, antes de examinar o desempenho no teste. Validação e teste não são balanceados artificialmente."""),
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
        "Executa ou lê a avaliação pareada do payload e da reconstrução. A unidade experimental é uma janela de teste de 4 s: o áudio original e todas as condições de codec são comparados usando o mesmo conteúdo.",
        [
            markdown(r"""
## O que este notebook faz

Para cada janela disponível no **split de teste**, o runner registra primeiro a referência PCM (`baseline/orig`) e depois uma linha para cada combinação de codec e configuração. Assim, uma linha não representa uma espécie nem um arquivo físico inteiro: representa **uma janela de 4 s sob uma condição de codificação**.

A chave que deve identificar uma linha de forma única é:

`dataset + clip_id + codec + setting`

O arquivo é incremental. Se a execução for interrompida, as chaves já gravadas são mantidas e o runner continua nas ausentes. Portanto, um `results.csv` existente pode ser apenas parcial enquanto o processamento não terminar. Se o protocolo ou a matriz de codecs mudar, resultados antigos não devem ser misturados silenciosamente com os novos.
"""),
            markdown(r"""
## Controle da execução

- `False`: não processa áudio; apenas lê e descreve `data/derived/experiment/results.csv`, se ele existir.
- `True`: percorre as janelas de teste com áudio disponível e acrescenta ao CSV as condições ainda ausentes.

O limite atual é de até **200 janelas por corpus**, amostradas de forma determinística com a semente 42. O notebook 04 deve passar antes desta execução completa.
"""),
            code("RUN_FULL_EVALUATION = False  # altere somente no EASNCloud após validar o notebook 04"),
            markdown(r"""
## Tabela detalhada: uma linha por janela e condição

O artefato primário é `data/derived/experiment/results.csv`. As colunas têm os seguintes significados:

| Coluna | Unidade | Interpretação |
|---|---:|---|
| `dataset` | — | Corpus de origem (`ecosoundset` ou `pteroset_ppa1`). |
| `clip_id` | — | Identificador estável da janela no manifest. É a unidade usada para parear original e reconstruções. |
| `codec` | — | Método avaliado. `baseline` é o áudio PCM de referência, sem compressão. |
| `setting` | — | Parametrização concreta, como bitrate, nível FLAC ou largura de banda do EnCodec. Duas configurações do mesmo codec são condições diferentes. |
| `nbytes` | bytes/janela | Payload de uma janela de 4 s. Para FFmpeg, é o tamanho real do arquivo codificado, incluindo o contêiner. Para EnCodec, é uma estimativa do código empacotado e escalas, não um arquivo serializado completo. Não inclui tamanho do modelo nem cabeçalhos de rede. A baseline PCM16 mono tem 384.000 bytes. |
| `bitrate_kbps` | kbit/s | Bitrate efetivo calculado como `nbytes × 8 / 4 s / 1000`; não é apenas o bitrate nominal solicitado ao encoder. Menor significa menor transmissão por segundo. |
| `compression_ratio` | razão | `bytes da baseline / nbytes`. Valor 1 significa nenhum ganho; 4 significa payload quatro vezes menor. Maior é melhor apenas quanto a tamanho. |
| `lag` | amostras | Deslocamento temporal removido antes das métricas. A 48 kHz, 48 amostras correspondem a 1 ms. Não é uma métrica de qualidade; documenta a compensação do atraso do codec. |
| `snr_db` | dB | Relação entre energia do sinal original e energia do erro após alinhamento. Maior indica menor erro de forma de onda. Pode ser negativa; `inf` aparece na baseline porque o erro é zero. |
| `mel_mae_db` | dB por célula | Média do erro absoluto entre os dois log-Mel espectrogramas. Menor é melhor. |
| `mel_rmse_db` | dB por célula | Raiz do erro quadrático médio no log-Mel; penaliza mais os erros grandes. Menor é melhor. |
| `mel_ssim` | adimensional | Similaridade estrutural entre os log-Mel espectrogramas. 1 indica identidade; maior é melhor. |
| `mel_cosine` | adimensional | Cosseno entre os log-Mel vetorizados após deslocamento do piso de −100 dB. Próximo de 1 indica orientação espectral semelhante, mas não garante igualdade de energia. |
| `idx_*` | depende do índice | Índices ecoacústicos calculados no original ou na reconstrução. Eles são preservados aqui por janela, mas a comparação pareada é feita no notebook 06. |

As métricas Mel usam potência (`power=2`), 128 bandas, FFT de 2.048 amostras, hop de 512 amostras, faixa de 0 a 24 kHz e piso de −100 dB. Todas são calculadas **depois** do alinhamento temporal.
"""),
            code(r"""
import pandas as pd
import config as C

if RUN_FULL_EVALUATION:
    from run_codec_eval import main as run_evaluation
    run_evaluation()

results_path = C.OUT_DIR / "results.csv"
if results_path.is_file():
    results = pd.read_csv(results_path, dtype={"clip_id": str})
    key_columns = ["dataset", "clip_id", "codec", "setting"]
    missing_columns = [column for column in key_columns if column not in results]
    if missing_columns:
        raise ValueError(f"results.csv incompatível; colunas ausentes: {missing_columns}")
    duplicates = results.duplicated(key_columns, keep=False)
    if duplicates.any():
        raise ValueError(f"Há {int(duplicates.sum())} linhas com chave experimental repetida.")

    coverage = (results.groupby(["dataset", "codec", "setting"], as_index=False)
                .agg(linhas=("clip_id", "size"),
                     janelas_unicas=("clip_id", "nunique")))
    print(f"Artefato detalhado: {results_path.relative_to(ROOT)}")
    print(f"{len(results):,} linhas; {results['clip_id'].nunique():,} clip_ids distintos")
    display(coverage)
    display(results.head(10))
else:
    results = pd.DataFrame()
    print(f"Ainda não há resultados em {results_path.relative_to(ROOT)}")
"""),
            markdown(r"""
### Como ler as duas saídas acima

- `coverage` tem uma linha por **corpus × codec × configuração**. `linhas` é o número de avaliações gravadas e `janelas_unicas` é o número de janelas distintas. Esses valores devem coincidir dentro de cada condição; diferenças entre condições indicam execução parcial.
- A segunda tabela mostra as primeiras dez linhas do artefato detalhado. Para um mesmo `dataset + clip_id`, compare a linha `baseline/orig` com as linhas dos codecs. Não se deve comparar uma janela de um corpus com outra janela diferente.
- Células ausentes (`NaN`) não significam automaticamente qualidade zero. Elas indicam métrica não calculada, falha anterior ou artefato incompatível e precisam ser investigadas antes da agregação.
"""),
            markdown(r"""
## Resumo por condição

A tabela seguinte agrega as janelas, mas mantém separados corpus, codec e configuração. Cada linha passa a representar **uma condição experimental inteira**, e não mais uma janela.

- `n_windows`: janelas distintas que contribuíram para a linha;
- colunas terminadas em `_mean`: média entre essas janelas;
- colunas terminadas em `_std`: desvio-padrão amostral entre janelas, isto é, a variabilidade do resultado dentro da condição;
- `compression_ratio_mean`: redução média relativa aos 384.000 bytes da baseline.

O resumo exclui a baseline porque suas métricas de reconstrução são triviais (`SNR = inf`, erros Mel = 0 e similaridades = 1). O CSV detalhado continua contendo a baseline para os pareamentos posteriores.
"""),
            code(r"""
if not results.empty:
    summary = (results[results.codec != "baseline"]
               .groupby(["dataset", "codec", "setting"], as_index=False, dropna=False)
               .agg(n_windows=("clip_id", "nunique"),
                    bytes_mean=("nbytes", "mean"), bytes_std=("nbytes", "std"),
                    bitrate_kbps_mean=("bitrate_kbps", "mean"),
                    compression_ratio_mean=("compression_ratio", "mean"),
                    snr_db_mean=("snr_db", "mean"), snr_db_std=("snr_db", "std"),
                    mel_mae_db_mean=("mel_mae_db", "mean"),
                    mel_rmse_db_mean=("mel_rmse_db", "mean"),
                    mel_rmse_db_std=("mel_rmse_db", "std"),
                    mel_ssim_mean=("mel_ssim", "mean"), mel_ssim_std=("mel_ssim", "std"),
                    mel_cosine_mean=("mel_cosine", "mean")))
    summary_path = REPORTS / "reconstruction_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Resumo por condição: {summary_path.relative_to(ROOT)}")
    display(summary)
"""),
            markdown(r"""
## Curva payload–distorção

Cada ponto do gráfico representa **uma linha da tabela-resumo**: a média de uma configuração de codec em um corpus. Não são janelas individuais.

- eixo x: bitrate efetivo médio em escala logarítmica; mais à esquerda significa menos dados transmitidos;
- eixo y: SSIM log-Mel médio; mais acima significa maior preservação da estrutura tempo–frequência;
- a região desejável é o canto superior esquerdo;
- linhas apenas conectam configurações do mesmo codec e corpus para facilitar a leitura; não representam uma interpolação medida;
- o gráfico é descritivo. Os desvios-padrão estão na tabela e não constituem intervalos de confiança.
"""),
            code(r"""
if not results.empty:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    for (dataset, codec), group in summary.groupby(["dataset", "codec"]):
        group = group.sort_values("bitrate_kbps_mean")
        ax.plot(group.bitrate_kbps_mean, group.mel_ssim_mean, "o-",
                label=f"{dataset} — {codec}")
    ax.set_xscale("log")
    ax.set(xlabel="bitrate efetivo médio (kbit/s)", ylabel="SSIM log-Mel médio")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    plt.show()
"""),
        ],
    ),
    "06_acoustic_indices.ipynb": notebook(
        "06 — Preservação dos índices ecoacústicos",
        "Compara, janela a janela, os índices calculados no áudio original e no áudio reconstruído. A análise separa erro absoluto, viés, preservação da ordenação e concordância, porque correlação isolada não demonstra que os valores foram preservados.",
        [
            markdown(r"""
## Pergunta e unidade de análise

Para cada janela de teste de 4 s, calculamos os mesmos índices duas vezes:

1. no áudio PCM de referência (`baseline/orig`);
2. no áudio reconstruído por uma configuração de codec.

O pareamento usa `dataset + clip_id`. Portanto, cada linha da tabela pareada representa **a mesma janela sob uma condição de codificação**, e não médias de espécies ou gravações inteiras. A condição é definida por `codec + setting`.

O objetivo não é exigir que todos os índices tenham a mesma escala. ACI, BIO, entropias e NDSI possuem domínios diferentes; por isso, erros brutos são interpretados dentro de cada índice e a visualização comparativa usa erro normalizado pelo desvio-padrão da referência.
"""),
            markdown(r"""
## Definições congeladas

O YAML abaixo é a especificação normativa usada pelo código: frequência de amostragem, STFT, bandas, limiares, fórmulas e casos especiais. Alterá-lo muda o experimento.

Resumo dos índices:

| Coluna | Domínio típico | O que resume | Papel no protocolo |
|---|---:|---|---|
| `idx_ACI` | ≥ 0 | Variação temporal da magnitude dentro de cada bin espectral. | Primário |
| `idx_ADI` | 0 a 1 | Diversidade de ocupação entre bandas de 1 kHz. | Primário |
| `idx_AEI` | 0 a 1 | Desigualdade da ocupação das bandas, via Gini. | Primário |
| `idx_Ht` | 0 a 1 | Entropia temporal do envelope RMS. | Componente reportado |
| `idx_Hf` | 0 a 1 | Entropia espectral da potência média. | Componente reportado |
| `idx_H` | 0 a 1 | Entropia acústica composta: `Ht × Hf`. | Primário |
| `idx_BIO_2_24` | ≥ 0 | Área espectral acima do nível mínimo entre 2 e 24 kHz. | Primário |
| `idx_BIO_2_8` | ≥ 0 | Versão entre 2 e 8 kHz, mantida como ponte com a literatura. | Sensibilidade |
| `idx_NDSI_CLASSIC` | −1 a 1 | Contraste entre 1–2 kHz e 2–11 kHz. | Sensibilidade |
| `idx_NDSI_EXTENDED` | −1 a 1 | Contraste entre 1–2 kHz e 2–24 kHz. | Sensibilidade |

Esses valores não possuem uma interpretação ecológica universal por janela. Aqui eles são usados para medir se a transformação do áudio altera uma estatística definida do sinal.
"""),
            code(r"""
from pathlib import Path
indices_config_path = ROOT / "configs" / "acoustic-indices-v1.yaml"
print(f"Configuração normativa: {indices_config_path.relative_to(ROOT)}\n")
print(indices_config_path.read_text())
"""),
            markdown(r"""
## Entrada e cobertura do pareamento

O notebook lê o mesmo `results.csv` produzido no notebook 05. Antes de calcular estatísticas, ele verifica:

- se existe no máximo uma baseline para cada `dataset + clip_id`;
- se cada reconstrução encontra sua janela de referência;
- quantas janelas de referência realmente aparecem em cada condição.

Na tabela `pair_coverage`, cada linha é uma condição em um corpus:

- `paired_rows`: pares reconstrução–referência disponíveis;
- `unique_windows`: janelas distintas nesses pares;
- `baseline_windows`: total de janelas de referência daquele corpus;
- `coverage_percent`: porcentagem das baselines coberta pela condição;
- `bitrate_kbps_mean`: bitrate efetivo médio da condição.

Cobertura inferior a 100% indica avaliação parcial ou falha em alguma condição; não deve ser confundida com desempenho do codec.
"""),
            code(r"""
import numpy as np
import pandas as pd
import config as C

results_path = C.OUT_DIR / "results.csv"
results = (pd.read_csv(results_path, dtype={"clip_id": str})
           if results_path.is_file() else pd.DataFrame())
index_columns = [column for column in results.columns if column.startswith("idx_")]

if results.empty:
    print("Execute primeiro a avaliação do notebook 05.")
else:
    required = {"dataset", "clip_id", "codec", "setting", "bitrate_kbps", *index_columns}
    missing = sorted(required - set(results.columns))
    if missing:
        raise ValueError(f"results.csv incompatível; colunas ausentes: {missing}")
    baseline = results.loc[results.codec == "baseline", ["dataset", "clip_id", *index_columns]].copy()
    duplicated_baseline = baseline.duplicated(["dataset", "clip_id"], keep=False)
    if duplicated_baseline.any():
        raise ValueError(f"Há {int(duplicated_baseline.sum())} baselines duplicadas.")
    reconstructed = results.loc[results.codec != "baseline"].copy()
    paired = reconstructed.merge(
        baseline,
        on=["dataset", "clip_id"],
        how="inner",
        suffixes=("", "_reference"),
        validate="many_to_one",
    )
    baseline_counts = (baseline.groupby("dataset", as_index=False)
                       .agg(baseline_windows=("clip_id", "nunique")))
    pair_coverage = (paired.groupby(["dataset", "codec", "setting"], as_index=False)
                     .agg(paired_rows=("clip_id", "size"),
                          unique_windows=("clip_id", "nunique"),
                          bitrate_kbps_mean=("bitrate_kbps", "mean"))
                     .merge(baseline_counts, on="dataset", validate="many_to_one"))
    pair_coverage["coverage_percent"] = (
        100 * pair_coverage["unique_windows"] / pair_coverage["baseline_windows"]
    )
    print(f"Entrada: {results_path.relative_to(ROOT)}")
    print("Índices encontrados:", ", ".join(index_columns))
    display(pair_coverage)
"""),
            markdown(r"""
## Tabela de preservação por condição e índice

Cada linha de `index_summary` resume **um corpus × codec × configuração × índice**. As colunas significam:

| Coluna | Interpretação |
|---|---|
| `n_pairs` | Número de janelas com valores finitos na referência e na reconstrução. |
| `reference_mean`, `reference_std` | Média e desvio-padrão do índice no áudio original. Definem a escala observada naquele corpus. |
| `reconstructed_mean`, `reconstructed_std` | Média e dispersão depois da codificação e reconstrução. |
| `bias` | Média de `reconstruído − referência`. Positivo indica deslocamento para cima; negativo, para baixo. Pode ser próximo de zero mesmo com erros grandes que se cancelam. |
| `mae` | Média de `|reconstruído − referência|`. Mede o tamanho típico do erro na unidade original do índice. Menor é melhor. |
| `rmse` | Raiz do erro quadrático médio. Dá maior peso a janelas com erros extremos. Menor é melhor. |
| `standardized_bias` | `bias / desvio-padrão da referência`. Permite comparar a magnitude do deslocamento entre índices. Zero é o alvo. |
| `normalized_mae` | `MAE / desvio-padrão da referência`. Zero é o alvo; 1 significa erro médio do tamanho de um desvio-padrão natural da referência. |
| `pearson` | Preservação de relação linear. Pode ser alta mesmo com deslocamento ou mudança de escala. |
| `spearman` | Preservação da ordenação das janelas. Pode ser alta mesmo que os valores absolutos estejam errados. |
| `ccc` | Coeficiente de concordância de Lin. Combina correlação, viés e diferença de escala; 1 representa concordância perfeita. |

Pearson, Spearman e CCC ficam `NaN` quando há menos de quatro pares válidos ou variância nula. Não usamos erro percentual como métrica principal porque vários índices podem assumir zero ou, no caso do NDSI, valores negativos.
"""),
            code(r"""
if not results.empty and index_columns:
    from scipy.stats import pearsonr, spearmanr

    rows = []
    for keys, group in paired.groupby(["dataset", "codec", "setting"]):
        for index in index_columns:
            a = group[f"{index}_reference"].to_numpy(float)
            b = group[index].to_numpy(float)
            ok = np.isfinite(a) & np.isfinite(b)
            if not ok.any():
                continue
            reference, decoded = a[ok], b[ok]
            delta = decoded - reference
            reference_std = float(reference.std(ddof=0))
            decoded_std = float(decoded.std(ddof=0))
            correlations_valid = len(reference) > 3 and reference_std > 0 and decoded_std > 0
            pearson_value = float(pearsonr(reference, decoded).statistic) if correlations_valid else np.nan
            spearman_value = float(spearmanr(reference, decoded).statistic) if correlations_valid else np.nan
            if correlations_valid:
                covariance = float(np.cov(reference, decoded, ddof=0)[0, 1])
                denominator = reference.var() + decoded.var() + (reference.mean() - decoded.mean()) ** 2
                ccc_value = float(2 * covariance / denominator) if denominator > 0 else np.nan
            else:
                ccc_value = np.nan
            bias = float(delta.mean())
            mae = float(np.abs(delta).mean())
            rows.append({
                "dataset": keys[0], "codec": keys[1], "setting": keys[2], "index": index,
                "n_pairs": int(len(reference)),
                "bitrate_kbps_mean": float(group.loc[ok, "bitrate_kbps"].mean()),
                "reference_mean": float(reference.mean()), "reference_std": reference_std,
                "reconstructed_mean": float(decoded.mean()), "reconstructed_std": decoded_std,
                "bias": bias, "mae": mae, "rmse": float(np.sqrt(np.mean(delta ** 2))),
                "standardized_bias": bias / reference_std if reference_std > 0 else np.nan,
                "normalized_mae": mae / reference_std if reference_std > 0 else np.nan,
                "pearson": pearson_value, "spearman": spearman_value, "ccc": ccc_value,
            })
    index_summary = pd.DataFrame(rows)
    summary_path = REPORTS / "index_preservation.csv"
    index_summary.to_csv(summary_path, index=False)
    print(f"Resumo por condição e índice: {summary_path.relative_to(ROOT)}")
    display(index_summary)
else:
    print("Execute primeiro a avaliação do notebook 05.")
"""),
            markdown(r"""
## Diagnósticos ordenados

As tabelas abaixo não impõem um limiar arbitrário de “aceitável”. Elas apenas destacam onde olhar:

1. `largest_normalized_errors`: maiores erros absolutos após normalização pela variabilidade natural do índice;
2. `largest_agreement_gaps`: casos em que Spearman permanece alto, mas CCC cai. Isso sugere que a ordenação das janelas foi preservada melhor que seus valores absolutos, geralmente por viés ou mudança de escala;
3. `best_ccc_by_index`: maior concordância observada para cada corpus e índice. Serve como referência interna, não como prova de equivalência.
"""),
            code(r"""
if not results.empty and not index_summary.empty:
    diagnostic_columns = [
        "dataset", "codec", "setting", "index", "n_pairs",
        "bitrate_kbps_mean", "standardized_bias", "normalized_mae",
        "pearson", "spearman", "ccc",
    ]
    largest_normalized_errors = (index_summary
        .sort_values("normalized_mae", ascending=False)
        .loc[:, diagnostic_columns]
        .head(15))
    agreement_gaps = index_summary.assign(agreement_gap=index_summary.spearman - index_summary.ccc)
    largest_agreement_gaps = (agreement_gaps
        .sort_values("agreement_gap", ascending=False)
        .loc[:, [*diagnostic_columns, "agreement_gap"]]
        .head(15))
    best_ccc_by_index = (index_summary.dropna(subset=["ccc"])
        .sort_values("ccc", ascending=False)
        .groupby(["dataset", "index"], as_index=False)
        .head(1)
        .loc[:, diagnostic_columns])

    print("Maiores erros normalizados")
    display(largest_normalized_errors)
    print("Ordenação preservada melhor que concordância absoluta")
    display(largest_agreement_gaps)
    print("Maior CCC observado em cada corpus e índice")
    display(best_ccc_by_index)
"""),
            markdown(r"""
## Figura 1 — mapa de concordância e erro normalizado

Para cada corpus, as linhas são configurações de codec e as colunas são índices:

- painel esquerdo: CCC, com 1 como concordância perfeita, 0 como ausência de concordância e valores negativos como discordância;
- painel direito: MAE normalizado pelo desvio-padrão da referência, com 0 como ausência de erro.

O painel de erro é limitado visualmente no percentil 95 para impedir que poucos valores extremos apaguem o contraste do restante. Os valores exatos permanecem em `index_preservation.csv`. Células cinzas indicam estatística indisponível.
"""),
            code(r"""
if not results.empty and not index_summary.empty:
    import matplotlib.pyplot as plt

    figures_dir = REPORTS / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    preferred_order = [index for index in C.PLOT_INDICES if index in index_columns]
    for dataset, dataset_summary in index_summary.groupby("dataset"):
        work = dataset_summary.copy()
        work["condition"] = work["codec"] + " | " + work["setting"]
        ccc_matrix = work.pivot(index="condition", columns="index", values="ccc").reindex(columns=preferred_order)
        nmae_matrix = work.pivot(index="condition", columns="index", values="normalized_mae").reindex(
            index=ccc_matrix.index, columns=preferred_order
        )
        finite_nmae = nmae_matrix.to_numpy(float)[np.isfinite(nmae_matrix.to_numpy(float))]
        nmae_limit = max(float(np.quantile(finite_nmae, 0.95)), 1e-9) if finite_nmae.size else 1.0

        height = max(5.0, 0.34 * len(ccc_matrix.index))
        fig, axes = plt.subplots(1, 2, figsize=(16, height), constrained_layout=True)
        ccc_image = axes[0].imshow(
            np.ma.masked_invalid(ccc_matrix.to_numpy(float)), aspect="auto",
            cmap="coolwarm", vmin=-1, vmax=1,
        )
        nmae_image = axes[1].imshow(
            np.ma.masked_invalid(nmae_matrix.to_numpy(float)), aspect="auto",
            cmap="viridis", vmin=0, vmax=nmae_limit,
        )
        for axis, title in zip(axes, ("CCC (maior é melhor)", "MAE normalizado (menor é melhor)")):
            axis.set_title(title)
            axis.set_xticks(range(len(preferred_order)), [name.removeprefix("idx_") for name in preferred_order], rotation=45, ha="right")
            axis.set_yticks(range(len(ccc_matrix.index)), ccc_matrix.index, fontsize=7)
            axis.set_xlabel("índice ecoacústico")
        axes[0].set_ylabel("codec | configuração")
        fig.colorbar(ccc_image, ax=axes[0], shrink=0.8)
        fig.colorbar(nmae_image, ax=axes[1], shrink=0.8, label=f"escala visual até P95 = {nmae_limit:.3g}")
        fig.suptitle(f"Preservação dos índices — {dataset}")
        output = figures_dir / f"index_preservation_heatmap_{dataset}.png"
        fig.savefig(output, dpi=160, bbox_inches="tight")
        print(f"Figura: {output.relative_to(ROOT)}")
        plt.show()
"""),
            markdown(r"""
## Figura 2 — referência contra reconstrução para toda a matriz

A célula abaixo gera automaticamente **uma figura para cada corpus × índice**. Dentro de cada figura há um painel para cada combinação de codec e configuração disponível. Portanto, uma única execução cobre todos os índices e bitrates presentes em `results.csv`.

- eixo x: índice no áudio original;
- eixo y: índice após reconstrução;
- linha tracejada: igualdade perfeita (`y = x`);
- nuvem deslocada paralelamente à diagonal: viés;
- inclinação diferente da diagonal: mudança de escala;
- dispersão ao redor da diagonal: erro variável entre janelas;
- poucos pontos distantes: janelas especialmente sensíveis;
- o título de cada painel informa configuração, bitrate efetivo médio, CCC e MAE normalizado.

`SCATTER_INDICES` recebe por padrão todos os índices encontrados. Ele pode ser reduzido apenas para uma inspeção rápida. `DISPLAY_SCATTERS=False` continua gerando e salvando todas as figuras, mas evita incorporá-las como saída visual do notebook — útil para não criar um `.ipynb` muito pesado.
"""),
            code(r"""
SCATTER_INDICES = list(index_columns)
DISPLAY_SCATTERS = True

if not results.empty and SCATTER_INDICES:
    import matplotlib.pyplot as plt

    unknown_indices = sorted(set(SCATTER_INDICES) - set(index_columns))
    if unknown_indices:
        raise ValueError(f"Índices inválidos: {unknown_indices}. Disponíveis: {index_columns}")

    condition_rates = (paired.groupby(["dataset", "codec", "setting"], as_index=False)
                       .agg(bitrate_kbps_mean=("bitrate_kbps", "mean")))
    figures_dir = REPORTS / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    scatter_outputs = []

    for dataset, dataset_conditions in condition_rates.groupby("dataset"):
        panels = list(dataset_conditions.sort_values(["codec", "bitrate_kbps_mean"]).itertuples(index=False))
        columns = min(4, len(panels))
        rows_count = int(np.ceil(len(panels) / columns))
        for scatter_index in SCATTER_INDICES:
            fig, axes = plt.subplots(
                rows_count, columns,
                figsize=(3.6 * columns, 3.45 * rows_count),
                squeeze=False,
            )
            for axis, condition in zip(axes.ravel(), panels):
                group = paired.loc[
                    (paired.dataset == dataset)
                    & (paired.codec == condition.codec)
                    & (paired.setting == condition.setting)
                ]
                reference = group[f"{scatter_index}_reference"].to_numpy(float)
                decoded = group[scatter_index].to_numpy(float)
                valid = np.isfinite(reference) & np.isfinite(decoded)
                reference, decoded = reference[valid], decoded[valid]
                axis.scatter(reference, decoded, s=12, alpha=0.5)
                if len(reference):
                    low = float(min(reference.min(), decoded.min()))
                    high = float(max(reference.max(), decoded.max()))
                    axis.plot([low, high], [low, high], "k--", linewidth=0.9)
                metrics_row = index_summary.loc[
                    (index_summary.dataset == dataset)
                    & (index_summary.codec == condition.codec)
                    & (index_summary.setting == condition.setting)
                    & (index_summary["index"] == scatter_index)
                ]
                if metrics_row.empty:
                    metric_label = "CCC/MAE-n indisponíveis"
                else:
                    metric = metrics_row.iloc[0]
                    metric_label = f"CCC={metric.ccc:.3f} | MAE-n={metric.normalized_mae:.3f}"
                axis.set_title(
                    f"{condition.codec} | {condition.setting}\n"
                    f"{condition.bitrate_kbps_mean:.1f} kbit/s | {metric_label}",
                    fontsize=7.5,
                )
                axis.set_xlabel("referência")
                axis.set_ylabel("reconstruído")
                axis.grid(True, alpha=0.2)
            for axis in axes.ravel()[len(panels):]:
                axis.set_visible(False)
            fig.suptitle(f"{dataset} — {scatter_index} — todas as condições", fontsize=12)
            fig.tight_layout()
            output = figures_dir / f"index_scatter_{dataset}_{scatter_index}.png"
            fig.savefig(output, dpi=160, bbox_inches="tight")
            scatter_outputs.append(output)
            if DISPLAY_SCATTERS:
                plt.show()
            plt.close(fig)

    print(f"{len(scatter_outputs)} figuras de dispersão geradas em {figures_dir.relative_to(ROOT)}")
    display(pd.DataFrame({"figura": [str(path.relative_to(ROOT)) for path in scatter_outputs]}))
elif not results.empty:
    print("Nenhum índice selecionado em SCATTER_INDICES.")
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
