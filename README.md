# CodEco Experiment

Pipeline reproduzível para avaliar o efeito de codecs de áudio em bioacústica.
O projeto compara a baseline PCM com reconstruções produzidas por codecs
tradicionais e neurais, mede fidelidade acústica, extrai embeddings de modelos
fundacionais e avalia classificação multirrótulo.

O pipeline foi organizado em etapas encadeadas, com cache, checkpoints,
retomada, logs e acompanhamento de progresso. Uma falha interrompe imediatamente
a execução e impede que etapas posteriores sejam iniciadas com dados
incompletos.

## Protocolo experimental

- Áudio mono com 4 segundos por clipe.
- Taxa de amostragem de 48 kHz.
- Baseline PCM de 16 bits.
- Datasets ECOSoundSet e PteroSet PPA1.
- Codecs FFmpeg: FLAC, MP3, AAC, Opus e LC3.
- Codec neural: EnCodec de 48 kHz nas bandas de 6, 12 e 24 kbps.
- Modelos de embeddings: BirdNET 2.4 e Google Perch v2.
- Classificação multirrótulo com MLP e os protocolos baseline e *matched*.
- Análise pareada por `dataset + clip_id` contra uma única baseline.

As definições do corpus ficam em `config/manifests.parquet`. O conjunto usado
pelas execuções autônomas fica em `config/standalone-examples.json`.

## Pipeline

O ponto de entrada executa as etapas nesta ordem:

```text
environment
    → datasets
    → flops
    → prepare_baseline
    → ffmpeg
    → encodec
    → embeddings
    → classify
    → analyze
    → graphs
```

| Etapa | Função |
| --- | --- |
| `environment` | Valida Python, `uv`, dependências, FFmpeg, codecs, CPU e GPUs. |
| `datasets` | Baixa, extrai e valida os datasets definidos pelo manifesto. |
| `flops` | Mede o custo computacional das condições experimentais. |
| `prepare_baseline` | Constrói o índice canônico e materializa a baseline PCM. |
| `ffmpeg` | Codifica, decodifica e calcula métricas para os codecs FFmpeg. |
| `encodec` | Executa EnCodec em GPU e calcula as mesmas métricas de reconstrução. |
| `embeddings` | Produz embeddings BirdNET e Perch para baseline e reconstruções. |
| `classify` | Treina e avalia o MLP conforme o protocolo experimental. |
| `analyze` | Calcula correlações e tabelas de taxa-distorção. |
| `graphs` | Gera dispersões baseline × codec e curvas de taxa-distorção. |

O estado de cada etapa é propagado pelo `Codeco.context`. As etapas de codecs e
embeddings recebem a mesma baseline preparada no início do experimento.

## Requisitos

- Python 3.12.
- [`uv`](https://docs.astral.sh/uv/) 0.13.8.
- FFmpeg com suporte aos encoders e decoders utilizados pelo experimento.
- GPU NVIDIA com CUDA para a etapa EnCodec. Não existe fallback de EnCodec para
  CPU.
- Espaço livre suficiente para datasets, modelos e resultados.
- Pelo menos 15 GiB livres no filesystem de `data/cache` para preparar a
  baseline PCM.

As dependências Python e suas versões estão registradas em `pyproject.toml` e
`uv.lock`.

## Instalação

Após clonar o repositório, entre em sua raiz e instale a versão requerida do
`uv`.

Linux:

```bash
curl -LsSf https://astral.sh/uv/0.13.8/install.sh | sh
```

Windows PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/0.13.8/install.ps1 | iex"
```

Sincronize o ambiente:

```bash
uv sync --locked
```

O comando cria a `.venv` do projeto. Não é necessário ativá-la para usar os
comandos documentados abaixo.

## Execução completa

Execute o pipeline científico completo:

```bash
.venv/bin/python -m codeco.main
```

O pipeline reutiliza downloads, modelos, caches e checkpoints compatíveis. Se
uma etapa falhar, corrija a causa e execute novamente o mesmo comando para
retomar o processamento.

Para descartar o estado gerado e iniciar uma nova execução científica:

```bash
.venv/bin/python -m codeco.main --clean
```

`--clean` remove:

- `data/cache/`;
- `data/checkpoints/`;
- `data/prepared/`;
- `data/samples/`;
- `results/`.

O comando preserva:

- `data/datasets/`;
- `data/downloads/`;
- `data/models/`;
- `data/tests/`.

## Datasets e modelos

`datasets()` verifica primeiro `data/datasets/`. Um dataset já extraído e
válido não é baixado novamente. Downloads incompletos permanecem em
`data/downloads/` para retomada; arquivos compactados são removidos depois da
extração e validação completas.

Os modelos são armazenados de forma persistente em:

```text
data/models/
├── birdnet/
├── huggingface/
└── tensorflow_hub/
```

Esses modelos não são removidos por `--clean`.

## Baseline, cache e checkpoints

`prepare_baseline()` cria `data/cache/clip_table.parquet` e materializa a
baseline em `data/cache/baseline_pcm/`. Cada dataset possui um arquivo NumPy
mapeável em memória, enquanto `index.parquet` relaciona `dataset + clip_id` à
posição correspondente.

O cache evita reler e reamostrar os WAVs em cada condição. FFmpeg, EnCodec e os
produtores de embeddings acessam o mesmo PCM preparado. O cache só é
reconstruído quando manifesto, protocolo ou sequência de clipes mudam.

Checkpoints ficam em `data/cache/checkpoints.sqlite3` ou nos diretórios
específicos da etapa. O processamento é registrado em lotes, permitindo retomar
execuções sem repetir resultados já concluídos.

Todos os caminhos de arquivos pertencentes ao projeto são persistidos em forma
relativa à raiz do repositório. Ao carregar caches, índices ou checkpoints, o
pipeline os resolve contra a raiz descoberta por `CodecoPath`, permitindo mover
ou copiar o checkout sem manter referências à localização anterior.

## Testes autônomos

Cada módulo pode ser executado separadamente sem iniciar `environment()`,
`datasets()` ou `prepare_baseline()`. As etapas que processam clipes usam os 80
exemplos reais e balanceados de `config/standalone-examples.json`:

- 40 clipes por dataset;
- duas classes por dataset;
- 20 clipes de treino, 10 de validação e 10 de teste por dataset.

O teste de FLOPs é independente desse conjunto e usa seu próprio sinal
determinístico de 4 segundos.

Para verificar o fluxo completo com esse conjunto, execute na ordem:

```bash
.venv/bin/python -m codeco.flops
.venv/bin/python -m codeco.ffmpeg
.venv/bin/python -m codeco.encodec
.venv/bin/python -m codeco.embeddings
.venv/bin/python -m codeco.classify
.venv/bin/python -m codeco.analyze
.venv/bin/python -m codeco.graph
```

As etapas dependentes reutilizam os artefatos autônomos correspondentes:
`classify` consome os índices de `embeddings`, `analyze` reúne as métricas de
FFmpeg e EnCodec, e `graph` consome a análise. `analyze` também verifica se a
baseline e todas as condições de codec contêm exatamente os 80 clipes
configurados; resultados parciais não são aceitos.

Todos os artefatos autônomos ficam sob `data/tests/`:

```text
data/tests/
├── flops/
├── ffmpeg/
├── encodec/
├── embeddings/
├── classify/
├── analyze/
└── graph/
```

Esses arquivos são descartáveis e não são misturados aos resultados
científicos.

## Resultados

A execução integrada grava saídas destinadas à inspeção humana sob `results/`:

```text
results/
├── benchmarks/
│   └── flops.csv
├── tables/
│   ├── ffmpeg_metrics.csv
│   ├── encodec_metrics.csv
│   ├── birdnet_embeddings.csv
│   ├── perch_embeddings.csv
│   ├── classification_*.csv
│   ├── index_correlations.csv
│   └── rate_distortion.csv
├── graphs/
│   ├── scatter_*.png
│   └── rate_distortion_*.png
└── logs/
```

`ffmpeg_metrics.csv` contém a baseline única e as condições FLAC, MP3, AAC,
Opus e LC3. `encodec_metrics.csv` contém somente as três condições EnCodec. A
análise reúne logicamente os dois arquivos e compara todas as reconstruções com
a mesma baseline por `dataset + clip_id`.

As correlações dos índices acústicos incluem Pearson, Spearman e o coeficiente
de correlação de concordância de Lin (CCC). As tabelas de taxa-distorção mantêm
bitrate, Mel-SSIM, SNR e Mel-RMSE.

## Estrutura do repositório

```text
.
├── config/             # Manifesto e exemplos autônomos
├── data/               # Datasets, modelos, caches, checkpoints e testes
├── results/            # Resultados científicos legíveis por humanos
├── src/codeco/         # Implementação das etapas do pipeline
├── pyproject.toml      # Metadados e dependências do projeto
├── uv.lock             # Resolução reproduzível das dependências
└── README.md
```

Diretórios grandes ou reproduzíveis sob `data/` e `results/` não devem ser
versionados. O código, as configurações e os arquivos de ambiente determinam a
execução reproduzível do experimento.
