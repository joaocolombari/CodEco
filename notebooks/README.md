# Notebooks

Os notebooks serão a interface legível para auditoria, exploração, execução
assistida e apresentação dos experimentos. Eles não serão a única fonte da
lógica do projeto: funções reutilizáveis e testáveis devem permanecer em
`src/`, parâmetros congelados em `configs/` e execuções completas ou remotas
em `scripts/`.

Cada notebook experimental deverá:

- consumir manifests e configurações versionados;
- poder ser executado do início ao fim sem depender de uma ordem oculta de
  células;
- chamar funções de `src/` em vez de manter cópias divergentes da lógica;
- registrar sementes, versões e parâmetros relevantes;
- salvar tabelas e figuras derivadas em `reports/` ou em outro diretório de
  resultados ignorado pelo Git, conforme o tamanho;
- evitar incorporar áudio, modelos ou artefatos grandes ao repositório.

## Sequência experimental implementada

Os notebooks ficam em `notebooks/experimental/` e são regenerados por
`python scripts/build_experiment_notebooks.py`. As células pesadas começam
desativadas para evitar downloads ou execuções completas acidentais.

1. **`01_corpus_audit.ipynb`** — formatos, taxas de amostragem, durações,
   classes, licenças e grupos de gravação.
2. **`02_class_selection.ipynb`** — distribuições, seleção determinística das
   classes elegíveis e congelamento do recorte balanceado.
3. **`03_preprocessing.ipynb`** — conversão comum para mono, 48 kHz e PCM de
   16 bits; janelas de 4 s; hop de 1 s; e geração dos manifests derivados.
4. **`04_codec_sanity_check.ipynb`** — round-trip sintético ativado por padrão
   em FLAC, MP3, AAC-LC, Opus, LC3 e EnCodec, seguido de um piloto opcional em
   áudio real. A grade de configurações, os resultados e as falhas são
   exibidos separadamente para que campos não aplicáveis não apareçam como
   resultados ausentes.
5. **`05_reconstruction_metrics.ipynb`** — bytes transmitidos, taxa de
   compressão, qualidade da reconstrução e similaridade de espectrogramas e
   log-Mel.
6. **`06_acoustic_indices.ipynb`** — comparação pareada de ACI, ADI, BIO,
   NDSI e Ht antes e depois da reconstrução.
7. **`07_foundation_models.ipynb`** — BirdNET e Perch, com macro-F1,
   acurácia balanceada, matrizes de confusão e distâncias entre
   representações.
8. **`08_compute_cost.ipynb`** — FLOPs, quantização e relação entre custo
   computacional, armazenamento e transmissão.
9. **`09_summary.ipynb`** — tabelas finais, curvas de compromisso e testes das
   hipóteses.

Os três primeiros notebooks funcionam apenas com os manifests versionados. As
rotinas pesadas permanecem disponíveis como scripts não interativos para o
EASNCloud e os notebooks usam os mesmos módulos de `src/codeco` e `scripts/`.
Os autoencoders permanecem desativados por padrão até a correção de comprimento
e a serialização auditável do payload.

O hop de 1 s está implementado para futuras segmentações de gravações
contínuas, mas não reescreve os manifests congelados atuais. Uma resegmentação
dos corpora existentes deve gerar uma nova versão do freeze e uma nova
auditoria.

Para abrir a sequência:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-notebook.txt
jupyter lab notebooks/experimental/01_corpus_audit.ipynb
```

As etapas com codecs neurais e modelos fundacionais também precisam das
dependências de `scripts/requirements.txt`.

## Notebook do seminário

`seminario_ecoacustica_parte_joao.ipynb` supports João's section of the AMST
seminar. It demonstrates the path from waveform to STFT, spectrogram,
mel-spectrogram and derived temporal variables, followed by lossless and lossy
codec comparisons.

## Local setup

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-notebook.txt
jupyter lab notebooks/seminario_ecoacustica_parte_joao.ipynb
```

The codec cells also require `ffmpeg`:

```bash
brew install ffmpeg                 # macOS
sudo apt-get install ffmpeg         # Ubuntu/Debian
./scripts/install_ffmpeg_lc3.sh     # FFmpeg local com LC3
```

O instalador local é necessário porque as distribuições usuais do FFmpeg nem
sempre habilitam `liblc3`. O notebook de inspeção mostra separadamente o
executável usado por cada família de codec.

The notebook looks for the frozen ECOSoundSet example at its normal corpus
path. If the corpus is unavailable, it uses an explicitly labelled synthetic
Orthoptera-like signal so every explanatory cell still runs. Generated audio
and figures are written under the ignored `notebooks/outputs/` directory.

For a live presentation, execute all cells beforehand and keep the outputs in
the notebook. Do not depend on downloads, package installation or codec
encoding while presenting.
