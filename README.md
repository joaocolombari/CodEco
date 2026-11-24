# 🌿 CodEco

### 🦗 Ecoacústica de insetos · 🐦 Bioacústica de aves · 🍃 Sensoriamento eficiente

**Codeco** é o espaço de trabalho do estudo sobre compressão de áudio e
representações compactas para nós de monitoramento ecoacústico. O projeto está
ligado ao sistema EASN e investiga quanta informação pode ser retirada do
payload sem comprometer as conclusões ecológicas produzidas no servidor.

O objetivo não é encontrar um codec universalmente melhor. O objetivo é medir
a fronteira entre **payload**, **fidelidade analítica**, **utilidade preditiva**
e **capacidade de reutilização**.

## Pergunta de pesquisa

> Em segmentos bioacústicos de quatro segundos, mono, PCM de 16 bits e
> amostrados a 48 kHz, quanto o
> payload pode ser reduzido por codecs de forma de onda e por representações
> aprendidas antes que índices acústicos, classificação e possibilidade de
> reanálise sejam materialmente degradados?

## Motivação

Os nós EASN operam com energia, armazenamento e conectividade limitados. Uma
conexão indisponível acumula gravações no nó, e o tamanho de cada registro
determina tanto o tempo de transmissão quanto a duração possível do backlog.
Reduzir o payload é, portanto, parte do sistema de medição e não apenas uma
decisão de armazenamento.

Codecs perceptuais são otimizados para audição humana e podem remover ou
redistribuir energia importante para análises ecoacústicas. Representações
aprendidas podem ser muito menores, mas podem preservar apenas a tarefa para a
qual foram treinadas. O projeto compara explicitamente esses compromissos.

## Base na literatura

Duas consultas complementares fundamentam o desenho:

```text
ALL=(("codec" OR "compressed" OR "lossy")
     AND
     ("ecoacoustics" OR "soundscape" OR "bioacoustics"))
```

```text
TS=(("audio codec*" OR compress* OR lossy OR bitrate
     OR "neural audio codec*" OR "latent representation*")
    AND
    (ecoacoustic* OR bioacoustic* OR soundscape*
     OR "passive acoustic monitoring"))
```

A primeira consulta resultou em 7 referências mantidas; a segunda, em 13
referências centrais e 18 de apoio metodológico ou de implantação. A triagem,
os critérios e a síntese estão em
[`docs/literature-screening.md`](docs/literature-screening.md).

O gap consolidado é a ausência de uma comparação pareada, sob os mesmos dados,
janelas, frequência de amostragem, partições e contabilização de bytes, entre:

1. codecs legados e modernos que recuperam a forma de onda;
2. codecs neurais reconstruíveis;
3. representações compactas orientadas à tarefa.

Também permanece pouco explicada a possível dissociação entre alterações nos
índices acústicos e estabilidade dos classificadores.

## Hipóteses

### H1 — dissociação entre fidelidade analítica e preditiva

Em payload equivalente, codecs perceptuais modernos preservarão melhor a
utilidade preditiva que MP3 e poderão ser não inferiores à referência sem
perdas acima de um limiar de bitrate. Ainda assim, os índices ecoacústicos
apresentarão vieses heterogêneos, dependentes do codec, bitrate e índice.

### H2 — vantagem condicionada da representação aprendida

Em payload equivalente, códigos aprendidos e quantizados alcançarão uma
fronteira taxa–utilidade superior à dos codecs de forma de onda para
classificação. A vantagem diminuirá quando a utilidade exigir reconstrução do
áudio, índices acústicos, classes não vistas ou uma tarefa diferente daquela
usada no treinamento.

## Escopo experimental

### Unidade de análise

- áudio mono;
- frequência de amostragem comum de 48 kHz;
- referência numérica PCM signed de 16 bits;
- janelas de 4 s;
- quando uma gravação longa exigir segmentação deslizante, hop temporal de 1 s,
  equivalente a um quarto da janela e 75% de sobreposição;
- conteúdo biologicamente relevante limitado a até 24 kHz;
- todas as derivações de uma janela pertencem sempre à mesma partição;
- comparação pareada: cada método recebe exatamente o mesmo sinal de entrada.

Sinais cuja identificação dependa de informação ultrassônica acima de 24 kHz
ficam fora do corpus principal. Reduções adicionais da frequência de
amostragem podem formar um estudo secundário, mas não serão misturadas ao
efeito dos codecs.

### Corpora

O protocolo foi concebido para dois domínios, analisados separadamente:

1. **🦗 Insetos — ECOSoundSet.** Corpus principal, com foco em Orthoptera e nos
   registros elegíveis após auditoria de proveniência e formato. A primeira
   versão liberada contém 14.954 janelas completas e 34 espécies-alvo. O depósito
   oficial usa licença CC BY-NC 4.0, que também limita o uso dos derivados.
2. **🐦 Aves — PteroSet/PPA1.** Réplica externa em áudio WAV original a 192 kHz,
   reamostrado uma única vez para 48 kHz. O recorte congelado tem 108 gravações,
   29 grupos de local–gravador, 10.368 janelas de 4 s e 10 espécies-alvo. O
   arquivo PPA1 tem 15,3 GB, evitando o download dos 86,4 GB do PteroSet inteiro.
   O ZIP oficial e os 108 WAVs foram integralmente verificados em 2026-10-08.

Não haverá mistura dos dois corpora em um único treinamento ou resultado. A
repetição em aves testa se os padrões de degradação se mantêm em outro domínio,
sem transformar diferenças de gravador, habitat ou taxonomia em efeito de
compressão.

O recorte de classificação é derivado separadamente para cada corpus. Sobre as
contagens de janelas do **treino**, aplica-se `log1p` e calcula-se o limiar
`exp(μ - kσ) - 1`. O hiperparâmetro versionado `k` controla o compromisso entre
número de classes e suporte por classe; também se exige um número mínimo de
grupos independentes. O notebook de seleção mostra a distribuição, a análise
de sensibilidade de `k`, cada inclusão/exclusão e o recorte balanceado. O
freeze completo permanece inalterado.

Se o cronograma permitir, o protocolo poderá ser replicado no **AnuraSet** e no
**Watkins Marine Mammal Sound Database**. Essas extensões dependem de auditoria
de licença, formato original, anotações, frequência de amostragem e
proveniência sem perdas.

## Instalação e execução, do zero

Esta é a sequência de referência para o João e para os demais integrantes do
grupo. Ela é suportada em **Ubuntu 24.04 ou superior** e **macOS**. No Windows,
use WSL2 com Ubuntu e siga a seção Linux. Use Python **3.10, 3.11 ou 3.12**;
Python 3.13/3.14 ainda não é compatível com todas as versões de TensorFlow,
PyTorch e bibliotecas bioacústicas usadas aqui.

### 1. Requisitos da máquina

- aproximadamente **75 GiB livres** para instalar os dois corpora mantendo os
  arquivos oficiais e os WAVs extraídos;
- pelo menos 16 GiB de RAM para o piloto; a execução completa é recomendada no
  EASNCloud;
- acesso à internet para os corpora e, no primeiro uso, para EnCodec, BirdNET
  e Perch;
- Git, compilador C, FFmpeg, `pkg-config` e `libsndfile`.

Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y git curl build-essential pkg-config ffmpeg \
  libsndfile1-dev liblc3-dev nasm xz-utils python3 python3-venv python3-dev
```

macOS com Homebrew:

```bash
brew install git curl pkg-config ffmpeg libsndfile nasm python@3.12
```

Se a conta do EASNCloud não tiver `sudo`, peça ao administrador para instalar
os pacotes do primeiro bloco. Os dados e o ambiente Python ficam no diretório
do projeto e não exigem acesso de administrador.

### 2. Clonar ou atualizar o repositório

Instalação nova:

```bash
cd /srv/easn/shared                       # no EASNCloud
git clone https://github.com/joaocolombari/CodEco.git codeco
cd codeco
```

Se o clone já existe:

```bash
cd /srv/easn/shared/codeco
git status
git pull --ff-only origin main
```

Não execute `git pull` sobre alterações locais sem antes salvá-las. Todos os
comandos seguintes devem ser executados na raiz do clone, onde este README e
`scripts/` estão visíveis.

### 3. Criar o ambiente Python

O pipeline integrado usa Python 3.12, `uv` 0.12.0 ou superior e o lock
versionado. Instale o `uv` na conta do usuário, sem `sudo`:

```bash
curl -LsSf https://astral.sh/uv/0.12.0/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version
```

Em CPU — inclusive macOS sem GPU NVIDIA — execute:

```bash
uv python install 3.12
uv sync --locked --extra notebook
. .venv/bin/activate
```

Em um servidor Linux com CUDA, a aceleração é opcional:

```bash
uv sync --locked --extra gpu --extra notebook
. .venv/bin/activate
```

Execute a sincronização no terminal, antes de iniciar ou reiniciar o kernel.
Durante a execução, os notebooks apenas conferem o lock e nunca instalam ou
removem pacotes do processo Python que está em uso.

O ambiente alternativo abaixo serve apenas para notebooks legados e inspeção.
Ele **não substitui o `uv` no pipeline integrado**, pois não valida `uv.lock`:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-notebook.txt -r scripts/requirements.txt
```

Confirme que o terminal está usando o ambiente correto:

```bash
which python
python --version
```

O primeiro comando deve apontar para `codeco/.venv/bin/python`. O arquivo
`requirements.txt` da raiz é propositalmente vazio: os instaladores dos
corpora usam apenas a biblioteca padrão. O ambiente experimental completo é a
combinação de `requirements-notebook.txt` e `scripts/requirements.txt` acima.
CUDA nunca é requisito funcional: EnCodec, BirdNET, Perch e a classificação
possuem caminho CPU. A execução sem GPU é substancialmente mais lenta, mas usa
os mesmos manifests, condições, checkpoints e formatos de resultado.

### 4. Instalar e validar o LC3

```bash
./scripts/install_ffmpeg_lc3.sh
```

O script compila uma cópia local do FFmpeg 9.0.2 com `liblc3`, sem substituir
o FFmpeg do sistema, e termina somente depois de um round-trip real. A saída
esperada termina com `LC3 round-trip verified`. O pipeline usa esse executável
somente para LC3; FLAC, MP3, AAC e Opus usam o FFmpeg do sistema.

Para o FFmpeg do sistema, a validação é baseada nas capacidades realmente
usadas: presença dos encoders e round-trip de FLAC, MP3, AAC e Opus. A versão
encontrada é registrada nos resultados, mas não precisa coincidir com a cópia
9.0.2 fixada para LC3.

### 5. Instalar exatamente os corpora congelados

Para os dois corpora:

```bash
./scripts/install_frozen_corpora.sh all
```

O instalador pode ser interrompido e executado novamente. Ele clona os
repositórios auxiliares nos commits congelados, retoma downloads, verifica
tamanho e hashes e audita os WAVs. Ao final, deve imprimir
`freeze 1.3.0 verification passed` e
`Corpus freeze 1.3.0 installation complete`.

Instalações parciais:

```bash
./scripts/install_frozen_corpora.sh metadata     # sem áudio
./scripts/install_frozen_corpora.sh ecosoundset  # insetos; requer 38 GiB livres
./scripts/install_frozen_corpora.sh pteroset     # aves; requer 34 GiB livres
```

Para conferir novamente uma instalação completa:

```bash
python scripts/verify_frozen_install.py --stage all --corpus all
```

A especificação legível por máquina está em
[`data/freeze-spec-v1.json`](data/freeze-spec-v1.json). Áudio e downloads ficam
em `data/audio/` e `data/downloads/`; checkouts auxiliares ficam em
`external/`. Eles são grandes, reproduzíveis e ignorados pelo Git.

O PPA1 é primeiro auditado dentro do ZIP e depois seus 108 WAVs são extraídos
atomicamente para `data/audio/pteroset_ppa1/`. Uma instalação correta ocupa
aproximadamente 15,3 GB no ZIP e 18,0 GB nos WAVs extraídos. A verificação
final exige os arquivos físicos e confere novamente tamanho e SHA-256; ter
somente o ZIP verificado não é considerado uma instalação completa.

### 6. Gerar as tabelas locais e os notebooks

Os manifests definem o recorte; os caminhos absolutos e a disponibilidade dos
WAVs dependem de cada máquina. Por isso, gere novamente apenas as tabelas
locais depois de instalar ou mover os corpora:

```bash
python scripts/data.py
python scripts/build_experiment_notebooks.py
```

A saída de `scripts/data.py` deve informar um número de áudios disponíveis
maior que zero para cada corpus instalado. Isso não altera o freeze nem as
classes: apenas resolve os caminhos na máquina atual.

Com `K_SIGMA=1`, `MINIMUM_GROUPS=3` e semente 42, os recortes atuais possuem
7.127 janelas do ECOSoundSet e 777 do PteroSet/PPA1. Esses números são menores
que os manifests brutos porque representam a seleção e o balanceamento do
treino; todos os 108 WAVs físicos do PPA1 continuam instalados.

### 7. Executar a sequência auditável

```bash
jupyter lab --no-browser --ip=0.0.0.0 notebooks/experimental
```

Abra e execute os notebooks nesta ordem:

1. `01_corpus_audit.ipynb` — confere manifests, ambiente e distribuição;
2. `02_class_selection.ipynb` — mostra a regra gaussiana em `log1p`, classes e
   balanceamento;
3. `03_preprocessing.ipynb` — confirma 4 s, 48 kHz, hop de 1 s e adaptadores;
4. `04_codec_sanity_check.ipynb` — executa o round-trip sintético de todos os
   codecs. Depois defina `RUN_CORPUS_PILOT=True` para testar uma janela real de
   cada corpus;
5. `05_reconstruction_metrics.ipynb` — defina `RUN_FULL_EVALUATION=True`
   somente depois que o notebook 04 terminar sem falhas;
6. `06_acoustic_indices.ipynb` — compara ACI, ADI, BIO, NDSI e Ht;
7. `07_foundation_models.ipynb` — BirdNET e Perch, macro-F1, acurácia
   balanceada e preservação dos embeddings;
8. `08_compute_cost.ipynb` — FLOPs para redes neurais e tempo/instruções para
   codecs tradicionais;
9. `09_summary.ipynb` — tabelas e curvas finais.
10. `10_integrated_pipeline.ipynb` — inspeciona a configuração única, o
    ambiente, a proveniência e executa as mesmas etapas do runner otimizado.

No notebook 04, `RUN_SMOKE_TEST=True` já vem ativado. EnCodec baixa os pesos
`facebook/encodec_48khz` no primeiro uso. O piloto dos corpora reconstrói as
tabelas locais antes de começar, mostra quantos WAVs encontrou e registra cada
condição executada; ele não termina mais silenciosamente quando o áudio está
ausente.

### 8. Execução otimizada no EASNCloud

Primeiro execute os notebooks 01–04. Para a matriz completa, use o runner
integrado, paralelo e retomável:

```bash
. .venv/bin/activate
python -m codeco.main
```

Ele valida primeiro o freeze 1.3.0, cria `results/run-metadata.json`, prepara o
cache PCM e executa FFmpeg, EnCodec, BirdNET, Perch, classificação, análises e
figuras. As reconstruções FFmpeg usam os núcleos físicos da CPU; a inferência
usa um worker persistente em CPU quando não há CUDA ou um worker por GPU quando
há. Filas limitadas evitam acumular todas as reconstruções em memória.

Sem CUDA, o EnCodec usa lotes de uma janela e grava checkpoint após cada
janela. Essa configuração limita a memória e torna interrupções retomáveis; o
resultado numérico é o mesmo, embora a execução neural em CPU continue muito
mais lenta que em GPU. Em CUDA, permanecem os lotes adaptativos orientados à
VRAM.

Para executar somente modelos ou corpora específicos, use os parâmetros do
`07_foundation_models.ipynb`. Defina `RUN_EMBEDDINGS=True` e, depois,
`RUN_CLASSIFICATION=True`; `DATASETS`, `MODELS`, `CODECS` e
`MAX_CLIPS_PER_DATASET` controlam o recorte. A execução final usa `CODECS=None`
e `MAX_CLIPS_PER_DATASET=None`.

Para inspecionar ou executar etapas individualmente, use os módulos
correspondentes:

```bash
python -m codeco.environment
python -m codeco.prepare_table
python -m codeco.ffmpeg
python -m codeco.encodec
python -m codeco.embeddings
python -m codeco.classify
python -m codeco.analyze
python -m codeco.graph
```

Checkpoints ficam em `data/checkpoints/` e caches em `data/cache/`. Interromper
e repetir o comando retoma somente as unidades ainda não concluídas. Use
`python -m codeco.main --clean` apenas quando quiser descartar deliberadamente
caches e resultados derivados; os corpora, downloads, modelos e manifests são
preservados.

Os autoencoders próprios permanecem desativados. Não use
`--include-autoencoders` até a validação de comprimento e serialização do
payload. EnCodec não é um desses autoencoders: ele já integra a grade normal.

### 9. Onde ficam os resultados

- `data/derived/experiment/` — tabelas, caches, métricas e classificações;
- `data/derived/experiment/cache/` — embeddings reaproveitáveis;
- `reports/` — auditorias e figuras produzidas pelos notebooks;
- `results/` — tabelas, logs, gráficos e proveniência do runner integrado;
- `data/cache/` e `data/checkpoints/` — cache PCM, embeddings e retomada;
- `.tools/ffmpeg-lc3/9.0.2/` — FFmpeg local usado exclusivamente para LC3.

Esses artefatos são ignorados pelo Git. Para compartilhar resultados, registre
o commit exibido pelo notebook 01 e envie separadamente as tabelas/figuras
necessárias.

### 10. Diagnóstico rápido

- **Piloto real não mostra nenhuma condição:** execute `python scripts/data.py`.
  Se aparecer `0 audio files available`, os WAVs não estão dentro deste clone
  em `data/audio/` ou o corpus foi instalado apenas parcialmente.
- **`FFmpeg com liblc3 nao encontrado`:** execute
  `./scripts/install_ffmpeg_lc3.sh` e confirme o round-trip.
- **EnCodec informa ausência de `torch` ou `transformers`:** ative `.venv` e
  reinstale `scripts/requirements.txt`.
- **A máquina não tem CUDA:** não é erro. O log deve informar o fallback CPU;
  reduza primeiro `max_clips_per_dataset` para estimar o tempo local.
- **O notebook aberto não reflete uma atualização:** feche-o, execute
  `python scripts/build_experiment_notebooks.py` e abra novamente.
- **Perch/BirdNET ou EnCodec iniciam um download:** isso é esperado apenas na
  primeira execução; mantenha `data/derived/experiment/cache/` para retomadas.
- **Processo remoto cai ao fechar o terminal:** execute a sessão dentro de
  `tmux` ou do gerenciador de jobs adotado no EASNCloud.

Cada arquivo do corpus é auditado quanto a codec de origem, frequência de
amostragem, canais, duração, licença e relação com outras gravações. A análise
principal usa somente fontes sem perdas. Os resultados da auditoria estão em
[`docs/corpus-audit.md`](docs/corpus-audit.md).

### Famílias comparadas

| Família | Métodos | Função |
|---|---|---|
| Referência | PCM e FLAC | Estabelecer sinal e payload sem perdas |
| Legado | MP3 | Fazer a ponte direta com a literatura |
| Perceptual opcional | AAC-LC | Comparador amplamente implantado, condicionado a encoder e perfil congelados |
| Perceptual moderno | Opus e LC3 | Testar eficiência em baixo bitrate |
| Neural reconstruível | EnCodec e/ou implementação equivalente compatível com 48 kHz | Transmitir código e reconstruir áudio |
| Aprendida para a tarefa | embedding quantizado e bottleneck treinado | Minimizar payload para classificação |

O núcleo usa **LC3 normal**, no perfil de 48 kHz e frames de 10 ms, e não
LC3plus. A referência normativa é a
[especificação LC3 do Bluetooth SIG](https://www.bluetooth.com/specifications/specs/low-complexity-communication-codec-1-0/),
e a implementação de referência proposta é
[`google/liblc3`](https://github.com/google/liblc3), distribuída sob
Apache-2.0. O LC3plus fica excluído do núcleo porque suas extensões não são
necessárias ao protocolo e sua adoção acrescenta uma trilha própria de licença
de software e patentes, descrita pela
[Fraunhofer IIS](https://www.iis.fraunhofer.de/en/ff/amm/communication/lc3/lc3-lc3plus-licensing.html).
A licença do código não substitui a verificação de
patentes, qualificação e framing antes de uma implantação comercial do LC3.
Os níveis dos codecs neurais serão definidos a partir das configurações
realmente suportadas pela implementação, e não apenas de valores publicados.

### Instalação reproduzível do LC3

O FFmpeg comum do Homebrew não inclui `liblc3`. Por isso o projeto constrói
uma cópia local e fixada do **FFmpeg 9.0.2**, ligada à implementação oficial
`google/liblc3`, sem substituir o FFmpeg do sistema:

```bash
./scripts/install_ffmpeg_lc3.sh
```

O script confere o SHA-256 do código-fonte, instala o binário em
`.tools/ffmpeg-lc3/9.0.2/` e verifica a presença do encoder. O pipeline escolhe
esse executável somente para LC3; FLAC, MP3, AAC e Opus continuam usando o
FFmpeg do sistema. O perfil experimental fica explícito em **mono, 48 kHz,
frames de 10 ms, modo high-resolution desligado**, com 32, 64 e 96 kbit/s.
Assim, a configuração permanece no LC3 normal e não ativa extensões LC3plus.

O AAC não integra o núcleo mínimo nesta versão, mas poderá entrar no piloto
como **AAC-LC mono/48 kHz**, inicialmente em 64, 96 e 128 kbit/s. A
implementação preferida para reprodutibilidade é o encoder AAC nativo do
FFmpeg, com versão, perfil e container registrados. HE-AAC e xHE-AAC não serão
tratados como equivalentes a AAC-LC; sua inclusão exigiria uma implementação e
uma verificação de licença próprias.

## Princípios de comparação

1. **Um único master padronizado.** A conversão para mono e 48 kHz ocorre uma
   vez, antes das condições experimentais.
2. **Mesmo conteúdo e mesmas partições.** Codec nunca altera a composição do
   treino, validação ou teste.
3. **Modelo congelado como análise principal.** O classificador é treinado no
   master sem perdas e testado em todas as condições.
4. **Codec augmentation como análise secundária.** Mede adaptação, não
   robustez nativa.
5. **Frontend fixo.** Janela, STFT, escala, normalização e arquitetura não
   mudam entre codecs.
   A janela temporal de 4 s e seu hop de 1 s são separados da janela e do hop
   internos da STFT ou do frontend de BirdNET e Perch.
   Após a reconstrução, BirdNET recebe o recorte central de 3 s; Perch recebe
   os 4 s reamostrados para 32 kHz com 0,5 s de zeros em cada extremidade. O
   mesmo adaptador é usado na referência e em todas as condições e não entra
   na contabilização do payload.
6. **Payload observado.** A unidade é byte por janela e bit/s efetivo, incluindo
   cabeçalhos e metadados quando presentes.
7. **Comparação por orçamento.** Além dos bitrates nativos, métodos são
   comparados em faixas comuns de payload.
8. **Métodos não equivalentes permanecem identificados.** Um embedding sem
   decoder não recebe métricas de forma de onda ou índices reconstruídos.
9. **Inferência pareada.** Incerteza e testes estatísticos respeitam a gravação
   original e os grupos de local, data e equipamento.

## Desfechos

### Primários

- macro-F1 de classificação no teste;
- acurácia balanceada no teste;
- erro relativo e viés dos índices acústicos selecionados;
- similaridade do espectrograma e do log-Mel antes e depois da reconstrução;
- payload efetivo em bit/s;
- FLOPs por janela para codificação, decodificação e inferência;
- fronteira de Pareto entre payload e utilidade.

A margem provisória de não inferioridade da classificação é de **2 pontos
percentuais de macro-F1** em relação ao master sem perdas. Ela deverá ser
confirmada antes de observar os resultados finais.

### Secundários

- average precision e recall por classe;
- matriz de confusão por condição;
- calibração das probabilidades;
- distância log-espectral e erro de energia por banda;
- preservação de séries de atividade, quando os metadados temporais permitirem;
- desempenho em gravações, locais ou fontes não vistos;
- tempo de codificação, pico de memória e energia em plataforma declarada.

Os limites de referência de **256 KiB de RAM e 1 MiB de flash** pertencem à
validação embarcada. Eles não funcionam como critério de exclusão da primeira
fase, que compara informação, payload e FLOPs em ambiente de servidor. Depois,
as alternativas finalistas serão avaliadas quanto a portabilidade, tempo,
memória, flash e energia no nó.

## Fases

| Fase | Entrega | Critério de passagem |
|---|---|---|
| 0 | Auditoria e manifesto dos corpora | Proveniência, licenças e grupos definidos |
| 1 | Pipeline PCM/FLAC e partições | Reprodução determinística e sem vazamento |
| 2 | MP3, Opus e LC3; AAC-LC opcional | Bitstreams decodificáveis e payload validado |
| 3 | Classificação e índices | Métricas pareadas com intervalos de confiança |
| 4 | Codec neural e representação orientada à tarefa | Payload serializado e decoder/tarefa documentados |
| 5 | Generalização e séries temporais | Avaliação externa sem reajuste no teste |
| 6 | Validação embarcada | Duas alternativas finalistas medidas no hardware |

As Fases 0–3 formam o núcleo mínimo do estudo. As Fases 4–6 ampliam o artigo e
a recomendação para o EASN, mas não devem bloquear o primeiro resultado.

## Matriz experimental

A especificação das condições, comparações, métricas e ablações está em
[`docs/experimental-matrix.md`](docs/experimental-matrix.md). A matriz separa o
núcleo confirmatório das extensões para manter o número de execuções
controlável.

## Estrutura prevista

```text
codeco/
├── README.md
├── configs/              # configurações versionadas dos experimentos
├── data/                 # manifests; áudio bruto não entra no Git
├── docs/
│   ├── experimental-matrix.md
│   ├── literature-screening.md
│   ├── savedrecs_screening_1.bib
│   └── savedrecs_screening_2.bib
├── notebooks/            # análises legíveis apoiadas em src/configs/scripts
├── reports/              # tabelas e figuras derivadas
├── scripts/              # aquisição e preparação reprodutível
├── src/                  # pipeline de codecs, features e modelos
└── tests/
```

## Reprodutibilidade e prevenção de vazamento

- manifests identificam arquivo original, janela, espécie, fonte, local,
  gravador e licença;
- o split ocorre antes da segmentação e de qualquer transformação;
- janelas da mesma gravação nunca atravessam partições;
- duplicatas e versões recomprimidas da mesma gravação são agrupadas;
- sementes, versões dos encoders e parâmetros ficam em configuração;
- arquivos derivados recebem hash do master e da configuração;
- áudio e artefatos grandes ficam fora do Git;
- o teste permanece fechado até o congelamento de métodos e margens.

## Estado atual

A ordem abaixo é deliberada: o frontend e os índices foram congelados antes da
escolha definitiva das implementações dos codecs, do pipeline de referência e
da grade do piloto. Assim, decisões de implementação não podem redefinir os
desfechos depois que os primeiros resultados forem vistos.

- [x] triagem bibliográfica das duas queries;
- [x] gap e hipóteses consolidados;
- [x] escopo comum de 4 s e 48 kHz;
- [x] primeira versão da matriz experimental;
- [x] auditar e congelar os manifests de metadados;
- [x] bloquear o recorte Xeno-Canto por insuficiência de fontes lossless;
- [x] selecionar PteroSet/PPA1 como réplica lossless de aves;
- [x] congelar janelas, espécies, grupos e partições dos dois corpora;
- [x] baixar e verificar o arquivo PPA1 do PteroSet;
- [x] mapear os 15.097 WAVs selecionados do ECOSoundSet no depósito oficial;
- [x] definir a lista final de índices e suas parametrizações;
- [x] baixar e auditar os 15.097 WAVs selecionados no EASNCloud; 14.954 foram liberados e 143 clipes curtos foram excluídos sem padding;
- [x] organizar o protocolo em nove notebooks auditáveis;
- [x] implementar seleção parametrizada de classes e adaptadores BirdNET/Perch;
- [x] integrar execução paralela, checkpoints, cache PCM e gravação atômica;
- [x] preservar execução CPU e aceleração CUDA opcional;
- [x] corrigir alinhamento temporal normalizado e entropias com suporte fixo;
- [x] salvar probabilidades e calibrar limiares exclusivamente na validação;
- [x] acrescentar erros pareados, bootstrap por grupo, ajuste BH e Pareto;
- [ ] confirmar licenças e implementações dos codecs;
- [x] implementar o pipeline de referência;
- [ ] validar a execução completa pós-integração nos dois corpora;
- [ ] executar o piloto de payload e escolher a grade final de bitrates.

## Documentação

- [Triagem e síntese da literatura](docs/literature-screening.md)
- [Matriz experimental](docs/experimental-matrix.md)
- [Auditoria da integração e correções metodológicas](docs/integration-audit.md)
- [Plano dos notebooks](notebooks/README.md)
- [Auditoria e congelamento dos corpora](docs/corpus-audit.md)
- [Capacidade local e plano de execução](docs/storage-plan.md)
- [Parâmetros congelados dos índices](configs/acoustic-indices-v1.yaml)
- [BibTeX — Query 1](docs/savedrecs_screening_1.bib)
- [BibTeX — Query 2](docs/savedrecs_screening_2.bib)
