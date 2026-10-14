# Auditoria e congelamento dos corpora

Data do congelamento: 2026-10-08
Versão do manifesto: 1.3.0

## Resultado

O **ECOSoundSet foi liberado para análise** como corpus confirmatório. Foram
auditados 15.097 WAVs e o manifesto analítico contém 14.954 janelas realmente
completas, 34 táxons-alvo de Orthoptera e os splits oficiais. Não há gravações
nem grupos local–data
compartilhados entre treino, validação e teste.

O **PteroSet/PPA1 foi liberado para análise** como réplica em aves. O recorte
usa fontes WAV a 192 kHz, 108 gravações, 29 grupos local–gravador e 10.368
janelas determinísticas de quatro segundos. Dez espécies satisfazem o suporte
mínimo. O arquivo de áudio necessário tem 15,3 GB, em vez dos 86,4 GB do corpus
completo.

O ZIP oficial do PPA1 foi verificado por tamanho, MD5 e SHA-256. Todos os 108
WAVs foram lidos integralmente dentro do arquivo, o que também valida o CRC do
ZIP, e tiveram estrutura RIFF, formato, frequência, canais, duração e SHA-256
conferidos. No ECOSoundSet, os dois arquivos oficiais passaram por tamanho,
MD5 e SHA-256; os 15.097 WAVs tiveram cabeçalhos e hashes conferidos.

## Fontes congeladas

| Corpus | Repositório | Revisão |
|---|---|---|
| ECOSoundSet | `DavidFunosas/ECOSoundSet_database_analysis` | `f3fdf9c354f97bd7d6ad1991743a2134a9cb9b0f` |
| Aves | `microsoft/PteroSet` | `b6aaf57ac32f93715e3570538f3580a78a80b741` |

Os hashes SHA-256 dos arquivos de origem e dos manifests gerados estão em
`data/manifest-lock-v1.json`. Os repositórios em `external/` não entram no Git e
podem ser recuperados pelas URLs e revisões registradas nesse arquivo.

## Regras comuns

- janela de 4 s;
- frequência de origem mínima de 48 kHz;
- frequência máxima da análise de 24 kHz;
- somente fontes sem perdas na análise causal;
- uma unidade por gravação ou janela original;
- nenhuma derivação atravessa partições;
- a verificação do arquivo de áudio ocorre antes da liberação para análise.

# ECOSoundSet

## Resultado da auditoria

| Medida | Resultado |
|---|---:|
| Linhas de anotação processadas | 111.701 |
| Janelas únicas no repositório | 41.964 |
| Janelas que passam os critérios de sinal | 16.556 |
| WAVs baixados e auditados | 15.097 |
| WAVs excluídos por duração real inferior a 4 s | 143 |
| Janelas no manifesto analítico liberado | 14.954 |
| Espécies-alvo | 34 |
| Treino | 8.916 |
| Validação | 3.531 |
| Teste | 2.507 |
| Conflitos de split por gravação | 0 |
| Conflitos de split por grupo local–data | 0 |

O artigo descreve 41.883 segmentos anotados, enquanto os CSVs do commit
congelado contêm 41.964 `recording_chunk_id` únicos. A diferença de 81 janelas
não foi corrigida manualmente. O manifesto preserva o estado observável do
repositório e registra a revisão exata para tornar a diferença auditável.

## Critérios de inclusão

Uma janela entra no manifesto quando:

1. dura exatamente quatro segundos;
2. tem frequência de amostragem de pelo menos 48 kHz;
3. contém ao menos um rótulo de Orthoptera considerado audível no limite de
   24 kHz;
4. não contém um Orthoptera marcado como ultrassônico;
5. todos os rótulos de Orthoptera podem ser associados à lista taxonômica;
6. o WAV publicado tem duração real de quatro segundos, sem padding;
7. contém ao menos uma espécie com suporte mínimo em todos os splits.

O suporte mínimo por espécie foi congelado em:

- 50 janelas no total;
- 30 no treino;
- 5 na validação;
- 5 no teste.

Esses limiares produzem 34 espécies. As contagens por espécie e split estão na
planilha `data/audit/corpus-audit-v1.xlsx` e no lock JSON.

## Exclusão de conteúdo ultrassônico

A exclusão usa duas evidências:

- campo `Ultrasons` ou frequência dominante a partir de 24 kHz na lista
  curada de espécies;
- anotação cuja frequência mínima seja igual ou superior a 24 kHz.

Se uma janela contiver simultaneamente uma espécie elegível e uma espécie
ultrassônica, a janela inteira é excluída. Isso impede que o downsampling para
48 kHz seja tratado como uma transformação neutra.

## Limitação residual

O dataset distribui os segmentos como WAV, mas os CSVs não demonstram toda a
cadeia de codificação anterior de cada gravação. O manifesto registra
`dataset_wav_prior_lossy_history_not_declared`. A auditoria confirmou que todos
os 15.097 arquivos distribuídos são WAVE/PCM de 16 bits, com frequência e
canais coerentes com os metadados, mas isso não prova a cadeia anterior à
publicação. Como o efeito de compressão será estimado de forma pareada a partir
do mesmo master, essa incerteza permanece uma limitação de generalização, não
uma diferença entre tratamentos.

## Plano de aquisição do ECOSoundSet

O DOI conceitual `10.5281/zenodo.15043892` resolve atualmente para o registro
`10.5281/zenodo.18636037`. O CSV oficial possui 41.883 segmentos únicos e foi
associado ao manifesto congelado por gravação, início, fim e split. As 15.097
janelas selecionadas tiveram correspondência única com 15.097 nomes de WAV.
O registro distribui o conjunto sob **CC BY-NC 4.0**; essa restrição de uso
não comercial é registrada no manifesto e deve ser preservada nos derivados.

Os dois arquivos oficiais de segmentos somam 22.869.982.595 bytes e passaram
pelos MD5 publicados. Os 15.097 WAVs extraídos ocupam 13.620.120.838 bytes.
Todos receberam SHA-256 e tiveram o cabeçalho lido; 143 têm duração real
inferior a quatro segundos e foram excluídos, deixando 14.954 janelas. Como a
máquina local dispõe de cerca de 17 GiB livres e já conserva o PPA1, a
aquisição integral foi concluída no EASNCloud. O arquivo
`data/manifests/ecosoundset_download_manifest_v1.csv` congela exatamente o que
foi extraído e verificado no servidor.

# Aves — PteroSet/PPA1

## Por que este recorte

O PteroSet completo ocupa aproximadamente 86,4 GB. O projeto PPA1 isolado ocupa
15.257.785.797 bytes e preserva a estrutura necessária ao estudo: arquivos WAV
originais a 192 kHz, anotações temporais por espécie, metadados de local e
gravador e licença CC BY 4.0. Ele substitui o recorte Xeno-Canto anteriormente
bloqueado por predominância de MP3.

| Medida | Resultado |
|---|---:|
| Gravações WAV | 108 |
| Grupos local–gravador | 29 |
| Eventos identificados por espécie | 2.242 |
| Espécies anotadas no PPA1 | 105 |
| Espécies representadas nas janelas retidas | 104 |
| Snapshots de 10 s | 5.184 |
| Janelas congeladas de 4 s | 10.368 |
| Duração útil derivada | 11,52 h |
| Espécies-alvo | 10 |
| Janelas com ao menos uma espécie-alvo | 1.259 |
| Janelas de background para as espécies-alvo | 9.109 |
| Treino | 7.392 janelas; 20 grupos |
| Validação | 1.248 janelas; 4 grupos |
| Teste | 1.728 janelas; 5 grupos |

## Segmentação sem duplicação

Cada snapshot temporal de 10 s gera duas janelas: `[0,4)` e `[4,8)`. Os 2 s
finais são descartados. No PPA1, snapshots consecutivos começam a cada 9 s e
compartilham 1 s de áudio. A regra escolhida deixa um intervalo de 1 s antes do
snapshot seguinte e, portanto, não inclui o crossfade em duas amostras.

Um evento entra nos rótulos da janela quando a interseção temporal é de pelo
menos 0,25 s. A amostra continua multilabel. Janelas sem uma das dez espécies
selecionadas permanecem no manifesto como background, inclusive quando contêm
outras aves identificadas.

## Espécies-alvo e partição

Uma espécie é selecionada quando aparece em pelo menos 70 janelas e 15 grupos
antes do split. A partição é otimizada de forma determinística com semente
`20261007`, mas sua unidade indivisível é a combinação local–gravador. Não há
grupo compartilhado entre treino, validação e teste.

| Espécie | Total | Treino | Validação | Teste | Grupos |
|---|---:|---:|---:|---:|---:|
| *Cacicus cela* | 71 | 48 | 7 | 16 | 19 |
| *Crypturellus cinereus* | 115 | 76 | 19 | 20 | 21 |
| *Crypturellus undulatus* | 80 | 58 | 11 | 11 | 17 |
| *Cyanocorax violaceus* | 364 | 270 | 46 | 48 | 29 |
| *Ortalis guttata* | 109 | 77 | 14 | 18 | 24 |
| *Orthopsittaca manilatus* | 81 | 61 | 7 | 13 | 16 |
| *Psarocolius angustifrons* | 172 | 119 | 19 | 34 | 26 |
| *Ramphastos tucanus* | 158 | 118 | 17 | 23 | 26 |
| *Rupornis magnirostris* | 147 | 101 | 23 | 23 | 23 |
| *Sclateria naevia* | 135 | 92 | 18 | 25 | 16 |

## Verificação do áudio

O arquivo `PPA1.zip` tem 15.257.785.797 bytes e MD5
`d5fce3e9d743bad4988131b57aa1c9bc`, iguais ao registro do Zenodo. Seu SHA-256
local é `7bf3b8fdea64cae5ebbb47a951c95e25dc9dca47d55feef45be270383fb1d605`.
Os 108 arquivos são RIFF/WAVE PCM inteiro, mono, 16 bits, 192 kHz e 433 s; todos
foram lidos integralmente e receberam SHA-256 individual.

O ZIP foi mantido sem extração permanente para preservar espaço em disco. Isso
não reduz a auditoria: a leitura em streaming descompacta todo o conteúdo,
aciona a verificação CRC da biblioteca ZIP e permite analisar cada cabeçalho e
calcular seu hash. A extração sob demanda e a padronização para 48 kHz partem,
portanto, de fontes sem perdas verificadas.

Os antigos arquivos `bird_frontends_*_v1.csv` permanecem somente como histórico
da alternativa rejeitada. Eles não fazem parte do lock ativo 1.3.0.

# Artefatos

| Arquivo | Conteúdo |
|---|---|
| `data/manifests/ecosoundset_catalog_v1.csv` | Todas as janelas auditadas e seus motivos de exclusão |
| `data/manifests/ecosoundset_manifest_v1.csv` | Janelas selecionadas para o estudo principal |
| `data/manifests/ecosoundset_download_manifest_v1.csv` | Associação das janelas selecionadas aos nomes oficiais dos WAVs |
| `data/manifests/pteroset_ppa1_catalog_v1.csv` | Todas as janelas PPA1 derivadas e seus rótulos antes da seleção final |
| `data/manifests/pteroset_ppa1_manifest_v1.csv` | Janelas, espécies e splits congelados para a réplica em aves |
| `data/manifest-lock-v1.json` | Regras, revisões, hashes e contagens congeladas |
| `data/audit/corpus-audit-v1.json` | Resumo legível por máquina |
| `data/audit/corpus-audit-v1.xlsx` | Resumo para revisão do grupo |
| `data/audit/pteroset_ppa1_audio_files_v1.csv` | Cabeçalho, integridade e SHA-256 de cada WAV do PPA1 |
| `data/audit/pteroset_ppa1_audio_verification_v1.json` | Resultado consolidado da verificação do arquivo PPA1 |
| `data/audit/ecosoundset_download_plan_v1.json` | Registro Zenodo, hashes, tamanhos e decisão de execução no EASNCloud |
| `data/audit/ecosoundset_audio_files_v1.csv` | Cabeçalho, duração, elegibilidade e SHA-256 dos 15.097 WAVs auditados |
| `data/audit/ecosoundset_audio_verification_v1.json` | Resultado consolidado da auditoria de áudio do ECOSoundSet |

Os manifests podem ser reproduzidos com:

```bash
python3 scripts/build_manifests.py
```

O script não acessa a rede nem baixa áudio.
