# CodEco

**Codeco** é o espaço de trabalho do estudo sobre compressão de áudio e
representações compactas para nós de monitoramento ecoacústico. O projeto está
ligado ao sistema EASN e investiga quanta informação pode ser retirada do
payload sem comprometer as conclusões ecológicas produzidas no servidor.

O objetivo não é encontrar um codec universalmente melhor. O objetivo é medir
a fronteira entre **payload**, **fidelidade analítica**, **utilidade preditiva**
e **capacidade de reutilização**.

## Pergunta de pesquisa

> Em segmentos bioacústicos de quatro segundos, amostrados a 48 kHz, quanto o
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
- janelas de 4 s;
- conteúdo biologicamente relevante limitado a até 24 kHz;
- todas as derivações de uma janela pertencem sempre à mesma partição;
- comparação pareada: cada método recebe exatamente o mesmo sinal de entrada.

Sinais cuja identificação dependa de informação ultrassônica acima de 24 kHz
ficam fora do corpus principal. Reduções adicionais da frequência de
amostragem podem formar um estudo secundário, mas não serão misturadas ao
efeito dos codecs.

### Corpora

O protocolo foi concebido para dois domínios, analisados separadamente:

1. **Insetos — ECOSoundSet.** Corpus principal, com foco em Orthoptera e nos
   registros elegíveis após auditoria de proveniência e formato. A primeira
   versão liberada contém 14.954 janelas completas e 34 espécies-alvo. O depósito
   oficial usa licença CC BY-NC 4.0, que também limita o uso dos derivados.
2. **Aves — PteroSet/PPA1.** Réplica externa em áudio WAV original a 192 kHz,
   reamostrado uma única vez para 48 kHz. O recorte congelado tem 108 gravações,
   29 grupos de local–gravador, 10.368 janelas de 4 s e 10 espécies-alvo. O
   arquivo PPA1 tem 15,3 GB, evitando o download dos 86,4 GB do PteroSet inteiro.
   O ZIP oficial e os 108 WAVs foram integralmente verificados em 2026-10-08.

Não haverá mistura dos dois corpora em um único treinamento ou resultado. A
repetição em aves testa se os padrões de degradação se mantêm em outro domínio,
sem transformar diferenças de gravador, habitat ou taxonomia em efeito de
compressão.

## Reproduzir exatamente os corpora congelados

O freeze **1.3.0** pode ser reconstruído em Linux ou macOS com Git, `curl`,
Python 3.10 ou mais recente e aproximadamente **55 GiB livres**. O instalador
não depende de bibliotecas Python de terceiros; `requirements.txt` registra
essa condição explicitamente.

Em uma máquina nova:

```bash
git clone <URL_DO_REPOSITORIO> codeco
cd codeco
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
./scripts/install_frozen_corpora.sh all
```

O último comando clona os repositórios auxiliares nos commits congelados,
recria e confere os manifests, baixa os depósitos oficiais com retomada,
verifica tamanho e hashes dos arquivos e audita o áudio selecionado. Ele falha
em vez de aceitar silenciosamente uma revisão, arquivo ou seleção diferente.
A especificação legível por máquina está em
[`data/freeze-spec-v1.json`](data/freeze-spec-v1.json).

Modos menores também estão disponíveis:

```bash
./scripts/install_frozen_corpora.sh metadata     # sem baixar áudio
./scripts/install_frozen_corpora.sh ecosoundset  # somente insetos, 38 GiB livres
./scripts/install_frozen_corpora.sh pteroset     # somente aves, 17 GiB livres
```

Os downloads ficam em `data/downloads/`, os WAVs do ECOSoundSet em
`data/audio/ecosoundset/` e os checkouts auxiliares em `external/`. Esses
diretórios são reproduzíveis e ignorados pelo Git. No EASNCloud, execute o
comando dentro de um volume com espaço persistente. Não apague a cópia local
antes de confirmar o push deste repositório e a verificação completa no cloud.

Cada arquivo será auditado quanto a codec de origem, frequência de amostragem,
número de canais, duração, licença e relação com outras gravações. A análise
causal principal usará arquivos de origem sem perdas. Arquivos originalmente
comprimidos poderão aparecer apenas em análise de sensibilidade, identificados
como tais. Os resultados da auditoria estão em
[`docs/corpus-audit.md`](docs/corpus-audit.md).

### Famílias comparadas

| Família | Métodos | Função |
|---|---|---|
| Referência | PCM e FLAC | Estabelecer sinal e payload sem perdas |
| Legado | MP3 | Fazer a ponte direta com a literatura |
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
- erro relativo e viés dos índices acústicos selecionados;
- payload efetivo em bit/s;
- fronteira de Pareto entre payload e utilidade.

A margem provisória de não inferioridade da classificação é de **2 pontos
percentuais de macro-F1** em relação ao master sem perdas. Ela deverá ser
confirmada antes de observar os resultados finais.

### Secundários

- average precision e recall por classe;
- calibração das probabilidades;
- distância log-espectral e erro de energia por banda;
- preservação de séries de atividade, quando os metadados temporais permitirem;
- desempenho em gravações, locais ou fontes não vistos;
- tempo de codificação, pico de memória e energia em plataforma declarada.

## Fases

| Fase | Entrega | Critério de passagem |
|---|---|---|
| 0 | Auditoria e manifesto dos corpora | Proveniência, licenças e grupos definidos |
| 1 | Pipeline PCM/FLAC e partições | Reprodução determinística e sem vazamento |
| 2 | MP3, Opus e LC3 | Bitstreams decodificáveis e payload validado |
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
- [ ] confirmar licenças e implementações dos codecs;
- [ ] implementar e validar o pipeline de referência;
- [ ] executar o piloto de payload e escolher a grade final de bitrates.

## Documentação

- [Triagem e síntese da literatura](docs/literature-screening.md)
- [Matriz experimental](docs/experimental-matrix.md)
- [Auditoria e congelamento dos corpora](docs/corpus-audit.md)
- [Capacidade local e plano de execução](docs/storage-plan.md)
- [Parâmetros congelados dos índices](configs/acoustic-indices-v1.yaml)
- [BibTeX — Query 1](docs/savedrecs_screening_1.bib)
- [BibTeX — Query 2](docs/savedrecs_screening_2.bib)
