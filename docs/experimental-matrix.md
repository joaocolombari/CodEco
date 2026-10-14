# Matriz experimental

Data da versão: 2026-10-13

Esta matriz operacionaliza as hipóteses descritas no
[`README`](../README.md). Ela foi desenhada para isolar o efeito da redução de
dados e impedir que diferenças de corpus, frontend, partição ou treinamento
sejam interpretadas como diferenças entre codecs.

## 1. Estrutura do estudo

O estudo tem dois blocos independentes:

| Bloco | Corpus | Papel | Resultado reportado |
|---|---|---|---|
| I | ECOSoundSet, 34 Orthoptera elegíveis | Experimento principal em insetos | Curvas, testes e conclusão próprios |
| B | PteroSet/PPA1, 10 espécies-alvo | Réplica externa em aves | Curvas, testes e conclusão próprios |

Os blocos reutilizam o protocolo e o código, mas não compartilham exemplos,
modelo ajustado, normalização aprendida ou estimativas agregadas. Uma conclusão
comum só será apresentada quando a direção do efeito for compatível nos dois.

## 2. Constantes experimentais

| Fator controlado | Valor principal | Observação |
|---|---|---|
| Canais | mono | Regra de downmix versionada |
| Frequência de amostragem | 48 kHz | Nyquist de 24 kHz |
| Comprimento | 4 s | 192.000 amostras por janela |
| Referência numérica | PCM signed 16-bit | 384.000 bytes por janela, sem container |
| Segmentação | determinística | Política de padding e descarte deve ser congelada |
| Split | por gravação original e grupo | Local/data/gravador quando disponíveis |
| Frontend | único por bloco | Mesma STFT, escala e normalização |
| Classificador | único por bloco | Pesos congelados na avaliação principal |
| Entrada de treino principal | master sem perdas | Nenhum codec augmentation |
| Unidade de incerteza | gravação original | Evita tratar janelas correlacionadas como independentes |

O PCM de 16 bits é a referência operacional após padronização, não uma
afirmação de que o arquivo publicado originalmente tinha essa resolução.

## 3. Elegibilidade e auditoria dos dados

| Regra | Análise principal | Sensibilidade |
|---|---|---|
| Codec de origem | WAV/FLAC ou cadeia comprovadamente sem perdas | Origem comprimida, estratificada |
| Frequência original | pelo menos 48 kHz | Não usar upsampling como evidência principal |
| Conteúdo relevante | até 24 kHz | Espécies ultrassônicas excluídas |
| Rótulo | espécie verificável e licença utilizável | Rótulos fracos analisados à parte |
| Duplicatas | agrupadas pela gravação-fonte | Nenhuma duplicata cruza splits |
| Duração | WAV com 4 s reais | Clipes curtos excluídos; não usar padding na análise principal |

No bloco B, cada snapshot temporal de 10 s do PPA1 produz duas janelas em
`[0,4)` e `[4,8)`. Os 2 s finais são descartados. Como snapshots PPA1
consecutivos começam a cada 9 s, essa regra evita duplicar o segundo de
crossfade. As 29 combinações local–gravador são indivisíveis entre os splits.
As dez espécies-alvo têm pelo menos 70 janelas positivas e presença em 15
grupos antes da partição.

O arquivo oficial PPA1 foi validado por tamanho e MD5; seus 108 WAVs foram
lidos integralmente e conferidos como PCM mono de 16 bits a 192 kHz. A
padronização para 48 kHz parte, portanto, de fontes sem perdas verificadas.

No bloco I, os dois arquivos oficiais do ECOSoundSet foram verificados por MD5
e SHA-256. Todos os 15.097 WAVs selecionados são PCM de 16 bits e têm cabeçalho,
frequência e canais coerentes; 143 arquivos com duração real inferior a 4 s
foram excluídos sem padding. O manifesto analítico liberado contém 14.954
janelas, preserva as 34 espécies e usa 8.916/3.531/2.507 janelas em
treino/validação/teste.

Saída da auditoria: um manifesto por bloco com, no mínimo, `recording_id`,
`source_id`, `window_id`, táxon, rótulo, fonte, licença, formato original,
frequência original, canais, duração, local, data, gravador, split e motivo de
inclusão ou exclusão.

## 4. Métodos e níveis de payload

### 4.1 Núcleo confirmatório

| ID | Família | Condição | Níveis iniciais | Reconstrói áudio? | Entra em índices? |
|---|---|---|---|---|---|
| R0 | Referência | PCM s16le | 768 kbit/s | Sim, referência | Sim |
| R1 | Sem perdas | FLAC | Taxa observada | Sim, exato | Sim |
| L1 | Legado | MP3 | 96 e 320 kbit/s | Sim, aproximado | Sim |
| M1 | Moderno | Opus | 16, 24, 32, 48, 64 e 96 kbit/s | Sim, aproximado | Sim |
| M2 | Moderno | LC3, 48 kHz, frame de 10 ms | 16, 24, 32, 48, 64 e 96 kbit/s, se suportados | Sim, aproximado | Sim |

MP3 em 96 e 320 kbit/s reproduz condições presentes na literatura. Ele não
precisa ocupar todos os pontos da curva de baixo bitrate. Para Opus e LC3, o
piloto poderá retirar níveis redundantes, mas não acrescentará níveis depois
de aberto o teste final. LC3plus não integra a matriz: suas extensões de alta
resolução e menor duração de frame não são necessárias ao sinal mono/48 kHz de
quatro segundos e acrescentariam uma condição com licenciamento distinto.

### 4.2 Representações aprendidas

| ID | Família | Condição | Orçamentos-alvo | Reconstrói áudio? | Entra em índices? |
|---|---|---|---|---|---|
| N1 | Codec neural | EnCodec ou equivalente a 48 kHz | Níveis nativos próximos à grade comum | Sim, aproximado | Sim |
| N2 | Encoder–código–decoder | AE/VQ-VAE treinado no corpus de treino | 8, 16, 24, 32 e 64 kbit/s efetivos | Sim, aproximado | Sim |
| T1 | Task-aware | Embedding quantizado | 8, 16, 32 e 64 kbit/s ou menores | Não | Não |
| T2 | Task-aware | Bottleneck supervisionado quantizado | Mesmos orçamentos de T1 | Não, salvo decoder explícito | Não |

Os orçamentos são alvos de payload efetivo, não apenas dimensões latentes. Cada
linha deve contabilizar valores quantizados, índices de codebook, framing,
versão do modelo e metadados necessários para interpretação no servidor.

### 4.3 Grade comum para comparações

As comparações de taxa–utilidade usarão as faixas abaixo. Não se presume que
todo método atinja exatamente o centro da faixa.

| Faixa | Payload nominal | Bytes máximos aproximados por 4 s |
|---|---:|---:|
| P1 — extremo | até 16 kbit/s | 8.000 |
| P2 — muito baixo | >16 a 24 kbit/s | 12.000 |
| P3 — baixo | >24 a 32 kbit/s | 16.000 |
| P4 — intermediário | >32 a 64 kbit/s | 32.000 |
| P5 — alto | >64 a 96 kbit/s | 48.000 |
| P6 — ponte histórica | >96 a 320 kbit/s | 160.000 |

O valor efetivo será calculado pelo tamanho final serializado dividido por
quatro segundos. PCM e FLAC funcionam como referências, mesmo fora das faixas
de baixo bitrate.

## 5. Matriz de perguntas, contrastes e desfechos

| ID | Pergunta | Condições | Desfecho primário | Contraste | Hipótese |
|---|---|---|---|---|---|
| E1 | Quanto cada transformação altera os índices? | R0 versus R1/L1/M1/M2/N1/N2 | Erro relativo e viés por índice | Pareado por janela | H1 |
| E2 | O classificador congelado tolera compressão? | R0 versus todos os métodos | Δ macro-F1 | Não inferioridade a R0 | H1 |
| E3 | Codecs modernos superam MP3? | L1 versus M1/M2 em payload comparável | macro-F1 e erro de índices | Diferença pareada/curvas | H1 |
| E4 | O efeito depende do índice? | Métodos reconstruíveis × índices | Interação método × índice | Modelo hierárquico | H1 |
| E5 | Representação aprendida melhora utilidade por bit? | N1/N2/T1/T2 versus L1/M1/M2 | Fronteira payload–macro-F1 | Dominância de Pareto | H2 |
| E6 | A vantagem persiste para reutilização? | N1/N2/T1/T2 | Índices, tarefa secundária e OOD | Queda relativa | H2 |
| E7 | O padrão se repete entre domínios? | Bloco I versus Bloco B | Direção e magnitude dos efeitos | Síntese, sem pooling ingênuo | H1/H2 |

## 6. Métricas

### 6.1 Payload e sistema

| Métrica | Unidade | Prioridade |
|---|---:|---|
| Payload puro | bytes/janela e kbit/s | Primária |
| Payload de aplicação | bytes/janela e kbit/s | Primária |
| Razão contra PCM | vezes | Primária |
| Tempo de codificação | ms/janela e fator de tempo real | Secundária |
| Pico de RAM | bytes | Validação embarcada |
| Energia | mJ/janela e mJ/byte útil | Validação embarcada |

O payload puro mede o produto do encoder. O payload de aplicação acrescenta
cabeçalho, identificadores, versão, timestamp e framing exigidos pelo EASN. Os
dois valores devem ser reportados.

### 6.2 Classificação

| Métrica | Papel |
|---|---|
| Macro-F1 | Desfecho primário, sensível a classes minoritárias |
| Macro average precision | Secundário para desbalanceamento e multilabel |
| Recall por classe | Identificar espécies particularmente afetadas |
| ECE/Brier score | Verificar alteração de calibração |
| Taxa de abstenção, se usada | Separar incerteza de erro forçado |

A tarefa será formulada conforme os rótulos finais do corpus. Se houver
multilabel forte, macro average precision passa a acompanhar macro-F1 como
desfecho co-primário; essa decisão deve ocorrer antes do teste final.

### 6.3 Fidelidade acústica

| Grupo | Métricas candidatas | Uso |
|---|---|---|
| Espectral | distância log-espectral, erro de energia por banda | Núcleo |
| Temporal | erro de envelope e atividade | Núcleo |
| Reconstrução | SI-SDR e multi-resolution STFT distance | Secundário |
| Perceptual humano | métricas de fala/música | Não usar como conclusão ecológica |

Bandas congeladas para o erro de energia: 0–6, 6–12, 12–18 e 18–24 kHz. Elas
não substituem os limites próprios de cada índice e serão sempre calculadas com
o mesmo frontend.

### 6.4 Índices acústicos congelados

Todos os índices usam o master ou a reconstrução em mono/48 kHz, sem
normalização de ganho depois do codec. O frontend comum é STFT com janela Hann
periódica de 1.024 amostras, hop de 512, `center=false`, sem padding e banda de
análise de 0,5 kHz até Nyquist, com limite superior exclusivo.

| Índice | Papel | Parâmetros congelados |
|---|---|---|
| ACI | Primário | magnitude; 0,5–24 kHz; ACI calculado em quatro blocos de 1 s e então promediado |
| ADI | Primário | 24 bandas de 1 kHz entre 0–24 kHz; ocupação acima de −50 dBFS; Shannon normalizado |
| AEI | Primário | mesmas bandas e limiar do ADI; coeficiente de Gini das ocupações |
| H | Primário | `H = Ht × Hf`; `Ht` sobre envelope RMS por frame e `Hf` sobre potência média em 0,5–24 kHz; componentes normalizados e também reportados |
| BIO_2_24 | Primário | integral do espectro médio acima do mínimo da banda entre 2–24 kHz |
| BIO_2_8 | Sensibilidade | mesma definição entre 2–8 kHz para comparação com o uso clássico |
| NDSI_CLASSIC | Sensibilidade | antropofonia 1–2 kHz; biofonia 2–11 kHz |
| NDSI_EXTENDED | Sensibilidade | antropofonia 1–2 kHz; biofonia 2–24 kHz |

As definições numéricas, tratamento de silêncio e fórmula de dBFS estão
congelados em [`configs/acoustic-indices-v1.yaml`](../configs/acoustic-indices-v1.yaml).
Os cinco índices primários formam a família confirmatória; as duas componentes
de entropia são diagnósticas e BIO clássico/NDSI permanecem análises de
sensibilidade. Isso preserva a ponte com a literatura sem fazer das bandas
tradicionais de aves uma descrição obrigatória de Orthoptera.

### 6.5 Séries temporais

Aplicável somente quando timestamps e amostragem temporal forem adequados:

- viés e MAE da atividade por espécie;
- erro da curva média por horário;
- deslocamento do pico diário;
- erro na atividade acumulada;
- preservação da autocorrelação e do espectro temporal.

Clipes sem sequência temporal não serão artificialmente ordenados para essa
análise.

## 7. Regimes de treinamento

| Regime | Treino | Validação | Teste | Interpretação |
|---|---|---|---|---|
| A — principal | R0 sem perdas | R0 | R0 e todas as condições | Robustez e domain shift |
| B — adaptação | R0 + codecs no treino | Mesma política | Todas as condições | Degradação recuperável |
| C — específico | Uma condição por modelo | Mesma condição | Condição correspondente | Limite superior por método |

O Regime A é confirmatório. B e C são análises secundárias e não podem substituir
um resultado negativo no Regime A.

## 8. Partições e generalização

### Partição principal

- treino, validação e teste agrupados pela gravação-fonte;
- agrupamento adicional por local, data e gravador quando houver metadados;
- prevalência de classes documentada por split;
- teste fechado durante seleção de hiperparâmetros.

### Testes de generalização

| Teste | Treina em | Avalia em | Pergunta |
|---|---|---|---|
| G1 | fontes/locais de treino | fonte ou local não visto | O efeito depende do domínio? |
| G2 | classes de treino | sons fora da taxonomia-alvo | O código preserva novidade? |
| G3 | tarefa primária | tarefa/descritor secundário | O payload pode ser reutilizado? |
| G4 | insetos | nenhuma transferência de pesos obrigatória | O padrão reaparece em aves? |

G2 mede detecção de novidade ou preservação da estrutura, não classificação
supervisionada de uma classe sem exemplos.

## 9. Plano estatístico

### H1

- estimar a diferença pareada entre cada condição e R0;
- usar intervalo de confiança obtido com bootstrap agrupado por gravação;
- declarar não inferioridade apenas se o limite desfavorável do intervalo
  permanecer dentro de **−0,02 de macro-F1**;
- modelar o erro dos índices com efeitos de codec, bitrate e índice, incluindo
  interceptos de gravação e espécie quando sustentados pelos dados;
- corrigir multiplicidade dentro das famílias confirmatórias.

### H2

- construir curvas payload–utilidade usando payload observado;
- identificar condições não dominadas na fronteira de Pareto;
- comparar métodos em faixas comuns P1–P6, sem extrapolar curvas além dos
  pontos medidos;
- repetir a comparação com uma utilidade ampliada que penalize ausência de
  reconstrução, perda em tarefa secundária e degradação OOD.

Uma diferença não significativa não será tratada automaticamente como
equivalência. A conclusão de preservação exige o teste de não inferioridade e a
margem pré-especificada.

## 10. Execuções mínimas e extensões

### Núcleo mínimo

1. auditoria e manifesto do ECOSoundSet;
2. R0, R1, L1, M1 e M2;
3. classificador congelado;
4. métricas de payload, classificação, espectro e índices;
5. intervalos agrupados e curvas taxa–utilidade;
6. réplica do mesmo protocolo no recorte PteroSet/PPA1.

### Extensão para o artigo completo

1. N1, N2, T1 e T2;
2. regimes B e C de treinamento;
3. generalização G1–G3;
4. séries temporais, se suportadas pelo corpus;
5. falhas prolongadas de conexão e capacidade de backlog;
6. medição das duas alternativas finalistas no hardware EASN.

## 11. Simulação operacional do EASN

Depois de medir o payload, cada condição será projetada em cenários de
conectividade, sem misturar essa etapa com a avaliação do codec:

| Cenário | Indisponibilidade | Saída |
|---|---:|---|
| C0 | conexão contínua | tráfego diário e energia de transmissão |
| C1 | 5 min | backlog máximo e tempo de recuperação |
| C2 | 30 min | backlog máximo e tempo de recuperação |
| C3 | 120 min | risco de buffer cheio e dados descartados |

Perdas de pacotes isoladas não serão injetadas dentro do bitstream quando o
transporte real usar entrega confiável. O fenômeno operacional relevante é a
indisponibilidade da conexão e a perda eventual de registros completos.

## 12. Critérios de decisão

Um método será candidato ao firmware quando:

1. satisfizer a margem de não inferioridade na tarefa principal;
2. estiver na fronteira de Pareto de payload–utilidade;
3. não produzir viés inaceitável nos produtos que promete preservar;
4. possuir encoder executável ou portável para o hardware-alvo;
5. tiver licença compatível e bitstream/versão claramente identificáveis;
6. melhorar materialmente a capacidade de backlog ou o custo de transmissão.

O resultado final poderá recomendar mais de uma estratégia: um codec
reconstruível para amostras auditáveis e um payload orientado à tarefa para
telemetria contínua.

## 13. Decisões que precisam ser congeladas após o piloto

- lista e suporte mínimo das espécies em cada bloco;
- elegibilidade de fontes originalmente comprimidas;
- política para rótulos fortes, fracos e multilabel;
- implementação, framing, versão e requisitos de licença/qualificação do LC3;
- codec neural de 48 kHz e níveis efetivamente disponíveis;
- arquitetura-base e embedding dos classificadores;
- margem de não inferioridade de 2 pontos percentuais;
- níveis removidos da grade após o piloto;
- campos exatos incluídos no payload de aplicação.

Essas decisões devem ser registradas em configuração antes de acessar o teste
final.
