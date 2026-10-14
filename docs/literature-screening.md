# Triagem da literatura sobre compressão em bioacústica

Data da consolidação: 2026-10-07

Este documento registra duas consultas complementares, seus critérios de triagem e a síntese dos textos selecionados. As duas consultas são usadas conjuntamente para posicionar o projeto; nenhuma delas é tratada isoladamente como revisão sistemática exaustiva.

Arquivos bibliográficos:

- `savedrecs_1.bib`: export integral da Query 1, com 22 registros;
- `savedrecs_screening_1.bib`: referências mantidas após o screening da Query 1;
- `savedrecs_2.bib`: export integral da Query 2, com 73 registros;
- `savedrecs_screening_2.bib`: referências centrais ou metodologicamente úteis mantidas após o screening da Query 2.

## Critérios comuns de triagem

A triagem foi feita por título e abstract. Quando o PDF estava disponível, a classificação e as conclusões foram verificadas no texto completo.

Um trabalho foi considerado **central** quando avaliava pelo menos um dos seguintes pontos em áudio ambiental ou bioacústico:

- codec ou compressão com perdas;
- redução de dados para armazenamento ou transmissão;
- efeito da compressão sobre índices, características acústicas, embeddings, detecção ou classificação;
- representação latente, quantização, sparse coding ou compressive sensing com objetivo explícito de reduzir dados;
- reconstrução de áudio a partir de uma representação transmitida.

Um trabalho foi mantido como **apoio metodológico ou de implantação** quando contribuía para:

- escolha e robustez de embeddings ou frontends;
- autoencoders e representações compactas;
- redução de custo computacional no edge;
- desenho de sensores, redes e monitoramento remoto;
- metodologia de avaliação de codecs.

Foram excluídos trabalhos nos quais os termos da consulta tinham outro significado, por exemplo compressor de gás, compressão anatômica, compressão de imagens, arte ou música, bem como estudos bioacústicos puramente descritivos sem relação com redução, transmissão ou preservação de informação.

# Query 1

## Consulta

```text
ALL=(("codec" OR "compressed" OR "lossy")
     AND
     ("ecoacoustics" OR "soundscape" OR "bioacoustics"))
```

A Query 1 privilegia menções explícitas a codec, áudio comprimido ou compressão com perdas dentro de ecoacústica, paisagens sonoras e bioacústica. Ela é relativamente restrita em terminologia, mas recupera trabalhos que tratam diretamente de MP3, formatos compactos e arquiteturas que descrevem suas representações como comprimidas.

## Resultado do screening

- 22 registros no export;
- 6 registros do export mantidos;
- 16 registros do export excluídos;
- 1 artigo adicional mantido por seleção manual, fornecido em PDF, mas ausente do export;
- corpus final da Query 1: 7 referências.

O artigo adicional é *In-Vivo Vibroacoustic Surveillance of Trees in the Context of the IoT*. Sua referência foi inserida manualmente em `savedrecs_screening_1.bib`, acompanhada de um comentário que registra sua ausência em `savedrecs_1.bib`.

## Artigos mantidos

### Efeito de compressão sobre resultados ecoacústicos

1. **Heath et al. (2021). How index selection, compression, and recording schedule impact the description of ecological soundscapes. DOI: 10.1002/ece3.8042.**

   Avalia MP3, duração e agenda de gravação sobre sete índices analíticos e um fingerprint produzido por uma CNN do AudioSet. É central para a dissociação entre alterações nos índices e estabilidade da classificação.

2. **MacPhail et al. (2024). Audio data compression affects acoustic indices and reduces detections of birds by human listening and automated recognisers. DOI: 10.1080/09524622.2023.2290718.**

   Compara WAV com MP3 a 320 e 96 kbit/s, além de diferentes frequências de amostragem. Avalia escuta, distância, dois reconhecedores de aves e índices acústicos.

### Representações compactas e preservação

3. **Thakur et al. (2018). Compressed Convex Spectral Embedding for Bird Species Classification.**

   Usa projeções aleatórias sobre superframes espectrais e archetypal analysis para produzir representações compactas de espécies de aves.

4. **Di Loreto e Montelpare (2025). A two-stage architecture for soundscape classification and preservation. DOI: 10.1038/s41598-025-22819-0.**

   Usa VAE para gerar representação latente de mel-espectrogramas e CNN para classificação. É uma inspiração arquitetural, embora não contabilize um bitstream transmitido.

5. **Martin et al. (2021). Hybrid millidecade spectra: A practical format for exchange of long-term ambient sound data. DOI: 10.1121/10.0003324.**

   Propõe uma representação espectral compacta para armazenamento e intercâmbio de dados acústicos de longo prazo. Não é um codec reconstruível, mas constitui uma alternativa de produto acústico reduzido.

### Sensores e transmissão remota

6. **Rigakis et al. (2021). TreeVibes: Modern Tools for Global Monitoring of Trees for Borers. DOI: 10.3390/smartcities4010017.**

   Demonstra uma arquitetura IoT de aquisição e transmissão de vibrações internas de árvores, contextualizando restrições de energia, armazenamento e conectividade.

7. **Potamitis et al. (2019). In-Vivo Vibroacoustic Surveillance of Trees in the Context of the IoT. DOI: 10.3390/s19061366.**

   Usa Opus a 24 kbit/s em sinais adquiridos a 8 kHz e apresenta transmissão remota de clipes para detecção de insetos perfuradores. É motivação operacional, não comparação sistemática de codecs.

## Exclusões da Query 1

Os 16 registros excluídos se distribuíram principalmente entre:

- áudio espacial, realidade virtual e cinema digital;
- artes, música e estudos culturais de soundscape;
- compressão de imagem ou usos não acústicos do termo;
- acústica humana, propagação ou modelagem física sem relação com PAM;
- descrição taxonômica ou comportamental sem avaliação de redução de dados;
- classificação ou edge computing sem análise de codec, payload ou preservação da informação.

Alguns desses registros são tangenciais e reaparecem como apoio metodológico na Query 2, cujos critérios incluem explicitamente bitrate, representações latentes e passive acoustic monitoring.

## Síntese dos textos completos da Query 1

Heath et al. mostram alterações consideráveis e específicas de cada índice. A classificação de habitats é menos afetada: mesmo a compressão mais forte reduz pouco o desempenho, e o fingerprint AudioSet supera os índices analíticos. Entretanto, o estudo usa apenas três locais em um gradiente muito distinto de habitats, um único tipo de embedding e áudio originalmente adquirido a 44,1 kHz.

MacPhail et al. mostram que a robustez observada por Heath não é geral. MP3 a 96 kbit/s reduz detecções humanas e o recall dos reconhecedores; 320 kbit/s apresenta efeito menor. A frequência de amostragem também altera índices e classificadores. Portanto, a literatura não sustenta a afirmação de que compressão normalmente não apresenta compromisso: a magnitude depende da tarefa, do bitrate e do tipo de sinal.

Thakur et al. reduzem a dimensão de 1285 para 500, aproximadamente 61%, sem perda de classificação em um corpus de 26 espécies. Entretanto, dimensão não equivale a payload: não são avaliados quantização, serialização, entropia, cabeçalhos ou reconstrução do áudio.

Di Loreto e Montelpare reduzem um mel-espectrograma de aproximadamente 353 mil valores para um latente de 256 dimensões e apresentam essa redução como próxima de 1400 vezes. O estudo não converte o latente em bitstream nem mede bitrate, custo no edge ou perda de pacotes. O domínio é predominantemente urbano, não classificação bioacústica de espécies.

Martin et al. alcançam reduções de aproximadamente 20:1 a 80:1 em relação a espectros de 1 Hz, preservando resolução adequada para diversos produtos espectrais. A representação não permite recuperar a forma de onda e não foi validada como payload para classificadores de espécies.

TreeVibes e In-Vivo demonstram a necessidade prática de transmitir menos dados em sistemas remotos. Ambos oferecem contexto de hardware e rede, mas não isolam o efeito da compressão sobre índices, embeddings ou desempenho preditivo.

Conclusão da Query 1: a evidência direta é concentrada em MP3 e em poucos contextos. Trabalhos de representação compacta reportam redução dimensional, mas geralmente não medem payload real. Os trabalhos de IoT justificam a aplicação, mas não fecham a questão metodológica.

# Query 2

## Consulta

```text
TS=(("audio codec*" OR compress* OR lossy OR bitrate
     OR "neural audio codec*" OR "latent representation*")
    AND
    (ecoacoustic* OR bioacoustic* OR soundscape*
     OR "passive acoustic monitoring"))
```

A Query 2 cobre terminologia usada por trabalhos que não se descrevem apenas como estudos de codec. Ela inclui bitrate, compressive sensing, autoencoders, quantização e representação latente, mantendo o vínculo explícito com ecoacústica, bioacústica, soundscape ou PAM.

## Resultado do screening

- 73 registros avaliados por título e abstract;
- 13 registros centrais ou quase diretamente relacionados;
- 18 registros mantidos como apoio metodológico ou de implantação;
- 42 registros excluídos;
- corpus final da Query 2: 31 referências.

## Evidência central mantida

1. Di Loreto e Montelpare (2025): VAE e classificação em dois estágios.
2. Srivats e Nayak (2026): compressão task-aware de embeddings para telemetria.
3. Bjorck et al. (2019): alocação aprendida de bits para PAM de elefantes.
4. MacPhail et al. (2024): MP3, frequência de amostragem, índices e detecção de aves.
5. Heath et al. (2021): MP3, índices e classificação de soundscapes.
6. Martin et al. (2021): espectros millidecade para intercâmbio de dados.
7. Thakur et al. (2018): compressed convex spectral embeddings para aves.
8. Razik et al. (2015): sparse coding para mineração de cantos de baleias.
9. Stowell e Plumbley (2014): estabilidade de características de modulação sob MP3.
10. Dong et al. (2013): representação compacta de eventos bioacústicos para busca.
11. Mennitt e Fristrup (2012): medidas acústicas calibradas a partir de áudio comprimido.
12. Diaz et al. (2012): compressive sensing e reconstrução de anuros em WSN.
13. Logue et al. (2005): artefatos do codec ATRAC em gravações bioacústicas.

## Apoio metodológico ou de implantação mantido

1. Ntalampiras e Potamitis (2021): VAE e detecção de espécies ou indivíduos desconhecidos.
2. Xie et al. (2020): autoencoder com atenção para identificação individual de aves.
3. Lee et al. (2023): avaliação controlada do Opus em diferentes contextos e bitrates.
4. Khalid et al. (2026): VQ-VAE e códigos discretos de vocalizações de aves.
5. Matsumoto et al. (2015): FLAC e aquisição PAM em plataformas móveis.
6. Jeantet et al. (2026): redução de bandas, parâmetros, memória e energia para PAM embarcado.
7. Fernandez et al. (2026): classificador quantizado de belugas e generalização entre locais.
8. Alipek et al. (2023): autoencoder e classificação de morcegos; apoio metodológico fora do corpus de 48 kHz.
9. Matsumoto et al. (2020): transmissão e compressão em tecnologias acústicas oceânicas remotas.
10. Ivanko et al. (2025): detecção acústica e edge networks em cidades inteligentes.
11. Ghaffari e Devos (2024): comparação de frontends estáticos e aprendidos para aves.
12. Tan et al. (2023): compressed sensing associado a CNN para eventos acústicos.
13. Rigakis et al. (2021): arquitetura TreeVibes.
14. Balestriero et al. (2021): representação tempo-frequência aprendível e interpretável.
15. Bergler et al. (2019): autoencoder undercomplete para clustering de chamadas de orcas.
16. Lostanlen et al. (2019): PCEN e robustez a ruído em soundscapes.
17. Gaona-Gonzalez et al. (2011): captura e transmissão Bluetooth de vocalizações.
18. Grivas et al. (2009): monitoramento remoto audiovisual com transmissão comprimida.

## Exclusões da Query 2

Os 42 registros excluídos incluem principalmente:

- usos semânticos não relacionados de *compress* ou *compressed*;
- medicina, audição humana, biomecânica e propagação acústica;
- geração artística ou espacialização de áudio sem contribuição ao protocolo;
- taxonomia, comportamento ou ecologia descritiva sem redução de dados;
- ruído de compressores industriais;
- modelos de classificação sem relação com payload, representação compacta ou implantação.

## Síntese dos cinco textos completos prioritários da Query 2

### Srivats e Nayak (2026): embeddings orientados à tarefa

O trabalho utiliza 819 clipes de dois segundos de 15 classes do Watkins Marine Mammal Sound Database, reamostrados para 32 kHz. O conjunto contém 654 clipes de treinamento e 165 de teste, com estratificação por espécie e sessão.

São comparados VGGish, PaSST e Perch. O Perch produz 1536 valores FP32, ou 6144 bytes por clipe, e alcança weighted F1 de 0,9149 com regressão logística. Um autoencoder supervisionado reduz o embedding e PQ ou OPQ produz o código final. O resultado principal é um código de oito bytes por clipe, equivalente a 32 bit/s na cadência usada, com 97,14% de retenção do F1. Uma configuração de quatro bytes retém 94,30%.

Limitações:

- a razão de 768 vezes usa como referência o embedding FP32, não o áudio PCM;
- não há reconstrução do áudio nem recálculo de índices;
- o bottleneck é supervisionado para 15 classes e pode não preservar classes novas;
- não há comparação com MP3, Opus, LC3 ou codec neural reconstruível;
- o conjunto é pequeno e único;
- não são avaliados fluxo contínuo, perda de pacotes ou mudança de domínio;
- o payload não inclui framing, correção de erros e metadados;
- OPQ, codebooks e modelos precisam estar previamente disponíveis nas duas pontas;
- o autoencoder adiciona pouco custo, mas é necessário executar o Perch, estimado em aproximadamente 500 milhões de MACs por clipe de dois segundos.

O artigo demonstra a viabilidade de **task-aware embedding compression**, mas não preservação geral da informação acústica.

### Bjorck et al. (2019): alocação aprendida de bits

O trabalho usa gravações de elefantes de quatro localidades. Os sinais são reduzidos a 1 kHz e limitados a frequências abaixo de 100 Hz. Cada exemplo tem 25,5 segundos e é representado por 64 passos temporais e 47 bandas.

O método aprende uma precisão diferente para cada banda. Ruído gaussiano aproxima o erro de quantização durante o treinamento; no uso, os bits menos significativos são truncados. A abordagem supera uma alocação baseada na sensibilidade humana e reporta redução aproximada de 116 vezes com pequena degradação na classificação.

Não é um codec completo, não reconstrói áudio e não compara codecs reais. O bitrate é uma soma de precisões por banda, sem bitstream ou container. A tarefa é binária e específica para frequências abaixo de 100 Hz.

### Diaz et al. (2012): compressive sensing e reconstrução

O trabalho utiliza quatro espécies, segmentos de dez segundos e uma base de reconstrução construída com as frequências dominantes das chamadas. O sensor envia uma subamostragem e o servidor reconstrói o áudio com GPSR. A classificação permanece próxima de 95% usando aproximadamente 10% dos dados.

A fidelidade geral é mais modesta: correlação de Pearson média de 0,6120 e coerência média de 0,2434. A classificação continua alta porque a representação preserva precisamente as frequências utilizadas pelo classificador. O estudo não compara codecs, não contabiliza payload completo e não demonstra generalização para espécies novas.

### Stowell e Plumbley (2014): robustez dependente da representação

São utilizados três corpora: 45 gravações de cinco espécies, 762 de 84 espécies e 1390 de 56 espécies. O áudio é convertido para 48 kHz mono, com análise entre 2 e 10 kHz.

O teste de compressão utiliza apenas as 45 gravações do primeiro corpus, degradadas com MP3 a 64 kbit/s. A avaliação mede correlação das características antes/depois, não classificação comparativa. Três métodos mantêm correlação superior a 0,95; matching pursuit apresenta robustez muito inferior. O resultado sustenta que a sensibilidade à compressão depende da representação.

### Ghaffari e Devos (2024): controle do frontend

O estudo utiliza dez espécies do Xeno-Canto: 8569 gravações de treinamento, 1219 de validação e 2207 de teste. O áudio é reamostrado para 32 kHz. O treinamento usa recortes aleatórios de dois segundos; o teste usa janelas de cinco segundos com sobreposição.

São comparados espectrograma, mel-espectrograma, SincNet+ e LEAF com backend EfficientNet-B0. Entre 48 configurações, PCEN e normalização explicam grande parte da diferença entre representações estáticas e aprendidas. O melhor mel frontend alcança macro-F1 de 93,62%, e o melhor LEAF, 94,11%, com maior custo computacional.

Neste artigo, compressão significa compressão de faixa dinâmica, não redução de payload. A consequência para o projeto é manter frontend, normalização e classificador fixos entre condições de codec.

# Síntese integrada das duas queries

As duas consultas identificam três famílias de redução de dados:

1. **áudio comprimido ou subamostrado**, com possibilidade de reconstrução total ou aproximada;
2. **representação espectral ou latente compacta**, nem sempre quantizada ou serializada;
3. **payload orientado à tarefa**, otimizado para classificação e não para preservar todos os usos futuros.

Os resultados anteriores não permitem concluir simplesmente que “a compressão não apresenta muito compromisso”. O quadro sustentado pelas duas queries é:

- índices acústicos podem mudar consideravelmente e de maneira específica por índice;
- classificadores podem ser mais robustos, mas essa robustez depende da espécie, do bitrate, da representação e do regime de treinamento;
- códigos orientados à tarefa alcançam payloads extremamente pequenos, mas descartam informação necessária para reconstrução, índices, novas classes ou novas perguntas;
- reduções dimensionais publicadas nem sempre correspondem a bytes efetivamente transmitidos;
- estudos de IoT demonstram a necessidade operacional, mas raramente realizam comparações controladas.

## Gap consolidado

> A literatura avalia separadamente codecs perceptuais, compressive sensing, representações espectrais, autoencoders e transmissão de embeddings em diferentes contextos bioacústicos. Falta uma comparação pareada e controlada entre codecs legados e modernos de forma de onda, codecs neurais reconstruíveis e representações orientadas à tarefa, usando os mesmos segmentos, frequência de amostragem, partições e contabilização de payload. Também permanece pouco compreendida a dissociação entre fidelidade analítica — índices e propriedades espectrais — e utilidade preditiva — embeddings e classificação — sobretudo em gravações de campo de insetos e aves.

O projeto deverá comparar três paradigmas sem tratá-los como equivalentes:

1. **waveform transport:** o servidor recupera áudio e pode executar análises futuras;
2. **learned reconstructible transport:** um código neural permite reconstrução aproximada do áudio;
3. **task-aware representation transport:** o payload é mínimo para a tarefa atual, mas a reutilização é limitada.

## Hipóteses

### H1 — dissociação entre fidelidade analítica e preditiva

Em janelas de quatro segundos a 48 kHz, codecs perceptuais modernos preservarão a utilidade preditiva melhor que MP3 em payload equivalente, podendo permanecer não inferiores à referência sem perdas acima de um limiar de bitrate. Entretanto, índices ecoacústicos apresentarão vieses heterogêneos, dependentes do codec, do bitrate e do índice.

### H2 — vantagem condicionada da representação aprendida

Em payload equivalente, representações aprendidas e quantizadas alcançarão uma fronteira taxa–utilidade superior à dos codecs de forma de onda para classificação. Essa vantagem diminuirá quando a utilidade incluir reconstrução, índices ecoacústicos, classes não vistas ou novas tarefas, evidenciando o compromisso entre especialização e reutilização.

## Decisões metodológicas resultantes

- usar segmentos de 4 s e frequência de amostragem de 48 kHz;
- restringir o corpus a sinais com informação relevante abaixo do limite utilizável nessa frequência;
- avaliar insetos e aves separadamente, repetindo o mesmo protocolo;
- separar treino, validação e teste por gravação original e, quando possível, por local, data e gravador;
- manter frontend, normalização e classificador fixos entre codecs;
- comparar bytes reais por janela, não somente dimensão do vetor;
- reportar payload puro e payload de aplicação, incluindo framing e metadados;
- treinar primariamente em áudio sem perdas e testar todas as condições comprimidas;
- usar codec augmentation somente como análise secundária;
- medir reconstrução e índices apenas em métodos que recuperam áudio;
- testar representações orientadas à tarefa em gravações, locais e, se viável, classes não vistas;
- medir tempo, memória e energia do encoder em hardware-alvo ou plataforma de referência declarada;
- expressar os resultados por curvas e fronteiras de payload versus utilidade, evitando declarar um único vencedor universal.

## Estado da revisão

O material fundamenta o README e a primeira versão da matriz experimental sem alegações excessivas de novidade. Antes dos downloads e da implementação, a revisão do grupo deve congelar codecs, níveis de payload, embeddings, índices, espécies, critérios de elegibilidade dos arquivos de origem e a margem de não inferioridade. Essas decisões estão explicitadas em `experimental-matrix.md`.
