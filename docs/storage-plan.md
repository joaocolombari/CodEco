# Capacidade local e execução dos corpora

Data da avaliação: 2026-10-13

## Resultado

O piloto cabe nesta máquina desde que use uma amostra estratificada e processe
as condições de codec em streaming, descartando reconstruções temporárias após
o cálculo das métricas. A preparação integral dos dois corpora não cabe com uma
margem operacional segura e deverá ser concluída no EASNCloud.

## Capacidade observada

| Recurso | Valor |
|---|---:|
| Memória física | 8 GiB |
| Memória livre informada por `memory_pressure` | 36% no momento da auditoria |
| CPUs lógicas | 8 |
| Espaço livre observado | aproximadamente 17 GiB; 92% do volume em uso |
| PPA1 já armazenado | 15.257.785.797 bytes; aproximadamente 14,21 GiB |

O processamento deve usar leitura incremental, batches pequenos e somente um
modelo residente por vez. O corpus nunca deve ser carregado integralmente na
RAM. Modelos fundacionais ou codecs neurais deverão iniciar com batch entre 1 e
8 e ajustar para baixo se houver pressão de memória.

## Orçamento de disco

| Produto | Estimativa |
|---|---:|
| ECOSoundSet: 15.097 WAVs selecionados | 12,69 GiB |
| Master ECOSoundSet, mono/48 kHz/PCM16 | 5,40 GiB |
| Master PPA1, mono/48 kHz/PCM16 | 3,71 GiB |
| Masters dos dois corpora | 9,11 GiB |
| Arquivos oficiais `Split recordings1/2` | 22.869.982.595 bytes; aproximadamente 21,30 GiB |

Os dois arquivos compactados do ECOSoundSet, sozinhos, já excedem o espaço
livre. Mesmo uma extração seletiva por streaming deixaria pouco espaço após os
12,69 GiB de WAVs e não comportaria simultaneamente os masters dos dois corpora.

## Piloto local

O piloto local deverá usar 10% ou menos de cada manifesto, com amostragem
determinística estratificada por corpus, split, espécie e grupo. Dez por cento
dos masters ocupam aproximadamente 0,91 GiB; 10% dos WAVs-fonte selecionados do
ECOSoundSet acrescentam cerca de 1,27 GiB. Mesmo reservando mais 1 GiB para uma
condição temporária, o piloto permanece abaixo de aproximadamente 3,2 GiB
adicionais. Bitstreams e reconstruções devem ser produzidos uma condição por
vez e removidos depois que hashes, payload e métricas forem persistidos.

O piloto local serve para validar:

- leitura e padronização dos WAVs;
- round trip dos codecs;
- contabilização do payload;
- implementação dos índices;
- estimativa de tempo e pico de memória;
- redução da grade de bitrates.

Ele não substitui o treinamento e a avaliação finais nos manifests completos.

## Execução no EASNCloud

O host `easn-cloud` foi inspecionado em 2026-10-13 e apresentou 92 GiB livres
antes da aquisição e 58 GiB após conservar os dois arquivos oficiais e os
15.097 WAVs extraídos; possui 14 GiB de RAM física e 4 GiB de swap. A aquisição
foi preparada em
`/srv/easn/shared/codeco`, isolada dos serviços existentes em `/srv/easn/app`.

Antes do download integral, o servidor deve manter pelo menos 50 GiB
livres para a preparação inicial. Uma margem de 100 GiB continua preferível para
manter os arquivos selecionados, os masters, caches e uma condição temporária
sem operar próximo da capacidade máxima.

O procedimento foi concluído com:

1. baixar cada arquivo oficial com retomada;
2. verificar o MD5 publicado antes da extração;
3. extrair apenas os membros presentes em
   `data/manifests/ecosoundset_download_manifest_v1.csv`;
4. conferir RIFF/WAVE, PCM, frequência, canais, duração e hash de cada membro;
5. excluir 143 clipes cuja duração real é inferior a 4 s, sem introduzir padding;
6. liberar 14.954 janelas completas no lock 1.3.0;
7. produzir o master mono/48 kHz sem normalização de ganho na próxima etapa;
8. conservar os arquivos de origem no armazenamento científico ou documentar
   uma política verificável de reaquisição;
9. executar codecs e reconstruções em streaming, uma condição por vez.
