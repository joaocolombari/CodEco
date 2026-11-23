# Auditoria da integração do pipeline

Este documento registra como a infraestrutura experimental foi incorporada sem
substituir os manifests congelados nem promover resultados preliminares como
evidência final.

## Fonte dos dados e configuração

- `data/manifest-lock-v1.json` e os dois manifests CSV continuam autoritativos.
- A tabela otimizada é um cache derivado; ela não redefine janelas, rótulos,
  grupos ou splits.
- `configs/experiment-v1.json` é a fonte única da grade executável de codecs.
- O protocolo permanece em 4 s, mono, 48 kHz e PCM16. O Perch recebe padding
  simétrico somente depois da reconstrução.
- Toda execução registra commit, estado do Git, hashes, versões, hardware e
  backend em `results/run-metadata.json`.

## Correções metodológicas aplicadas

| Problema identificado | Tratamento na integração |
|---|---|
| Correlação não normalizada e somente lag positivo | Correlação cruzada normalizada, lag assinado, score e métricas antes/depois do alinhamento |
| Entropia normalizada pelo número de bins não nulos | Bins nulos contribuem zero, mas o suporte completo permanece no denominador |
| Validação aleatória interna do MLP | Desativada; treino, validação e teste seguem os grupos congelados |
| Limiar fixo em 0,5 | Um limiar por classe é escolhido no split de validação e aplicado ao teste |
| Probabilidades descartadas | Probabilidades, limiares, classes, grupos e rótulos de teste são persistidos |
| Apenas correlações dos índices | Acrescentados viés, MAE, RMSE e bootstrap por grupo de gravação |
| Múltiplas comparações | Ajuste Benjamini–Hochberg nas famílias de correlação e não inferioridade |
| Sem inferência de não inferioridade | Delta e IC bootstrap por grupo para macro-F1 e acurácia balanceada, margem provisória de 0,02 |
| Sem análise multicritério | Fronteira de Pareto para bitrate efetivo, Mel-SSIM e Mel-RMSE |
| CUDA obrigatória | CPU é fallback funcional; CUDA permanece aceleração opcional |
| FFmpeg compilado versionado | Binários ficam em `.tools/`, fora do Git, e são instalados pelo script fixado |
| Resultados antigos misturados ao código | Permanecem apenas na branch histórica; a integração exige nova execução |

## Pontos que continuam sendo validações empíricas

As mudanças abaixo não devem ser declaradas resolvidas sem uma execução
específica:

1. comparar numericamente BIO e os demais índices com uma implementação de
   referência e registrar eventuais diferenças de definição;
2. medir a banda efetiva do frontend BirdNET, em vez de inferi-la apenas pela
   taxa de amostragem;
3. quantificar o efeito do Nyquist de 16 kHz do Perch nas espécies retidas;
4. escolher e congelar definitivamente as margens de não inferioridade antes
   de abrir os novos resultados completos;
5. medir energia, RAM e flash diretamente nas plataformas ESP e Nordic nRF.

Os limites de 256 KiB de RAM e 1 MiB de flash permanecem uma validação
secundária. Eles não excluem métodos na primeira fase de comparação de
informação, payload, armazenamento e FLOPs.

## Critério para uma nova execução válida

Uma rodada completa pós-integração só deve ser promovida quando:

- os testes rápidos de todas as condições terminarem sem falhas;
- `verify_frozen_install.py --stage all` passar;
- `run-metadata.json` registrar o commit limpo e os hashes esperados;
- nenhuma classe ou grupo atravessar splits;
- todas as condições possuírem o mesmo conjunto de chaves de janela;
- falhas, retries e deslocamentos de alinhamento forem revisados;
- as análises forem reconstruídas a partir das probabilidades e resultados
  pareados dessa mesma rodada.
