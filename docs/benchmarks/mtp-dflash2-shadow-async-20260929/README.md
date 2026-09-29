# Observação paralela assíncrona do DFlash2 (Etapa 1, itens 2–4) — 2026-09-29

> **Revisão de 2026-09-29:** os contadores de coincidência e prontidão destes
> runs foram invalidados pela revisão do observador: bônus lido do batch de
> entrada, resultados descartados antes do commit e prazo medido depois do MTP.
> A KV auxiliar também podia reter linhas rejeitadas. Os tempos de geração são
> registros reais desta implementação, mas não aprovam/rejeitam a hipótese de
> reaproveitamento. A separação antiga de etapas exclui as sincronizações dos
> getters; o residual não comprova custo de preparo/amostragem. Ver a nova
> [medição corrigida](../mtp-dflash2-shadow-corrected-20260929/README.md).

Objetivo: rodar o bloco do auxiliar DFlash2 na Radeon, a partir do **mesmo
prefixo/âncora do MTP**, em um worker, e medir prontidão na decisão, coincidência
do prefixo (incluindo bônus) e sufixo utilizável. Nenhum token auxiliar entra na
resposta.

## Implementação

- `common_speculative::shadow` (impl DFlash2 próprio, split local) sincronizado
  com o prefixo confirmado; o worker roda `draft()` do auxiliar via
  `std::async` um bloco por vez.
- O thread principal nunca toca o contexto auxiliar com worker em voo:
  `process()` copia as features do alvo para uma fila e injeta quando o worker
  termina; o join só ocorre quando o future já está pronto (`wait_for(0)`) ou em
  `begin()/free()`; ciclos com worker ocupado são pulados e contados.
- Cruza o resultado com a sequência realmente confirmada, com `c = n_accepted +
  1` (inclui bônus) pareado pela âncora do draft verificado.
- Flags: `--spec-draft-shadow-*`; throttle opcional
  `GGML_DFLASH_SHADOW_EVERY=N` (1 bloco a cada N drafts primários); perfil por
  ciclo com `GGML_DFLASH_SHADOW_PROF=1`.

## Resultados de observação (bateria Etapa 0: 3 curtas + 3 de código + 2 longas)

Run final (`run3`, gate "só com draft primário" + pareamento corrigido):

- 411 blocos lançados, 193 observações com verificação correspondente;
- prontidão: 407 prontos na decisão, 4 atrasados (~99%);
- coincidência completa do prefixo (incluindo bônus): **7 de 193 (~3,6%)**;
- sufixo utilizável: **7 blocos, 33 tokens** em ~1600 tokens gerados;
- distância típica até a coincidência completa: 3 tokens (102 casos; o auxiliar
  costuma acertar os tokens aceitos do MTP e divergir no bônus);
- sem falhas (`errors=0`).

Runs anteriores: `run1-pre-fix/` (bug de pareamento do `c`) e `run2-paired/`
(pareamento corrigido, 177 observações, 3 coincidências). Todos com hashes de
saída idênticos ao MTP isolado.

## Custo

| Configuração | Repetição | Código | Longo |
| --- | ---: | ---: | ---: |
| MTP n=4 isolado (Etapa 0) | 84,50 | 59,08 | 57,41 |
| + auxiliar sincronizado, sem blocos | 81,65 | 56,61 | 54,76 |
| + observação assíncrona | 57,3–58,4 | 43,3–44,5 | 38,7–42,4 |

A observação paralela custa ~30% no MVP. É o caso 3 da matriz da seção 9: o
trabalho paralelo cabe no prazo, mas raramente coincide por completo e hoje
custa mais do que o valor medido.

## Instrumentação por etapa e throttle (runs 4–5)

| Configuração | Repetição | Código | Longo | CPU efetiva | PPT |
| --- | ---: | ---: | ---: | ---: | ---: |
| MTP n=4 isolado | 84,50 | 59,08 | 57,41 | 2,80 GHz | 8,0 W |
| + auxiliar sincronizado | 81,65 | 56,61 | 54,76 | 2,76 GHz | 10,1 W |
| + observação, `EVERY=1` | 59,48 | 46,20 | 41,43 | 1,40 GHz | 18,2 W |
| + observação, `EVERY=4` | 80,32 | 55,57 | 54,24 | 2,61 GHz | 12,1 W |

- `GGML_DFLASH_SHADOW_EVERY=4` devolve quase todo o desempenho (−1,6% sobre o
  piso síncrono) e preserva a amostragem de blocos para observação.
- Estágios por bloco (probe dedicado, 40 blocos, `../mtp-dflash2-stage-probe-20260929/`):
  worker 55,3 ms (Radeon ~15,7 ms + seletor na 4070 ~4,9 ms + ~35 ms de preparo
  e amostragem), injeção 2,2 ms e cópia 1,1 ms no thread principal.
- A causa dominante da perda não é o thread principal (~4% do ciclo): o bloco
  contínuo na Radeon consome o orçamento compartilhado da APU e derruba o clock
  efetivo da CPU (~2,8 → ~1,4 GHz; PPT 8 → 18 W), desacelerando o caminho
  primário. O throttle resolve porque deixa a APU ociosa na maior parte do tempo.

## Condições e ressalvas

- Run3 (`cpu-original.json` no diretório raiz) rodou enquanto um ator externo
  mudou a política de CPU (EPP `balance_power`, máx 5,13 GHz); o controlador
  detectou a divergência na restauração e a política foi restaurada ao baseline
  documentado (powersave/`power`/3,3 GHz) e verificada (diferença NONE). Os runs
  1 e 2 usaram a política original normal e deram os mesmos números.
- As ressalvas originais de ausência de throttle/perfil se aplicavam aos runs
  1–3. Os runs 4–5 e o probe posterior exercitaram esses caminhos, com as
  limitações de medição descritas no aviso de revisão acima.

`run1-pre-fix/`, `run2-paired/` e o diretório raiz (`run3`) contêm comandos,
logs, respostas, telemetria e hashes. `SUDO_PASS=<senha> python3 runner.py`
reproduz (para e restaura os serviços).
