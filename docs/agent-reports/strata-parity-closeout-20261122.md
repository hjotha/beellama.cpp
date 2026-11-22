# Fechamento da paridade Strata × BeeLlama

A paridade de prefill ainda está aberta. Na A100-40G, o Qwen3.8-Flash-Next
Q2_0 fez **1.946,6 t/s em 28k** com a melhor configuração integrada. A
Strata 0.2.17 fez **4.629,3 t/s**, 2,38× o throughput do BeeLlama. Com
DFlash2 n-max 8, o BeeLlama superou a Strata no decode após 4k
(**95,4 vs 67,4 t/s**) e 28k (**87,3 vs 75,0 t/s**). No prompt curto,
a Strata ficou na frente (**85,9 vs 74,0 t/s**).

## Integrado e validado

| Entrega | Commit no histórico de main | Resultado |
|---|---|---|
| QSA esparsa com KV Q8_0 | `5f46a18da` | 28/28 casos da matriz CUDA. No A/B isolado, 1.513,9 → 1.516,4 t/s em pp28k e 39,62 → 43,53 t/s no decode pós-28k (+9,9%). |
| Residência de experts em prefill | `1e682081c` | 1.514,0 → 1.567,4 t/s em pp28k no A/B inicial (+3,5%). |
| Admissão parcial do cache MoE | `6cefde1a5` | IQ3_S, `-ncmoe 44`, 647,2 → 818,9 t/s em pp28k com cache de 16.000 MiB (+26,5%) e saída idêntica. Com cache de 20.000 MiB, 914,8 t/s em pp28k e 25,5 t/s no decode. |
| MMQ com Q8_1 compartilhado | `f0e2648a8` | +1,0–1,6% de prefill, bit a bit idêntico. |
| MMQ Q2_0 com gate/up/GLU fundidos | `3fd591f0b` | **+5,3%** de prefill pp28k com compartilhamento Q8_1 nos dois braços, bit a bit idêntico. |
| Perfil por operação na GPU | `9f0159147` | No IQ3_S, `MUL_MAT_ID` foi 42,9% do tempo medido na GPU. Os grafos CUDA somaram 4,665 s de 43,8 s de parede. |
| DFlash2 no Qwen4Exp e correção M-RoPE | `1d0ecd6de`, `b1e56c896` | 95,4/87,3 t/s pós-4k/pós-28k com n-max 8 e drafter Q8_0. |
| Fixtures RKVA e companion atualizados | `26d4ba616` | 3/3 passaram no CTest. Os fixtures Python extraem o helper KVarN real e incluem `llama.h`. O fixture E4 verifica que Q8 usa o layout 5 com capability própria. |

A receita de prefill usa `-b 2048 -ub 2048`, GLU e SHARED_Q8. A receita
DFlash2 usa `-ub 1024` para caber na A100. O braço que solicitou
`--spec-dm-controller profit` mediu 96,6/88,0 t/s pós-4k/pós-28k. O
controlador já é default e o seletor DFlash2 o desativa no código atual.
A configuração reproduzível usa n-max 8.

[Receitas e parâmetros completos](strata-parity-status-20261122.md#3-recipes-a100-40g)

## Medido e fora da integração

**Reuso do mapa MMQ no down.** A100-40G, Q2_0, `parity14` (`48a6c549a`),
três rodadas por braço com GLU e SHARED_Q8 ativos. Medianas pp4k **1.882,35 →
1.896,82** t/s, pp28k **1.948,68 → 1.962,36 t/s** (+0,70%) e decode pós-28k
**45,17 → 45,47** t/s. Em pp28k, as faixas foram 1.947,92–1.948,87 e
1.962,19–1.962,81 t/s. Hashes iguais nos dois prompts (`09f5c1647f89`,
`39d5087ff47a`), pico de 38.906 MiB nos dois braços e NMSE 0 no fixture da
4070 com e sem CUDA Graphs. O ganho ficou abaixo do piso de 3%.
`LLAMA_MMQ_MOE_REUSE_MAP` permanece default-off no código arquivado em
`archive/strata-expert-major-20261122` (`05f1d7d76`, tag local).

## Fallback CPU keyed

O build CUDA SM80 original em `/work-cpufb` terminou na .57 com o commit
`38e525698`. Na RTX 4070, `test-llama-archs -a '^deepseek2$' -b CUDA0 -s 12345`
passou 3/3 com `LLAMA_MOE_CPU_FALLBACK=0` e com `=1`. Os filtros
`MUL_MAT_ID` e `MUL_MAT` de `test-backend-ops` reportaram 2/2 cada
(1.088/1.088 e 2.604/2.604 casos executados em CUDA). O fixture CPU keyed
passou **63/63 com NMSE 0**, incluindo três misses por token, nos tipos
Q4_K, Q3_K e F16. No fixture CUDA de cache parcial, os logits de
prefill+decode também deram NMSE 0. O contador de misses computados no
fallback ficou sem registros nesse fixture.

O smoke adicional com OLMoE Q2_K executou o fallback de fato. Com o mesmo
prompt greedy, 32 tokens de saída e `-cmoe --moe-cache-mib 64`, o braço on
contou **3.794 IDs computados na CPU** e 182 atendidos pelo cache. O hash
mudou de `942dc5fee967` para `4467fad8fbeb` em uma rodada por braço.
O fixture CPU isola o roteamento e a soma por posição. A equivalência em um
GGUF real depende também da aritmética das duas backends.

O A/B usou a mesma A100-SXM4-40GB e o mesmo binário `38e525698` para o
Qwen3.8-Flash-Next IQ3_S. O comando do servidor fixou `-ngl 99 -c 32768 -b 4096
-ub 2048 -ncmoe 36 --moe-cache-mib 20000 --parallel 1 --fit off -fa on
-ctk q8_0 -ctv q8_0 --load-mode mmap --lazy-mode on --jinja --no-warmup -lv 4`.
Nos dois braços, `LLAMA_MOE_CACHE_PREFILL`, `LLAMA_MOE_MAP_DEVICE` e
`LLAMA_MOE_CACHE_PARTIAL_PREFILL` receberam `1`. O `bench.py` usou
três execuções por braço, dois warmups por execução, `temperature 0`,
`cache_prompt false` e o caso `pp28k+dec` (28.547 tokens de prompt,
até 128 de saída). [GGUF IQ3_S fixado](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/tree/2a55d75962e22f7a4a1d9963ab6eae3678537831/IQ3_S)

| `LLAMA_MOE_CPU_FALLBACK` | Prefill pp28k, mediana (faixa), t/s | Decode pós-28k, mediana (faixa), t/s | Pico de VRAM | SHA-256 da saída |
|---|---:|---:|---:|---|
| 0 | 1.070,27 (1.051,07–1.075,74) | 26,80 (25,35–26,81) | 40.158 MiB | `9fb4c487bd56` |
| 1 | 1.057,99 (1.051,23–1.061,93) | 11,24 (11,06–12,59) | 40.174 MiB | `198b55638dd8` |

Foram 3/3 execuções completas em cada braço, com o hash estável dentro de
cada braço. Os hashes entre braços divergiram (86 vs 98 tokens gerados),
o prefill variou **−1,15%** com sobreposição das faixas e o decode perdeu
**58,06%**. Cada execução com a flag ligada registrou **14.548 IDs calculados
na CPU** e 24.692 no cache. O patch ficou fora de `main` e a branch
`feat/cpufb-verify` preserva o protótipo em `38e525698`.

O Colab apresentou falha no canal de notebook e interrompeu uma transferência
HTTP/2 antes do A/B. A rodada válida usou a terceira VM alocada, o arquivo do
build original verificado por SHA-256 e os dois shards baixados por faixas
validadas. A medição ocorreu pelo `colab exec`, sem compilação na VM. Os
artefatos locais estão em `/tmp/opencode/strata-closeout/cpufb.json` e
`cpufb-logs.tar.gz`. O JSON tem SHA-256
`d16f785f28b205a60d243b7dfa4c64670f731b166b4713f8dcee837c3daf7db2`.

## Pendências

- **QSA 4c.** O desenho e o A/B para levar os IDs selecionados à FA CUDA estão
  no plano. Implementação, comparação numérica e medição continuam abertos.
- **Fusão completa gate/up/GLU/down.** A validação depende de reproduzir os
  blocos Q8_1 intermediários e a ordem da redução do down. O plano registra
  as condições numéricas e o fixture necessário.
- **Janela de 256k.** Falta uma medição pareada de throughput e pico de VRAM
  com `-c 262144`.
- **Hybrid RPC prefix-state.** Compila e permanece desabilitado pelo contrato
  obsoleto de SWA e exact-tail, com motivo no CMake.
- **Replay CUDA por largura de verificação e hidden MTP via D2D.** Seguem como
  itens Decode-5 e Decode-5b do plano original.

[Plano e critérios](../PLAN-strata-parity.md)

## Estado final

No gokaya, `llama-server-root.service` voltou a `active` e
`127.0.0.1:8090/health` respondeu `{"status":"ok"}`. O Colab confirmou zero
sessões ativas e `pgrep -c nvcc` retornou 0, sem processos CMake em execução.
`df -h /home/hjotha` mostrou 48G livres e 90% de uso. Os worktrees temporários
da tarefa e os binds do chroot foram removidos. O reuso ficou na tag local
de arquivo e o fallback ficou na branch local acima.
