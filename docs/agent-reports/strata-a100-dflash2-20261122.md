# A100-40G: DFlash2 sobre Qwen3.8-Flash-Next Q2_0, perfil de prefill e PLE

Data: 2026-11-22 · worktree `/home/hjotha/worktrees/beellama-strata-opt` ·
branch `feat/strata-opt` @ `9f59e300f` · binário `collabosm-a100-parity6-20261121`
(Ubuntu 24.04 chroot, sm_80, CUDA FA + CUDA Graphs).

Alvo: `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0` (shard 1 + 2, 62 GB),
`-ncmoe 2 -c 32768 -b 2048 -ub 1024 -fa on -ctk q8_0 -ctv q8_0 --lazy-mode on
--load-mode mmap --fit off --parallel 1`, NVIDIA A100-SXM4-40GB (40.960 MiB).
Drafter: `Qwen3.8-Flash-Next-DFlash2-Q8_0.gguf` (622 MiB, HF
`nazeshinjite/Qwen3.8-Flash-Next-DFlash2-GGUF`), convertido do safetensors
oficial com `conversion/qwen.py::DFlashModel` e verificado 81/81 tensores
bit-exatos (relatório `dflash-qwen4exp-20261121.md`).

Harness: `/content/parity/bench.py`, casos `short`, `pp4k`, `pp4k+dec`,
`pp28k`, `pp28k+dec`; `temperature 0`, `cache_prompt false`, 2 warmups,
2 execuções por config, mediana. Prompts de 4.662 e 28.542 tokens.

## 1. Throughput (mediana de 2 execuções)

| config | short tg | pp4k | pp4k+dec tg | pp28k | pp28k+dec tg |
|---|---:|---:|---:|---:|---:|
| `m1-np1` (sem spec) | 60,15 | 1.559,36 | 46,93 | 1.575,14 | 45,09 |
| `dflash7` (n-max 7) | 66,83 | 1.341,26 | **89,91** | 1.392,97 | **84,17** |
| `dflash4` (n-max 4) | 59,28 | 1.353,47 | 90,33 | 1.399,12 | 79,31 |

t/s. `pp*` é prefill, `tg` é decode. Ganho de decode do `dflash7` sobre o alvo
sem spec: **+91,5%** em 4k e **+86,7%** em 28k. Prefill cai **14,0%** (4k) e
**11,6%** (28k) porque o draft também processa o prompt e o alvo passa a copiar
os taps das 5 camadas. Para comparação, o MTP sidecar Q4_K_M medido em rodada
separada fez 64,8/58,5 t/s (4k/28k), +38%/+33% sobre o mesmo alvo sem spec.

VRAM (MiB): `m1-np1` 37.098 pronto / 37.838 pico; `dflash7` 38.796 / 39.624;
`dflash4` 38.458 / 39.286. Carga 12,7-12,8 s nos três.

`--parallel 1` é obrigatório nesta configuração: com o default do server
(`n_slots = 4`) o buffer de estado recorrente (GDN) do alvo sobe de 112,6 MiB
para **3.602,3 MiB** e o DFlash2 não cabe (OOM de 478 MiB no compute do pp).
O `--moe-cache-mib` **não** altera VRAM (é cache host-side): 6144 e 2048 dão os
mesmos 37.488 MiB prontos.

### Aceitação (limite inferior)

Cada passo de decode especulativo verifica `n_accepted + 1` tokens e custa pelo
menos o mesmo que um passo de 1 token, então
`tokens_por_passo >= tps_spec / tps_base`. Para `dflash7`: **≥ 1,87 t/ passo**
em 28k (1,92 em 4k); `dflash4`: ≥ 1,76. Consistente com aceitação ~85-90% em
`n-max 7`; o server não expõe contador de aceitação no log em `-lv 3`.

## 2. Atribuição do prefill (`LLAMA_PREFILL_PROFILE=1`)

**Achado operacional:** as linhas `[PREFILL …]` são `LLAMA_LOG_INFO` e o
`llama-server` em `-lv 3` as filtra. O A/B "prof" anterior rodou sem
atribuição; o perfil só aparece com **`-lv 4`** (as linhas `[GRAPHPREP]`, que
usam `fprintf`, apareciam em qualquer nível — por isso o engano). Com `-lv 4`:

`prof4`, 1 execução, `pp4k` + `pp28k` no mesmo processo: 37 ubatches, 33.177
tokens. `scripts/strata-prefill-profile.py --log …`:

| seção | total | leitura |
|---|---:|---|
| `SCHED/all wall_us` | 15,669 s | parede dos 37 ubatches, 74 splits |
| `CLASS/mixed device_host_us` | 15,159 s | host submetendo trabalho (96,7% da parede) |
| `PREP/0 prep_us` | 2,614 s | build/alloc/aplicação de grafo por ubatch |
| `SYNC/0 wait_us` | 2,714 s | 22 amostras, 33.177 tokens na fila |
| `SCHED transfer_host_us` | 0,181 s | cópias |
| `SCHED h2d_bytes` | 1,595 GB | |

Bytes lógicos por classe: `moe` 507,3 GB, `other` 1.215,8 GB, `gdn` 163,2 GB,
`qsa_fa` 27,4 GB. **Todos os 74 splits caem em `mixed`**, então o perfil não
separa GDN/QSA/MoE por tempo — para isso é preciso o trace de kernels
(Nsight/`GGML_RKVA_TRACE_FILE`). O que já está medido: a parede do prefill é
**submissão host**, não compute do dispositivo, e o `prep` de grafo custa ~17%
do total. Isso casa com o throughput plano em ~1,57k t/s de 4k a 28k (custo por
token constante no host), enquanto a Strata cresce de 2.394 para 4.629 t/s.

Sem overhead mensurável: `prof4` deu pp4k 1.543,78 e pp28k 1.574,15, dentro do
ruído do `m1-np1` (1.559,36 / 1.575,14).

## 3. PLE (`LLAMA_PLE_PREFETCH=1`) — A/B frio

`/proc/sys/vm/drop_caches` é **read-only** no container do Colab
(`Read-only file system`), então a primeira rodada do A/B foi quente nos dois
braços (1.551,01 vs 1.551,46 t/s, +0,03%, ruído). O A/B válido usa
`posix_fadvise(POSIX_FADV_DONTNEED)` nos dois shards e no drafter antes de cada
carga: `Cached` cai de 80 GB para 43 GB no teste de controle e a carga sobe de
12,7 s (quente) para **26,0 s** nos dois braços.

| braço | pp28k (3 execuções) | mediana |
|---|---|---:|
| `LLAMA_PLE_PREFETCH=0` | 1.551,91 / 1.550,44 / 1.548,73 | 1.550,44 |
| `LLAMA_PLE_PREFETCH=1` | 1.551,62 / 1.551,49 / 1.551,23 | **1.551,49** |

**+0,07% (1,05 t/s), dentro da dispersão do próprio braço de controle (3,2 t/s).**
Sem ganho mensurável mesmo com a tabela fria: o `WILLNEED` de uma ubatch à
frente não muda a parede porque o prefill é limitado por submissão host (§2), não
por latência de leitura da tabela. A flag fica default-off.

## 4. Corretude do DFlash2

O capture dos taps é bit-exato no alvo (teste CPU
`test-qwen4exp-dflash-capture`: "tap-off: target logits bit-identical"), e o
`batch_inject` do M-RoPE foi corrigido (`b1e56c896`). No harness, `short` e
`pp4k+dec` saem **idênticos** entre `m1-np1`, `dflash7` e `dflash4`
(`all_configs_identical: true`), mas `pp4k`, `pp28k` e `pp28k+dec` divergem já
no 4º-5º token gerado — e `dflash7` diverge de `dflash4`.

**Controle decisivo (mesmo alvo, mesmo prompt de 28.542 tokens, sem spec):**

| config | texto gerado (greedy, 16 tokens) |
|---|---|
| `m1` `-ub 1024` | "The provided code **defines a function** named `process` that iterates through a list" |
| `m1` `-ub 512` | "The provided code **consists of hundreds** of identical, redundant definitions of a function named `" |
| `dflash7` | "The provided code **consists of numerous** identical definitions of a function named `process` that" |

Só trocar o ubatch, sem nenhum drafter, já muda o texto no 5º token. O greedy
deste alvo (Q2_0 + KV q8_0) **não é invariante à geometria do batch**: matmul e
FA com formas diferentes mudam a aritmética o suficiente para virar empates
próximos no argmax. No primeiro token do `pp28k` o top-1 é o mesmo nos dois
(`We`, logprob -0,00074 no `m1` e -0,00082 no `dflash7`), com 19/20 de
sobreposição no top-20, mas a cauda difere 0,002-0,27 nats — compatível com
seleção de kernel/split diferente, não com um erro de plumbing.

Consequências práticas:

- "Texto idêntico ao sem spec" **não serve** como critério de aceite neste alvo;
  o critério válido é top-1 no mesmo batch, NMSE do verify contra single-token e
  aceitação/throughput. É o mesmo critério que o MTP já usava (`n_accepted ±1 p.p.`,
  NMSE ≤1e-6).
- Um erro nos taps do drafter reduziria a aceitação, não mudaria o argmax do
  alvo — os taps só alimentam o draft.
- Não há evidência de corretude quebrada no DFlash2; o que existe é sensibilidade
  numérica do alvo, já presente sem spec.

## 5. O que ainda separa da Strata 0.2.17

| métrica | Bee (`dflash7`, Q2_0, q8 KV) | Strata 0.2.17 (INT8 + DFlash2) | razão |
|---|---:|---:|---:|
| decode 4k | 89,91 | 85,9 | **1,05× a favor do Bee** |
| decode 28k | 84,17 | 67,4-75,0 | **1,12-1,25× a favor do Bee** |
| prefill 4k | 1.341 | 2.395 | 0,56× |
| prefill 28k | 1.393 | 4.629 | 0,30× |

Decode: com o DFlash2 Q8_0 o Bee passa a Strata nas três medições publicadas,
mesmo com pesos Q2_0 e KV q8 (a Strata usa INT8 proprietário). Prefill
continua sendo a lacuna, e agora com causa medida: **submissão host por
ubatch** (96,7% da parede) mais `prep` de grafo (~17%), não compute do
dispositivo nem H2D (0,18 s, 1,6 GB). Próximos passos verificáveis, em ordem:
reduzir splits/ubatch (menos submissão por token), cachear o plano de grafo
entre ubatches de mesma forma (o `prep_us` de 2,6 s), e só depois investigar
kernels por classe com trace.

## 6. Comandos

```bash
# DFlash2 (A100, 32k, q8 KV, 1 slot)
llama-server -m Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf \
  --alias q -ngl 99 -c 32768 --n-cpu-moe 2 --moe-cache-mib 2048 \
  -b 2048 -ub 1024 -fa on --load-mode mmap --lazy-mode on \
  -ctk q8_0 -ctv q8_0 --fit off --parallel 1 \
  --spec-type draft-dflash \
  --spec-draft-model Qwen3.8-Flash-Next-DFlash2-Q8_0.gguf \
  --spec-draft-n-max 7

# perfil de prefill (precisa de -lv 4)
LLAMA_PREFILL_PROFILE=1 llama-server … -lv 4 2>&1 | tee prefill.log
python3 scripts/strata-prefill-profile.py --log prefill.log
```
