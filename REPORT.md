# A3 — Encurtar o caminho crítico do decode RKVA (nq=1)

Branch: `perf/rkva-decode-host-20261026` (base `40f22aa6b`) no clone
`/home/hjotha/beellama-agents/decode-host`. Nada foi escrito em
`/home/hjotha/beellama.cpp`, em `~/.codex/`, em `~/llama-slot-cache/`, em
`/home/hjotha/models/` nem no Xbox. Nenhum protocolo RKVA foi alterado.

## 1. Resumo

| entrega | arquivo | estado |
|---|---|---|
| cache da amostra do guard de VRAM (host CUDA) | `ggml/src/ggml-cuda/graph-guard.h` (novo), `common.cuh`, `ggml-cuda.cu` | implementado, lógica testada em CPU |
| fence com spin limitado no engine nq=1 (Xbox) | `tools/xbox/uwp/rkva/remote-attn-engine.cpp` | implementado, **não compilado** (sem MSVC) |
| contadores `fence_spin` no perfil do engine | `tools/xbox/uwp/rkva/rkva-range-profile.h` | aditivo, não compilado |
| runner A/B/B/A inerte + atribuição do trace | `tools/xbox/scripts/run-rkva-decode-host-ab.py`, `analyze-rkva-decode-trace.py` | testado (21 testes) |
| doc do caminho crítico | `docs/rkva-decode-critical-path.md` | novo |

Ganho estimado (marcado como **estimativa**, ver §7): **3,4–3,9 ms/token** em
57 ms ⇒ ~53,1–53,6 ms/token (~18,7–18,8 t/s, +6,4% a +7,4%). Somente 2,85
ms/token vêm de medição direta (guard); o resto é estimativa por camada.

## 2. Diagnóstico verificado (não presumido)

Reproduzi a atribuição publicada com uma ferramenta própria, sobre o trace
citado na tarefa:

```bash
python3 tools/xbox/scripts/analyze-rkva-decode-trace.py \
  /home/hjotha/.codex/artifacts/beellama-xbox-rkva-20261026/wf4-cached-p102400-long-r1-trace.3091043.jsonl
```

Saída literal (recortada; 512 tokens de decode, `first_pos >= 130560`):

```
scanned=... decode_events=... decode_tokens=512
phase                              n     p50_us     p95_us     sum_ms  gpu_p50_us
cuda_graph_host                47616         63       1307  14083.680       0.000
cuda_graph_bookkeeping         47616         32         57   1781.681       0.000
cuda_graph_memory_guard        47616         29         49   1552.377       0.000
cuda_producer_sync             15872        782        808   6801.789       0.000
rpc_total                       8192       1278       1450  10707.299     730.000
rpc_header_wait                 8192       1158       1319   9703.233       0.000
range_wait                      8192        417        607   3544.055       0.000
job_prepare                     8192        125        142   1103.503       0.000
rpc_body_recv                   8192         89        148    788.073       0.000
kv_producer                     8192         42         50    420.004       0.000
out_decode                      8192         31         52    286.498       0.000
q_pack                          8192         30         41    256.054       0.000
k_readback                      8192         14         16    118.656       0.000
v_readback                      8192         13         13    103.139       0.000

per-token milliseconds:
  rpc_total                        20.913
  rpc_header_wait                  18.952
  range_wait                        6.922
  job_prepare                       2.155
  kv_producer                       0.820
  q_pack                            0.500
  k_readback                        0.232
  v_readback                        0.201
  out_decode                        0.560
```

Isso confirma, contra os artefatos: RTT 20,913 ms/token (o texto de origem diz
20,914), espera exposta `range_wait` 6,922 ms (origem: 6,934), preparo nativo
2,155 ms (origem: 2,057), output decode 0,560 (origem: 0,544) e **memory guard
3,032 ms/token** (`1552,377 / 512`; origem: 3,089) em 93 chamadas/token de
~29–32 µs.

Números novos, que a análise de origem não separava:

* `cuda_graph_bookkeeping` (pai do guard) = **3,480 ms/token**; tirando o guard,
  sobram 0,45 ms/token de key cache/decisão.
* `rpc_header_wait` (18,95 ms) contra o wall do engine no console (11,72 ms):
  sobram **~0,42 ms/camada** de rede/enquadramento/fila, consistente com o RTT
  vazio histórico de ~0,56 ms.
* No perfil de decode gravado em 102400, `engine_us` p50 = **313 µs** por
  chamada com GPU ≈ 180 µs. Em 130560 o wall do engine é 733 µs/chamada; o tier
  de overflow cresce de 1 para ~28k linhas, então a diferença (~420 µs) é
  trabalho de GPU do kernel nq=1. **Não existe perfil de engine de decode em
  130560** — essa é a medição que falta (§6).

## 3. O que foi implementado

### 3.1 Guard de VRAM com amostra cacheada (host)

Problema: `ggml_backend_cuda_context::check_memory_pressure_before_graph()`
chama `cudaMemGetInfo()` antes de **toda** submissão de grafo — 93 vezes por
token de decode, 3,03 ms/token medidos.

Solução: uma amostra de memória livre continua sendo um **limite inferior
válido** enquanto nada reservar memória de dispositivo. Todas as rotas de
reserva agora incrementam `ggml_cuda_graph_guard::note_device_reservation()`:

* `ggml_cuda_device_malloc` (todos os `cudaMalloc`/`cudaMallocManaged`, inclusive
  os dois pools e os buffers de compute);
* `cuMemCreate` no crescimento do pool VMM;
* `cudaGraphInstantiate`, `cudaGraphExecUpdate`, `cudaStreamEndCapture`;
* `ggml_backend_cuda_buffer_type_alloc_buffer`.

A amostra só aprova uma submissão sem consulta quando (a) tem menos de
`GGML_CUDA_GRAPH_GUARD_SAMPLE_MS` (default 10 ms) e (b) a geração de reservas não
mudou. Qualquer rejeição cai na consulta exata. Consequências:

* decisão de pressão idêntica: amostra abaixo do headroom nunca aprova;
* as máquinas de estado de liberação/recuperação de grafos não foram tocadas;
* `GGML_CUDA_GRAPH_GUARD_SAMPLE_MS=0` restaura exatamente o comportamento antigo;
* a linha `const int headroom_mb = env != nullptr ? atoi(env) : 18;` ficou
  byte-idêntica (o `memory_policy()` do `bench-rkva-range.py` continua validando:

```
{'variable': 'GGML_CUDA_GRAPH_RECOVERY_HEADROOM_MB', 'effective_mib': 128,
 'source_default_mib': 18, 'inherited_value': None, ...}
```

Prova de ativação para a janela de hardware: ao destruir o contexto o backend
loga `headroom guard submissions fastpath=<n> queried=<m> sample_age_ms=10`.

### 3.2 Fence com spin limitado (engine nq=1 do console)

`RemoteAttnEngine::submit()` esperava `SetEventOnCompletion` +
`WaitForSingleObject` a cada camada. No perfil de decode a 102400,
`fence_wait_cpu` = 230 µs contra `gpu_total` = 180 µs ⇒ ~50 µs/camada de
transição de kernel + round trip de scheduler. Agora o engine gira no valor
user-mode do fence (`GetCompletedValue`) por até `RKVA_FENCE_SPIN_US`
(default 2000 µs; spin apertado com `YieldProcessor` nos primeiros 100 µs,
depois `SwitchToThread`) e **mantém** o caminho antigo por evento como fallback
quando o orçamento acaba (ou quando `RKVA_FENCE_SPIN_US=0`). O tratamento de
device-removal é o mesmo (o `UINT64_MAX` continua sendo validado depois).

Contadores novos em `counters.fence_spin` (`hits`, `fallbacks`, `spin_us`), para
a próxima janela provar que caminho cada chamada usou.

## 4. O que NÃO foi mudado (e por quê)

* **Rede/transporte.** O host já usa `TCP_QUICKACK` em leitura grande e um único
  `sendmsg` com header+payload; o listener responde com dois `send()`. Nagle não
  é o limitante (header de 40 B seguido de payload ≥ MSS sai imediatamente). O
  custo restante (~0,42 ms/camada) é latência por pacote no caminho USB 1 GbE +
  socket do console, não serialização (~10,3 KiB TX + 6,3 KiB RX por camada).
  Recomendações **sem medição**: NIC PCIe 2.5 GbE no host (em vez de USB 1 GbE),
  desligar interrupt moderation na NIC, avaliar `SO_BUSY_POLL` no socket do host
  e manter o console em Ethernet sem power saving.
* **WHT da query no CPU do console.** Medi a implementação de produção na
  máquina local: **9,98 µs/camada** para 24 cabeças (`g++ -O2 -mavx2`,
  `.oc-artifacts/wht_bench.cpp`). Pequeno demais para justificar mexer em código
  vizinho da numérica.
* **Readback de validação das células K/V.** `rkva_run_range()` lê de volta
  todos os `n_live` ids (~112 KiB com 28k linhas) antes de cada submissão nativa
  assíncrona — estimativa **derivada** de ~50 µs/camada (~0,8 ms/token), sem
  fase de trace própria. Validar só o sufixo exigiria provar que todo caminho que
  reescreve células existentes incrementa `range_epoch`; essa auditoria não foi
  feita, então o scan exato ficou.
* **Kernel de decode / split_k.** `kvarn4_decode_split()` já usa `split_k = 64`
  com 28k linhas (grid 4 KV-heads × 1 query × 64 splits = 256 workgroups), então
  não se pode atribuir o custo do console a falta de paralelismo sem o perfil de
  engine na posição de produção.
* **Protocolo RKVA.** Intocado (sem HELLO/capabilities novo).

## 5. Validação offline (com saídas reais)

Ambiente: `gokaya`, Ryzen Z1 Extreme, sem toolkit CUDA (`/usr/local/cuda` não
existe) e sem MSVC. Codex com `llama-server` ativo no início da sessão
(PID 3899876) ⇒ **nenhum build completo, nenhum nvcc, nenhum teste de GPU**.

### 5.1 Teste unitário do guard (compila e roda em CPU)

```bash
$ g++ -std=c++17 -O2 -Wall -Wextra -I ggml/src/ggml-cuda \
      tests/test-cuda-graph-guard.cpp -o .oc-artifacts/test-cuda-graph-guard
$ ./.oc-artifacts/test-cuda-graph-guard
graph-guard: all checks passed
```

Cobre: aprovação com folga, fronteira `free == headroom` (igual ao teste
`free < headroom` do caminho exato), pressão nunca aprovada, amostra velha /
geração nova / timestamp para trás / `max_age = 0` sempre rejeitam, amostra
default rejeita, extremos de `size_t` não estouram, contador de geração é
monotônico.

### 5.2 Checagem de sintaxe do TU do host (sem CUDA)

```bash
$ g++ -std=c++17 -fsyntax-only -Iggml/include -Iggml/src \
      -Iggml/src/ggml-remote-attn -Iggml/src/ggml-cpu \
      ggml/src/ggml-remote-attn/ggml-remote-attn.cpp
exit=0
```

### 5.3 Scripts da janela de hardware

```bash
$ python3 -m unittest discover -s tools/xbox/tests -p test_run_rkva_decode_host_ab.py
Ran 21 tests in 0.001s
OK
$ python3 -m unittest discover -s tools/xbox/tests -p test_bench_rkva_range.py
Ran 47 tests in 0.046s
OK
```

### 5.4 Regressão estática do repo

```bash
$ for t in tests/test-*-static.py; do python3 "$t"; done
static python tests: pass=34 fail=0
```

`tests/test-rkva-cuda-singleton-fastpath.py` falha, mas **já falhava antes**: a
asserção `compute.count('check_memory_pressure_before_graph()') == 1` não bate
com o código atual (2 ocorrências). Verifiquei contra `git show HEAD:` numa
árvore limpa e reproduzi a mesma falha; não é regressão desta tarefa e não mexi.

### 5.5 Limites da validação

* `ggml-cuda.cu` / `common.cuh` **não foram compilados** (sem nvcc e sem build
  permitido). O diff foi revisado linha a linha e a lógica pura está testada,
  mas a compilação CUDA fica pendente.
* O C++ do app UWP **não foi compilado** (sem MSVC); o diff do engine foi
  revisado e usa apenas APIs já presentes no arquivo (`prof_clock`, `YieldProcessor`
  via `<intrin.h>`, `SwitchToThread` via `<Windows.h>` de `xllama/d3d12_dyn.h`).
* Não reexecutei o `pgrep` imediatamente antes de cada `g++` isolado; a checagem
  mais próxima (antes e depois) não mostrou `llama-server`/`bench-rkva`/`ninja`/
  `cmake --build` ativos. Os três compilados foram de 1 TU, com
  `nice -n 19 ionice -c3`, sem nvcc.

## 6. Pendente de hardware (janela futura)

1. Compilar o host com CUDA e rodar `ctest -R 'cuda-graph-guard|remote-attn'`
   (pendente: sem nvcc aqui).
2. Build MSVC/UWP do app e instalação do pacote — responsabilidade do revisor.
3. Rodar o A/B/B/A (não executado aqui):

```bash
python3 tools/xbox/scripts/run-rkva-decode-host-ab.py \
  --server-binary build/bin/llama-server --model model.gguf \
  --prompt-tokens ids.json --xbox-host 192.168.1.200:50054 \
  --xbox-build-id <pacote instalado> --threads 6 \
  --prefix-snapshot <prefixo congelado> --output ~/rkva-decode-ab-r1 --execute
```

O runner é inerte por default, recusa rodar com processo de benchmark do Codex
(ou a CLI do Codex, salvo `--allow-codex-idle`), compara os 256 IDs gerados dos
quatro passes, exige `fastpath > 0` no braço B do guard e grava
`rkva-decode-host-ab.json` + atribuição do trace por pass.
4. Coletar o **perfil de engine de decode na posição de produção** (130560):
   é o dado que falta para decidir entre kernel nq=1 e transporte.

## 7. Estimativa de ganho (claramente marcada)

Medido: guard = 3,032 ms/token. Com janela de 10 ms em ~57 ms/token sobram ~6
consultas/token (~0,18 ms) ⇒ **economia medida/deduzida de ~2,85 ms/token
(~5,0%)**.

Estimado (sem medição): fence spin 30–60 µs/camada × 16 = **0,5–1,0 ms/token
(0,9–1,8%)**. Total estimado 3,4–3,9 ms/token ⇒ 57 → 53,1–53,6 ms/token
(~18,7–18,8 t/s, +6,4% a +7,4%). O teto de "zerar a espera exposta" (≈50 ms,
+14%) continua fora de alcance: exige cortar ~0,47 ms/camada de RTT e ~0,42
ms/camada são rede/console.

## 8. Riscos

* **Guard**: se algum consumidor de VRAM do processo não for coberto pela
  contabilidade, o fast path pode aprovar com folga real menor que a amostra por
  até 10 ms. Mitigações: toda reserva conhecida incrementa a geração; qualquer
  rejeição volta à consulta exata; `=0` restaura o comportamento antigo. O
  caminho de *pressão* (liberar grafos) não muda de semântica.
* **Fence spin**: queima de CPU (até 1,6 ms/token no pior caso) e possível
  disputa com a thread de submissão do driver; mitigado por `SwitchToThread`
  após 100 µs, orçamento finito e fallback idêntico ao atual. Device removal
  segue tratado.
* **Contadores**: `counters.fence_spin` é chave JSON nova e aditiva; nenhum
  campo existente mudou.

## 9. Arquivos alterados

* `ggml/src/ggml-cuda/graph-guard.h` (novo)
* `ggml/src/ggml-cuda/common.cuh`
* `ggml/src/ggml-cuda/ggml-cuda.cu` (regiões: `ggml_cuda_device_malloc`,
  pool VMM, destrutor do contexto, `alloc_buffer`, `graph_instantiate`,
  `graph_update_executable`, fim do capture — **fora** do bloco mask-pack que o
  Codex está editando)
* `tests/test-cuda-graph-guard.cpp` (novo), `tests/CMakeLists.txt` (bloco no fim,
  longe do bloco mask-pack do Codex)
* `tools/xbox/uwp/rkva/remote-attn-engine.cpp`, `tools/xbox/uwp/rkva/rkva-range-profile.h`
* `tools/xbox/scripts/run-rkva-decode-host-ab.py` (novo),
  `tools/xbox/scripts/analyze-rkva-decode-trace.py` (novo),
  `tools/xbox/tests/test_run_rkva_decode_host_ab.py` (novo)
* `docs/rkva-decode-critical-path.md` (novo)
* `.oc-artifacts/` (bench do WHT, resumo do trace, binário do teste ignorado)
