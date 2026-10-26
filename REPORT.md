# REPORT — A1: teto real do Xbox Series X dentro do sandbox UWP

Branch: `research/xbox-ceiling-probe-20261026` (clone `/home/hjotha/beellama-agents/xbox-ceiling`,
base `40f22aa6b`). Nenhum C++ foi compilado e nada foi instalado ou executado no Xbox
nesta tarefa; a única compilação foi a de shaders com DXC na 193 (permitida e isolada).
A sessão começou com um `llama-server` do Codex ativo (PID 3899876) e ele voltou a rodar
no fim, então a regra de build local (`pgrep -af 'llama-server|bench-rkva|test-rkva|ninja|cmake --build'`)
nunca deixou de valer. O que foi validado de fato: 11 alvos DXIL na 193, testes Python em
Linux e inspeção do código do listener/app.

## 1. O que existe agora

| Peça | Caminho | Estado |
| --- | --- | --- |
| Driver da sonda | `tools/xbox/src/bridge/gpupeak.cpp`, `include/xllama/gpupeak.h` | escrito, **não compilado** |
| Modo headless | `gpupeak.flag` em `uwp/App.cpp` -> `bridge::run_gpupeak()` | escrito, não compilado |
| Shaders | `tools/xbox/shaders/gpupeak_stream.hlsl`, `gpupeak_alu.hlsl`, `gpupeak_empty.hlsl` | **compilados com DXC 1.8.2502.11** |
| DXIL embutido | `shaders/generated/gpupeak_*_dxil.h` + `gpupeak-shaders.json` | commitado, com SHA256 |
| Regenerador | `scripts/compile-gpupeak-shaders.py` | local ou `--ssh-host 193` |
| Sweep do `pass1` | `tests/rkva_pass1_sweep.py` | offline + `--self-test` OK; `--execute` não rodado |
| Modelo de roofline | `tests/rkva_ceiling_roofline.py` | 17 testes Python verdes |
| Orquestrador | `scripts/run-xbox-ceiling.py` | inerte por padrão, `--execute` não rodado |
| Doc | `tools/xbox/docs/gpu-ceiling.md` | plano, fórmulas, tabela nominal vs medido |

Teste C++ novo (`tests/test_gpupeak.cpp`) e entrada CTest `xllama-ceiling-roofline`
(Python, sem hardware) ficam pendentes de build.

## 2. Estado de partida (medidas de terceiros, não minhas)

Usadas só como alvo de comparação; nenhuma foi reproduzida aqui.

- `/home/hjotha/.codex/artifacts/beellama-xbox-rkva-20261026/mq295-gpu-bucket-comparison.json`,
  bucket `attention/kvarn4-prefix/optimized-kvarn4_mq2_fine/nq2-16/r16k-32k/warm`:
  `samples=4112`, `n_tokens.mean=8`, `active_rows.mean=27135.767`, `split_k.mean=63.837`,
  `gpu_us.pass1.mean=2724.8`, `reduce.mean=59.516`, `readback_copy.mean=1.215`,
  `total.mean=2785.531`.
- `/home/hjotha/build/rkva-next-20261021/reports/real-fp8-profile-r5c-102400.json`:
  `gpu_pass1_us` com `n=416, mean=13770.163, min=144.722, p50=13421.915` por envelope
  (mistura prefill e decode), `bulk_queries=98320`, `fp8_gpu_calls` presente.
- Envelope de prefill citado na tarefa: 256 queries x 1 camada x ~28k linhas = 65,3 ms de
  `pass1`, ~2,7 TFLOP/s derivados, 22% dos 12,15 TFLOPS nominais.

## 3. Evidência real desta sessão

### 3.1 Compilação dos shaders na 193 (DXC isolado)

Comando (diretório remoto próprio, `C:\Users\hjotha\build\oc-xbox-ceiling-20261026`):

```
$ python3 tools/xbox/scripts/compile-gpupeak-shaders.py \
    --ssh-host hjotha@192.168.1.193 \
    --remote-root 'C:/Users/hjotha/build/oc-xbox-ceiling-20261026'
dxc: dxcompiler.dll: 1.8 - 1.8.2502.11 (239921522); dxil.dll: 1.8(1.8.2502.11)
stream_read      3816 bytes  shaders/generated/gpupeak_stream_read_dxil.h
stream_write     3664 bytes  shaders/generated/gpupeak_stream_write_dxil.h
stream_copy      3980 bytes  shaders/generated/gpupeak_stream_copy_dxil.h
fma_f32          3992 bytes  shaders/generated/gpupeak_fma_f32_dxil.h
fma_f16          3944 bytes  shaders/generated/gpupeak_fma_f16_dxil.h
dot2_f16         4032 bytes  shaders/generated/gpupeak_dot2_f16_dxil.h
dot4_i8          3784 bytes  shaders/generated/gpupeak_dot4_i8_dxil.h
latency_fma      3748 bytes  shaders/generated/gpupeak_latency_fma_dxil.h
cu_probe_32      4140 bytes  shaders/generated/gpupeak_cu_probe_32_dxil.h
cu_probe_16      4140 bytes  shaders/generated/gpupeak_cu_probe_16_dxil.h
empty            3344 bytes  shaders/generated/gpupeak_empty_dxil.h
manifest: shaders/generated/gpupeak-shaders.json
```

Perfis: `cs_6_0` (F32/stream/empty/LDS), `cs_6_2 -enable-16bit-types` (F16 packed),
`cs_6_4 [-enable-16bit-types]` (`dot2add`, `dot4add_i8packed`). O manifesto registra
`source_sha256`, `dxil_sha256`, entry, profile, defines e flags de cada alvo.

### 3.2 Os intrínsecos sobrevivem ao DXIL (não são dobrados)

Desmontagem (`dxc -Fc`) guardada em `.oc-artifacts/isa/` (ignorado pelo git):

```
$ for f in .oc-artifacts/isa/*.asm; do echo "-- $f"; \
    grep -oE "Dot2AddHalf|Dot4AddI8Packed|fmul|fadd|fma" "$f" | sort | uniq -c; done
-- .oc-artifacts/isa/dot2.asm
      4 Dot2AddHalf
     11 fadd
      8 fmul
-- .oc-artifacts/isa/dot4.asm
      4 Dot4AddI8Packed
-- .oc-artifacts/isa/f16.asm
     20 fadd
     12 fmul
```

Ou seja: 4 chamadas por iteração (as 4 cadeias do kernel), dentro do loop. O F16
empacotado sai como `fmul/fadd half`; se o backend AMD não empacotar em `v_pk_fma_f16`,
a taxa medida para `fma_f16` fica em ~1x FP32 e `dot2_f16` (garantidamente empacotado)
mostra o dobro — a própria sonda revela qual dos dois aconteceu.

### 3.3 Modelo e testes offline

```
$ python3 -m unittest tests.test_rkva_ceiling_roofline -v
...
Ran 17 tests in 0.003s
OK

$ python3 tests/rkva_pass1_sweep.py --self-test
self-test: OK (4 shapes, 3 fail-closed paths, aggregation)
```

O `--self-test` prova: extração de `pass1` por envelope nos 4 formatos, rejeição de
kernel diferente do qualificado, rejeição de relatório sem timestamp GPU, rejeição de
`fp8_cpu_fallback_calls != 0`, e agregação best/median sobre execuções.

Plano offline (abre nenhum socket):

```
$ python3 tests/rkva_pass1_sweep.py
"model_table": [
 {"rows": 28672, "n_queries": 256, "segments": 32, "gflop": 180.388626432,
  "kv_mib": 31.625, "qo_mib": 12.0}, ...
]
```

### 3.4 Orquestrador inerte e guard de processo (as duas metades verificadas)

```
$ python3 tools/xbox/scripts/run-xbox-ceiling.py | python3 -c \
    "import json,sys; d=json.load(sys.stdin); print('busy:', d['busy_processes_now'])"
busy: []
```
(naquele instante só o agente estava ativo)

```
$ python3 tools/xbox/scripts/run-xbox-ceiling.py | python3 -c \
    "import json,sys; d=json.load(sys.stdin); print('phases:', [p['name'] for p in d['phases']], 'busy:', d['busy_processes_now'])"
phases: ['gpupeak', 'pass1-sweep'] busy: ['pid=3967184 /home/hjotha/beellama.cpp/build-upstream-gpu-20261025/bin/llama-server --batch-size-l 256 ...']
```
(com o `llama-server` do Codex ativo; nesse estado o `--execute` recusa antes de abrir
qualquer socket)

O guard compara o **basename do executável** (`llama-server`, `llama-bench`, `ninja`,
`cmake`, prefixos `bench-rkva`/`test-rkva`) varrendo `/proc`, não `pgrep -f`: a linha de
comando de um agente (esta sessão) carrega o texto da tarefa inteiro e faria um
`pgrep -f` disparar ou esconder o guard conforme o caso. As duas execuções acima mostram
que ele distingue os dois casos.

### 3.5 Fatos de código verificados (para o sweep não medir outra coisa)

- O bulk nativo revision-3 **exige** `append_rows == n_queries` em segmento sem RESET
  (`tools/xbox/uwp/rkva/rkva-kvarn-bulk.h:292`), então uma envelope de atenção pura não
  existe nesse caminho; o sweep anexa exatamente as queries de cada segmento, como a
  produção faz.
- `op 20` com `RESET` sem `ENABLE` **desliga** o profiling (`profiling_ = enabled`,
  `remote-attn-engine.cpp:1082-1140`). O sweep usa `ENABLE|RESET` (flags=3), igual a
  `rkva_kvarn_bulk_probe.py`.
- Observação (não corrigida, fora do escopo): `tests/rkva_range_q4_kernel_probe.py`
  chama `profile(client, PROFILE_RESET, mode)` com flags=2, ou seja, desabilita o
  profiling antes de medir; os buckets de decode daquele probe tendem a sair vazios.
- O listener só existe com `api.flag` em LocalState (`uwp/App.cpp:281-305`, porta 50054),
  por isso o orquestrador sobe o app em modo XAML para o sweep e em modo headless para
  o `gpupeak`.

## 4. Teto teórico nominal vs o que a suíte mede

| Grandeza | Nominal publicado (Series X) | O que a suíte mede (campo do JSON) |
| --- | --- | --- |
| FP32 | 12,15 TFLOP/s (52 CU x 64 lanes x 2 x 1,825 GHz) | melhor `fma_f32` do sweep de grupos -> `ceiling_measured.fp32_tflops` |
| FP16 packed | 24,30 TFLOP/s | `fma_f16` e `dot2_f16` -> `fp16_tflops`, `dot2_f16_tflops` |
| INT8 | 48,6 TOPS | `dot4_i8` -> `dot4_i8_tops` |
| Banda | 560 GB/s (pool 10 GB, 320-bit) / 336 GB/s (6 GB) | stream read/write/copy -> `bandwidth_gbs`, `read_gbs`, `write_gbs`, `copy_gbs` |
| CUs | 52 dual-CU | degraus de ocupação 32 KiB e 16 KiB; só publica se as duas estimativas concordarem em 15% -> `compute_units.dual_cu` |
| Clock | 1,825 GHz | contador de timestamp (medido) + clock implícito da cadeia FMA (derivado) -> `clocks.*` |
| Orçamento de GPU | ~4,35 GB reportados | `DXGI_QUERY_VIDEO_MEMORY_INFO` -> `device.local_budget_bytes` |
| Custo fixo por chamada | não publicado | dispatch vazio (GPU), submit+fence, `Map` de readback -> `dispatch_latency.*` |

O nominal é só denominador. Um valor medido abaixo dele é resultado, não erro: o sandbox
pode limitar clock, memória ou ocupação.

## 5. Fórmulas do `pass1` (as mesmas do código e do relatório)

```
MAC  = 2 * nq * n_head * rows * head_dim        # QK + PV
FLOP = 2 * MAC = 4 * nq * 24 * rows * 256
```

Envelope de produção: `nq=256`, `rows=28672` -> 90.194 G MAC = **180.389 GFLOP**
(`gflop: 180.388626432` na tabela acima). A 65,3 ms isso dá 2,763 TFLOP/s, batendo com o
número derivado da tarefa (~2,7).

Bytes únicos de KV (KVarN4 revision 3):

```
records = floor(rows/128) * n_head_kv * 2 fatias * 8960 B * 2 planos
stage   = min(256, grupo_parcial + 128) linhas * n_head_kv * head_dim * 2 B * 2
tail    = min(rows, 128) linhas * n_head_kv * head_dim * 2 B * 2
```

Em 28.672 linhas: 30,6 MiB de records + ~1 MiB de stage/tail = **31,625 MiB**, o ~31 MB
usado na análise do kernel. O JSON reporta dois limites de banda, sem inventar um valor
único: `gbs_unique` (tráfego compulsório) e `gbs_issued_upper` (`unique * nq`, sem
reuso). Mesmo o limite superior fica abaixo de 1 GB/s em 28k linhas, então o `pass1` está
no lado de compute do roofline — e o campo `bound` do JSON diz isso explicitamente.
O WHT de Q e o inverso de OUT somam `2*nq*24*2*(8*256)` FLOP (<0,1% das matmuls).

## 6. O que foi validado offline, e como

| Item | Como |
| --- | --- |
| 11 shaders compilam para DXIL | DXC 1.8.2502.11 na 193, saída acima |
| Intrínsecos preservados | `-Fc` + contagem de `Dot2AddHalf`/`Dot4AddI8Packed` |
| Aritmética FLOP/bytes do pass1 | 17 testes Python (inclui 180,4 GFLOP e 31,2 MiB) |
| Contrato do JSON da sonda | teste Python que aceita o documento válido e rejeita checksum falho, `d3d12_ran=false` e schema errado |
| Fit de concorrência / CUs | teste Python com degrau sintético `ceil(N/104)` -> 104; duas estimativas 32K/16K -> 52 |
| Sweep falha fechado | `--self-test` (kernel errado, sem timestamp, fallback FP8) |
| Guard do orquestrador | execução sem `--execute` e `busy: []` com só o agente ativo |
| Planejamento de dispatch/limites D3D12 | testes C++ escritos (`test_gpupeak.cpp`), **não compilados** |

## 7. PENDENTE de hardware (não executado)

1. **Compilar o C++** (`gpupeak.cpp`, `test_gpupeak.cpp`, `inference-bridge.cpp`,
   `App.cpp`) em Windows/MSVC e rodar `ctest`. Nenhum build foi feito; este é o maior
   risco aberto.
2. Instalar o pacote que contém o modo `gpupeak` (build/assinatura/instalação são do
   Claude; o script não instala nada).
3. Rodar a janela:
   `tools/xbox/scripts/run-xbox-ceiling.py --execute --expect-pfn '<pfn>' \
    --output .oc-artifacts/xbox-ceiling-result.json`.
   Passo a passo: `gpupeak.flag` -> `gpupeak-result.json`; depois `api.flag` -> listener
   :50054 -> `rkva_pass1_sweep.py --execute`; no fim `xbox-ceiling-result.json`.
4. Coletar também `remote-attn.jsonl`/`xllama.log` da execução (o orquestrador já
   registra comandos e códigos de retorno no JSON).

## 8. Riscos

- **C++ não compilado** (regra de build + Codex ativo). Erros de compilação/API D3D12 são
  possíveis; a revisão manual foi feita, mas não substitui MSVC. Principais pontos de
  atenção: `IDXGIAdapter1::GetDesc1`, root signature com 4 ranges, barreiras de transição
  (restauradas para COMMON a cada dispatch) e `D3D12_FEATURE_DATA_D3D12_OPTIONS1`.
- **SM 6.4 no console**: `dot2add`/`dot4add_i8packed` compilam, mas a criação do PSO pode
  falhar no runtime do Xbox. A sonda registra em `pso_errors` e segue; não inventa taxa.
- **Memória**: o passe de 1 GiB é opt-in (`run_big_stream=1`); o padrão usa 2 x 256 MiB +
  2 MiB de sink. O orçamento reportado é 4.348.654.388 B; o `gpupeak` roda headless, sem
  modelo carregado.
- **Sweep**: o bootstrap do KVarN4 (records + stage + tail) leva dezenas de segundos em
  NumPy e ~150 MB de upload TCP por sessão de 28k linhas; o sweep usa 4 sessões. O
  `active_rows` medido (não o alvo) entra no roofline, então o crescimento de linhas
  durante as execuções não distorce a conta.
- **Clock**: o contador de timestamp da AMD não é o clock do shader (18.024 ticks para
  ~0,18 ms implicam ~100 MHz). O clock implícito depende do latência 4 clocks da VALU
  RDNA2; está rotulado como derivado.
- **Observação alheia**: `rkva_range_q4_kernel_probe.py` mede com profiling desligado
  (flags=2). Não alterei; vale corrigir antes de usar os números de decode dele.

## 9. Estimativa de ganho (ESTIMATIVA, não medida)

O que a suíte entrega é decisão, não ganho direto. Cenários para o `pass1` de prefill,
usando 180,4 GFLOP por envelope, 16 camadas e o custo não-`pass1` de 2,25 ms/envelope
(67,55 - 65,30):

| Pico FP32 medido | `pass1` a 50% dele | envelope | teto só-Xbox | ponta a ponta (modelo aditivo) |
| --- | --- | --- | --- | --- |
| 4 TFLOP/s | 2,0 TFLOP/s | 92,4 ms | 173 t/s | ~156 t/s |
| 6 TFLOP/s | 3,0 TFLOP/s | 62,4 ms | 256 t/s | ~221 t/s |
| 8 TFLOP/s | 4,0 TFLOP/s | 47,3 ms | 338 t/s | ~278 t/s |
| 12,15 TFLOP/s (nominal) | 6,1 TFLOP/s | 31,9 ms | 501 t/s | ~380 t/s |

Premissas: 50% do pico medido é o que um kernel com dequant/WHT/exp costuma atingir;
"ponta a ponta" usa o resíduo atual de ~0,635 ms/token (`1/206 - 1/237`), que pode mudar.
Se a suíte medir 4 TFLOP/s, otimizar o `pass1` rende pouco e o gargalo é outro; se medir
8-12, o kernel tem 3-4x de espaço e a frente A2 é a certa. **Nada disso é medição.**

Para o decode, a latência de dispatch/fence/readback decide a fronteira da frente A3:
hoje sobram ~0,5 ms/camada fora da GPU; se o piso de D3D12 medido for ~0,05 ms, ~90% do
overhead é atacável; se for ~0,3 ms, o teto de encurtar o host é pequeno.

## 10. Commits

Cinco commits sobre `40f22aa6b`, nesta ordem (hashes em `git log`):

```
feat(xbox): add GPU ceiling probe shaders and DXIL
feat(xbox): add headless GPU ceiling suite and roofline model
test(xbox): add pass1 sweep over the existing RKVA diagnostics
feat(xbox): add inert one-window ceiling orchestration
docs(xbox): report the ceiling probe design, evidence and open risks
```

(o `.gitignore` com `.oc-artifacts/` entrou no commit de orquestração)

Nenhum arquivo fora de `tools/xbox/` e do `.gitignore` foi tocado; nada foi escrito em
`/home/hjotha/beellama.cpp`, no Xbox ou em `~/.codex`. Artefatos locais (desmontagens
DXIL) estão em `.oc-artifacts/`, ignorado pelo git.
