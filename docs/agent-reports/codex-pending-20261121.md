# Pendências da paridade Strata × BeeLlama

Base `1327435a1`, branch `feat/strata-pending`. Implementação e testes no commit `53de7ad41`. O ponto de partida na A100 é pp28k de 1.567,4 t/s com cache residente no prefill, ante 4.629 t/s na referência Strata. A série anterior mede +3,5% para residência e efeitos pequenos ou ausentes para as demais flags de prefill. Esta entrega prioriza medir a diferença por etapa.

| Item | Entrega |
|---|---|
| Atribuição de prefill | `LLAMA_PREFILL_PROFILE=1` registra preparação, submissão, sincronizações já existentes, eventos do scheduler, cópias e inventário GDN, QSA/FA, MoE e restante. O resumo agrega os logs e o perfil de kernels, cópias e ociosidade do dispositivo. |
| PLE | `LLAMA_PLE_PREFETCH=1` aconselha o SO a carregar linhas do próximo ubatch após submeter o atual. Somente Qwen4Exp com tabela lazy/mmap e próximo ubatch contíguo, de texto e uma sequência. As primeiras `n_gram-1` linhas ainda dependem do histórico comprometido. O caminho normal resolve todas as linhas, inclusive fronteiras, lacunas, multimodal e rollback. |
| Regressão MTP | O log separa preparo do alvo, extensão do último bloco com NextN ativo, exportação de linhas/bytes e sincronização. O trace existente separa espera e escrita de input no draft. O alvo usa um reshape do residual, anterior ao mixer final. Com NextN desativado o último bloco pode restaurar a poda anterior ao FFN. Com MTP cada `h[p]` alimenta o KV draft de `p+1` e o último hidden vira carry. Cortar linhas só pelos logits muda o histórico. O primeiro corte seguro já está disponível em `LLAMA_MTP_KV_STORE_ONLY=1`, preservando todas as escritas do catch-up. |
| IDs direto à FA | A geometria registra o teto de bytes de I32 por query e número de células do corpo. O marcador CUDA existente mede os bytes de máscara varridos e a capacidade da união. A seleção, a máscara e a FA continuam pelo caminho atual. |
| Replay | O resumo cruza larguras 1/2/3 com decisão, invalidação, fallback e modo executado via `graph_run`. A política de CUDA Graphs continua a mesma. |
| Overlap com misses | O bypass seletivo e os fills do banco já chamam `llama_moe_upload`. Os dois leem IDs e sincronizam o roteamento antes de emitir uploads. O stream de compute espera o upload no início do split consumidor. A instrumentação identifica D2H dos IDs, bytes H2D enviados, D2D de hits no bypass, uso de staging, esperas pelo ring e submissão da dependência. A razão dominante para o empate anterior depende do trace. |
| Hidden D2D | A instrumentação registra bytes e espera do D2H e as escritas de estado do draft. O desenho abaixo exige vida útil do staging, rearme e carry por sequência. |

## Leitura dos perfis

- `LLAMA_PREFILL_PROFILE=1` seleciona ubatches de 4+ tokens fora de warmup. `ctx=0` identifica alvo e `ctx=1` identifica draft MTP. Lotes largos de verificação também podem aparecer. Para atribuir prefill sem ambiguidade, escolha a janela do prompt no trace. `PREP` mostra `nq`, outputs, NextN e reuso. Com `LLAMA_GRAPH_PREP_PROFILE=1` também aparecem apply, check, reset, build, alloc e inputs individualmente.
- `PREFILL CLASS` contabiliza bytes **lógicos de saídas de nós**. Esses bytes indicam geometria e volume produzido. `cpu_us` mede wall time de splits síncronos da CPU e `device_host_us` mede o intervalo do host submetendo trabalho ao dispositivo. Um split de várias classes é registrado uma vez em `mixed`. Para uma sessão diagnóstica extra, `GGML_SCHED_SPLIT_TIMING=1` sincroniza cada split e inclui a espera nesse intervalo. Remova a variável para medir o comportamento normal.
- `PREFILL SCHED` agrupa cópias genéricas executadas, sem cache hits de input. Os callbacks de experts têm bytes próprios no trace. `transfer_host_us` inclui `input_wait_host_us` e `callback_host_us`. `PREFILL SYNC` e a espera do NextN podem cobrir o mesmo intervalo. Use a união de intervalos de kernel e de DMA no Nsight para calcular o tempo de cópia fora de compute e os gaps do dispositivo. `idle_ns` é o intervalo sem kernel ou cópia rastreados, que pode incluir espera do host, dependências ou submissão. Veja as APIs CUDA/OSRT no mesmo intervalo antes de atribuir a causa.
- `strata_moe_upload` usa bit 0 para stream dedicado e bit 1 para destino host. `strata_moe_route` usa bit 0 para fill do banco e zero para bypass. `strata_moe_cuda_upload` usa bit 0 no primeiro upload do lote, bit 1 para staging, bit 2 para origem pinned confirmada e bit 3 para origem sem consulta. `value` conta chunks de staging. Cruze o custo de `cudaMemcpyAsync`, `strata_moe_ring_wait` e `strata_moe_stream_join` com o DMA e com a espera do roteamento. O join é submissão de dependência, enquanto o ring pode bloquear a thread.
- `cuda_graph_decision` usa os códigos do enum em `ggml-cuda.cu`, por exemplo 2 para env, 4 para pressão, 9 para MMID incompatível, 10 para warmup e 11 para propriedades/pool alterados. `cuda_graph_invalidated` distingue propriedades (bit 0) e pool (bit 1). `cuda_local_span_gpu` usa 0 para execução direta, 1 para replay e 2 para captura com launch. Exija zero perda de trace antes de comparar proporções.
- O resumo SQLite classifica kernels pelos nomes. `mmq_moe_candidates` também pode conter MMQ genérico, então confirme em quais splits/camadas houve `MUL_MAT_ID`. Kernels genéricos podem ficar em `other`. Selecione no Nsight o intervalo contínuo de cada prefill, sem load nem decode, com limites fora das atividades que quer contar. As somas por família podem se sobrepor. O cálculo de atividade do dispositivo usa a união. [Esquema SQLite do Nsight Systems](https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html)

## Trabalho condicionado ao perfil

- **QSA IDs.** Começar pelo corpo Q8_0 local, CUDA Ampere+, head 256 e os dispatches esparsos de uma ou oito queries por tile. Construir uma saída auxiliar de IDs I32 e contagem por query após top-k e a expansão dos pools, filtrando -inf, sentinelas e slots repetidos. Preservar o tail causal parcial e os offsets de cada stream. O consumidor unifica os IDs por tile/GQA e carrega bits de pertinência por query. A mesma máscara fica disponível para layouts excluídos, união acima da capacidade, KVarN, tail de precisão explícita e CPU. A proteção do overflow precisa selecionar um caminho denso que realmente escreva todos os outputs. Qualificar com conjunto de células idêntico por query, overflow forçado, cauda de 1/2/3, GQA incompatível, duas sequências e NMSE ≤1e-6, além dos casos de matriz abaixo.
- **MTP D2D.** Se a passagem host for material, usar staging de propriedade do draft com trilha experimental single-device, uma sequência e offsets contíguos. O produtor deve publicar completion de cada faixa e o alvo só pode reusar sua saída depois que o draft a copiar. Usar o carry anterior na primeira linha e promover a última só após sucesso. Associar posição e ID de decode a cada faixa. Em erro, restauração, mudança de sessão ou geometria fora desse contrato, preservar a rota host. Testar prefixo restaurado, rejeição parcial e cruzamento de ubatch com logits/hidden NMSE ≤1e-6 e aceitação dentro de 1 p.p.
- **PLE.** As dicas `WILLNEED` são best-effort e podem custar chamadas de sistema ou competir com o mapa de experts. O pipeline evita worker, cópia da tabela ou buffer além dos IDs do próximo ubatch. Mantém a dica da leitura normal. O tempo de GPU em paralelo, os major faults, o RSS e o efeito de disco quente precisam do A/B.
- **Replay/overlap.** Alterar a política apenas após identificar uma causa dominante recorrente. O bypass já usa o stream dedicado quando elegível. Em hits totais essa trilha pode ficar sem uploads. Com misses, DMA pageable, custo de copiar ao ring e a espera imediata do consumidor são hipóteses distinguíveis pelas novas marcas.

## A/B na RTX 4070 e na A100

Use o mesmo build por GPU. Compile com sm_89 na 4070 e sm_80 na A100, com CUDA FA e CUDA Graphs ligados. Esta etapa roda em um processo de benchmark dedicado. O harness encerra apenas os filhos que ele próprio iniciou. No host de build da A100, mantenha a configuração de Glibc/CCCL/linker do plano. [Build A100](../PLAN-strata-parity.md#build-a100-nota-curta)

```bash
# RTX 4070
ARCH=89
TARGET=/modelos/OLMoE-1B-7B-Q2_K.gguf
base=(-m "$TARGET" -cmoe -c 4096 -b 512 -ub 128)
FULL=3072
export PROMPTS=64,3840

# A100, substitua o bloco anterior por este
ARCH=80
TARGET=/modelos/Flash-Next-Q2_0-00001-of-00002.gguf
SIDECAR=/modelos/MTP/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf
base=(-m "$TARGET" -ncmoe 2 -c 32768 -b 2048 -ub 1024)
FULL=6144
export PROMPTS=64,4096,28672
sha256sum "$TARGET" "${TARGET/00001-of-00002/00002-of-00002}" "$SIDECAR"
```

```bash
# Comum aos dois hosts. Abra um shell Bash sem flags experimentais herdadas.
CUDA_BUILD="build-strata-sm$ARCH"
cmake -S . -B "$CUDA_BUILD" -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DGGML_CUDA=ON -DGGML_CUDA_FA=ON -DGGML_CUDA_GRAPHS=ON \
  -DGGML_VULKAN=OFF -DCMAKE_CUDA_ARCHITECTURES="$ARCH"
cmake --build "$CUDA_BUILD" -j4 --target llama-server test-llama-archs test-backend-ops
BIN="$PWD/$CUDA_BUILD/bin"
OUT=$(mktemp -d /tmp/strata-pending.XXXXXXXX)
export PORT=18090
git rev-parse HEAD > "$OUT/commit"
sha256sum "$TARGET" > "$OUT/model.sha256"
if [[ $ARCH == 80 ]]; then
  sha256sum "${TARGET/00001-of-00002/00002-of-00002}" "$SIDECAR" >> "$OUT/model.sha256"
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$OUT/gpu"

cat > "$OUT/load.py" <<'PY'
import json, os, time, urllib.request

def call(path, data):
    url = "http://127.0.0.1:" + os.environ["PORT"] + path
    req = urllib.request.Request(url, json.dumps(data).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as response:
        return json.load(response)

text = ("def process(rows):\n    return [(row[0], sum(row[1:])) for row in rows if row is not None]\n" * 1600)
tokens = call("/tokenize", {"content": text, "add_special": True})["tokens"]
for n in map(int, os.environ["PROMPTS"].split(",")):
    assert len(tokens) >= n
    for tag in ("cold", "warm"):
        started = time.perf_counter()
        r = call("/completion", {"prompt": tokens[:n], "n_predict": 128, "temperature": 0,
                                 "seed": 12345, "ignore_eos": True, "cache_prompt": False,
                                 "return_tokens": True, "stream": False})
        assert not r["truncated"] and r["timings"]["prompt_n"] == n and len(r["tokens"]) == 128
        print(json.dumps({"n": n, "tag": tag, "wall_s": time.perf_counter()-started,
                          "timings": r["timings"], "tokens": r["tokens"]}), flush=True)
PY

run() (
  set -e
  set +m
  label=$1
  shift
  python3 -c 'import os,socket; s=socket.socket(); s.bind(("127.0.0.1",int(os.environ["PORT"])))'
  launch=()
  if [[ ${NSYS:-0} == 1 ]]; then
    launch=(nsys profile --trace=cuda,osrt --cuda-graph-trace=node --force-overwrite=true -o "$OUT/$label")
  fi
  if [[ ${TRACE:-0} == 1 ]]; then
    export GGML_RKVA_TRACE_FILE="$OUT/$label.trace"
    export LLAMA_CUDA_VERIFY_TRACE=1 LLAMA_MTP_HIDDEN_TRACE=1 LLAMA_QSA_FA_TRACE=1 LLAMA_PLE_PREFETCH_TRACE=1
  fi
  start=$SECONDS
  CUDA_VISIBLE_DEVICES=0 setsid "${launch[@]}" "$BIN/llama-server" "${base[@]}" \
    -ngl 99 -fa on -ctk q8_0 -ctv q8_0 --load-mode mmap --lazy-mode on \
    --no-warmup -cram 0 -np 1 --host 127.0.0.1 --port "$PORT" "$@" >"$OUT/$label.log" 2>&1 &
  server=$!
  nvidia-smi -i 0 --query-gpu=timestamp,memory.used --format=csv,noheader,nounits -lms 250 >"$OUT/$label.vram.csv" &
  telemetry=$!
  trap 'kill -TERM "$telemetry" 2>/dev/null || true; wait "$telemetry" || true; kill -INT -- "-$server" 2>/dev/null || true; wait "$server" || true' EXIT
  for ((i=0; i<900; ++i)); do
    if curl -fsS "http://127.0.0.1:$PORT/health" >"$OUT/$label.health.json" 2>/dev/null; then break; fi
    kill -0 "$server"
    sleep 1
  done
  curl -fsS "http://127.0.0.1:$PORT/health" >"$OUT/$label.health.json"
  printf '%s\n' "$((SECONDS-start))" >"$OUT/$label.ready-s"
  rss_faults() {
    awk '{ print "minor_faults=" $10, "major_faults=" $12 }' "/proc/$server/stat"
    awk '/^VmHWM:/ { print "peak_rss_kib=" $2 }' "/proc/$server/status"
  }
  if [[ ${NSYS:-0} == 0 ]]; then rss_faults >"$OUT/$label.before.txt"; fi
  python3 "$OUT/load.py" >"$OUT/$label.requests.jsonl"
  if [[ ${NSYS:-0} == 0 ]]; then rss_faults >"$OUT/$label.after.txt"; fi
)
```

O primeiro caso de cada processo é realmente frio para o cache de experts. Para medir pp28k frio, rode `PROMPTS=28672` sozinho. Para pp4k frio use `PROMPTS=4096`. No roteiro completo, os demais rótulos `cold` marcam o primeiro pedido daquele tamanho no processo. Use três pares alternados por hipótese. As sessões com profiler servem para atribuição.

```bash
# Ambas as GPUs, overhead da atribuição e caminho de misses.
for rep in 1 2 3; do
  for arm in 0 1; do
    LLAMA_PREFILL_PROFILE=$arm LLAMA_MOE_CACHE_PREFILL=1 \
      run "profile-$rep-$arm" --moe-cache-mib "$FULL"
    LLAMA_MOE_UPLOAD_OVERLAP=$arm \
      run "overlap-$rep-$arm" --moe-cache-mib 1024
    LLAMA_MOE_UPLOAD_OVERLAP=1 LLAMA_MOE_PINNED_RING=$arm \
      run "ring-$rep-$arm" --moe-cache-mib 1024
  done
done
for arm in 0 1; do
  TRACE=1 NSYS=1 LLAMA_PREFILL_PROFILE=1 LLAMA_MOE_CACHE_PREFILL=$arm \
    run "resident-trace-$arm" --moe-cache-mib "$FULL"
  TRACE=1 NSYS=1 LLAMA_PREFILL_PROFILE=1 LLAMA_MOE_UPLOAD_OVERLAP=$arm \
    run "overlap-trace-$arm" --moe-cache-mib 1024
  TRACE=1 NSYS=1 LLAMA_PREFILL_PROFILE=1 LLAMA_MOE_UPLOAD_OVERLAP=1 LLAMA_MOE_PINNED_RING=$arm \
    run "ring-trace-$arm" --moe-cache-mib 1024
  TRACE=1 NSYS=1 LLAMA_PREFILL_PROFILE=1 GGML_CUDA_DISABLE_GRAPHS=1 \
    LLAMA_MOE_UPLOAD_OVERLAP=1 LLAMA_MOE_PINNED_RING=$arm run "ring-direct-$arm" --moe-cache-mib 1024
done

# A100, PLE com a tabela original. Rode com disco frio e quente separadamente.
for rep in 1 2 3; do
  for arm in 0 1; do
    PROMPTS=28672 LLAMA_PLE_PREFETCH=$arm LLAMA_MOE_CACHE_PREFILL=1 \
      run "ple-$rep-$arm" --moe-cache-mib 6144
  done
done
for arm in 0 1; do
  PROMPTS=28672 TRACE=1 NSYS=1 LLAMA_PREFILL_PROFILE=1 LLAMA_PLE_PREFETCH=$arm \
    LLAMA_MOE_CACHE_PREFILL=1 run "ple-trace-$arm" --moe-cache-mib 6144
done

# A100, MTP. Primeiro isole o custo do NextN com o mesmo orçamento nos três braços.
draft=(-md "$SIDECAR" --spec-type draft-mtp --spec-draft-n-max 2 -ngld 99 -devd CUDA0)
MTP_CACHE=0
for rep in 1 2 3; do
  run "mtp-plain-$rep" --moe-cache-mib "$MTP_CACHE"
  for arm in 0 1; do
    LLAMA_MTP_KV_STORE_ONLY=$arm run "mtp-$rep-$arm" --moe-cache-mib "$MTP_CACHE" "${draft[@]}"
  done
done
TRACE=1 NSYS=1 LLAMA_PREFILL_PROFILE=1 run mtp-plain-trace --moe-cache-mib "$MTP_CACHE"
for arm in 0 1; do
  TRACE=1 NSYS=1 GGML_MTP_PROF=1 LLAMA_PREFILL_PROFILE=1 LLAMA_MTP_KV_STORE_ONLY=$arm \
    run "mtp-trace-$arm" --moe-cache-mib "$MTP_CACHE" "${draft[@]}"
done
# Braço de produção em separado. Registre slots, placement e orçamento efetivo.
LLAMA_MTP_RESERVE_WEIGHTS=1 LLAMA_MTP_KV_STORE_ONLY=1 \
  run mtp-production --moe-cache-mib -1 --vram-reserve-mib 4096 "${draft[@]}"

# 4070 e A100, larguras 1/2/3. Na A100 repita também com "${draft[@]}" e n-max 1/2.
for mode in on off; do
  if [[ $mode == off ]]; then export GGML_CUDA_DISABLE_GRAPHS=1; else unset GGML_CUDA_DISABLE_GRAPHS; fi
  TRACE=1 run "replay-$mode-plain" --moe-cache-mib "$FULL"
  for n in 1 2; do
    TRACE=1 run "replay-$mode-ngram$n" --moe-cache-mib "$FULL" --spec-type ngram-simple \
      --spec-ngram-simple-size-n 2 --spec-ngram-simple-size-m "$n" --spec-ngram-simple-min-hits 1
  done
  if [[ $ARCH == 80 ]]; then
    for n in 1 2; do
      TRACE=1 LLAMA_MTP_KV_STORE_ONLY=1 run "replay-$mode-mtp$n" --moe-cache-mib "$MTP_CACHE" \
        -md "$SIDECAR" --spec-type draft-mtp --spec-draft-n-max "$n" -ngld 99 -devd CUDA0
    done
  fi
done
unset GGML_CUDA_DISABLE_GRAPHS

for arm in 0 1; do
  LLAMA_TEST_MOE_FULL=1 LLAMA_MOE_UPLOAD_OVERLAP=$arm LLAMA_MOE_PINNED_RING=$arm \
    "$BIN/test-llama-archs" -a '^deepseek2$' -b CUDA0 -s 12345 -v 4
done
"$BIN/test-backend-ops" test -b CUDA0 -j 1 --seed 12345 -o FLASH_ATTN_EXT \
  -p 'hsk=256.*kv=(4096|16384|32768),nb=(1|2|3|4|5|13|512),.*type_K=q8_0,type_V=q8_0.*n_kv_max=(0|2048)'
"$BIN/test-backend-ops" test -b CUDA0 -j 1 --seed 12345 -o GATED_DELTA_NET_CACHE_FUSION
```

A comparação de lazy on/off exige RAM para a tabela inteira e mede uma hipótese separada. Na 4070, OLMoE serve de controle sem PLE. Para smoke de PLE, a fixture Qwen4Exp pode ser usada em CUDA. A tabela sintética qualifica o dispatch e o trace sem reproduzir a pressão de I/O do GGUF real.

```bash
# RTX 4070, smoke com a fixture Qwen4Exp
"$BIN/test-llama-archs" --test-ple-prefetch "$OUT/ple-fixture.gguf" -s 12345
(
  base=(-m "$OUT/ple-fixture.gguf" -c 512 -b 64 -ub 16)
  for arm in 0 1; do
    PROMPTS=64 TRACE=1 LLAMA_PLE_PREFETCH=$arm run "ple-fixture-$arm" --moe-cache-mib 0
  done
)
```

```bash
python3 scripts/strata-prefill-profile.py --log "$OUT/overlap-trace-1.log"
python3 scripts/strata-prefill-profile.py --trace "$OUT"/replay-on-*.trace.*.jsonl
python3 scripts/strata-prefill-profile.py --trace "$OUT"/ring-trace-1.trace.*.jsonl
nsys export --type sqlite -o "$OUT/ring-trace-1.sqlite" "$OUT/ring-trace-1.nsys-rep"
nsys stats --report cuda_gpu_trace,cuda_api_trace,cuda_gpu_mem_size_sum "$OUT/ring-trace-1.nsys-rep" \
  > "$OUT/ring-trace-1.timeline.txt"
# Defina START_NS e END_NS em ns relativos à sessão, a partir do intervalo de um prompt no timeline.
python3 scripts/strata-prefill-profile.py --sqlite "$OUT/ring-trace-1.sqlite" \
  --start-ns "$START_NS" --end-ns "$END_NS"
```

### Critérios numéricos

- **Atribuição.** Em três pares, exigir saída greedy idêntica, overhead mediano do perfil in-engine ≤3% em pp4k/pp28k e nenhuma perda de trace. No Nsight contabilizar kernel busy, cópia fora de kernel e ociosidade para 100% da janela. Na A100 investigar a maior família ou gap com ≥10% da janela antes de propor outra flag. Registrar bytes de cada direção por stream, tempo de eventos, TTFT/prefill e placement. Aceitar classe `other` e revisão manual dos nomes genéricos.
- **PLE.** Exigir mesmos IDs greedy, nenhum OOM, NMSE da fixture ≤1e-6, e pp28k ≥103% do controle acima da dispersão ou ≥10% menos tempo de fault com throughput ≥97%. Medir major faults e RSS de pico do processo no host, bem como tempo até `/health`. Confirme marcadores de lookahead apenas no braço lazy on elegível e rode ablações de 4k/decode para verificar ≥97% do controle.
- **MTP e D2D.** Compare linhas e bytes absolutos e ms de Target, NextN D2H, espera, H2D e catch-up sem comparar braço auto com fixo como se fossem iguais. A otimização futura precisa recuperar ≥3% de pp28k acima da dispersão, decode ≥97%, aceitação dentro de 1 p.p. e pico ≤39.005 MiB se o teto de 40,9 GB for aplicável. Mantendo os braços fixos iguais, ajuste `MTP_CACHE` somente se ambos completarem e registre o orçamento.
- **QSA.** Exigir PASS da matriz Q8, zero overflow inesperado e fallback corretos. Considerar a troca por IDs apenas se seleção → união → FA representar ≥10% da janela ou reduzir o pico/bytes intermediários em ≥10%, com cauda e fallback validados. Compare a geometria de IDs do ubatch com os bytes de máscara de cada camada compatível.
- **Replay.** Para cada largura, coletar ≥100 amostras reais de alvo por modo e medir contagem e p50/p95 de submissão host e span GPU. Uma alteração futura exige ≥3% de melhora no p95 acima da dispersão e mesma saída greedy. Se a carga n-gram não gerar larguras 2/3 suficientes, contar como amostra incompleta e usar o sidecar n-max 1/2 na A100.
- **Overlap.** Com misses, exigir NMSE ≤1e-4 quando a fixture de experts for usada, mesmos tokens e mesmo volume H2D. Um ganho exige redução ≥10% de DMA fora de compute ou pp28k ≥103%, acima da dispersão. O braço residente e os demais casos precisam de throughput ≥97%.

## Validação CPU e arquivos

Build `RelWithDebInfo` com `GGML_CUDA=OFF`, Vulkan, HIP, SYCL, OpenCL e Metal desligados. Alvos compilados com `-j4`. O teste `test-strata-prefill` cria e abre uma fixture GGUF Qwen4Exp, verifica lazy off/on, prefetch off/on, EOS, lacuna e replay após restore. Também exercita o perfil in-engine, a contagem de linhas NextN, classes da CPU e os modos de leitura dos relatórios com SQLite/replay sintéticos. `test-strata-trace` cobre geometria de QSA em CPU e a correlação das larguras com o graph run.

- Scheduler e preparação em `ggml/include/ggml-backend.h`, `ggml/src/ggml-backend.cpp`, `src/llama-context.{cpp,h}` e `src/llama-impl.{cpp,h}`.
- PLE/QSA em `src/models/qwen4exp.cpp` e `src/llama-memory-hybrid.h`.
- Trace MoE em `src/llama-moe-cache.cpp` e `ggml/src/ggml-cuda/ggml-cuda.cu`.
- Harness e testes em `scripts/strata-prefill-profile.py`, `tests/test-strata-prefill.py`, `tests/test-strata-trace.py`, `tests/test-llama-archs.cpp` e `tests/CMakeLists.txt`.

| Teste executado | Resultado |
|---|---:|
| CTest focado | 34/34 |
| CTest MTP, KVarN e QSA com as dez flags em `1` | 5/5 |
| FA Q8, KV 4096, `nb=1` | 3/3 |
| `GATED_DELTA_NET_CACHE_FUSION` | 14/14 |
| Decisão de lookahead | 56/28 IDs por ubatch de 16/9 tokens, repetidos após restore, só com lazy on e flag em `1` |

A seleção focada segue o relatório anterior, acrescida de `test-strata-prefill`. Os blocos Bash e o cliente Python acima passaram na validação de sintaxe. [Comandos da suíte focada](codex-opt-implementation-20261121.md#código-e-testes-cpu)

O perfil de CUDA, o efeito de I/O de PLE e as hipóteses de replay/overlap ficam pendentes do hardware. A alteração CUDA desta branch é telemetria de upload e eventos. O build e os testes desta entrega foram todos CPU-only.
