# Strata × BeeLlama · implementação de otimizações

Worktree `beellama-strata-opt`, branch `feat/strata-opt`, base `b51af9d9c`, implementação `933c8746e`. Os experimentos exigem valor `1` no respectivo env. Os números de throughput abaixo são critérios para o A/B de hardware.

## Entrega

| Item | Flag | Implementação e limite |
|---|---|---|
| 1. Mapa residente | `LLAMA_MOE_MAP_DEVICE` | Grupos CUDA de capacidade integral com espaço para o mapa recebem uma faixa contígua por camada. O construtor termina o upload do banco, publica I32 e só então disponibiliza os tensores ao scheduler. O lookup lê o mapa como peso do device e o callback SLRU deixa de participar. Os endereços e os slots vivem pelo contexto inteiro, inclusive em capture/replay. O gate de ubatches acima de 32 continua em `LLAMA_MOE_CACHE_PREFILL`. |
| 2. Ring pinned | `LLAMA_MOE_PINNED_RING` com `LLAMA_MOE_UPLOAD_OVERLAP=1` | Dois blocos de 4 MiB por backend CUDA, alocados no primeiro upload. A origem pinned passa direto. O ring divide a origem pageable sem mudar o volume e só reusa um bloco depois de seu evento DMA. Os eventos existentes ordenam a reutilização do destino e o consumo. Falha ou desabilitação de pinned usa a cópia pageable anterior. Compare o tempo de submissão com misses antes de optar pelo ring. A cópia host adicional e a espera ao dar a volta no ring podem custar mais que a economia. |
| 3. Limites do indexer | `LLAMA_QSA_INDEXER_BOUNDS` | Qwen4Exp de quatro cabeças em CPU/CUDA. Dois I32 por query usam o mesmo intervalo causal e por sequência que preenchia a F16. As rotas CUDA vetorial e tiled pulam tiles inteiramente invisíveis e gravam `-inf`. Scores visíveis mantêm a redução existente. A máscara de células da FA ainda é gerada. |
| 4. IDs direto à FA | `LLAMA_QSA_FA_TRACE` | A FA continua consumindo a máscara atual. O trace registra o tamanho lógico varrido, largura da união e capacidade. O kernel diagnóstico informa os casos em que a união real excede a capacidade. Só use a impressão CUDA para diagnóstico. O desenho está abaixo. |
| 5. GDN conv+SiLU+L2 | `LLAMA_GDN_CONV_L2` | Qwen4Exp. O epílogo CUDA usa um bloco de 128 canais por cabeça Q/K e mantém o caminho SiLU em V. Usa o kernel de conv existente nas larguras curtas e longas. A variante CPU usa o mesmo contrato e fatia o trabalho por cabeças. O grafo segue escrevendo o estado de conv a partir da entrada e os estados GDN a partir do scan existente. A normalização fundida altera a ordem de redução e o arredondamento. |
| 6. Replay por largura | `LLAMA_CUDA_VERIFY_TRACE` + `GGML_RKVA_TRACE_FILE` | Marca `graph_run`, largura real do ubatch 1/2/3, posições inicial/final, número de saídas e tipo de contexto. Correlaciona os splits CUDA com captura, replay e execução direta. Acrescenta bits para mudança de propriedades e mutação do pool. A política de captura precisa de uma causa dominante medida antes de ser alterada. |
| 7. NextN D2D | `LLAMA_MTP_HIDDEN_TRACE` + `GGML_RKVA_TRACE_FILE` | Mede bytes e submissão D2H, espera no getter sincronizado e escrita do estado do draft. A latência do getter inclui todo o trabalho pendente anterior. O trace CUDA é necessário para isolar o DMA. A passagem de hidden continua via host. |
| 8. PLE | `LLAMA_PLE_PREFETCH_TRACE` + `GGML_RKVA_TRACE_FILE` | Registra se o lote emitiu o prefetch existente, quantidade de linhas pedidas e tempo host. `bytes` estima os bytes pedidos antes de coalescência e page cache. Nenhum lookahead ou carregamento adicional da tabela. |

### Próximas mudanças dependentes dos traces

**QSA direto à FA.** Começar com CUDA local sm80+, Q8_0/Q8_0, D=256, GQA >4 e sem KV tail explícito ou KVarN. Emitir uma lista de células por query com contagem real, descartando pools com score inválido e o sentinela da cauda. A compactação por tile 1/8 deve produzir uma união de IDs e bits de pertinência por query. Usar a pertinência para aplicar a máscara no mesmo ponto do cálculo atual. O consumidor precisa de um ramo denso no device para overflow da união, com a máscara e workspace disponíveis até que a exclusão do fallback seja provada. A rota também deve aceitar a última tile incompleta e várias sequências sem misturar células. O limite atual da união é `min(n_kv, largura × limite_por_query)`. Se aparecer truncamento, investigar a origem antes de trocar o formato. Evitar decisão via D2H no caminho quente.

**Replay.** Juntar `strata_verify_ubatch` e `cuda_graph_*` por `graph_run` e contar por `nq` e comprimento do prefixo. Em `cuda_graph_enqueue`, 0 é direto, 1 replay e 2 captura. O contador de decisão marca a rota escolhida antes da execução. Verificar também `cuda_graph_fallback` para identificar a tentativa de captura que terminou na execução direta. Razões relevantes de `cuda_graph_decision` incluem 9 (MMID com sync), 10 (warmup), 11 (mudança) e 14 (SAW/NC com sync). `cuda_graph_invalidated` usa bit 0 para propriedades e bit 1 para o pool. `cuda_graph_pool_changed_during_execution` sinaliza mudança observada durante o próprio split. Se alternância de largura realmente invalidar o cache, estudar executáveis por largura, ainda validados por identidade, bindings e epoch do pool. O mapa imutável elimina uma causa externa ao replay, o callback de roteamento dos grupos integrais.

**MTP D2D.** Parear primeiro o sidecar com store-only e separar `strata_nextn_read` (bit 0 host, bit 1 draft), `strata_nextn_wait` e `strata_mtp_state_write` (bit 0 host). As escritas de input ocorrem antes do submit do graph do draft. Alinhar pelo timestamp e sucessão de ubatches. Para prototipar, usar staging pertencente ao contexto de destino em CUDA single-device e uma sequência contígua. A cópia espera o produtor e precisa de uma barreira de vida útil antes de reutilizar a saída do alvo. Conservar o carry anterior como primeira linha, deslocar as linhas em uma posição e atualizar a última linha só após sucesso. Associar cada faixa ao ID do decode para rejeitar dados obsoletos após rollback ou mudança de contexto. Tratar múltiplas sequências, cadeias de heads, embeddings mistos e rollback sem proveniência pelo caminho host existente. O A/B pede queda dos bytes host e da latência de sincronização, logits/hidden equivalentes e aceitação dentro de 1 p.p.

## A/B de hardware

Rodar em um processo de teste dedicado, com os modelos locais completos. Registrar SHA-256 dos GGUFs, commit, GPUs, driver, settings de cache e linhas de placement. Usar sm_89 na 4070 e sm_80 na A100. Os comandos abaixo são para execução posterior em cada máquina.

```bash
ARCH=89                         # na A100 use 80
CUDA_BUILD="build-cuda-sm${ARCH}"
cmake -S . -B "$CUDA_BUILD" -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DGGML_CUDA=ON -DGGML_CUDA_GRAPHS=ON -DGGML_CUDA_FA=ON \
  -DGGML_VULKAN=OFF -DCMAKE_CUDA_ARCHITECTURES="$ARCH"
cmake --build "$CUDA_BUILD" -j2 --target llama-server test-strata-opt test-llama-archs test-backend-ops
export BIN="$PWD/$CUDA_BUILD/bin"
export OUT="$(mktemp -d /tmp/strata-opt.XXXXXXXX)"
export PORT=8090
git rev-parse HEAD | tee "$OUT/commit.txt"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv | tee "$OUT/gpu.txt"
```

Na RTX 4070:

```bash
export TARGET=/caminho/OLMoE-1B-7B-Q2_K.gguf
sha256sum "$TARGET" | tee "$OUT/model.sha256"
base=(-m "$TARGET" -cmoe -ngl 99 -c 4096 -b 512 -ub 128)
full=3072
export PROMPTS=64,3840
```

Na A100:

```bash
export TARGET=/caminho/Flash-Next-Q2_0-00001-of-00002.gguf
export SIDECAR=/caminho/MTP/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf
sha256sum "$TARGET" "${TARGET/00001-of-00002/00002-of-00002}" "$SIDECAR" | tee "$OUT/model.sha256"
base=(-m "$TARGET" -ncmoe 2 -ngl 99 -c 32768 -b 2048 -ub 1024)
full=6144
export PROMPTS=64,4096,28672
```

Use a carga abaixo em ambos os braços. `nq` vem do runtime. A carga usa o texto do harness LP30, tokenizado e cortado por tamanho exato via `/tokenize`. Envia tokens na API nativa com seed fixa, greedy, cache de prompt desligado e sem chat template. Isso separa os braços locais de comparação histórica de chat. Os casos pareados são cold e warm para cada processo. `PAIR=1` manda duas sequências de até 4k com sufixos diferentes. Compare dois processos com `-np 2 -kvu`. Verifique os slots e os ubatches realmente agrupados no trace. Para testar duas janelas de 28k ao mesmo tempo, defina `PAIR_TOKENS=28672` e use `-c 65536` nos dois braços com orçamento verificado separadamente.

```bash
cat > "$OUT/load.py" <<'PY'
import concurrent.futures
import json
import os
import time
import urllib.request

base = "http://127.0.0.1:" + os.environ["PORT"]
filler = ("def process(rows):\n    out = []\n    for row in rows:\n"
          "        if row is None:\n            continue\n"
          "        out.append((row[0], sum(row[1:])))\n    return out\n\n")

def call(path, data):
    req = urllib.request.Request(base + path, data=json.dumps(data).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=1800))

def measure(case, n, n_predict, slot=0):
    content = ("Analyze the following program.\n" if slot == 0 else "Explain this implementation.\n") + filler*1200
    ids = call("/tokenize", {"content": content, "add_special": True})["tokens"][:n]
    assert len(ids) == n
    started = time.perf_counter()
    r = call("/completion", {"prompt": ids, "n_predict": n_predict, "id_slot": slot,
                             "seed": 12345, "temperature": 0, "ignore_eos": True,
                             "cache_prompt": False, "return_tokens": True, "stream": False})
    assert r["timings"]["prompt_n"] == n and not r["truncated"], r
    assert r["timings"]["predicted_n"] == n_predict and len(r["tokens"]) == n_predict, r
    print(json.dumps({"case": case, "slot": r["id_slot"], "elapsed_s": time.perf_counter()-started,
                      "timings": r["timings"], "step": r.get("f5_per_token"),
                      "tokens": r["tokens"]}), flush=True)

if os.getenv("PAIR") == "1":
    depth = int(os.getenv("PAIR_TOKENS", "4096"))
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        for tag in ("cold", "warm"):
            a = executor.submit(measure, tag, depth, 128, 0)
            b = executor.submit(measure, tag, depth, 128, 1)
            a.result()
            b.result()
else:
    for tag in ("cold", "warm"):
        for n in map(int, os.environ["PROMPTS"].split(",")):
            measure(tag, n, 16 if n > 64 else 128)
            if n > 64:
                measure(tag + "-decode", n, 128)
PY

# Cada execução escreve seus próprios logs e termina só o servidor de teste que iniciou.
run() (
  set -e
  set +m
  name=$1
  shift
  launch=()
  if [[ ${PROFILE:-0} == 1 ]]; then
    launch=(nsys profile --trace=cuda,osrt,nvtx --force-overwrite=true --output "$OUT/$name")
    export GGML_RKVA_TRACE_FILE="$OUT/$name.trace"
  fi
  check_port() {
    python3 -c 'import socket,os; s=socket.socket(); s.bind(("127.0.0.1", int(os.environ["PORT"])))'
  }
  check_port
  CUDA_VISIBLE_DEVICES=0 LLAMA_SERVER_F5_PER_TOKEN_TIMING=1 \
    setsid "${launch[@]}" "$BIN/llama-server" "${base[@]}" -fa on -ctk q8_0 -ctv q8_0 \
      --load-mode mmap --lazy-mode on --no-warmup -cram 0 -np 1 \
      --host 127.0.0.1 --port "$PORT" "$@" >"$OUT/$name.server.log" 2>&1 &
  server=$!
  nvidia-smi -i 0 --query-gpu=timestamp,memory.used --format=csv,noheader,nounits -lms 250 \
    >"$OUT/$name.vram.csv" &
  telemetry=$!
  cleanup() {
    kill -TERM "$telemetry" 2>/dev/null || true
    wait "$telemetry" 2>/dev/null || true
    kill -INT -- "-$server" 2>/dev/null || true
    wait "$server" || true
  }
  trap cleanup EXIT
  for ((i=0; i<900; ++i)); do
    if curl -fsS "http://127.0.0.1:$PORT/health" >"$OUT/$name.health.json" 2>/dev/null; then
      break
    fi
    kill -0 "$server"
    sleep 1
  done
  curl -fsS "http://127.0.0.1:$PORT/health" >"$OUT/$name.health.json"
  python3 "$OUT/load.py" >"$OUT/$name.requests.jsonl"
)
```

### Mapas e ring, em ambas as GPUs

```bash
unset GGML_CUDA_DISABLE_GRAPHS GGML_CUDA_NO_PINNED GGML_RKVA_TRACE_FILE LLAMA_MOE_CACHE_PREFILL \
  LLAMA_MOE_MAP_DEVICE LLAMA_MOE_PINNED_RING LLAMA_MOE_UPLOAD_OVERLAP LLAMA_QSA_INDEXER_BOUNDS \
  LLAMA_GDN_CONV_L2 LLAMA_QSA_FA_TRACE LLAMA_MTP_HIDDEN_TRACE LLAMA_CUDA_VERIFY_TRACE \
  LLAMA_MTP_RESERVE_WEIGHTS LLAMA_MTP_KV_STORE_ONLY LLAMA_PLE_PREFETCH_TRACE
for rep in 1 2 3; do
  for arm in 0 1; do
    LLAMA_MOE_CACHE_PREFILL=1 LLAMA_MOE_MAP_DEVICE=$arm \
      run "map-${rep}-${arm}" --moe-cache-mib "$full"
    LLAMA_MOE_UPLOAD_OVERLAP=1 LLAMA_MOE_PINNED_RING=$arm \
      run "ring-${rep}-${arm}" --moe-cache-mib 1024
  done
done
for arm in 0 1; do
  PROFILE=1 LLAMA_MOE_CACHE_PREFILL=1 LLAMA_MOE_MAP_DEVICE=$arm \
    run "map-trace-$arm" --moe-cache-mib "$full"
  PROFILE=1 LLAMA_MOE_UPLOAD_OVERLAP=1 LLAMA_MOE_PINNED_RING=$arm \
    run "ring-trace-$arm" --moe-cache-mib 1024
  PROFILE=1 GGML_CUDA_DISABLE_GRAPHS=1 LLAMA_MOE_CACHE_PREFILL=1 LLAMA_MOE_MAP_DEVICE=$arm \
    run "map-direct-$arm" --moe-cache-mib "$full"
  PROFILE=1 GGML_CUDA_DISABLE_GRAPHS=1 LLAMA_MOE_UPLOAD_OVERLAP=1 LLAMA_MOE_PINNED_RING=$arm \
    run "ring-direct-$arm" --moe-cache-mib 1024
done
GGML_CUDA_NO_PINNED=1 LLAMA_MOE_UPLOAD_OVERLAP=1 LLAMA_MOE_PINNED_RING=1 \
  run ring-pageable-fallback --moe-cache-mib 1024
PROFILE=1 LLAMA_MOE_CACHE_PREFILL=1 LLAMA_MOE_MAP_DEVICE=1 \
  run map-partial-control --moe-cache-mib 1024
for ub in 32 33; do
  for arm in 0 1; do
    LLAMA_MOE_CACHE_PREFILL=1 LLAMA_MOE_MAP_DEVICE=$arm \
      run "map-ub${ub}-$arm" --moe-cache-mib "$full" -ub "$ub"
  done
done

for arm in 0 1; do
  LLAMA_TEST_MOE_FULL=1 LLAMA_MOE_CACHE_PREFILL=1 LLAMA_MOE_MAP_DEVICE=$arm \
    LLAMA_MOE_UPLOAD_OVERLAP=1 LLAMA_MOE_PINNED_RING=$arm \
    "$BIN/test-llama-archs" -a '^deepseek2$' -b CUDA0 -s 12345 -v 4 \
    >"$OUT/moe-fixture-$arm.log" 2>&1
done
```

- Mapa. Exigir logs de 100% de residência, publicação única e mesma saída greedy. Confirmar 0 bytes D2H dos IDs, 0 sync atribuível ao callback e 0 H2D de mapa/pesos no quente desses grupos. A rota interna de MMID pode sincronizar em geometrias/tipos sem kernel device-side. Separar esse caso da sincronização do callback. A mediana de pp28k aquecido deve superar a dispersão entre pares, com TTFT frio e VRAM do mapa registrados. Grupos parciais continuam contabilizando hits/misses SLRU. Uma execução cache parcial com `LLAMA_MOE_MAP_DEVICE=1` deve produzir as mesmas saídas e sem publicação para esses grupos. Para a fronteira de admissão, acrescentar `-ub 32` e `-ub 33` ao comando de mapa com os dois budgets.
- Ring. Exigir mesmo volume H2D de pesos, NMSE da fixture ≤1e-4 e mesmos tokens. Confirmar wrap na estatística `chunks > 2`, fallback `active=0` e no máximo 8 MiB novos pinned por backend ativo. Compare duração de API `cudaMemcpyAsync`, `cudaEventSynchronize`, gap de submissão no stream de compute e H2D descoberto. Aceitar se o tempo descoberto cair ou a mediana melhorar ≥3% além da variabilidade, com os demais casos ≥97% do controle.
- Os pesos da fixture sintética são F32. Confirme a rota quantizada MMQ e a paridade greedy com os GGUFs reais, além do resultado de NMSE da fixture.

### QSA/GDN, FA e replay

```bash
"$BIN/test-strata-opt" CUDA0 | tee "$OUT/strata-fixture.log"
"$BIN/test-backend-ops" test -b CUDA0 -j 1 --seed 12345 -o FLASH_ATTN_EXT \
  -p 'hsk=256.*kv=(4096|16384|32768),nb=(1|2|3|4|5|13|512),.*type_K=q8_0,type_V=q8_0.*n_kv_max=(0|2048)' \
  | tee "$OUT/flash-fixture.log"
"$BIN/test-backend-ops" test -b CUDA0 -j 1 --seed 12345 -o GATED_DELTA_NET_CACHE_FUSION \
  | tee "$OUT/gdn-fixture.log"

# Controle A/B do diagnóstico de união na 4070 (também pode rodar na A100).
for arm in 0 1; do
  LLAMA_QSA_FA_TRACE=$arm GGML_RKVA_TRACE_FILE="$OUT/fa-fixture-$arm.trace" \
    nsys profile --trace=cuda,osrt -o "$OUT/fa-fixture-$arm" \
    "$BIN/test-backend-ops" test -b CUDA0 -j 1 --seed 12345 -o FLASH_ATTN_EXT \
    -p 'hsk=256.*kv=(4096|16384|32768),nb=(1|2|3|4|5|13|512),.*type_K=q8_0,type_V=q8_0.*n_kv_max=(0|2048)' \
    >"$OUT/fa-fixture-$arm.log" 2>&1
done

# A100 Flash-Next. Cada A/B muda só um mecanismo.
for rep in 1 2 3; do
  for arm in 0 1; do
    LLAMA_QSA_INDEXER_BOUNDS=$arm run "bounds-${rep}-${arm}" --moe-cache-mib "$full"
    LLAMA_GDN_CONV_L2=$arm run "gdn-${rep}-${arm}" --moe-cache-mib "$full"
    PAIR=1 LLAMA_QSA_INDEXER_BOUNDS=$arm run "bounds-dual-${rep}-${arm}" -np 2 -kvu --moe-cache-mib "$full"
  done
done
for arm in 0 1; do
  PROFILE=1 LLAMA_QSA_FA_TRACE=1 LLAMA_QSA_INDEXER_BOUNDS=$arm \
    run "bounds-trace-$arm" --moe-cache-mib "$full"
  PROFILE=1 PAIR=1 LLAMA_QSA_FA_TRACE=1 LLAMA_QSA_INDEXER_BOUNDS=$arm \
    run "bounds-dual-trace-$arm" -np 2 -kvu --moe-cache-mib "$full"
  PROFILE=1 LLAMA_GDN_CONV_L2=$arm run "gdn-trace-$arm" --moe-cache-mib "$full"
  PROFILE=1 LLAMA_QSA_FA_TRACE=$arm run "fa-trace-$arm" --moe-cache-mib "$full"
done

replay() {
  local label=$1
  shift
  local rep mode
  for rep in 1 2 3 trace; do
    for mode in on off; do
      if [[ $mode == off ]]; then export GGML_CUDA_DISABLE_GRAPHS=1
      else unset GGML_CUDA_DISABLE_GRAPHS
      fi
      local profile=0
      if [[ $rep == trace ]]; then profile=1; fi
      PROFILE=$profile LLAMA_CUDA_VERIFY_TRACE=1 LLAMA_MTP_HIDDEN_TRACE=1 \
        LLAMA_MTP_RESERVE_WEIGHTS=1 LLAMA_MTP_KV_STORE_ONLY=1 \
        run "replay-$label-$rep-$mode" "$@"
    done
  done
  unset GGML_CUDA_DISABLE_GRAPHS
}
# 4070, base sem draft e ngram para provocar verificação de 2/3 linhas.
replay plain --moe-cache-mib "$full"
for width in 2 3; do
  replay "ngram${width}" --moe-cache-mib "$full" --spec-type ngram-simple \
    --spec-ngram-simple-size-n 2 --spec-ngram-simple-size-m "$((width-1))" --spec-ngram-simple-min-hits 1
done
# A100, repetir o controle plain acima e acrescentar o sidecar.
replay mtp --moe-cache-mib -1 --vram-reserve-mib 4096 \
  -md "$SIDECAR" --spec-type draft-mtp --spec-draft-n-max 2 -ngld 99 -devd CUDA0


for f in "$OUT"/*.nsys-rep; do
  nsys stats --report cuda_gpu_mem_size_sum,cuda_gpu_trace,cuda_api_sum,cuda_api_trace "$f" \
    >"${f%.nsys-rep}.stats.txt"
done
```

- **Bounds/GDN.** Exigir mesmos IDs e scores válidos no indexer e `-inf` em todos os inválidos da fixture. Comparar tempo de preparação e kernels, redução de H2D do input de pools de `2 × n_pool × nq` para `8 × nq` por cópia. Para GDN, comparar preparação e V/Q/K com tolerância absoluta 1e-6 na fixture, NMSE do estado/logits ≤1e-6 no replay da fixture e tokens greedy nos GGUFs reais. Exigir pp28k maior que a dispersão e pico de VRAM ≤ controle.
- **IDs/FA.** Este A/B mede apenas o diagnóstico sobre a rota de máscara. Exigir zero `QSA union overflow` em geometria válida, PASS da matriz Q8 na tolerância do harness, e confirmar uso da união 1/8 e dense fallback com GQA ≤4. Para um futuro caminho por IDs, comparar conjuntos por query, NMSE ≤1e-6, menos bytes e tempo seleção→FA, overflow deliberado com fallback denso, cauda incompleta e duas sequências.
- **Replay.** Calcular por largura p50/p95 das latências `token_ms`/`iteration_ms`, junto de número de execuções diretas/capturas/replays e razões. OLMoE sem draft exercita largura 1. Ngram na 4070 e MTP na A100 só qualificam 2/3 se o trace realmente observar essas larguras. Se uma classe faltar, ampliar prompts e alternar n-max 1/2 em ensaios pareados. A futura correção precisa reduzir p95 ou a proporção de execução direta acima da dispersão, com IDs greedy iguais. Com MTP, usar o sidecar de 1.907.151.936 bytes e pico ≤39.005 MiB (40,9 GB decimais).
- **Registro comum.** Nas sessões Nsight, atribuir bytes e duração de H2D, D2H e D2D pelo stream e fase (publicação a frio, prefill e decode). Associar as esperas `cudaStreamSynchronize`/eventos ao callback ou split pelo tempo e thread. Conferir TTFT, picos em MiB, medianas pp/decode e tokens efetivos nos JSONL. Usar somente as rodadas sem profiler para throughput, com pp e decode fora do caso-alvo ≥97% do controle, nenhum OOM e sem drift entre GPUs. O trace de host contém tempos inclusivos. Não somar intervalos que se sobrepõem.

## Código e testes CPU

- `src/llama-moe-cache.cpp` e `src/llama-graph.cpp` publicam e ligam os bancos e mapas. `ggml/src/ggml-cuda/common.cuh`, `ggml-cuda.cu` e `moe-upload-ring.h` controlam staging e eventos.
- `src/models/qwen4exp.cpp`, `src/llama-memory-hybrid-idx.cpp`, a API GGML e as implementações CPU/CUDA mantêm os limites do indexer e o epílogo GDN. A leitura do trecho final do conv CUDA usa apenas colunas válidas, inclusive na cauda do caminho de referência.
- `ggml/src/ggml-cuda/fattn.cu`, `ggml-cuda.cu`, `src/llama-context.cpp` e `src/llama-graph.cpp` ligam os traces. As fixtures em `tests/test-strata-opt.cpp`, `tests/test-strata-trace.py` e `tests/test-llama-archs.cpp` cobrem as ablações e o rollback. A fixture de experts agora encaminha os overrides de buffer para que os casos host/cache realmente usem pesos no host.

```bash
cmake -S . -B build-cpu -DCMAKE_BUILD_TYPE=RelWithDebInfo -DGGML_CUDA=OFF \
  -DGGML_VULKAN=OFF -DGGML_HIP=OFF -DGGML_SYCL=OFF -DGGML_OPENCL=OFF -DGGML_METAL=OFF \
  -DLLAMA_BUILD_TESTS=ON
cmake --build build-cpu -j2 --target test-strata-opt test-llama-archs test-cuda-graph-policy \
  test-arg-parser test-kv-mixed-mtp-params test-kv-mixed-mtp-budget test-kv-tail-input test-nextn-flags \
  test-recurrent-zero test-speculative-rejection-sampling test-save-load-state test-backend-ops
ctest --test-dir build-cpu --output-on-failure -j1 -R '^(test-(arg-parser|kv-tail-input|nextn-flags|recurrent-zero|speculative-rejection-sampling|kv-mixed-mtp-(params|budget)|mtp-(ubatch-sync|request-reset|kvarn-routing|qwen4exp|qwen4exp-store-only|sidecar-weight-reserve|carry-state|context-cache-transfer|target-only-bootstrap|bootstrap-native-rollback)|context-lifecycle-(target|prepare-failure|yarn-zero-orig|mtp)|qsa-exact-tail|kvarn-state-restore|qwen4exp-qsa-kvarn-static|upstream-merge-keepers-static|kvarn-mtp-sharing-static|std-kv-tail-static|strata-(opt|trace)|cuda-graph-policy(-source-static)?))$'
env LLAMA_MOE_CACHE_PREFILL=1 LLAMA_MTP_RESERVE_WEIGHTS=1 LLAMA_MTP_KV_STORE_ONLY=1 \
  LLAMA_MOE_UPLOAD_OVERLAP=1 LLAMA_MOE_MAP_DEVICE=1 LLAMA_MOE_PINNED_RING=1 \
  LLAMA_QSA_INDEXER_BOUNDS=1 LLAMA_GDN_CONV_L2=1 \
  ctest --test-dir build-cpu --output-on-failure -j1 \
  -R '^test-(mtp-(qwen4exp-store-only|kvarn-routing|sidecar-weight-reserve)|kvarn-state-restore|qsa-exact-tail)$'
build-cpu/bin/test-backend-ops test -b CPU -j 1 --seed 12345 -o FLASH_ATTN_EXT \
  -p 'hsk=256.*kv=4096,nb=1,.*type_K=q8_0,type_V=q8_0.*n_kv_max=(0|2048)'
build-cpu/bin/test-backend-ops test -b CPU -j 1 --seed 12345 -o GATED_DELTA_NET_CACHE_FUSION
```

| Executado | Resultado |
|---|---|
| CTest focado original | 29/29 em 8,46 s |
| CTest final com as quatro entradas extras | 33/33 em 7,25 s |
| CTest com as oito flags de execução em 1 | 5/5 em 4,60 s |
| FA Q8_0 CPU, KV de 4096 | 3/3 |
| `GATED_DELTA_NET_CACHE_FUSION` CPU | 14/14 |
| `test-strata-opt` CPU | Scores válidos e IDs iguais em 4/4 casos, com duas sequências, contextos equivalentes a 4.096/28.672 e 1/9 queries. No caso de 28.672 com 9 queries, 258.048 → 144 bytes de entrada lógica. GDN em 1/32/33/65 tokens com duas sequências teve delta máximo Q/K de 4,47×10⁻⁸ e V igual nesses casos. O teste do ring percorreu slots 0→1→0 e cauda de 17 bytes. |
| `test-strata-trace` CPU | Marcadores 1/2/3 associados aos graph runs, bytes NextN e do input de estado, marcador de espera, input compacto confirmado e decisão de PLE sem mmap. A fixture com duas sequências passou o replay após alterar e restaurar o prefixo, com NMSE 0. O teste exact-tail teve erro máximo 0. |
| Gate do mapa | Asserções em compilação cobrem flag desligada, backend fora de CUDA, grupo parcial, capacidade integral e falta de espaço para o mapa. |

A compilação CUDA, DMA real, binding/replay na GPU, ganhos, pico de VRAM e as corridas propostas na 4070/A100 ficam pendentes da qualificação de hardware. O único build desta entrega foi CPU-only. Nenhum serviço foi reiniciado.
