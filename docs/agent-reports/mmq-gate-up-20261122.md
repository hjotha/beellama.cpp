# Reutilização de roteamento e Q8_1 entre gate e up no CUDA MMQ

Base `1fb324283`. Implementação `f0e2648a8` em `feat/mmq-gate-up`.

## Mudança

O MMQ batelado calcula a permutação e os limites por expert em `mmid.cu`. Depois quantiza a ativação em Q8_1 e espalha as linhas de cada token na ordem compactada. O atalho de broadcast já evitava quantizar um token por expert em cada projeção. Gate e up ainda repetiam a preparação inteira. A fusão gate/up/GLU existente tem outra lista de tipos e o dispatch dessa variante está desabilitado nesta base. [MMQ MoE](../../ggml/src/ggml-cuda/mmq.cu) · [Agrupamento](../../ggml/src/ggml-cuda/mmid.cu) · [Fusão](../../ggml/src/ggml-cuda/mmq.cuh)

`LLAMA_MMQ_GATE_UP_SHARED_Q8=1` permite que dois `MUL_MAT_ID` adjacentes compartilhem a permutação, os limites e os dados de ativação Q8_1 quando leem os mesmos objetos de ativação e IDs. O teste de elegibilidade exige pesos do mesmo tipo, Q2_0 de **pesos** ou IQ3_S, layouts e parâmetros idênticos e ativação F32 broadcast. A seleção também exige o caminho MMQ NVIDIA acima do limite MMVQ, dentro de um único split e sem região CUDA concorrente ativa ou bifurcação entre os dois nós. A flag é lida uma vez por processo. [Seleção e dispatch](../../ggml/src/ggml-cuda/ggml-cuda.cu)

Cada projeção usa o mesmo kernel, geometria de tiles e fixup que usaria separadamente. O scratch e o padding de leitura seguem o dimensionamento já usado pelo primeiro lançamento, válido para os dois layouts iguais. O segundo lançamento troca peso e destino e repete a limpeza de padding do próprio peso. Os dois consumidores entram no mesmo stream antes da devolução do scratch ao pool. CUDA graphs capturam a preparação antes dos dois consumidores e a reexecutam no replay. O código enfileira uma preparação e duas projeções, ante duas preparações e duas projeções. Um par de pesos staged em splits separados, gate/up fundido no GGUF, tipos mistos, demais formatos e decode MMVQ seguem o dispatch individual. [Execução e scratch](../../ggml/src/ggml-cuda/mmq.cu)

## Validação no gokaya

Fixture com pesos quantizados independentes para gate e up, 16 experts, top-4, duas sequências sintéticas com experts quentes diferentes e seis execuções com IDs e ativações novos. Para cada projeção, o resultado do par é comparado a um grafo de referência de um único `MUL_MAT_ID` no mesmo backend, com os mesmos endereços de pesos/entradas e geometria. Um traço `cuda_mmq_gate_up_shared_q8` distingue uma execução que realmente entra no caminho compartilhado. [Fixture](../../tests/test-mmq-gate-up.cpp)

Sanidade executada em CPU:

```sh
cmake -S . -B build-host -DGGML_CUDA=OFF -DGGML_VULKAN=OFF \
  -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF \
  -DLLAMA_BUILD_SERVER=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-host --target test-mmq-gate-up -j2
./build-host/bin/test-mmq-gate-up CPU
```

```text
CPU q2_0 tokens=1024 seq=2: max NMSE 0, bit-exact yes
CPU q2_0 tokens=2048 seq=2: max NMSE 0, bit-exact yes
CPU iq3_s tokens=1024 seq=2: max NMSE 0, bit-exact yes
CPU iq3_s tokens=2048 seq=2: max NMSE 0, bit-exact yes
```

A validação CUDA usa o chroot Noble no gokaya. O build aguarda `grep -c 'build_rc' /home/hjotha/chroots/noble/tmp/build-iq3.log` a cada 300 s. Configuração preparada com CUDA 13.4.92, SM 89, gráficos CUDA ligados, FA/KVarN/EXL3 desabilitados para isolar o teste MMQ, build próprio `build-cuda-sm89` e `-j3`. Veja `/tmp/mmq-gate-up-cuda-build.log` no host para o estado do build.

```sh
# Dentro do chroot Noble, no worktree feat/mmq-gate-up
cmake -S . -B build-cuda-sm89 -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=89-real \
  -DCMAKE_CUDA_FLAGS=-I/opt/cuda/targets/x86_64-linux/include/cccl \
  -DGGML_CUDA=ON -DGGML_CUDA_GRAPHS=ON -DGGML_CUDA_FA=OFF \
  -DGGML_CUDA_FA_QUANTS=f16-f16 -DGGML_CUDA_KVARN=OFF -DGGML_EXL3=OFF \
  -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF \
  -DLLAMA_BUILD_SERVER=OFF -DLLAMA_CURL=OFF
cmake --build build-cuda-sm89 --target test-mmq-gate-up test-backend-ops -j3
```

Validação prevista com a RTX 4070 livre. Cada execução CUDA deve retornar 0. A fixture exige NMSE ≤ 1e-6 e o traço da flag 1 precisa confirmar o dispatch compartilhado.

```sh
# No gokaya, com llama-server-root.service parado durante a janela de teste
export LD_LIBRARY_PATH="$PWD/build-cuda-sm89/bin:/opt/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
for flag in 0 1; do
  for op in MUL_MAT_ID MUL_MAT; do
    LLAMA_MMQ_GATE_UP_SHARED_Q8=$flag ./build-cuda-sm89/bin/test-backend-ops -b CUDA0 -o "$op"
  done
  env -u GGML_CUDA_DISABLE_GRAPHS LLAMA_MMQ_GATE_UP_SHARED_Q8=$flag \
    GGML_RKVA_TRACE_FILE="/tmp/mmq-gate-up-graph-$flag" ./build-cuda-sm89/bin/test-mmq-gate-up CUDA0
  GGML_CUDA_DISABLE_GRAPHS=1 LLAMA_MMQ_GATE_UP_SHARED_Q8=$flag \
    GGML_RKVA_TRACE_FILE="/tmp/mmq-gate-up-direct-$flag" ./build-cuda-sm89/bin/test-mmq-gate-up CUDA0
done
rg '"phase":"cuda_mmq_gate_up_shared_q8"' /tmp/mmq-gate-up-{graph,direct}-1.*.jsonl
# Após religar o serviço
curl -fsS http://127.0.0.1:8090/health
```

## A/B de prefill na A100

Use o mesmo binário SM 80 nos dois braços e informe a revisão, o GGUF e as configurações a partir da recipe medida. O exemplo fixa b2048, Q8_0/Q8_0, contexto 32768, um slot e greedy, com a matriz cache 2048/6144 e ubatch 1024/2048 MiB/tokens, respectivamente. Execute separadamente com o GGUF Q2_0 e com IQ3_S. Use o mesmo prompt de referência em ambos e os mesmos envs de cache da recipe. O JSON guarda `timings.prompt_n`, `timings.prompt_ms` e `timings.prompt_per_second`. O loop alterna as duas ordens para reduzir viés entre processos.

```bash
set -euo pipefail
: "${BIN:?caminho do llama-server SM80 deste commit}"
: "${MODEL:?caminho do primeiro shard do GGUF}"
: "${PROMPT_FILE:?arquivo com o prompt fixo}"
out="mmq-gate-up-ab/$(basename "$MODEL" .gguf)"
mkdir -p "$out"
realpath "$MODEL" > "$out/model.txt"
sha256sum "$PROMPT_FILE" | tee "$out/prompt.sha256"
"$BIN" --version > "$out/version.txt" 2>&1
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$out/gpu.csv"
jq -n --rawfile prompt "$PROMPT_FILE" \
  '{prompt:$prompt,n_predict:1,temperature:0,seed:42,cache_prompt:false}' > "$out/request.json"
pid=
trap 'if [[ -n $pid ]]; then kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true; fi' EXIT
for cache in 2048 6144; do
  for ub in 1024 2048; do
    for round in 0 1 2 3; do
      for flag in "$((round % 2))" "$((1 - round % 2))"; do
        name="cache${cache}-ub${ub}-round${round}-flag${flag}"
        env -u GGML_RKVA_TRACE_FILE LLAMA_MMQ_GATE_UP_SHARED_Q8=$flag "$BIN" \
          -m "$MODEL" -ngl 999 -ncmoe 2 --moe-cache-mib "$cache" \
          -c 32768 -b 2048 -ub "$ub" -ctk q8_0 -ctv q8_0 -fa on \
          --parallel 1 --host 127.0.0.1 --port 18090 > "$out/$name.log" 2>&1 &
        pid=$!
        ready=0
        for check in {1..300}; do
          kill -0 "$pid"
          if curl -fsS http://127.0.0.1:18090/health 2>/dev/null | jq -e '.status == "ok"' >/dev/null; then
            ready=1
            break
          fi
          sleep 2
        done
        [[ $ready == 1 ]]
        for request in 1 2; do
          curl -fsS -H 'Content-Type: application/json' \
            --data-binary "@$out/request.json" \
            http://127.0.0.1:18090/completion > "$out/$name-request${request}.json"
        done
        kill "$pid"
        wait "$pid" || true
        pid=
      done
    done
  done
done
```

Para DFlash2, acrescente os mesmos `--spec-type draft-dflash --spec-draft-model "$DRAFTER" --spec-draft-n-max "$DRAFT_MAX"` aos dois braços e use a recipe de verificação já medida. Em uma execução separada de throughput, exporte `GGML_RKVA_TRACE_FILE` e procure a fase `cuda_mmq_gate_up_shared_q8` no braço on para identificar os splits que entram no atalho. O comando de throughput remove o trace deliberadamente.

## Cobertura pendente

A sanidade CPU comprova o fixture e o comparador. Ficam pendentes build CUDA, `test-backend-ops` 2/2 nos dois filtros, NMSE no caminho compartilhado CUDA, replay com a RTX 4070 e o ciclo de parada/retorno do serviço. A janela de validação depende do encerramento do build IQ3. Nesta sessão `sudo -n` exige senha e o PolicyKit pede autenticação para gerenciar a unidade. O health que foi observado antes da parada retornou `{"status":"ok"}`.

A fixture cobre duas sequências sintéticas na mesma geometria batelada; ela não abre uma sessão de servidor com dois slots. Também ficam fora da cobertura atual os GGUFs da recipe, os splits com `-ncmoe 2` em modelo real, DFlash2, execução A100 e performance na RTX 4070. Nenhum número de ganho foi medido.
