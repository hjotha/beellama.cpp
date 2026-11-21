# Paridade Strata no Flash-Next

## Alterações

- **`LLAMA_MOE_CACHE_PREFILL=1`** admite ubatches acima de 32 quando o banco do grupo comporta todas as combinações camada × expert. Grupos parciais usam o bypass, que pode copiar hits do banco. O gate de lotes pequenos e a política SLRU foram preservados. No auto-size de um dispositivo, o orçamento pode cobrir o slot extra de padding exigido por MMQ. O alocador limita cada grupo ao número de experts. Confirme a capacidade nas linhas de slots por grupo e a ativação em `large batch of ... uses the MoE cache`. A alocação do grafo pode mudar com a flag.
- **`LLAMA_MTP_RESERVE_WEIGHTS=1`** desconta do orçamento automático do alvo, em um dispositivo, o tamanho do sidecar local e de seus shards GGUF convencionais. O desconto satura em zero. Para caminhos ausentes ou ilegíveis, o log identifica o arquivo e a estimativa fica em zero. O tamanho em disco é uma aproximação para pesos. A margem existente continua necessária para alocações do draft, e o estimador de KV baseado nas camadas do alvo pode dar zero em um alvo trunk-only. O caminho fixo e perfis sem draft-MTP ficam fora do desconto. Corrigida a propagação de `moe_cache_auto` para o draft, onde o código já desabilitava o cache fixo.
- **`LLAMA_MTP_KV_STORE_ONLY=1`** elimina projeção Q/gate, a leitura de atenção e sua projeção de saída em ubatches de catch-up do Qwen4Exp sem outputs. Mantém as escritas K/V, incluindo os pools QSA. KVarN, tail explícito, memória paginada, treinamento e ubatches com outputs seguem pelo caminho existente. No alvo, cada `h[p]` alimenta `KV[p+1]` do draft, inclusive o carry entre batches. O residual antes do mixer final já é o hidden consumido pelo draft. Usar só a última linha ou mascarar pelos logits faria o draft perder seu histórico. O corte implementado reduz o trabalho no draft e preserva o histórico completo.
- **`LLAMA_MOE_UPLOAD_OVERLAP=1`** envia fills do banco e uploads seletivos do bypass por um stream CUDA não bloqueante. Um evento ordena a reutilização do destino após a última operação do stream principal. Outro ordena o consumo dos novos pesos antes do split, fora de capture/replay. O slot map e as cópias D2D continuam no stream de compute, com possibilidade de sobreposição dentro da callback. O roteamento de experts ainda precisa terminar para os IDs serem conhecidos. Em cache frio ou parcial, meça a transferência residual. Em regime com hits, o caminho novo não cria nem usa stream para fills vazios. Buffer pageable pode bloquear a submissão ou limitar o ganho. O caminho usa os buffers existentes, sem arena de staging adicional.
- A regressão CPU de QSA com KVarN reproduziu um segfault em `llama_memory_hybrid_idx::init_batch`. O batch, remoção de sequência e estado passam pela memória de atenção base. A restauração preserva o encaminhamento dos slots ao indexer e seleciona o leitor de estado conforme o tipo de KV. O teste de roteamento exercita também remoção e restauração.

Os quatro envs exigem valor `1` e ficam desligados por padrão. Os arquivos afetados estão em `common/`, `include/llama.h`, `src/`, `ggml/src/ggml-cuda/` e `tests/`. A interface C ganhou um campo ao final de `llama_context_params`. Recompile junto o executável e as bibliotecas.

## Stream-to-VRAM, PLE e próximos passos

O sintoma informado foi download concluído, VRAM em 36.548 MiB e `/health` pendente com cache 1024 e contexto 4096. O worktree não contém o log dessa execução. A conclusão do tee de disco precede a drenagem de DMAs pendentes no loader. Depois o processo cria contexto, KV, bancos e compute. Assim o download completo é insuficiente para localizar a espera. Em streaming, `common/common.cpp` seleciona `LLAMA_LOAD_MODE_NONE`. O loader coloca `TENSOR_READ_LAZY` em RAM host gravável, inclusive as tabelas PLE do Qwen4Exp, porque os arquivos de staging podem ser incompletos ou transitórios. O teto de VRAM sozinho não captura a pressão de RAM desse caminho.

A recipe Flash-Next com offload em host deve usar os dois shards já completos em disco com `-m`, `--load-mode mmap --lazy-mode on` e sem streaming. `LLAMA_STREAM_VRAM` é a entrada da recipe, enquanto o binário aceita `--hf-stream-vram` ou `LLAMA_ARG_HF_STREAM_VRAM`. Para diagnosticar o streaming separadamente, use cache remoto frio em disco e o mesmo modelo servido por HTTP/HF. Compare `--hf-stream-vram --hf-stream-disk-mode sync` contra o baseline local. Guarde carimbos do progresso de download, das linhas de buffers de modelo/KV/cache/compute, de `sched_reserve` e do primeiro HTTP 200. Amostre também RSS, swap e eventos de OOM. O primeiro marcador ausente delimita a etapa a investigar. O loader streaming existente é compartilhado com outros modelos e o sintoma informado ainda não isola uma falha nesse código.

Para o smoke a 4k, use o URL direto do primeiro shard em `TARGET_URL` e confirme a presença dos dois arquivos completos no braço mmap.

```bash
small=(-ngl 99 --n-cpu-moe 2 --moe-cache-mib 1024 -c 4096 -b 2048 -ub 1024
       -fa on -ctk q8_0 -ctv q8_0 --lazy-mode on --no-warmup
       --host 127.0.0.1 --port 8090)
"$BIN/llama-server" -m "$TARGET" --load-mode mmap "${small[@]}"
"$BIN/llama-server" -mu "$TARGET_URL" --hf-stream-vram --hf-stream-disk-mode sync "${small[@]}"
# Em outro shell durante cada braço
curl --fail --show-error http://127.0.0.1:8090/health
```

PLE fica condicionado ao A/B abaixo, incluindo o custo de RAM. CPU no miss fica condicionado à taxa de misses e ao tempo H2D após o gate de residência. A proposta precisaria de resultados parciais CPU/GPU concorrentes e teste de redução numérica. Prefetch SLRU entre camadas depende de evidência de misses residuais e de uma previsão de roteamento que não expulse entradas usadas.

## A/B de hardware

Use build CUDA limpo da mesma revisão para os dois braços. Compile sm_89 para a 4070 ou sm_80 para A100 conforme o plano, com `-DGGML_CUDA_GRAPHS=ON`. Use variáveis `BIN` para o diretório dos binários, `TARGET` para o primeiro shard Flash-Next Q2_0 e `SIDECAR` para `MTP/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf` (SHA-256 `f521868a9e143718bef513772f6e04d9642551e362cf2439636d2abdbd149dfc`). Rode um processo de benchmark por vez, com os demais envs experimentais ausentes. Os comandos abaixo usam Bash.

```bash
base=(-m "$TARGET" --alias q -ngl 99 --n-cpu-moe 2 -c 32768 -b 2048 -ub 1024
      -fa on -ctk q8_0 -ctv q8_0 --load-mode mmap --jinja --no-warmup
      --host 127.0.0.1 --port 8090)
draft=(-md "$SIDECAR" --spec-type draft-mtp --spec-draft-n-max 2)
run() { "$BIN/llama-server" "${base[@]}" "$@"; }

# A100, MoE residente no prefill
LLAMA_MOE_CACHE_PREFILL=0 run --lazy-mode on --moe-cache-mib 6144
LLAMA_MOE_CACHE_PREFILL=1 run --lazy-mode on --moe-cache-mib 6144

# A100, lado a lado com a mesma reserva base e o mesmo draft
LLAMA_MTP_RESERVE_WEIGHTS=0 run --lazy-mode on --moe-cache-mib -1 --vram-reserve-mib 4096 "${draft[@]}"
LLAMA_MTP_RESERVE_WEIGHTS=1 run --lazy-mode on --moe-cache-mib -1 --vram-reserve-mib 4096 "${draft[@]}"

# A100, catch-up com reserva já ativada (repita sem GGML_MTP_PROF para medir throughput)
GGML_MTP_PROF=1 LLAMA_MTP_RESERVE_WEIGHTS=1 LLAMA_MTP_KV_STORE_ONLY=0 run --lazy-mode on --moe-cache-mib -1 --vram-reserve-mib 4096 "${draft[@]}"
GGML_MTP_PROF=1 LLAMA_MTP_RESERVE_WEIGHTS=1 LLAMA_MTP_KV_STORE_ONLY=1 run --lazy-mode on --moe-cache-mib -1 --vram-reserve-mib 4096 "${draft[@]}"

# A100, PLE lazy A/B, somente em host com RAM suficiente para o braço off
run --lazy-mode on  --moe-cache-mib 6144
run --lazy-mode off --moe-cache-mib 6144

# A100, overlap com misses residuais. Repita com 6144 e LLAMA_MOE_CACHE_PREFILL=1
# como controle de cache residente. Rode ambos os braços novamente com graphs desligados.
LLAMA_MOE_UPLOAD_OVERLAP=0 run --lazy-mode on --moe-cache-mib 1024
LLAMA_MOE_UPLOAD_OVERLAP=1 run --lazy-mode on --moe-cache-mib 1024
```

Para perfil de cópia, prefixe o lançamento dentro de `run` com `nsys profile --trace=cuda,osrt -o "$TRACE"` e exporte a sessão com `nsys stats --report cuda_gpu_mem_size_sum,cuda_gpu_trace "$TRACE.nsys-rep"`. Para desligar graphs use `GGML_CUDA_DISABLE_GRAPHS=1`. Para ligar, remova essa variável. Capture ao menos três repetições alternadas no mesmo commit com o profiler desligado. Guarde logs, H2D em bytes por fase e por stream, tempo esperando o evento, taxa de hits/misses, t/s por fase e pico de VRAM via `nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 250`. Use os prompts LP30 e LP30+dec do relatório de prefill, além dos prompts de 4k e decode curto, com `cache_prompt=false`, temperatura zero, seed fixa e thinking desligado. Registre os tokens reais. Desligue o processo de benchmark normalmente após o último request para obter as estatísticas de cache. [Harness de referência](parity-moe-prefill.md)

- **MoE prefill.** Confirme 100% de slots, uma carga fria de até um working set e poucos bytes residuais em requests repetidos. O alvo do plano continua +25% no prefill 28k. Use +10% como limiar para o experimento e exija decode 28k ≥97% do baseline pareado. Repita com grupo parcial e ubatches 32/33 para checar bypass e ausência de abort. Execute os testes de janela e NMSE descritos abaixo.
- **MTP.** Verifique `reserving 1818.80 MiB of draft-MTP weights` para 1.907.151.936 bytes, o orçamento efetivo, `/health` e primeiro request de 28k. Repita com contextos liberados entre braços. Para catch-up, compare `MTPPHASE phase=sync` e prefill total com 4096/28k tokens, tokens de saída greedy, taxa de aceitação dentro de 1 ponto percentual e decode ≥97% do A/B pareado. Não use a variante de sidecar com ratio 4. Aplique o teto solicitado de 40,9 GB de pico em todos os braços. Registre o valor bruto em MiB e a capacidade reportada pelo dispositivo. 40,9 GB decimais correspondem a ~39.005 MiB, abaixo dos 39.052 MiB registrados na execução anterior.
- **PLE.** Registre RSS/major faults e tempo até `/health`, inclusive restart com disco quente (~13 s na referência), além de prefill/decode. Se o braço off exceder a RAM disponível, encerre o ensaio e registre o limite. Uma fase de prefetch só se justifica com custo de fault comprovado.
- **Overlap.** Em casos com misses veja H2D no stream dedicado, fila correta entre eventos e replay sem erro. Compare NMSE ≤1e-4 da fixture CPU/CUDA, output greedy de requests repetidos e decode ≥97% do baseline. Um ganho exige sobreposição visível ou ≥3% nas medianas com misses. Com hits, espere poucos H2D e nenhuma regressão consistente.
- **RTX 4070.** Faça o mesmo A/B com OLMoE-1B-7B Q2_K local (`-cmoe -ngl 99 -c 4096 -b 512 -ub 128 -fa on -ctk q8_0 -ctv q8_0 --load-mode mmap --lazy-mode on --no-warmup`). Use cache 3072 para forçar capacidade total se os logs confirmarem espaço e 1024 para exercitar misses. Teste graphs on/off em ambos. Para conferir NMSE da rota de experts em host/cache, rode `LLAMA_MOE_UPLOAD_OVERLAP=1 "$BIN/test-llama-archs" -a '^deepseek2$' -b CUDA0 -s 12345` e repita com a flag desligada. Para repetir os fixtures do catch-up e QSA/KVarN no build CUDA, rode `ctest --test-dir "$CUDA_BUILD" --output-on-failure -R 'test-mtp-qwen4exp|test-mtp-kvarn-routing|test-qsa-exact-tail'`. O fixture de MTP não mede a reserva real do sidecar.

## Validação CPU

Build `RelWithDebInfo` com `GGML_CUDA=OFF` e demais backends acelerados desabilitados. Configurei `build-cpu` e compilei estes alvos com `-j4`.

```bash
cmake -S . -B build-cpu -DGGML_CUDA=OFF
cmake --build build-cpu -j4 --target test-llama-archs test-arg-parser   test-kv-mixed-mtp-params test-kv-mixed-mtp-budget test-kv-tail-input   test-nextn-flags test-recurrent-zero test-speculative-rejection-sampling   test-save-load-state test-backend-ops
ctest --test-dir build-cpu --output-on-failure -j1   -R '^(test-(arg-parser|kv-tail-input|nextn-flags|recurrent-zero|speculative-rejection-sampling|kv-mixed-mtp-(params|budget)|mtp-(ubatch-sync|request-reset|kvarn-routing|qwen4exp|qwen4exp-store-only|sidecar-weight-reserve|carry-state|context-cache-transfer|target-only-bootstrap|bootstrap-native-rollback)|context-lifecycle-(target|prepare-failure|yarn-zero-orig|mtp)|qsa-exact-tail|kvarn-state-restore|qwen4exp-qsa-kvarn-static|upstream-merge-keepers-static|kvarn-mtp-sharing-static|std-kv-tail-static))$'
```

- CTest focado retornou **29/29** em **6,21 s**. Inclui ciclo de vida com e sem MTP, warmup, carry, bootstrap/rollback, restauração KVarN, QSA exact-tail e parser. Corrigi o warmup que tentava preencher o tensor de posições descartado pelo grafo. Atualizei os testes antigos de tail, contrato NextN estático e a expectativa de host pinned para o backend presente.
- Rodada com os quatro envs em `1` retornou **5/5** em **4,55 s** para MTP Qwen4Exp store-only, roteamento KVarN, restauração KVarN, QSA exact-tail e reserva do sidecar. A fixture Q8_0 comparou logits e hidden com catch-up completo após rollback e replay. NMSE observado foi **0** nas seis combinações de ratios 0/4 e ubatch configurado 32/512/1024, em cada braço. O prompt sintético usa 40 tokens. O teste de reserva passou **42 checks** em cada braço e o de orçamento KV passou **57 checks**.
- `test-backend-ops test -b CPU -j 1 --seed 12345 -o FLASH_ATTN_EXT -p 'hsk=256.*kv=4096,nb=1,.*type_K=q8_0,type_V=q8_0.*n_kv_max=(0|2048)'` retornou **3/3**. O gate de MoE passou cinco verificações em compilação e o simulador SLRU arquivado passou seis cenários, todos sem fills nos ubatches seguintes ou no decode após o prefill. O banco GPU e os eventos precisam de testes em hardware.
- A fixture geral `test-generate-models` falhou em `sensenova_loop.attention.recurrent_layers`, ausente no GGUF sintético gerado. O CTest de ciclo de vida agora gera isoladamente os dois GGUFs Qwen35 de que precisa. A suíte geral ainda tem essa pendência.
- Sem build CUDA ou execução GPU nesta sessão. Os ensaios de throughput, pico de VRAM, tempo de load quente e o travamento de streaming continuam pendentes. Nenhum serviço foi reiniciado.
