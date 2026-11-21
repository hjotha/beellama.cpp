# DFlash2 sobre Qwen3.8-Flash-Next (Qwen4Exp): capture das features do alvo

Data: 2026-11-21/22 · worktree `/home/hjotha/worktrees/beellama-dflash-qwen4exp` ·
branch `feat/dflash-qwen4exp` · base `main` `1327435a1` · CPU-only, sem GPU/A100,
sem serviços. Nada foi feito push/merge.

## Veredito

O capture do alvo **já existia** no `main`: o commit `d9ee007a0`
("model : fix qwen4exp layer inputs for DFlash drafters (#30927)", 2026-10-22)
publica exatamente a média HC no `t_layer_inp[il]`, sob a flag default-off
`cparams.embeddings_layer_inp[il]`. Não foi necessário escrever um capture novo.

O que **faltava** era o teste e uma correção no encanamento do drafter: o
`batch_inject` do DFlash2 (batch só de embeddings, usado no prefill do draft)
alocava **uma** posição por linha, mas o caminho M-RoPE escreve **quatro**
(posições `[t, y, x, z]`), corrompendo o heap. O drafter oficial do DFlash2 para
o Qwen3.8-Flash-Next é M-RoPE (`dflash.rope.dimension_sections=[64,0,0,0]`), então
esse overflow atingiria o alvo real no primeiro prefill. Corrigido e coberto pelo
teste de CPU.

Resultado da validação local (build CPU, fixture Qwen4Exp com `hc=4`):

```
tap-off: target logits bit-identical with the capture enabled
prefill: tokens=4 hc=4 max|tap-mean| = 1.490e-08, max stream spread = 2.681e-01
ubatch: tokens=16 hc=4 max|tap-mean| = 1.490e-08, max stream spread = 5.374e-01
decode: tokens=1 hc=4 max|tap-mean| = 1.304e-08, max stream spread = 2.874e-01
rollback: tokens=1 hc=4 max|tap-mean| = 1.490e-08, max stream spread = 2.928e-01
two-seq: tokens=4 hc=4 max|tap-mean| = 1.490e-08, max stream spread = 2.964e-01
spec-prefill: tokens=5 hc=4 max|tap-mean| = 1.490e-08, max stream spread = 2.874e-01
spec: drafted 7 tokens from a 5-token prompt (first id 28)
OK: Qwen4Exp HC target capture and DFlash2 handoff
```

O critério final continua sendo a A100 com o alvo real (greedy idêntico ao sem
spec + aceitação/throughput); a fixture prova a semântica e o fluxo, não o
desempenho.

## 1. Plumbing real (com linhas)

- O drafter declara `dflash.target_layers` no GGUF; o loader exige a chave e a
  usa para dimensionar a entrada do encoder
  (`src/models/dflash.cpp:53-57`, `n_embd_inp_enc = n_target_layers * n_embd`).
  O conversor grava **id HF + 1**, porque o runtime tapa a **entrada** do bloco:
  `conversion/qwen.py:945-948`.
- No alvo, `llama_set_embeddings_layer_inp(ctx, lid, true)` liga o tap
  (`src/llama-context.cpp:3721`) e reserva o buffer `n_embd * n_batch`
  (`output_reserve`, `src/llama-context.cpp:5662`).
- O grafo do Qwen4Exp só constrói o tap quando `cparams.embeddings_layer_inp[il]`
  é verdadeiro (`src/models/qwen4exp.cpp:554-559`), usando
  `hc_stream_mean` (`src/models/qwen4exp.cpp:388-400`):
  soma os `hc` streams do resíduo `[n_embd, hc, n_tokens]` e multiplica por
  `1/hc`. Sem a flag, nenhum nó é criado (default-off).
- O `llm_graph_result::set_outputs` marca o tensor como saída
  (`src/llama-graph.cpp:2419-2423`), e `llama_context::extract_layer_inputs`
  copia as linhas do ubatch para o buffer do host, por `token_offset`
  (`src/llama-context.cpp:5514` e `5833`). `get_embeddings_layer_inp` só lê esse
  buffer (`src/llama-context.cpp:3459`).
- No `common/speculative.cpp`, a inicialização do DFlash2 liga os taps do alvo
  e o `embeddings_nextn` do draft (`:2887`), e cada `process()` monta
  `[n_tokens, n_target_layers * n_embd]` copiando `target_layer_ids[k]` em ordem
  (`:3204`). O buffer injetado usa a largura `n_embd_enc`.

Isso é genérico: qualquer alvo que publique `t_layer_inp[il]` com largura
`n_embd` funciona. O Qwen4Exp publica a média HC nessa largura, então está
coberto.

## 2. Semântica validada para o Qwen3.8

- **Média HC exata.** A referência SGLang
  (`incoai/Qwen3.8-Flash-Next-DFlash2/patches/qwen38-dflash2-capture.patch`)
  captura, depois de completar a camada `i`, o hidden `[T, hc*hidden]` e faz
  `hidden_states.unflatten(-1, (hc_count, hidden_size)).mean(dim=-2)`.
  O beellama tapa `res_hc` no topo da iteração `il`, que é exatamente o resíduo
  concluído da camada `il-1` (atenção + FFN + HC combine), antes do PLE da
  camada `il`. Para `target_layers=[6,16,26,36,46]`, o tap `il` é a saída
  concluída de HF `[5,15,25,35,45]`. O PLE real fica na camada 2, abaixo de todos
  os taps, então não há deslocamento.
- **Prefill, decode, ubatch.** O tap é raiz de grafo, calculado antes de
  qualquer crop de `inp_out_ids`; `extract_layer_inputs` grava por
  `token_offset`. Testado com 4 tokens, 1 token de decode e um decode de 16
  tokens com `n_ubatch=10` (duas ubatches).
- **Duas sequências no mesmo batch.** As linhas são por token, na ordem do
  batch; a checagem por linha com valores diferentes por token também detecta
  troca de ordem.
- **Rollback.** Com `n_rs_seq=7` (mesmo valor que o servidor deriva de
  `--spec-draft-n-max`), `seq_rm` do sufixo rejeitado e re-decode na mesma
  posição produz tap correto.
- **Multimodal.** Fora do escopo, como pedido. O `process()` do DFlash já pula
  linhas de imagem com posição pinada (`pos_pinned`), mas não foi exercitado.
- **Default-off.** O teste decodifica o mesmo prompt com o tap desligado, liga o
  tap, limpa a memória e decodifica de novo: os logits são **bit-idênticos**
  (`memcmp`), provando que o capture não altera a aritmética do alvo.

## 3. Correção encontrada: posições M-RoPE do `batch_inject`

`common/speculative.cpp:2825` cria

```cpp
batch_inject = llama_batch_init(llama_n_ubatch(ctx_dft), n_embd_enc, n_seq);
```

como batch **só de embeddings**. `llama_batch_init` aloca `pos` com
`n_tokens_alloc` entradas (`src/llama-batch.cpp:1109`), mas
`llama_batch_compat::init` lê, para batch de embeddings,
`n_pos_per_embd` posições por linha no layout section-major
(`src/llama-batch.cpp:1444-1452`). O `process()` grava as quatro linhas
`[p, p, p, 0]` (`common/speculative.cpp:3217-3221`).

Com um drafter M-RoPE — o caso do DFlash2 oficial, cujas seções são
`[64,0,0,0]` — isso escreve `4*n_chunk` posições num buffer de `n_ubatch`.
Antes da correção, a fixture abortava no destrutor com
`double free or corruption (out)` (free do `batch_inject` corrompido); o mesmo
vale para o alvo real, onde o prefill do draft usa o batch de features.

Correção em `common/speculative.cpp:2829-2843`: quando `is_mrope`, realocar
`batch_inject.pos` para `n_ubatch * GGML_MROPE_SECTIONS` (mesmo padrão já usado
no buffer `inject` do caminho shadow, `:2371-2378`). Caminho não-M-RoPE
inalterado. O teste novo falha sem esse trecho e passa com ele.

## 4. Teste de CPU e fixture

Arquivos novos:

- `tests/gen-tiny-qwen4exp-hc.py` — gera dois GGUFs:
  - alvo `qwen4exp`: 2 camadas, `n_embd=256`, `hc=4`, camada 0 linear com PLE,
    camada 1 full attention com indexer QSA; **todos os pesos da camada 1 são
    zero**, então o bloco é no-op e o resíduo da camada 0 chega intacto ao
    `h_nextn`;
  - draft `dflash`: 2 camadas, `block_size=8`, conv 2/16, selector rank 8 /
    top_k 4, `target_layers=[1]`, vocabulário sintético `test` de 128 tokens
    compartilhado com o alvo, sem `token_embd`/`output` (empresta do alvo via
    `ctx_other`).
- `tests/test-qwen4exp-dflash-capture.cpp` — carrega os dois GGUFs e verifica:
  1. tap desligado × ligado → logits bit-idênticos;
  2. prefill 4 tokens;
  3. decode de 16 tokens com `n_ubatch=10`;
  4. decode de 1 token;
  5. rollback de sufixo + re-decode;
  6. duas sequências no mesmo batch;
  7. fluxo real `common_speculative`: `begin` → `process` (taps → KV do draft) →
     `draft` (selector), exigindo 1..7 tokens dentro do vocabulário.

  A comparação é `tap[i*n_embd+j]` contra a média em `double` dos `hc` streams
  de `llama_get_embeddings_nextn` (layout `[token][stream][hidden]`), com
  tolerância `1e-6`; a fixture também exige que os streams realmente difiram
  (`max spread > 1e-4`), para a média não ser vacuamente igual.

Wiring em `tests/CMakeLists.txt:518-539`: teste `generate-qwen4exp-dflash-fixture`
(fixture de CTest, `PYTHONPATH=gguf-py`) + `test-qwen4exp-dflash-capture`.

Execução real:

```
$ ctest --test-dir build-dflash -R "qwen4exp-dflash" --output-on-failure
1/2 Test #80: generate-qwen4exp-dflash-fixture ...   Passed    0.41 sec
2/2 Test #81: test-qwen4exp-dflash-capture .......   Passed    0.10 sec
100% tests passed out of 2
```

Regressões próximas (mesmo build): `test-dflash-proposal`,
`test-dflash-pipeline-candidate`, `test-dflash-shadow-observation`,
`test-dfly-fusion`, `test-nextn-flags`, `test-kv-tail-input`,
`test-generate-speculative-target`, `test-speculative-rejection-sampling` —
todas passaram.

Build usado: `cmake -S . -B build-dflash -DGGML_CUDA=OFF -DCMAKE_BUILD_TYPE=Release`,
`cmake --build build-dflash -j4 --target llama test-qwen4exp-dflash-capture`.

## 5. Arredondamento da média HC

- **beellama:** as ativações do alvo são F32; `hc_stream_mean` faz soma
  sequencial dos `hc` streams em F32 e multiplica por `1/hc` (F32). O drafter
  recebe F32.
- **SGLang:** `hidden_states` é BF16; `.mean(dim=-2)` reduz com acumulador
  float32 e devolve BF16 (o vetor que o drafter consome é BF16).
- **Pack DF2P:** o drafter foi treinado com as features BF16 do alvo; a
  diferença é só a quantização final para BF16 (~0,4% relativo), não uma
  mudança de layout ou de camada. O beellama é mais preciso; a expectativa é
  aceitação igual ou marginalmente melhor, mas isso só a A100 mede.
- Não há equivalência com o Strata: o critério é o alvo real na A100 (greedy
  idêntico ao sem spec, aceitação e throughput), não paridade bit a bit de
  features.

## 6. Comando exato para a A100 (não executado aqui)

Levar o build CUDA sm_80 da branch e os GGUFs do drafter
(`/tmp/opencode/dflash2-gguf/dflash2-bf16.gguf` e `dflash2-q8_0.gguf`; não
alterar esses arquivos) para `/models/`.

Smoke de load/geometria (contexto curto, mesmas flags de placement):

```bash
BIN=./build-a100/bin
TARGET=/models/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
DRAFT=/models/dflash2-q8_0.gguf

$BIN/llama-server \
  -m "$TARGET" -md "$DRAFT" \
  -c 4096 -b 512 -ub 128 -fa on \
  --load-mode mmap --lazy-mode on \
  -ctk q8_0 -ctv q8_0 \
  -ncmoe 2 --moe-cache-mib 1024 --vram-reserve-mib 4096 \
  --spec-type draft-dflash --spec-draft-n-max 7 \
  -np 1 --host 127.0.0.1 --port 8190 --metrics
```

Medição greedy com cache 6144 (primeiro BF16 para corretude; repetir com Q8_0):

```bash
# controle sem spec
$BIN/llama-server \
  -m "$TARGET" \
  -c 6144 -b 2048 -ub 512 -fa on \
  --load-mode mmap --lazy-mode on \
  -ctk q8_0 -ctv q8_0 \
  -ncmoe 2 --moe-cache-mib -1 --vram-reserve-mib 4096 \
  --spec-type none \
  -np 1 --host 127.0.0.1 --port 8191 --metrics

# DFlash2 BF16 (trocar DRAFT por /models/dflash2-bf16.gguf)
$BIN/llama-server \
  -m "$TARGET" -md /models/dflash2-bf16.gguf \
  -c 6144 -b 2048 -ub 512 -fa on \
  --load-mode mmap --lazy-mode on \
  -ctk q8_0 -ctv q8_0 \
  -ncmoe 2 --moe-cache-mib -1 --vram-reserve-mib 4096 \
  --spec-type draft-dflash --spec-draft-n-max 7 \
  -np 1 --host 127.0.0.1 --port 8192 --metrics
```

Para o mapa de prefill/throughput sem spec (baseline do alvo), o `llama-bench`
não usa drafter:

```bash
$BIN/llama-bench -m "$TARGET" \
  -c 6144 -b 2048 -ub 512 -fa 1 \
  --load-mode mmap -ctk q8_0 -ctv q8_0 \
  -ncmoe 2 --moe-cache-mib -1 --vram-reserve-mib 4096 \
  -p 512,2048,4096 -n 0
$BIN/llama-bench -m "$TARGET" \
  -c 6144 -b 2048 -ub 512 -fa 1 \
  --load-mode mmap -ctk q8_0 -ctv q8_0 \
  -ncmoe 2 --moe-cache-mib -1 --vram-reserve-mib 4096 \
  -p 0 -n 128
```

Critérios, nesta ordem:

1. **Load:** log com `DFlash2 conv kernel = 2, group = 16, selector rank = 256,
   top-k = 16`, `block_size=8`, `n_extract=5`, `target_layers=[6,16,26,36,46]`;
   sem `failed to create draft context`, sem OOM.
2. **Corretude:** greedy (`temperature=0`) em ≥8 prompts fixos com `n_predict`
   fixo → mesmos token ids do controle sem `--spec-*` (requisito absoluto).
3. **Aceitação:** `llamacpp:spec_decode_num_drafts_total > 0` e
   `..._accepted_tokens_per_pos_total{position="0".."6"}` com pos-0 > 0;
   acceptance length média por passo.
4. **Throughput:** decode com e sem spec nas mesmas flags; pico de VRAM
   registrado; depois repetir com `dflash2-q8_0.gguf` e comparar.
5. Se a aceitação for ~0 com o load OK, investigar primeiro o mapeamento das
   camadas (tap de entrada de blk 6/16/26/36/46) e depois o empréstimo de
   `token_embd`/`output` do alvo.

## 7. Riscos e limites

- **A100 não executada** (regra da tarefa). O capture é F32 e a referência é
  BF16; o efeito em aceitação/throughput é desconhecido até a medição.
- **CUDA:** `hc_stream_mean` usa `ggml_view_2d` com `nb[1] = x->nb[2]` e soma
  `ggml_add`/`ggml_scale`; o mesmo padrão já é usado pelo deepseek4 e pelo
  commit upstream `d9ee007a0`, mas a execução em sm_80 não foi feita aqui.
- **Multimodal** (Qwen4Exp-VL) não foi exercitado; o DFlash já pula linhas de
  imagem pinadas, e o capture em si é o mesmo grafo de texto.
- **Duas sequências** foram validadas no capture; o `process()`/`draft()` do
  DFlash2 foi exercitado com uma sequência (o caminho multiseq do spec não foi
  alvo desta mudança).
- O `begin()` do DFlash avisa `process() did not run on every prefill ubatch`
  quando chamado antes do prefill; é o comportamento esperado do pedido novo,
  não um erro do capture.
- O GGUFs em `/tmp/opencode/dflash2-gguf/` não foram alterados.

## 8. Arquivos alterados

| arquivo | mudança |
| --- | --- |
| `common/speculative.cpp` | realoca `batch_inject.pos` para 4 posições/linha em drafters M-RoPE (fix do overflow) |
| `tests/gen-tiny-qwen4exp-hc.py` | novo gerador da fixture alvo Qwen4Exp HC + drafter DFlash2 |
| `tests/test-qwen4exp-dflash-capture.cpp` | novo teste CPU de semântica do tap e fluxo do spec |
| `tests/CMakeLists.txt` | registra a fixture e o teste no CTest |

`src/models/qwen4exp.cpp` **não** foi alterado: o capture já estava correto no
`main` desde `d9ee007a0`.
