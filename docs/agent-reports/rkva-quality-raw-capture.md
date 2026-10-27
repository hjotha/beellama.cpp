# Captura CAPI de logits full-vocab — entrega CPU (2026-10-27)

Branch: `perf/rkva-quality-raw-capture-20261027` (clone
`/home/hjotha/beellama-agents/quality-gate`, base `2b5f3ed65`, derivada de
`perf/rkva-quality-gate-20261026`). Nenhuma worktree.

**Esta entrega não executou hardware.** Não houve `llama-server`, CUDA runtime,
`llama_decode`, carregamento de modelo, Xbox, LAN/SSH, `systemctl`, build
UWP/MSVC, push ou merge. O único build foi o driver C++ de host; a suíte final
roda apenas `--help`, `--self-test` e `preview`. Durante o desenvolvimento, dois
invokes exploratórios de `--execute` confirmaram somente rejeição antes de
`llama_backend_init` (contexto 160001 e ambiente inválido); ambos foram
removidos da suíte final para respeitar a instrução de não executar
`--execute`. Nenhum backend/modelo foi inicializado. O root continua dono da
execução serializada de GPU; o Astra high revisa antes de integrar.

## 1. Arquivos entregues

| Arquivo | Conteúdo |
| --- | --- |
| `tools/xbox/tests/rkva_quality_capture.cpp` | Driver CAPI inerte por padrão; `--execute` explícito |
| `tools/xbox/tests/test_rkva_quality_capture.py` | 27 testes CPU offline; compila o driver com g++ |
| `docs/agent-reports/rkva-quality-raw-capture.md` | Este relatório |

O harness aprovado `tools/xbox/scripts/rkva-quality-gate.py`,
`tools/xbox/tests/test_rkva_quality_gate.py`, o checkout root
`/home/hjotha/beellama.cpp`, modelos, slots, `~/.codex`, outros clones e
configurações não foram alterados.

O driver ficou com 1427 linhas (1332 não vazias). A meta de ~450 linhas não foi
atingida: o excedente é SHA-256 próprio (~95 linhas), self-test CPU (~120),
parser/validador estrito de argv e envelope (~180) e escrita/manifesto. Não há
framework de 3000 linhas nem dependência de teste externa. Se o Astra exigir um
driver menor, o corte natural é remover o SHA-256 e o relatório JSON do
self-test, não a validação de contrato.

## 2. Por que capturar pela CAPI

A QA HTTP com `n_probs=248320` custa cerca de 2,46 s por passo por causa do JSON
com todos os IDs, strings e bytes. Os dois primeiros casos (128 passos)
passaram PPL mas falharam KL; a matriz 6 caudas × 3 comprimentos × 128 passos
precisa de caracterização semântica e ficaria muito lenta nesse protocolo.

Esta captura salva **todos** os logits em binário, na mesma trajetória e
cadência do servidor, sem mudar modelo, shaders ou bibliotecas numéricas. Ela
não decide INT8 default: o rótulo é `RAW_CAPTURE_ONLY`.

## 3. CLI e modos

```text
rkva_quality_capture --help
rkva_quality_capture --self-test [--self-test-json FILE]
rkva_quality_capture [--preview] \
  --common-args-json FILE --prompt-json FILE --prefix-snapshot FILE \
  --p 102400 --context 131072 --teacher-steps 128 \
  [--teacher-json FILE] --output DIR
rkva_quality_capture --execute ...   # somente root, com GPU serializada
```

- Sem `--execute`, o modo é `preview`: lê arquivos, imprime o plano e **não**
  chama backend, modelo, rede ou escrita.
- `--help` sai antes de qualquer init.
- `--self-test` é CPU puro: parser do envelope, cadência, writer dummy,
  argmax/teacher e guarda de prefixo.
- `--context` tem hardcap 160000; para este protótipo o valor tem de ser
  exatamente 131072. `--p` tem de ser 102400.
- `--output` precisa ser um diretório novo, sem symlink. Nada é sobrescrito.
- `--teacher-steps` é configurável (1..1024); a matriz GOLD usa 128.
- O prompt precisa ter `P+512 <= L`, cauda múltipla de 512 e
  `L + teacher_steps <= 131072`.

## 4. Contrato de argumentos comuns

`--common-args-json` é um array JSON de argv, no formato de
`H.server_command` do harness (o root substitui o argv0 pelo binário do
driver). O preview valida lexicalmente; o `--execute` valida de novo e depois
usa `common_params_parse(..., LLAMA_EXAMPLE_SERVER)` e
`common_init_from_params`.

Flags exigidas: `--ctx-size 131072`, `--batch-size 1024`,
`--ubatch-size 512`, `--parallel 1`, `--gpu-layers 99`, `--device CUDA0`,
`--flash-attn on`, `--fit off`, `--cache-type-k kvarn4`,
`--cache-type-v kvarn4`, `--kv-position-split-p 102400`,
`--remote-attn-layers full`, `--remote-attn-cache-type-k q4_0`,
`--remote-attn-cache-type-v q4_0`, `--remote-attn-format kvarn4`,
`--remote-attn-wire-format fp8`, `--remote-attn` não vazio, `--spec-type none`,
`--spec-draft-n-max 0`, `--no-context-shift`, `--no-warmup` e um `-m` local.
`--help`, `--version`, `--list-devices` e `--completion` dentro do array são
recusados, porque `common_params_parse` poderia sair do processo.

O driver não inicia HTTP. Flags de servidor (`--host`, `--port`,
`--no-webui`, `--metrics`, `--slot-save-path`, ...) são aceitas pelo parser
para manter a configuração exata, mas não abrem socket.

## 5. Inicialização e config exata

`--execute` faz, nesta ordem: valida ambiente, `llama_backend_init()`,
`common_params_parse`, validação dos campos parseados, `common_init_from_params`
(um modelo, um contexto), restauração do prefixo, prefill, teacher, destruição
de contexto/modelo, `llama_backend_free()`, e só então grava `manifest.json`.

O driver força:

```text
warmup = false
load_mtp = false
speculative.types = {NONE}
speculative.draft.n_max = 0
speculative.n_rs_seq_target = 0
```

`load_mtp=false` é o campo público de `include/llama.h`; os ~330 MB do head MTP
não cabem em B1024. A config declarada é checada contra a runtime:
`n_ctx=131072`, `n_batch=1024`, `n_ubatch=512`, `n_ubatch_physical=256`,
`n_seq_max=1`, `n_rs_seq=0`, `n_outputs_max=1`, `n_outputs_max_per_seq=1`,
FA on, KVarN `K4V4_G128` com backing `q4_0/q4_0`, remote
`full`/`kvarn4`/`fp8`/`q4_0`, `kv_position_split_p=P`, `kv_tail_tokens=0` e
device `CUDA0`.

O limite de saída reproduz `tools/server/server-context.cpp::server_output_limits`
via `common_speculative_get_output_limits(n_batch=1024, n_parallel=1, n_draft=0)`,
que retorna `{1,1}`. `params.n_outputs_max` e `n_outputs_max_per_seq` são
pinados em 1 antes do init; com o default 0 o contexto interpretaria
`n_outputs_max=n_batch=1024` e poderia alocar ~970 MiB de logits. O runtime lê
`ctx->get_cparams().n_outputs_max`/`n_outputs_max_per_seq` e exige 1/1.

O dispositivo segue o contrato real de `common/arg.cpp::parse_device_list`:
`--device CUDA0` produz `{CUDA0, nullptr}` (dois elementos, um GPU e o
terminador). O driver exige exatamente um ponteiro não nulo, terminador null,
sem null inicial/buraco, tipo diferente de CPU e nome `CUDA0`; recusa `none`,
duas GPUs e listas malformadas. O self-test valida a contagem/terminador com
ponteiros opacos, sem consultar GPU.

O ambiente é validado antes de inicializar o backend. São exigidos
`GGML_RKVA_NATIVE_KVARN4_EXPERIMENTAL=1`, `NATIVE_PRODUCER=cuda`,
`RANGE_STORAGE=kvarn4`, `RANGE_KERNEL=fine`, `NATIVE_TAIL_TOKENS=128`,
`PREFILL_QUERY_CHUNK=8`, `NATIVE_BULK=f16`, `WIRE_FORMAT=fp8`,
`WIRE_STRICT=1`, `ASYNC_LOCAL_REMOTE=1`, `CUDA_FP8=mapped`,
`DISK_SNAPSHOTS=0`, `WAVEFRONT_TILE=256`, `WAVEFRONT_RPC_BATCH=1`,
`TCP_PREFETCH=1`, `LOCAL_INPUT_REUSE=0`, `PROFILE=1`,
`GGML_KVARN_WINDOW_CHUNK=4096`, headroom 128 MiB, guard 10 ms/64 MiB,
`LLAMA_N256_LOOKAHEAD=0` e `GGML_CUDA_MASK_PACK` **ausente**. O manifesto grava
todo o ambiente `GGML_/LLAMA_/CUDA_/XLLAMA_` observado.

## 6. Prefixo congelado

O driver restaura **apenas** o prefixo P com
`llama_state_seq_load_file`. O primeiro call lê só a contagem; a alocação é
limitada a `P` ou `P+4` (envelope server v1). `P-128` é recusado neste
protótipo, assim como qualquer outro tamanho. Depois do load:

- o envelope v1 `[LLAMA_TOKEN_NULL, 1, count, IDs, 0]` é validado e **nunca**
  cai para raw em caso de truncamento, versão errada, mídia ou dado extra;
- `unpack_ids == prompt[0:P]`;
- `llama_memory_seq_pos_max == P-1`.

O SHA do snapshot de 2,2 GB é delegado ao wrapper do root, que já valida o
contrato H. O driver grava tamanho, palavras empacotadas e formato; não rehash
do arquivo gigante.

## 7. Cadência idêntica à do servidor

A cadência copia apenas o algoritmo legítimo de
`tests/test-rkva-wavefront-2x256-live.cpp` (sem rodar dois contextos):

```text
count = min(1024, remaining)
if remaining > 512 && count == remaining: count -= 512
if final e count == 512: decode 256, 128, 128
else: decode count
```

Só o último 128 do seam pede logits. Os parents de 1024 são divididos pela API
em 2 UB512; o parent final de 512 vira 256/128/128. Contagens exatas:

| cauda | UB512 | seam |
| --- | --- | --- |
| 4096 | 7 | 256/128/128 |
| 16384 | 31 | 256/128/128 |
| 28160 | 54 | 256/128/128 |
| 57344 | 111 | 256/128/128 |

O gate valida a cobertura total do plano e a equação
`ubatch_512*512 + seam_256*256 + seam_128*128 == tail`, nunca `tail/512`
(a substituição do último 512 pelo seam reduz uma unidade UB512). O self-test
inclui `P=102400, length=159744` (cauda 57344) sob o hardcap 160000; a execução
continua exigindo C131072. A ausência do checkpoint de CPU pode mudar
tempo/alocação, não a matemática; o root deve comparar GOLD com o HTTP antes de
adotar a captura. Não se deve generalizar `RAW != HTTP` sem investigar.

## 8. Teacher e passos

Se `--teacher-json` for fornecido, cada passo usa `teacher_ids[step]` fixo. O
argmax do candidato nunca substitui o teacher. Sem teacher, o próximo token é o
argmax dos logits capturados (menor ID em empate), como o baseline da QA.

Mapeamento exato do HTTP com 128 passos:

```text
passo 0: logits em L-1; teacher_ids[0] escolhido
passo 1: logits em L;   teacher_ids[1] escolhido
...
passo 127: logits em L+126; teacher_ids[127] escolhido (não realimentado)
```

O último token realimentado é `teacher_ids[126]`, na posição `L+126`. O estado
final tem `L+127` tokens (posições 0..L+126), que é o mesmo estado da 128ª
requisição HTTP do harness. O relatório usa “último estado L+127” no sentido de
contagem de tokens, não de índice máximo.

Token history = prompt completo + teacher realimentado. É gravado em
`tokens.u32.bin` (little endian, 4 bytes/ID) com SHA-256; cada passo grava também o `prompt_sha256` do prompt daquele passo (prompt + teacher já realimentado), no mesmo formato `struct.pack("<I")` do harness; não há RNN nem dump de
188 MB por passo.

## 9. Saída

Cada passo grava imediatamente:

```text
0000.logits.f32.bin   # float32 LE, index == token id, exatamente 4*vocab bytes
0000.json             # step, position, token_count, teacher_id, argmax,
                      # vocab, filename, bytes, finite, sha256, cadence
```

No fim:

```text
tokens.u32.bin
manifest.json         # PASS, RAW_CAPTURE_ONLY, após contexto/modelo destruídos
failed.json           # somente em falha de execução
```

Antes de ler cada linha, o driver exige o contrato de logits densos:
`sampled_logits_ith == nullptr`, `sampled_logits_count_ith == vocab` e
`sampled_candidates_count_ith == 0`. O getter público de candidatos pode
devolver o fallback identidade full-vocab (não um conjunto compacto); o count
zero é o sinal autoritativo. `params.sampling.backend_sampling=false` é pinado
para o HTTP `n_probs`. O manifesto registra `capture.logits_contract`
(`sampled_logits_active=false`, contagens, `sampled_candidates_source`,
`dense_row_index=count-1`, `outputs_requested=last-row-only`).

O manifesto contém geometria runtime (incluindo `n_rs_seq`, `n_outputs_max`,
`n_outputs_max_per_seq`), config declarada, modelo (desc, params, tamanho,
vocab, camadas, heads), ambiente, prefixo, plano de cadência, teacher, hashes
do histórico e de cada arquivo de logits, e `llama_print_system_info()`.
`llama_perf_context_print()` é chamado antes da destruição para preservar o
diagnóstico de perfil; o root faz o parsing com o helper já aprovado.

## 10. Build CPU real

```bash
ROOT=/home/hjotha/beellama.cpp
g++ -std=c++17 -O2 -Wall -Wextra \
  -I$ROOT/include -I$ROOT/common -I$ROOT/ggml/include -I$ROOT/ggml/src \
  -I$ROOT/vendor -I$ROOT/src tools/xbox/tests/rkva_quality_capture.cpp \
  -o /tmp/rkva_quality_capture \
  -L$ROOT/build-optimized/bin -Wl,-rpath,$ROOT/build-optimized/bin \
  -lllama-common -lllama -lggml -lggml-base -pthread -ldl -fopenmp
```

Saída real: exit 0, sem warnings. Nenhum backend CUDA foi compilado. O binário
linka as libs do root (incluindo `libcudart`/`libggml-cuda` via `libllama`),
mas `--help`, `--self-test` e `preview` não chamam `llama_backend_init`.

## 11. Validação CPU real

```text
$ ./rkva_quality_capture --help
rkva_quality_capture - raw full-vocab logits (RAW_CAPTURE_ONLY)
...
Defaults: preview. No backend/model/network is touched without --execute.
```

```text
$ ./rkva_quality_capture --self-test
SELF-TEST PASS
```

O self-test cobre: envelope v1 válido, raw, truncado, versão errada, mídia,
dado extra e contagem impossível; guarda P/P+4/P-128/P+1; contrato de device
`{GPU,null}` vs `none`/duas GPUs/null inicial/buraco/sem terminador; cadência
exata 4096→7, 16384→31, 28160→54, 57344→111 e 106496/130560/159744; writer
`4*vocab`, SHA, overwrite, NaN e linha curta; teacher fixo vs argmax; empate no
menor ID; SHA do hash de prompt no formato do harness.

Testes Python (compilam o driver com g++, sem CUDA):

```text
$ RKVA_CAPTURE_REQUIRE_DRIVER=1 python3 -m unittest -v \
    tools/xbox/tests/test_rkva_quality_capture.py
Ran 27 tests in 10.963s (última execução)
OK
```

Prévia real com fixture de 106496 tokens:

```text
mode: preview
execute: false
starts_child: false
writes_output: false
label: RAW_CAPTURE_ONLY
context: 131072 (hardcap 160000)
p: 102400
prompt_tokens: 106496
tail_tokens: 4096
teacher_steps: 128
teacher_source: fixed-file
prefix_snapshot: .../prefix.bin (4096 bytes)
output: .../out (not created)
common_args: 62
cadence_api_calls: 7
cadence_ubatch_512: 7
cadence_seam_256: 1
cadence_seam_128: 2
```

Trecho real do relatório do self-test:

```text
device one_gpu_terminated: non_null=1 terminated=true valid=true
device none:                non_null=0 leading_null=true valid=false
device two_gpus:            non_null=2 valid=false
cadence tail=4096  ubatch_512=7  seam=1/2 api_calls=7
cadence tail=16384 ubatch_512=31 seam=1/2 api_calls=19
cadence tail=28160 ubatch_512=54 seam=1/2 api_calls=30
cadence tail=57344 ubatch_512=111 seam=1/2 api_calls=59
```

O caminho de falha (`failed.json`) é coberto por contrato estático e pelo
código. A suíte final não executa `--execute`; um invoke exploratório inicial
confirmou apenas a rejeição de ambiente antes do backend e foi removido. O root
é quem exercita o caminho completo no hardware.

## 12. Uso do GOLD pelo root

Casos GOLD já existentes:

```text
/home/hjotha/.codex/artifacts/rkva-priorities-20261026/quality-wide-r1/cases/portuguese-106496/
  tokens.json
  off/teacher.json
  off/distributions.bin.gz
```

O root deve rodar o driver uma vez, por GPU serializada, com o
`H.environment` congelado e o snapshot C131072, e comparar os primeiros 8 passos
com o HTTP baseline. Critério antes de adotar a captura: max abs ~<= 1e-6 e KL
próximo de zero. A matriz HTTP está ativa agora e **não** foi tocada.

## 13. Limitações e o que não foi validado

- Nenhum `--execute` válido (só rejeições exploratórias antes do backend, removidas da suíte), modelo, CUDA, KVarN, remote attention, TCP, Xbox,
  perfil RPC ou snapshot real foi executado/lido.
- Nenhum logit real foi comparado ao HTTP; a equivalência numérica é do root.
- `P-128`, prefill frio, replay de prefixo e FULL overflow continuam fora do
  escopo deste protótipo.
- O driver usa um processo, um modelo, um contexto e nenhum sampler; não há
  braço OFF/ON nem decisão INT8.
- O SHA do snapshot grande é do wrapper; o driver não o recalcula.
- A captura é `RAW_CAPTURE_ONLY`, não uma qualificação automática.
- O contrato de output limits, o `backend_sampling=false`, os counts de
  logits/candidatos e o `ctx->get_cparams()` são validados por código/compilação,
  mas não foram exercitados em hardware nesta entrega.

## 14. Revisão Astra e próximos passos

Checklist para o Astra high:

1. Conferir que o preview/help/self-test não chamam backend/modelo/rede.
2. Conferir o contrato de argv e a recusa de flags que não fixam geometria,
   modelo, device, cache e wire.
3. Conferir a cadência 256/128/128, as contagens exatas 7/31/54/111 e a
   cobertura total do plano (sem a fórmula `tail/512`).
4. Conferir o contrato de device `{CUDA0,nullptr}` e a recusa de CPU/duas GPUs.
5. Conferir o pin `n_outputs_max=1`/`per_seq=1` e a auditoria runtime de
   `ctx->get_cparams()`.
6. Conferir `backend_sampling=false`, `sampled_logits==nullptr`,
   `sampled_logits_count==vocab` e `sampled_candidates_count==0`.
7. Conferir teacher fixo, hashes e a não-substituição pelo argmax candidato.
8. Conferir o envelope v1, a recusa de P-128 e `load_mtp=false`.
9. Conferir que `manifest.json` só é gravado após destruir contexto/modelo e
   liberar o backend.

Depois da revisão, o root pode: (a) compilar o driver, (b) rodar `--preview` e
`--self-test`, (c) rodar um `--execute` controlado com o caso
`portuguese-106496`, (d) comparar os primeiros passos com o GOLD HTTP e só
então considerar a captura para a matriz. Nenhum commit/merge/push foi feito
nesta entrega além da própria branch de trabalho.
