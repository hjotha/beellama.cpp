> Draft MoE-prefill cache report. The patch is applied in the worktree; the A100 A/B and the final commit are pending.
>
> Historical agent report (2026-11-21). Current status and numbers: docs/PLAN-strata-parity.md.

# MoE expert cache no prefill (working set do grupo residente)

Status: implementado, compilado CPU-only, **não commitado**. Nenhuma GPU, CUDA ou A100 foi
usada; o `llama-server` local não foi tocado.

## Checkout

- Origem: `/home/hjotha/worktrees/beellama-strata-parity` (branch `feat/strata-parity`,
  HEAD `e0ee7098a`), copiada para `/tmp/opencode/impl-moe-prefill` sem `build*`, `.git`,
  `tmp`, `models`, `bench-results`.
- O HEAD da worktree avançou para `72565df76` durante o trabalho (outra sessão commitou
  `docs/PLAN-strata-parity.md`); `git diff e0ee7098a..72565df76 -- src/` está vazio, então o
  baseline dos dois arquivos é o mesmo e a worktree não tem modificação local neles.
- Edições apenas em `/tmp/opencode/impl-moe-prefill/src/llama-moe-cache.cpp` e
  `/tmp/opencode/impl-moe-prefill/src/llama-moe-cache.h`.
- Patch: `/tmp/opencode/moe-prefill.patch` (112 linhas, só os 2 arquivos; aplica com
  `patch -p1` na raiz do repo). sha256
  `92f186905ccf0cf4955f6b5c1b84b9f093e416f0d65fafa68651252d04cac613`.

## Arquivos e linhas

`src/llama-moe-cache.cpp`

| linha | mudança |
|---|---|
| 10 | `#include <cinttypes>` (para `PRId64` no log novo) |
| 208 | `int32_t n_layers` em `struct group` (camadas que compartilham os slots) |
| 251-252 | estado `mutable log_ig/log_tokens` para logar a decisão uma vez por grupo e tamanho de lote |
| 301 | `it->n_layers = it->layers.size()` ao preencher o grupo no construtor |
| 486-523 | `get_slot_map`: nova condição de admissão |
| 541 | comentário de `copy_experts` corrigido (aquele caminho só lê, nunca preenche slots) |
| 603-609 | `GGML_ABORT` de `plan()` em `copy()` com mensagem/invariante explícita |

`src/llama-moe-cache.h`

| linha | mudança |
|---|---|
| 22-25 | documenta que lote grande só usa o cache com o grupo inteiro residente |

## Nova condição (`get_slot_map`, linhas 486-523)

```cpp
if (n_tokens == 0) return nullptr;

const int32_t n_expert = (int32_t) g.ref[0]->ne[2];
const bool group_resident = (int64_t) g.n_layers*n_expert <= g.n_slots;

if (n_tokens > max_batch && !group_resident) return nullptr;          // bypass antigo p/ grupo parcial

if (std::min(n_tokens*n_expert_used, l.slot_map->ne[1]) > g.n_slots)   // seleção de 1 camada
    return nullptr;                                                   // não cabe nos slots
```

Antes: `n_tokens == 0 || n_tokens > 32 || min(...) > n_slots`. Agora `n_tokens > 32` só é
bypass quando o grupo **não** está 100% residente; a segunda checagem continua valendo para
qualquer tamanho de lote.

- `n_tokens <= 32` (decode, batch pequeno, verificação especulativa): caminho idêntico ao
  anterior, inclusive o log (não loga).
- `n_tokens > 32` com `n_layers*n_expert <= n_slots`: usa o slot map. Como o construtor
  limita `n_slots <= n_layers*n_expert` (`max_slots`, linha 359), a prática é
  `n_slots == n_layers*n_expert`, isto é, 100% dos experts do grupo.
- Log novo (uma linha INFO, só em lote grande, deduplicado por grupo+tamanho de lote):

```
get_slot_map: large batch of 1024 tokens uses the MoE cache of group 0: 2 layers, N experts, 2*N slots, the whole group is resident
```

## Por que não expulsa o decode

1. `plan()` precisa de no máximo `n_expert` slots por camada (`uniq` é deduplicado por
   expert). Com o grupo inteiro residente sobram, no mínimo, `n_slots-(n_layers-1)*n_expert
   >= n_expert` slots que nenhuma outra camada do grupo está usando.
2. O gate de seleção (`min(n_tokens*n_expert_used, n_expert) <= n_slots`) é exatamente a
   pré-condição de `plan()`: `uniq.size() <= min(ids, n_expert) <= n_slots`. Logo `plan()`
   não pode falhar nesse caminho.
3. Estado estacionário: depois do primeiro ubatch, todo expert selecionado é hit
   (`fills` vazio), não há `victim()` e nada é reescrito. O pior caso é o primeiro ubatch,
   que enche o grupo uma vez (~1,5 GB no cenário medido) — o mesmo custo de um único
   ubatch do caminho antigo (1,42 GB por ubatch).
4. O caminho antigo de lote grande (`sched_copy_experts` para `MUL_MAT_ID`, que copia
   experts usados para um dst compacto) não é acionado para essas camadas, porque o grafo
   passa a usar o banco (`ggml_mul_mat_id(ctx0, moe_cache->get_experts(w), cur_mm, slots)`).
   Não há dois caminhos ativos na mesma camada: `build_moe_cache_slots` exige que **todos**
   os tensores de experts da camada estejam no cache.

## Riscos

- **Working set**: o gate é residência total, não uma fração. Se o orçamento não cobre
  `n_layers*n_expert`, o comportamento antigo (bypass em lote grande) permanece. Em
  `-ncmoe 2` + 6144 MiB o grupo é 100% residente; em `-ncmoe 8` com cache apertado, não.
- **SLRU**: política inalterada. Com 100% de residência o SLRU degenera em mapeamento
  identidade; `victim()` só roda no preenchimento inicial. Nenhuma mudança em
  probation/protected.
- **Grupos por device**: o gate usa `n_slots` do grupo, que é por device. Com vários GPUs e
  `--tensor-split`, cada device precisa do seu grupo 100% residente; os demais mantêm o
  bypass. Cache auto (`--moe-cache-mib -1`) resolve para `min(livre-reserva, bytes host)`;
  se ficar abaixo de 100%, o caminho novo simplesmente não engaja. Cache fixo 6144 no
  cenário medido dá 100%.
- **Pré-condição de split (não validável sem GPU)**: `copy()` aborta se os experts
  selecionados forem calculados no mesmo split do lookup
  (`"the experts of layer %d are selected in the same split as their MoE cache lookup"`).
  Isso já vale para decode; agora passa a valer para ubatch de 1024, o que **só o A/B na
  A100 pode confirmar**. O abort é explícito — não há resultado silenciosamente errado.
- **Custo novo**: leitura host de `n_tokens*n_expert_used*4` bytes por camada host por
  ubatch (32 KiB com ub 1024 e 8 experts usados) + `plan()` O(ids). É desprezível contra os
  ~1,42 GB de H2D que substitui; o caminho antigo já sincronizava para ler os ids.
- **Splits extras**: 2 splits por ubatch (as 2 camadas host de `-ncmoe 2`) que antes não
  existiam no prefill. O caminho antigo também tinha um sync por camada para ler os ids,
  então não é uma classe nova de parada; o ganho esperado é remover o upload.
- **Thread safety**: o estado de log é `mutable` e escrito na construção do grafo
  (single-thread por contexto), como o resto do `impl`.

## Sem teto extra de `n_tokens` (decisão)

Não foi adicionado um teto tipo 4096. A correção é o gate de capacidade do grupo; o custo
do caminho novo é linear no ubatch (`-ub`, controlado pelo usuário) e limitado por ele, e
qualquer teto arbitrário criaria um degrau silencioso (ex.: `-ub 8192` cairia no bypass sem
motivo de capacidade). Se for desejado um limite conservador, é uma linha:
`if (n_tokens > 4096 && !group_resident) return nullptr;` — mas ele não protege nada que o
gate já não proteja.

## Validação feita (CPU-only)

```
cmake -B build-cpu -S . -DCMAKE_BUILD_TYPE=RelWithDebInfo -DGGML_CUDA=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF          # exit 0
cmake --build build-cpu --target llama -j4                   # exit 0
```

- `src/llama-moe-cache.cpp.o` compilado e linkado em `bin/libllama.so`, **zero**
  warnings/errors do arquivo (grep no log; os warnings restantes são de
  `llama-quant.cpp`/`llama-kv-cache-kvarn.cpp`, pré-existentes).
- `git status --short` (worktree git + `GIT_WORK_TREE` no checkout isolado, índice
  copiado para não tocar no índice real): apenas
  ` M src/llama-moe-cache.cpp` e ` M src/llama-moe-cache.h`.
- `git diff --check`: exit 0 (sem whitespace).
- `git diff --stat`: 2 arquivos, 43 inserções, 4 remoções.
- Invariante do gate checada por simulação fiel do `plan/move/touch/victim` em
  `/tmp/opencode/moe-slru-invariant.py` (2/1/8 camadas, 64/128/256 experts, ub 1024,
  top-8, decode antes e depois): `plan()` nunca falha, o 1º ubatch preenche só o que falta
  e todo ubatch seguinte tem `fills == 0`, inclusive o decode posterior.
- Não rodou `test-backend-ops`/runtime: o cache exige backend GPU.

## A/B na A100-40G (a executar)

Baseline pareado: build **sem** o patch em `/content/build-a100/bin` (HEAD `e0ee7098a`,
QSA q8, já usado nas medições de 1.514/1.516 t/s). Candidato: mesmo HEAD + este patch,
build CUDA novo (nunca build incremental) gerado na `.57` em chroot conforme
`docs/PLAN-strata-parity.md` §"Build A100 (nota curta)", copiado para
`/content/build-parity3/bin` na VM `fn40keep`.

Servidor, idêntico ao harness que produziu os números (1 processo por vez, 2 runs por
config, alternando baseline/candidato para cancelar drift):

```bash
LD_LIBRARY_PATH=/content/build-parity3/bin:/usr/local/cuda/lib64:/usr/lib64-nvidia \
/content/build-parity3/bin/llama-server \
  -m /content/fn40/real/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf \
  --alias q -ngl 99 --n-cpu-moe 2 --moe-cache-mib 6144 \
  -c 32768 -b 2048 -ub 1024 -fa on \
  -ctk q8_0 -ctv q8_0 \
  --load-mode mmap --lazy-mode on --jinja --no-warmup \
  --host 127.0.0.1 --port 8090
```

Carga (mesmos prompts/casos de `/tmp/opencode/vm_bench_parity.py`, `cache_prompt: false`,
`temperature 0`, `enable_thinking: false`; pp28k = `LP30`, 16 tokens; pp28k+dec = 128):

```python
import json, urllib.request
FILLER = ("def process(rows):\n    out = []\n    for row in rows:\n"
          "        if row is None:\n            continue\n"
          "        out.append((row[0], sum(row[1:])))\n    return out\n\n")
LP30 = FILLER*620 + "\nSummarize the code above in one sentence."
def chat(prompt, n):
    body = json.dumps({"model": "q", "messages": [{"role": "user", "content": prompt}],
        "max_tokens": n, "temperature": 0, "stream": False, "cache_prompt": False,
        "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request("http://127.0.0.1:8090/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    t = json.loads(urllib.request.urlopen(req, timeout=1800).read())["timings"]
    return round(1000*t["prompt_n"]/t["prompt_ms"], 1), round(1000*t["predicted_n"]/t["predicted_ms"], 2)
print("pp28k", chat(LP30, 16))
print("pp28k+dec", chat(LP30 + " Explain the time complexity.", 128))
```

VRAM: amostrar `nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits` a 1 Hz
durante a carga (como no harness) e guardar o pico.

### Critério de aceite

1. **pp28k >= 1.666 t/s** (>= +10% sobre 1.514 t/s), mediana de 2 runs; spread <= 2%.
2. **decode pós-28k (tg de 128 tokens em pp28k+dec) >= 97% do baseline pareado** —
   >= 42,2 t/s se o baseline for 43,53 t/s; sem regressão > 3%.
3. **VRAM pico <= 40.900 MiB** (`nvidia-smi`).
4. **Sem abort do cache**: ausência de `GGML_ABORT`, `the MoE cache cannot plan` e
   `selected in the same split` no log do servidor; processo vivo nos 5 casos.
5. Evidência positiva: a linha `large batch of ... uses the MoE cache` aparece no pp28k
   com `n_group_layers*n_expert <= n_slots`; no start, a linha do grupo mostra
   `100.0%` de slots; com `kill -TERM` (o harness usa SIGKILL e não imprime stats), a
   linha final `ubatch > 8: ... uploaded = X MiB` deve mostrar ~1,5 GB no total (primeiro
   ubatch) em vez de dezenas de GB.

Sanidade da meta: 28,6k tokens a 1.514 t/s = 18,9 s; os 39,7 GB de H2D serializados valem
1,7 s a 24 GB/s (pinned) ou 3,3 s a 12 GB/s, ou seja +9% a +21%. Um resultado nulo
significa que o H2D já estava sobreposto ao compute; nesse caso o próximo passo é o
overlap H2D/lookup, não mexer no gate.

## O que não foi validado

- Runtime GPU (por instrução): correção do caminho CUDA do banco com ids de 1024 tokens,
  pré-condição de split e hit rate real só saem do A/B acima.
- `n_expert`/`n_slots` reais do Qwen3.8-Flash-Next não foram assumidos: o log novo imprime
  os valores e o critério usa a relação `n_layers*n_expert <= n_slots`.
