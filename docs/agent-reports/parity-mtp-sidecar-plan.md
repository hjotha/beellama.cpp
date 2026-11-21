> Historical MTP sidecar research. The working path is -md + --spec-type draft-mtp with the unsloth Q4_K_M sidecar; the merge-into-shards route is a no-go.
>
> Historical agent report (2026-11-21). Current status and numbers: docs/PLAN-strata-parity.md.

# Plano: bloco NextN/MTP para o Qwen3.8-Flash-Next Q2_0 (beellama)

Status: **pesquisa read-only** (2026-11-21). Nada foi compilado, baixado (>500 MB) ou executado
com GPU. Repo não foi tocado. Branch de referência: `feat/strata-parity` @ `5f46a18da`
(worktree `/home/hjotha/worktrees/beellama-strata-parity`).

## 0. Decisão (TL;DR)

**Não converter, não fundir, não re-quantizar.** Já existe um GGUF de sidecar MTP pronto,
gerado do mesmo checkpoint base, com exatamente os nomes que o loader do fork espera:

```
unsloth/Qwen3.8-Flash-Next-GGUF  MTP/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf
  1.907.151.936 B  sha256 f521868a9e143718bef513772f6e04d9642551e362cf2439636d2abdbd149dfc
```

Ele tem `qwen4exp.block_count = 49`, `qwen4exp.nextn_predict_layers = 1`, 32 tensores
`blk.48.*` (6 deles `blk.48.nextn.*`) e **não** tem `token_embd.weight`/`output.weight`
(variante "shared": o fork empresta os dois do modelo alvo via `ctx_other`). O fork já
suporta esse caminho: `-md <sidecar> --spec-type draft-mtp`.

Custo: **1,9 GB de download**, zero conversão, zero alteração no Q2_0 publicado.
Verificação estática feita neste plano (seções 1–5); falta só o smoke/medição na A100.

Fallback (só se a aceitação for ruim): converter o bloco MTP do checkpoint com o conversor do
fork (`--remote --mtp`, 5,21 GB de download) e/ou quantizar para Q2_0 (seção 4).

---

## 1. Onde estão os pesos MTP (verificado via HF API + HTTP Range)

### 1.1 Checkpoint

- Repo: **`Qwen/Qwen3.8-Flash-Next`**, revision `812d96e10a0dd82b69d749e68693698cf6c36daa`.
- **Não é gated** (`"gated": false` no `https://huggingface.co/api/models/Qwen/Qwen3.8-Flash-Next`),
  não precisa de login/token.
- Formato: `model.safetensors.index.json` (170.726 B, 1658 tensores) + **131 shards**
  `model-00001-of-00131.safetensors` … `model-00131-of-00131.safetensors`,
  `total_size = 359.999.963.128` B (~335,3 GiB em disco), BF16.
- `config.json` → `text_config`: `num_hidden_layers = 48`, `hidden_size = 2560`,
  `mtp_num_hidden_layers = 1`, `mtp: {hybrid: true, layer_types: ["full_attention"],
  num_hidden_layers: 1}`, `num_experts = 512`, `num_experts_per_tok = 10`,
  `moe_intermediate_size = 640`, `hc_count = 4`, `hc_lowrank = 320`.
- Tensores MTP: prefixo **`mtp.`** (não `model.mtp.` neste checkpoint).

### 1.2 Onde estão os 31 tensores `mtp.*`

`model.safetensors.index.json` → `weight_map`:

- **31 tensores** `mtp.*`, espalhados por **28 shards** distintos (quase um tensor por shard).
- Payload total: **5.214.301.696 B = 4,86 GiB** (BF16).
- Baixar os 28 shards inteiros custaria **51,38 GiB**; o checkpoint inteiro, 335,3 GiB.
- **Dá para baixar só os tensores**: `https://huggingface.co/<repo>/resolve/<rev>/<shard>`
  responde **HTTP 206** a `Range` (mesmo sendo Xet/CDN, testado: `content-range: bytes 0-7/1900676064`).
  Custo mínimo real: 5,21 GB + ~5 MB de headers (2 requests por shard: 8 bytes + header JSON).

Mapa completo (nome → shard → offset → bytes), gerado por leitura dos headers dos 28 shards:

| tensor | dtype | shape | shard | offset | bytes |
|---|---|---|---|---|---|
| `mtp.fc_embedding.weight` | BF16 | 2560x2560 | 00120 | 1.935.408.064 | 13.107.200 |
| `mtp.fc_hidden.weight` | BF16 | 2560x2560 | 00124 | 1.716.597.760 | 13.107.200 |
| `mtp.hyper_connection_mixer.hc_norm.weight` | BF16 | 10240 | 00037 | 1.742.323.904 | 20.480 |
| `mtp.hyper_connection_mixer.input_mix_weight_down.weight` | BF16 | 320x10240 | 00041 | 1.935.408.064 | 6.553.600 |
| `mtp.hyper_connection_mixer.input_mix_weight_up.weight` | BF16 | 10240x320 | 00043 | 1.832.515.008 | 6.553.600 |
| `mtp.layers.0.attn_hyper_connection.block_inject_weight.weight` | BF16 | 4x10240 | 00047 | 1.716.597.760 | 81.920 |
| `mtp.layers.0.attn_hyper_connection.hc_norm.weight` | BF16 | 10240 | 00049 | 1.935.408.064 | 20.480 |
| `mtp.layers.0.attn_hyper_connection.input_mix_weight_down.weight` | BF16 | 320x10240 | 00053 | 1.832.515.008 | 6.553.600 |
| `mtp.layers.0.attn_hyper_connection.input_mix_weight_up.weight` | BF16 | 10240x320 | 00057 | 1.935.408.064 | 6.553.600 |
| `mtp.layers.0.mlp.experts.down_proj` | BF16 | 512x2560x640 | 00062 | 1.832.515.008 | 1.677.721.600 |
| `mtp.layers.0.mlp.experts.gate_up_proj` | BF16 | 512x1280x2560 | 00060 | 115.917.152 | 3.355.443.200 |
| `mtp.layers.0.mlp.gate.weight` | BF16 | 512x2560 | 00066 | 1.716.597.760 | 2.621.440 |
| `mtp.layers.0.mlp.shared_expert.down_proj.weight` | BF16 | 2560x640 | 00076 | 1.935.408.064 | 3.276.800 |
| `mtp.layers.0.mlp.shared_expert.gate_proj.weight` | BF16 | 640x2560 | 00072 | 1.832.515.008 | 3.276.800 |
| `mtp.layers.0.mlp.shared_expert.up_proj.weight` | BF16 | 640x2560 | 00072 | 1.835.791.808 | 3.276.800 |
| `mtp.layers.0.mlp.shared_expert_gate.weight` | BF16 | 1x2560 | 00070 | 1.832.515.008 | 5.120 |
| `mtp.layers.0.mlp_hyper_connection.block_inject_weight.weight` | BF16 | 4x10240 | 00078 | 1.832.515.008 | 81.920 |
| `mtp.layers.0.mlp_hyper_connection.hc_norm.weight` | BF16 | 10240 | 00082 | 1.935.408.064 | 20.480 |
| `mtp.layers.0.mlp_hyper_connection.input_mix_weight_down.weight` | BF16 | 320x10240 | 00086 | 1.935.408.064 | 6.553.600 |
| `mtp.layers.0.mlp_hyper_connection.input_mix_weight_up.weight` | BF16 | 10240x320 | 00088 | 1.832.515.008 | 6.553.600 |
| `mtp.layers.0.self_attn.indexer.index_qk_proj.weight` | BF16 | 640x2560 | 00090 | 1.832.515.008 | 3.276.800 |
| `mtp.layers.0.self_attn.indexer.k_layernorm.weight` | BF16 | 128 | 00094 | 1.935.408.064 | 256 |
| `mtp.layers.0.self_attn.indexer.q_layernorm.weight` | BF16 | 128 | 00100 | 1.716.597.760 | 256 |
| `mtp.layers.0.self_attn.k_norm.weight` | BF16 | 256 | 00102 | 1.935.408.064 | 512 |
| `mtp.layers.0.self_attn.k_proj.weight` | BF16 | 512x2560 | 00108 | 1.832.515.008 | 2.621.440 |
| `mtp.layers.0.self_attn.o_proj.weight` | BF16 | 2560x6144 | 00106 | 1.832.515.008 | 31.457.280 |
| `mtp.layers.0.self_attn.q_norm.weight` | BF16 | 256 | 00112 | 1.935.408.064 | 512 |
| `mtp.layers.0.self_attn.q_proj.weight` | BF16 | 12288x2560 | 00108 | 1.835.136.448 | 62.914.560 |
| `mtp.layers.0.self_attn.v_proj.weight` | BF16 | 512x2560 | 00108 | 1.898.051.008 | 2.621.440 |
| `mtp.pre_fc_norm_embedding.weight` | BF16 | 2560 | 00114 | 1.832.515.008 | 5.120 |
| `mtp.pre_fc_norm_hidden.weight` | BF16 | 10240 | 00118 | 1.716.597.760 | 20.480 |

(Os offsets são **relativos ao bloco de dados**, como no `data_offsets` do safetensors. Para
fazer `Range` no arquivo, some `8 + header_len` — `header_len` são os primeiros 8 bytes
little-endian do shard. Ex.: `mtp.fc_embedding.weight` está em
`00120`, offset de arquivo `1.935.408.064 + 8 + header_len`.)

### 1.3 Atalho: sidecars MTP já publicados

| repo/arquivo | bytes | uso |
|---|---|---|
| `unsloth/.../MTP/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf` | **1.907.151.936** | **recomendado** (32 tensores, shared) |
| `unsloth/.../MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` | 2.786.568.256 | +qualidade, +VRAM |
| `unsloth/.../MTP/mtp-Qwen3.8-Flash-Next-shared-BF16.gguf` | 5.227.963.456 | fonte p/ re-quantizar |
| `unsloth/.../MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf` | 4.137.429.120 | self-contained (traz `output.weight`) |
| `freriadoc/.../MTP/mtp-...-shared-IQ3K-IQ4NL-FR65k.gguf` | 1.484.237.088 | **não usar**: tipos/ids do fork Avalon |

O `MTP/README.md` do repo da unsloth documenta exatamente `-md ... --spec-type draft-mtp`
e diz que "shared heads borrow the token embedding and output projection from the model you
are already running".

---

## 2. Conversão (só como fallback)

O conversor do fork suporta exportar só o MTP:

- `conversion/qwen4exp.py`: `supports_mtp_export = True`, `_MTP_EXTRA` mapeia
  `fc_embedding→nextn_fc_embedding`, `fc_hidden→nextn_fc_hidden`,
  `hyper_connection_mixer→nextn_hc_head`; `modify_tensors` funde os dois `fc_*` em
  `nextn.eh_proj` e divide `index_qk_proj` em `indexer.q_proj`+`indexer.k_proj`.
- `conversion/qwen.py::_QwenMtpMixin` remapeia `mtp.layers.0.*` → `model.layers.48.*`,
  soma 1 em `block_count` e escreve `qwen4exp.nextn_predict_layers`.
- `convert_hf_to_gguf.py --mtp` (`mtp_only`) filtra tudo que não é MTP.

Comando (baixa só os tensores filtrados; `--remote` lê os headers dos 131 shards e busca
cada tensor por Range):

```bash
cd /home/hjotha/worktrees/beellama-strata-parity
python3 convert_hf_to_gguf.py Qwen/Qwen3.8-Flash-Next --remote --mtp --outtype bf16 \
  --outfile /models/mtp-qwen38-bf16.gguf
```

Custo: 5,21 GB (MTP) **+ 1,27 GB** (`lm_head.weight`, porque o keep-list de `mtp_only`
inclui `lm_head.weight`; o `token_embd` do checkpoint se chama
`model.language_model.embed_tokens.weight` e é descartado) → **~6,5 GB** de saída.
Metadados gerados: `block_count=49`, `nextn_predict_layers=1`,
`attention.compress_ratios[48]=4` (**QSA ligado**, diferente do sidecar da unsloth, que grava 0).

Para gerar um sidecar "shared" de verdade (sem `output.weight`), o patch mínimo é em
`conversion/qwen4exp.py::filter_tensors`, antes do `super()`: quando `cls.mtp_only` e o nome
for `lm_head.weight`, retornar `None`. Sem esse patch, o draft usa o próprio LM head
(bf16/Q2_0) em vez do Q2_0 GSQ do alvo — funciona, mas gasta ~1,27 GB e muda os logits.

Alternativa ainda mais barata que converter: **patch in-place** do sidecar da unsloth,
trocando `qwen4exp.attention.compress_ratios[48]` de `0` para `4` (mesmo tamanho, sem mexer
em offsets). Só faz sentido se a aceitação com o arquivo original ficar baixa; o checkpoint
tem `mtp.layers.0.self_attn.indexer.*`, então QSA é o que o modelo treinou.

---

## 3. Fundir no GGUF Q2_0 publicado (possível, mas **não recomendado**)

### 3.1 Estado real dos shards

| | shard 1 | shard 2 |
|---|---|---|
| arquivo | `Q2_0/...-00001-of-00002.gguf` | `Q2_0/...-00002-of-00002.gguf` |
| bytes | 37.623.740.192 | 28.800.138.432 |
| tensores | 1223 | 1 (`per_layer_token_embd.weight`, IQ4_NL) |
| `split.no` / `split.count` | 0 / 2 | 1 / 2 |
| `split.tensors.count` | 1224 | 1224 |
| data offset / alignment | 11.024.672 / 32 | 192 / 32 |
| `qwen4exp.block_count` | 48 | — |
| `qwen4exp.attention.compress_ratios` | 48 entradas | — |
| `qwen4exp.nextn_predict_layers` | **ausente** | — |

Tipos de tensor no shard 1: `{13:16, 30:483, 0:292, 11:91, 12:38, 23:56, 42:202, 6:7, 1:1, 2:16, 14:12, 20:7, 8:2}`
— 202 tensores no tipo **42 = Q2_0** (GSQ/RCO), que o `gguf-py` do fork conhece
(`gguf/constants.py:6662`), então um rewriter em Python consegue copiar os bytes crus.

### 3.2 O que teria que mudar

No shard 1 (metadados): `block_count 48→49` (u32, in-place), **adicionar**
`qwen4exp.nextn_predict_layers=1` (u32) e **estender** `compress_ratios` de 48 para 49
entradas (o loader exige comprimento exato: `get_key_or_arr(..., n_layer_all)` →
`key ... has wrong array length; expected 49, got 48`, `src/llama-model-loader.cpp:498`).
Adicionar uma KV e crescer um array **aumenta o header** → todos os offsets de tensores do
shard precisam ser rebaseados e os 37,6 GB de dados copiados para o novo offset.
Nos dois shards: `split.tensors.count 1224→1256` (u64, in-place).
Nos 32 tensores novos: `blk.48.attn_*`, `blk.48.ffn_*`, `blk.48.hc_*`,
`blk.48.indexer.*`, `blk.48.nextn.{eh_proj,enorm,hnorm,hc_head_norm,hc_head_down,hc_head_up}`
(copiados crus do sidecar, alinhados a 32).

O loader confere `split.tensors.count` contra o total registrado
(`src/llama-model-loader.cpp:725-731`, erro `corrupted model: %d tensors expected but %d found`)
e exige `split.no=0` no arquivo principal.

### 3.3 Ferramentas existentes

- `tools/gguf-split --merge`: junta shards, **não adiciona tensores**.
- `gguf-py/gguf/scripts/gguf_new_metadata.py`: copia metadados, não mexe em tensores.
- `gguf_set_metadata.py`: altera KV reescrevendo o arquivo inteiro; não adiciona tensor.
- `gguf_editor_gui.py` / `gguf_dump.py`: inspeção/edição de KV, não de tensor.
- **Não existe** utilitário "append tensor + rewrite offsets" no repo.

### 3.4 Utilitário mínimo (se alguém insistir)

Python com `gguf-py` (`GGUFReader`/`GGUFWriter`, copiando `tensor.data` cru):
1. ler shard 1 + shard 2 + sidecar;
2. novo shard 1: KVs com as 3 mutações acima, 1223 tensor infos + 32 novas, offsets
   rebaseados por `delta = novo_data_offset - 11.024.672`, dados copiados em streaming,
   depois os 32 tensores alinhados a 32;
3. patch de `split.tensors.count` no shard 2 (mesmo tamanho, sem shift);
4. validar com o parser da seção 6.1 e com o loader (`--check-tensors`).

Custo: ~38,5 GB livres, 37,6 GB lidos + ~38,5 GB escritos (≈5–20 min em NVMe), e **risco
real de corromper o único arquivo de produção**. Alternativa "terceiro shard"
(`split.count=3`, renomear para `-of-00003`, `split.no=2` no sidecar) evita mexer nos dados
do shard 2, mas **não** evita o rewrite do shard 1, porque os metadados do alvo continuam
tendo que mudar (block_count/nextn/compress_ratios).

**Conclusão:** o sidecar entrega o mesmo resultado sem tocar no Q2_0. Merge = no-go.

---

## 4. Quantização

- O loader **não exige** que o tipo do MTP case com o do tronco: ele valida shape
  (`create_tensor` com `ne[]`) e `ggml_validate_row_data`; o tipo vem do próprio tensor.
  O grafo usa `build_lora_mm`, que aceita qualquer tipo suportado pelo backend.
- Q2_0 (tipo 42, 2,25 bpw, bloco 64) é suportado como peso e é o tipo dos 202 tensores de
  experts do alvo. Tamanho esperado do bloco MTP em Q2_0 ≈ **750–800 MB**
  (3 matrizes 640×2560×512 × 2,25/8 ≈ 708 MB + atenção/normas).
- `llama-quantize` **aceita** um GGUF só-MTP: `llama-quant.cpp:946` usa
  `n_layer_all` (=49) como limite de camada, então `blk.48.*` não cai no erro
  "Bad layer"; `llama-quant.cpp:1007` carrega com `load_mtp=true`.
- Produzir Q2_0 a partir do BF16 (não re-quantizar Q4_K_M → Q2_0):

```bash
./build-cpu/bin/llama-quantize /models/mtp-qwen38-bf16.gguf \
  /models/mtp-qwen38-q2_0.gguf Q2_0
# controle fino por tensor:
#   --tensor-type 'blk\.48\.ffn_(up|down|gate)_exps=Q4_K'
#   --tensor-type 'blk\.48\.nextn\.=Q8_0'
```

- Trade-off: o README da unsloth mede ~2 pontos de aceitação a menos de Q4_K_M para Q8_0;
  Q2_0 tende a piorar mais. Como o draft é verificado pelo alvo, isso afeta **velocidade**,
  não corretude. Começar com Q4_K_M published (1,9 GB); Q2_0 só se a VRAM apertar.

---

## 5. Sidecar: **sim, é suportado** (e é o caminho do fork)

- `-md/--spec-draft-model` + `--spec-type draft-mtp` → `common_speculative_init_result`
  carrega o arquivo como **modelo draft separado**, cria o contexto com
  `ctx_type = LLAMA_CONTEXT_TYPE_MTP` e `ctx_other = ctx_tgt`
  (`common/speculative.cpp:5930-5981`).
- `src/models/qwen4exp.cpp` foi escrito para isso: se `model.tok_embd`/`model.output` são
  nulos, `qwen4exp_shared_model(cparams, model, ...)` empresta do modelo do `ctx_other`
  (linhas 663-680 e 755-760); se o sidecar for usado sozinho (sem `-md`), o grafo lança
  `"this draft head has no 'token_embd' of its own; load it as a draft of its target model (-md)"`.
  O sidecar da unsloth traz `qwen4exp.nextn_shared_target_tensors = True`, chave que o fork
  **não conhece e ignora**; o empréstimo funciona mesmo assim porque `tok_embd`/`output`
  simplesmente não existem no arquivo.
- `nextn_flags()` (`src/llama-model.cpp:5233`) detecta "arquivo só-MTP" pela ausência de
  `blk.0.hc_attn_norm.weight` e marca o tronco como `TENSOR_NOT_REQUIRED`; os 32 tensores de
  `blk.48` são obrigatórios e existem.
- Memória do draft: `llama-model.cpp:3979-3985` filtra atenção/recorrente/indexer para
  `il >= n_layer` no contexto MTP → só a camada 48 aloca KV (QSA/idx incluso, ~16 MB a 32k).
- **O que NÃO existe:** carregar o MTP como *split weights* do alvo. `split_mtp_weights` é
  só para `qwen35`/`sensenova_loop` (`llama-model.cpp:1721` lança
  `"separate MTP weights require an allocated qwen35 target with embedded MTP"`).
- **Importante:** o cache de experts compartilhado da branch de impl
  (`/tmp/opencode/impl-mtp`) exige `&params.ctx_other->model == &model`
  (`src/llama-context.cpp:1071-1073`), ou seja **só vale para MTP embutido no mesmo modelo**.
  Com `-md` o draft **não** compartilha o `llama_moe_cache`; portanto mantenha os experts do
  `blk.48` na GPU (1,8 GB) ou aceite upload por passo. `-otd` só compensa depois de
  implementar cache por (modelo, camada).

---

## 6. Plano executável

### 6.1 Pré-flight do sidecar (sem build, sem GPU)

```bash
cd /models
curl -L -o mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf \
  "https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/main/MTP/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf"
sha256sum mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf
# esperado: f521868a9e143718bef513772f6e04d9642551e362cf2439636d2abdbd149dfc
#           1907151936 bytes

python3 - <<'EOF'
import struct
p='mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf'
f=open(p,'rb'); assert f.read(4)==b'GGUF'
ver=struct.unpack('<I',f.read(4))[0]; nt=struct.unpack('<Q',f.read(8))[0]; nkv=struct.unpack('<Q',f.read(8))[0]
SZ={0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'Q',11:'q',12:'d'}
def rs():
    n=struct.unpack('<Q',f.read(8))[0]; return f.read(n).decode()
def rv(t):
    if t==8: return rs()
    if t==9:
        at=struct.unpack('<I',f.read(4))[0]; n=struct.unpack('<Q',f.read(8))[0]
        return [rv(at) for _ in range(n)]
    fmt=SZ[t]; return struct.unpack('<'+fmt,f.read(struct.calcsize(fmt)))[0]
kv={}
for _ in range(nkv):
    k=rs(); t=struct.unpack('<I',f.read(4))[0]; kv[k]=rv(t)
names=[]
for _ in range(nt):
    nm=rs(); nd=struct.unpack('<I',f.read(4))[0]; f.read(8*nd); f.read(12); names.append(nm)
print('version',ver,'tensors',nt)
print('block_count',kv.get('qwen4exp.block_count'),'nextn',kv.get('qwen4exp.nextn_predict_layers'))
print('nextn tensors',sum('.nextn.' in n for n in names),'blk.48',sum(n.startswith('blk.48.') for n in names))
print('has tok_embd/output', any(n in ('token_embd.weight','output.weight') for n in names))
print('compress_ratios[48]', kv['qwen4exp.attention.compress_ratios'][-1])
EOF
# esperado: tensors 32 | block_count 49 | nextn 1 | nextn tensors 6 | blk.48 32
#           has tok_embd/output False | compress_ratios[48] 0
```

### 6.2 Smoke (A100-40G; **não** rodar na `.57` durante o build CUDA)

Mesmo placement da produção, contexto pequeno só para validar load + MTP:

```bash
cd /home/hjotha/beellama.cpp           # ou a worktree com build-a100
./build-a100/bin/llama-server \
  -m /models/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf \
  -md /models/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf \
  -c 4096 -b 512 -ub 128 -fa on \
  --load-mode mmap --lazy-mode on -ctk q8_0 -ctv q8_0 \
  -ncmoe 2 --moe-cache-mib 1024 \
  --spec-type draft-mtp --spec-draft-n-max 2 \
  --spec-draft-type-k q8_0 --spec-draft-type-v q8_0 \
  -np 1 --host 127.0.0.1 --port 8190 --metrics
```

(`-ncmoe 2` + `--load-mode mmap --lazy-mode on` é o placement de produção do
`IMPL_REPORT` §6.2; sem `-ngl 99`, que não cabe em 40 GB.)

Logs que provam o caminho:
- `llama_model_load: n_layer_nextn         = 1` (draft) — e `n_layer_all = 49`;
- `common_speculative_init_result: loading draft model '/models/mtp-...gguf'`;
- **sem** `failed to create draft context` e **sem** o `WRN` `declares no NextN/MTP block`;
- `llama_context: MTP draft context ...` / `MoE cache auto: reserving ... of draft-MTP KV`.

Se o load falhar antes de qualquer decode, o problema é de metadata/nome de tensor (volte ao
6.1), não do grafo.

### 6.3 Produção + medição

```bash
./build-a100/bin/llama-server \
  -m /models/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf \
  -md /models/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf \
  -c 32768 -b 2048 -ub 512 -fa on \
  --load-mode mmap --lazy-mode on \
  -ctk q8_0 -ctv q8_0 \
  -ncmoe 2 --moe-cache-mib -1 --vram-reserve-mib 4096 \
  --spec-type draft-mtp --spec-draft-n-max 4 \
  --spec-draft-type-k q8_0 --spec-draft-type-v q8_0 \
  -np 1 --host 127.0.0.1 --port 8190 --metrics
```

- `--vram-reserve-mib 4096` (em vez de 1536) porque o auto-size do cache **não conhece** os
  ~1,9 GB de pesos do draft (ele só reserva KV+compute do draft na branch de impl).
  Ajustar depois de ver `MoE cache auto: <M> MiB for <H> MiB of host experts` e
  `nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits`.
- O draft herda `-ngld auto` (−1) → todos os 32 tensores na GPU. **Não** usar `-otd` para
  `blk\.48\.ffn_(up|down|gate)_exps` sem o cache compartilhado (ele não cobre modelo
  separado, seção 5): viraria upload por passo.

Critérios de aceite (do `IMPL_REPORT` §6.3):
- `/metrics`: `llamacpp:spec_decode_num_drafts_total > 0` e
  `llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position="0".."3"}` com pos-0 > 0;
- +20% decode vs. controle `--spec-type none` em ≥2 de 3 prompts, pisos 65 / 55 / 54 t/s;
- pico ≤ 39,4 GiB; `/health` 200; 3 rodadas sem `GGML_ASSERT`/OOM;
- respostas greedy idênticas ao controle.

### 6.4 Riscos e go/no-go

| risco | mitigação |
|---|---|
| OOM ao criar o draft (1,9 GB não reservados) | `--vram-reserve-mib 4096` ou `--moe-cache-mib` fixo; medir com `nvidia-smi` |
| Aceitação baixa por `compress_ratios[48]=0` (QSA desligado) no arquivo da unsloth | A/B com patch in-place 0→4 (§2) ou reconverter com `--mtp` (ratio 4) |
| `blk.48` com QSA ligado e cache idx não alocado | só ocorre no patch 0→4; o fork aloca idx para `il >= n_layer` |
| Draft lento por LM head/tok_embd emprestados do alvo Q2_0 | é o desenho "shared"; comparar com `shared-Q8_0` |
| Cache de experts não compartilhado com `-md` | manter experts do bloco na GPU (default) |
| Branch sem os fixes de `/tmp/opencode/impl-mtp` ainda aborta em `failed to create MTP context` | só ocorre quando o MTP pedido não existe (target sem bloco); com sidecar válido o contexto é criado |

**Go/no-go:** GO se 6.1 passar e o server logar `n_layer_nextn = 1` + drafts > 0.
NO-GO para o merge no Q2_0 (§3) — risco desnecessário. Se a aceitação ficar abaixo do
critério, o próximo passo é o fallback da §2 (conversão QSA-on) e, só depois, Q2_0.

### 6.5 Rollback

Nada em `/models/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-*` é alterado. Basta remover `-md` e
`--spec-type draft-mtp` para voltar ao baseline (o server já sobe sem especulação quando não
há NextN, com o guard da branch de impl; em `5f46a18da` o baseline é o mesmo comando sem
essas duas flags).
