# Decode KVarN4 row-per-lane (nq=1)

2026-10-26. Clone `decode-kernel`, branch `perf/rkva-decode-rowlane-20261026`,
base `8ac3e9d10`. Implementação experimental opt-in; **sem medição de GPU ou
qualificação UWP/MSVC nesta entrega**. Shader: commit `75e7c7d2f`; integração e
perfil: `a3c38e267`. Não houve push, merge, build CUDA/MSVC/UWP ou execução de
serviços. DXC foi usado somente pelo wrapper `.oc-artifacts/dxc/dxc193.py`, em
`C:/Users/hjotha/build/oc-decode-kernel-20261026` no host 193.

## Implementação e avaliação do desenho

- Alvo `attn_kvarn4_decode_rowlane`, entry `CSRangeAttnKvarn4Decode`, SM 6.0,
  defines `RKVA_SERIAL_PREFIX=1`, `RKVA_DECODE_OPT=1`, `RKVA_DECODE_ROWLANE=1`.
  Sem `-enable-16bit-types`.
- Um grupo de 128 threads percorre um grupo de 128 registros; cada wave64
  cobre 64 posições, uma por lane. As seis cabeças compartilham os nibbles de K.
- **`other_k[pos]` difere entre os dois slices físicos.** Calculamos separadamente
  `other_s[p] * (sum_d Q_s[d]*scale_s[d]*nib_s[d,p] + Z_s)` para cada slice de
  128 dimensões, depois somamos e aplicamos `kq_scale`. Um único `other` para
  256 dimensões seria incorreto.
- `Q*scale` permanece em registradores distribuídos por dimensão. A wave faz
  broadcasts com `WaveReadLaneAt(qs[g].component, slice*32+du)`, cujo índice é
  uniforme. PV usa `WaveReadLaneAt(weight[g], t+j)`, também uniforme. Não existe
  shuffle com índice dependente da lane. O teste de auditoria verifica as
  dependências SSA dos índices no DXIL, com controle negativo lane-dependent.
- QK percorre quatro dimensões por iteração; PV desenrola oito linhas, carregando
  os oito scales/zeros com `Load4`. Não há LDS nem barreiras **dentro do loop de
  registros**. A LDS de 12 KiB do kernel anterior continua no WHT e no merge.
- A primeira variante com Q escalado/pesos em LDS foi descartada: sua contagem
  ponderada era maior que a do baseline. Broadcasts uniformes e PV em blocos
  de oito reduziram esse custo. **A expectativa de ~80 instruções por linha e
  ~115 us não foi confirmada pela contagem; não é uma previsão desta entrega.**
- Sink, stage, bordas de grupo e tail F16 original usam as fases existentes.
  Não mudou o contrato `[split][qh][258]`, LSE natural, WHT ou reducer.
  Split policy é a mesma do decode existente: em 28672 linhas, modo 2 usa
  56 splits / 512 linhas e modo 1 usa 32 splits / 896 linhas.

## Seleção e observabilidade

Criar `LocalState/rkva-decode-rowlane.flag` e abrir uma **nova conexão HELLO**.
A flag é desligada por padrão. Seleção exige descriptor 3, nq=1, modos 1/2,
shape KVarN4 já qualificado, wave min=max=64 e PSO disponível. Mode 0 continua
sendo referência; nq>1 continua nos PSOs anteriores. Falta/falha do novo PSO
mantém o fallback existente.

Perfil JSON do engine:

- `kvarn4_decode_rowlane.{available,enabled,status,calls}`;
- bucket `attention/kvarn4-prefix/optimized-kvarn4_decode_rowlane/nq1/...`;
- contador próprio é um subconjunto do agregado `kvarn4_decode.calls`;
- `BulkSegmentProfile.decode_rowlane` é preenchido pelo dispatch **e copiado
  para `attn_sample.decode_rowlane` em `finish_bulk_profile`**, que também
  incrementa o contador próprio. Há regressão de fonte para essa cópia e teste
  C++ de bucket/JSON/reset, incluindo Sample marcado como segmento BULK.

`rkva_pass1_sweep.py --nq 1 --expect-decode-kernel kvarn4_decode_rowlane` exige
esse bucket e rejeita fallback. A opção também fica nos metadados do resultado.

## DXIL e contagem reproduzível

Recompilados os três alvos que compartilham a fonte alterada: `attn_kvarn4_opt`,
`attn_kvarn4_opt_prefix`, `attn_kvarn4_decode_prefix`. Comparados os **98 headers
preexistentes** com a base: todos byte-idênticos exceto as três linhas de hash
da fonte. Os demais alvos/fontes/defines não mudaram.

Novo payload: **49696 bytes**, SHA256
`0668df8afce0288fae460a2bed5e0922b88d80a07a415be1a706927fc22fa644`.
O header também registra o hash do include `rkva_decode_rowlane.hlsli`.

`rkva_decode_rowlane_cost.py` conta o loop natural com multiplicidade real:
QK = dois loops de 32 unidades; PV = oito iterações de oito linhas. Normaliza
por 64 linhas/wave; o baseline por oito. Custo DXIL: operação SSA=1, dot4=4,
WaveActiveOp=12. Inclui LDS, FMad, PHI, endereços, controle e os dois braços de
branches de softmax. É contagem **pré-ISA**, não tempo, VGPRs ou ocupação.

| Por linha KV | Decode anterior | Rowlane |
| --- | ---: | ---: |
| Unidades DXIL ponderadas, metadados carregados | 281.875 | 188.375 |
| Baseline com metadados amortizados por grupo | 276.296875 | 188.375 |
| WaveActiveOp por linha | 6 | 0.375 |
| Broadcasts uniformes por linha | 0 | 30 |
| Loads LDS no loop | 0 | 0 |
| Barreiras no loop | 0 | 0 |

Redução ponderada amortizada: **1.467x**. A convenção aproximada que duplica
as unidades para as duas metades de wave64 dá **552.59 -> 376.75**, próxima
dos ~550 do diagnóstico inicial. Essa duplicação não modela fielmente cada
instrução escalar/VALU; comparar as duas implementações com a mesma convenção.
O custo das reduções cai muito, mas broadcasts, endereçamento e PV continuam
relevantes. O modelo não autoriza afirmar speedup de 1.467x no console.

```sh
# Todos os outputs locais permanecem no clone.
mkdir -p .oc-artifacts/rowlane/tmp
export TMPDIR="$PWD/.oc-artifacts/rowlane/tmp"
export DXC="$PWD/.oc-artifacts/dxc/dxc193.py"
tools/xbox/scripts/compile-rkva-range-shaders.sh attn_kvarn4_decode_rowlane
"$DXC" -T cs_6_0 -E CSRangeAttnKvarn4Decode \
  -D RKVA_SERIAL_PREFIX=1 -D RKVA_DECODE_OPT=1 -D RKVA_DECODE_ROWLANE=1 \
  -Fc tools/xbox/shaders/rkva/rkva_range_attn_kvarn4_opt.hlsl \
  -Fo .oc-artifacts/rowlane/rowlane.dxil
python3 tools/xbox/tests/rkva_decode_rowlane_dxil_audit.py .oc-artifacts/rowlane/rowlane.txt
python3 tools/xbox/tests/rkva_decode_rowlane_cost.py \
  .oc-artifacts/decode.txt .oc-artifacts/rowlane/rowlane.txt
```

Auditoria: **96 chamadas estáticas / 10 expressões de índice, todas uniformes**;
controle com índice derivado de `WaveGetLaneIndex` rejeitado.

## Prova numérica CPU e regressões

Modelo novo `rkva_decode_rowlane_model.py`: transliteração FP32 do algoritmo,
**não um interpretador DXIL nem uma execução GPU**. Lê os bytes residentes
pelos loaders com verificação de limites; preserva ordem QK/PV, splits, waves,
softmax, WHT e merge. `mad` QK é explicitamente fusionado; executamos também as
duas opções de contração de fmul/fadd fast no restante. Oráculo independente
`serial_sources` + `oracle_attention` em FP64, o mesmo usado pelo probe;
comparação usa diretamente `rkva_kvarn_serial_prefix_probe.compare`:

- OUT: `abs(error) <= 3e-4 + 3e-4*abs(oracle)`;
- LSE: `abs(error) <= 3e-4 + 3e-5*abs(oracle)`; finitude deve coincidir.

**65 casos x dois modos**, com e sem contração, passaram. Maior erro absoluto:
OUT **1.27561606e-4**, LSE **1.40178712e-5**. Em 28672 linhas: OUT **5.90353e-5**
(com contração), **5.23596e-5** (sem), LSE **7.29234e-6**. Cobertura: origens 0 e
72704; tail 0/128; limites 127/128/129, 255/256/257, 511/512/513, 1025; retorno
direto, splits normais e forçados/desalinhados/vazios, query mascarada,
rewind/reappend com conteúdo novo e 28672 linhas. Dados têm zeros/escalas
assinados, magnitudes por linha e outliers.

```sh
OPENBLAS_NUM_THREADS=1 python3 tools/xbox/tests/rkva_decode_rowlane_model.py --long
OPENBLAS_NUM_THREADS=1 python3 tools/xbox/tests/rkva_decode_rowlane_model.py --long --contract
```

Outros checks executados:

- Os cinco `rkva_*_cpu.cpp`, compilados com g++ C++17 `-O2 -Wall -Wextra
  -Werror`, passaram: bulk FP8, execution lifetime, bulk, range e profile.
  JSON de profile validado por parser JSON, com stderr separado.
- **97 testes Python relacionados a RKVA** (`test_*rkva*.py`) passaram.
- `rkva_range_reduce_finite_model.py`: sete testes passaram.
- `rkva_kvarn_serial_prefix_probe.py --self-test`: 36 checks, sem GPU.
- `rkva_kvarn_serial_prefix_model.py --quick`: 117 batches, 626 queries,
  264 execuções otimizadas, duas rejeições esperadas; passou.
- Sweep `--self-test`, auditoria DXIL, `bash -n`, `git diff --check`: passaram.

A descoberta Python global também tentou dois testes fora do RKVA que não
carregam nesta base: `test_make_shape_gguf` depende do ausente
`tools/xbox/scripts/make-shape-gguf.py`; `test_validate_mtp_protocol` depende do
ausente `scripts.validate_mtp_protocol`. Não foram reparados nem contados como
pass. Isso não é uma qualificação geral de todos os testes da árvore.

Logs, DXIL/dump, contagem e comparação de headers estão em
`.oc-artifacts/rowlane/` (não versionados).

## O que Claude deve rodar no console

Compilar o pacote pelo fluxo UWP autorizado do Claude. Confirmar criação do
PSO, `wave_lane_min == wave_lane_max == 64`, `available=true`, `enabled=true`,
`status=ready` depois do HELLO com a flag. Manter o mesmo pacote e alternar
somente a flag, com nova conexão a cada braço. Não usar timestamps históricos
como o braço A atual.

Primeiro, corretude com a flag presente:

```sh
python3 tools/xbox/tests/rkva_kvarn_serial_prefix_probe.py \
  --execute --host "$RKVA_HOST" --modes 0 1 2 3 --precision-tails 128 0 --full \
  --output .oc-artifacts/rowlane/console-oracle.json
```

`--full` agora inclui **nq=1 em 28672 linhas**, além do caso de capacidade já
existente. Exigir `success=true`, os limites do oráculo acima e o bucket
rowlane em nq1/modes 1/2; nq>1 e modes 0/3 devem manter sua seleção anterior.
Verificar também uma chamada BULK com segmento nq=1: bucket e contador rowlane
precisam aparecer, sem atribuição ao bucket `kvarn4_decode_fine`.

A/B de GPU (mesma seed, tail, geometria, pacote e condição térmica):

```sh
# A: flag ausente; nova conexão HELLO.
python3 tools/xbox/tests/rkva_pass1_sweep.py --execute --host "$RKVA_HOST" \
  --nq 1 --rows 4096,8192,16384,28672 --tail 128 --seed 4242 \
  --warmup 3 --runs 10 --output .oc-artifacts/rowlane/console-A.json
# B: flag presente; nova conexão HELLO.
python3 tools/xbox/tests/rkva_pass1_sweep.py --execute --host "$RKVA_HOST" \
  --nq 1 --rows 4096,8192,16384,28672 --tail 128 --seed 4242 \
  --warmup 3 --runs 10 --expect-decode-kernel kvarn4_decode_rowlane \
  --output .oc-artifacts/rowlane/console-B.json
```

Repetir A/B invertendo a ordem. Critério proposto: corretude integral, nenhum
fallback, splits/grupos idênticos entre braços, **pelo menos 10% de redução da
mediana de pass1+reduce em 28672 linhas**, repetível nas duas ordens, sem
regressão maior que 5% nos outros contextos. Registrar pass1 e reduce separados;
a mediana de pass1 sozinha não decide. O sweep usa modo 2; avaliar modo 1 em
uma rodada separada, sem misturar buckets/resultados (nome rowlane é comum aos
dois modos; split_k identifica a política).

Depois disso, Claude pode medir decode real e espera remota exposta. Registrar
modelo exato/hash, prompt, sampling, comando, hardware, pacote e commit usado.
Os números anteriores informados pelo chamador não são resultados deste patch.
