# Consolidação final RKVA — 2026-10-26

Branch `integ/rkva-final-20261026`, base `8ac3e9d10`, checkout exclusivo
`/home/hjotha/beellama-agents/final`. Sem push ou merge em main. Sem acesso ao
Xbox/4070, execução de servidor, systemctl, build CUDA/MSVC/UWP ou polling de
hardware. DXC executado apenas pelo wrapper `.oc-artifacts/dxc/dxc193.py`, via
SSH 192.168.1.193, com arquivos em `C:/Users/hjotha/build/oc-final-20261026/`.
Não havia arquivos `.oc-artifacts` versionados; o wrapper permanece local.

## Resultado e proveniência

- Rowlane trazido pelos cherry-picks de `75e7c7d2f`, `a3c38e267`, `a02255f11`.
  Agora ligado por padrão; `rkva-decode-rowlane-off.flag` desliga, lida no HELLO.
  Descriptor 3, nq1, modos 1/2, shape qualificado, wave min=max=64 e PSO continuam
  obrigatórios. Ausência/falha do PSO conserva o fallback anterior.
- INT8 trazido do diff `8ac3e9d10..23af356f2`, com conflito apenas na emissão de
  proveniência do script: preservados hash do include rowlane e hash do HLSL
  incluído pelo wrapper INT8. O prefixo é agora incondicional dentro de INT8.
  `rkva-mq-i8.flag` permanece opt-in/default OFF, com os gates anteriores.
- Removidos do produto: INT8 XOR, masked, F32 bisect, seleção por extra defines,
  quantizador de medição e os seletores desses braços nos modelos/scripts.
  As constantes por lane continuam corretas; o modelo mantém um controle
  negativo da multiplicação indevida da constante por slice.
- Removido todo fence spin: configuração/flag/env/setter, `allow_spin`, loops,
  contadores, JSON e consumidores. Mantida espera por evento com timeout e
  verificações de terminal failure, fence incompleta e device removal.
- A3 guard CUDA preservado byte a byte, incluindo 64 MiB de slack. Rescale-skip
  continua ausente. A1/B2 e quatro probes wave/SM6.5/6.6 mantidos; nenhuma PSO
  de shader model superior foi inferida como suportada no console.
- Preservados rótulos INT8 bulk, contadores e identidade nos `BulkSegmentProfile`
  e samples. O sweep reconhece rowlane enabled+available no perfil e ainda
  rejeita fallback quando `--expect-decode-kernel` é explícito.

Vereditos **fornecidos pelo usuário**, não reexecutados nesta consolidação:

| Mudança | Resultado | Política |
| --- | --- | --- |
| Guard VRAM | Decode +2,2%, prefill neutro, tokens idênticos | Mantido |
| Fence spin | +0,29%, ruído | Removido |
| Rescale-skip | Mais lento | Ausente |
| Rowlane | Kernel −35%, decode +6,2%, oráculo ~1e-5, 254/256 tokens | Default ON |
| INT8 prefix | Pass1 −17,1%, prefill +7,96%, 256/256 tokens, Q8 ~1% | Flag/default OFF |
| Teto | Console aceita até SM 6.4 | Ferramenta mantida |

Modelos, prompt, sampling, comandos completos e commits de execução desses
resultados não acompanharam a tabela recebida. Não são novos benchmarks deste
checkout nem uma qualificação geral de qualidade. Política de produto:
[RKVA final optimizations](../rkva-final-optimizations.md).

## Prova DXIL

Todos os **101 headers `*_dxil.h` do repositório** estão em
`tools/xbox/shaders/generated/` e foram auditados. O
[manifesto completo](astra-final-dxil-manifest.json) registra tamanho, SHA256 do
payload, igualdade à base e igualdade aos branches de origem.

- **95 payloads preexistentes byte-idênticos à base**, inclusive todos os alvos
  não alterados intencionalmente. Comparação adicional do texto de cada header,
  retirando apenas comentários de SHA256, também idêntica.
- Uma substituição intencional: INT8 anterior → INT8 prefix.
- Cinco adições: rowlane + quatro probes de capacidades. Todas iguais aos
  respectivos branches de origem.

| Payload | Bytes | SHA256 final = entregue/medido |
| --- | ---: | --- |
| `attn_kvarn4_decode_rowlane` | 49696 | `0668df8afce0288fae460a2bed5e0922b88d80a07a415be1a706927fc22fa644` |
| `attn_kvarn4_mq2_i8` | 140356 | `a2b66069596fe3b8111dd19d99b4b65a652b2b2e376f44014015ade13f2c2d3f` |

**Nenhum dos dois payloads medidos mudou.** Remover branches pré-processados
inativos e tornar prefix incondicional mudou a fonte/proveniência, não o DXIL.
As compilações adicionais com `-Fc` também produziram esses mesmos hashes.

Regeneração (exit 0; tamanho em bytes):

```bash
DXC="$PWD/.oc-artifacts/dxc/dxc193.py" \
  bash tools/xbox/scripts/compile-rkva-range-shaders.sh \
  attn_kvarn4_mq2 attn_kvarn4_mq4 attn_kvarn4_mq2_i8 \
  attn_kvarn4_opt attn_kvarn4_opt_prefix \
  attn_kvarn4_decode_prefix attn_kvarn4_decode_rowlane
```

Saída por alvo: `mq2 118452`, `mq4 236896`, `mq2_i8 140356`,
`opt 73612`, `opt_prefix 73560`, `decode_prefix 51664`, `decode_rowlane 49696`
bytes (prefixo comum `attn_kvarn4_`).

Os quatro alvos opt/decode foram regenerados novamente após atualizar apenas
os comentários da política default-on; os hashes de payload permaneceram iguais.
GPUpeak não precisou de regeneração: fontes e headers importados não mudaram;
`python3 tools/xbox/scripts/compile-gpupeak-shaders.py --check` aprovou os 16
alvos, e a comparação dos bytes confirmou os 12 antigos e as quatro adições.

Reprodução da auditoria a partir da raiz (sem DXC, console ou refs remotas):

```bash
python3 - <<'PY'
import hashlib, json, re, subprocess
from pathlib import Path
base = '8ac3e9d10'
root = Path('tools/xbox/shaders/generated')
manifest = json.loads(Path('docs/agent-reports/astra-final-dxil-manifest.json').read_text())
def payload(data):
    arrays = re.findall(rb'(?:uint8_t|BYTE|unsigned char)\s+\w+\[\]\s*=\s*\{(.*?)\};', data, re.S)
    assert arrays
    return b''.join(bytes(int(x, 16) for x in re.findall(rb'0x([0-9a-fA-F]{2})\b', a)) for a in arrays)
def strip_hashes(data):
    return re.sub(rb'^// .*SHA256:.*\r?\n', b'', data, flags=re.M)
assert {p.name for p in root.glob('*_dxil.h')} == {r['header'] for r in manifest}
for row in manifest:
    path = root / row['header']
    data = path.read_bytes()
    raw = payload(data)
    assert len(raw) == row['bytes']
    assert hashlib.sha256(raw).hexdigest() == row['sha256'], path
    if row.get('base_identical'):
        old = subprocess.check_output(['git', 'show', base + ':' + str(path)])
        assert raw == payload(old), path
        assert strip_hashes(data) == strip_hashes(old), path
print('PASS: 101 payload hashes; 95 unchanged payloads/headers, except source hashes')
PY
```

## Resíduos e escopo

```bash
git grep -n -E 'RESCALE_SKIP|rescale_skip|fence_spin|rkva-fence-spin|MASKED_REDUCE_PROBE|F32_PROBE|lane \^ st([^[:alnum:]_]|$)' -- ':!docs/agent-reports'
git grep -n -E 'RKVA_FENCE_SPIN|allow_spin|RKVA_EXTRA_DEFINES|RKVA_MQ_I8_MEASURE|f32-probe|rkva-decode-rowlane.flag' -- ':!docs/agent-reports'
git ls-files .oc-artifacts
git diff 8ac3e9d10 -- ggml/src/ggml-cuda tests/test-cuda-graph-guard.cpp
git diff --check
bash -n tools/xbox/scripts/compile-rkva-range-shaders.sh
```

Ambos os greps: sem saída, exit 1. Demais: sem saída, exit 0. A expressão
`lane ^ st` usa limite de identificador: WHTs legítimos preexistentes usam
`lane ^ stride`/`stride_wht` em vários backends e não são o shuffle INT8 rejeitado.
Esses WHTs não foram alterados para contornar uma busca por substring.

## Validação CPU/Python

Saídas completas, comandos e códigos de retorno estão no
[registro de validação](astra-final-validation.json). Artefatos intermediários
ficam em `/tmp/rkva-final-validation/`; nenhum binário foi versionado.
`OPENBLAS_NUM_THREADS=1` nos modelos e unittests. Nenhum teste de hardware foi
executado. Os testes de origem/C++ portátil não compilam o engine UWP.

Compilação e execução dos cinco testes C++ portáteis:

```bash
for name in rkva_range_profile_cpu rkva_execution_lifetime_cpu \
            rkva_kvarn_range_cpu rkva_kvarn_bulk_cpu rkva_bulk_fp8_cpu; do
  g++ -std=c++17 -O2 -Wall -Wextra -Werror -pthread \
    tools/xbox/tests/$name.cpp -o /tmp/rkva-final-validation/$name
  /tmp/rkva-final-validation/$name
done
g++ -std=c++17 -O2 -Wall -Wextra -Werror -Iggml/src/ggml-cuda \
  tests/test-cuda-graph-guard.cpp -o /tmp/rkva-final-validation/test-cuda-graph-guard
/tmp/rkva-final-validation/test-cuda-graph-guard
```

Todos exit 0; perfil emite JSON válido; guard: `graph-guard: all checks passed`.

Unittests (comandos independentes, todos exit 0):

```bash
python3 -m unittest discover -s tools/xbox/tests -p 'test_rkva*.py'
python3 -m unittest discover -s tools/xbox/tests -p 'test_gpupeak*.py'
python3 -m unittest discover -s tools/xbox/tests -p 'test_bench*.py'
python3 -m unittest discover -s tools/xbox/tests -p 'test_run_rkva*.py'
```

Saídas: respectivamente **28, 5, 48, 21 testes; OK** (102 no total).
A descoberta ampla `python3 -m unittest discover -s tools/xbox/tests -p 'test_*.py'`
executou 104 entradas e terminou `FAILED (errors=2)`: `test_make_shape_gguf`
não encontra `tools/xbox/scripts/make-shape-gguf.py`; `test_validate_mtp_protocol`
não encontra `scripts.validate_mtp_protocol`. Esses scripts já não existem na
base, e não foram restaurados nesta tarefa RKVA. As outras 102 entradas passaram.

Sweeps offline:

```bash
python3 tools/xbox/tests/rkva_pass1_sweep.py --self-test
python3 tools/xbox/tests/rkva_pass1_sweep.py --self-test \
  --expect-decode-kernel kvarn4_decode_rowlane --expect-mq2-kernel kvarn4_mq2_i8
python3 tools/xbox/tests/rkva_kvarn_serial_prefix_probe.py --self-test
```

Saídas: `self-test: OK (4 shapes, 3 fail-closed paths, aggregation)` em ambos
os sweeps; serial-prefix: `self_test=ok`, 18 checks em cada ptt 128/0,
`offline fake server; no GPU`.

Compilação da assembly para auditar o programa emitido:

```bash
.oc-artifacts/dxc/dxc193.py -T cs_6_4 -E CSRangeAttnKvarn4Mq \
  -D RKVA_MQ_QT=2 -Fc tools/xbox/shaders/rkva/rkva_range_attn_kvarn4_mq_i8.hlsl \
  -Fo /tmp/rkva-final-validation/int8.dxil
.oc-artifacts/dxc/dxc193.py -T cs_6_0 -E CSRangeAttnKvarn4Decode \
  -D RKVA_SERIAL_PREFIX=1 -D RKVA_DECODE_OPT=1 -D RKVA_DECODE_ROWLANE=1 \
  -Fc tools/xbox/shaders/rkva/rkva_range_attn_kvarn4_opt.hlsl \
  -Fo /tmp/rkva-final-validation/rowlane.dxil
python3 tools/xbox/tests/rkva_mq_i8_dxil_model.py \
  /tmp/rkva-final-validation/int8.txt --require-uniform-reads
python3 tools/xbox/tests/rkva_decode_rowlane_dxil_audit.py \
  /tmp/rkva-final-validation/rowlane.txt
python3 tools/xbox/tests/rkva_dxil_loop_cost.py /tmp/rkva-final-validation/int8.txt
```

Saídas: INT8 **8832 scores, max error 8.34e-06, PASS**; rowlane **96 broadcasts,
10 uniform index expressions; lane-dependent negative rejected**. O contador
estático atual imprime 364,9 unidades/linha (inclui braços de branch); é um
modelo pré-ISA, não medição de GPU nem alteração do payload medido.

Modelos completos (todos exit 0):

```bash
python3 tools/xbox/tests/rkva_decode_rowlane_model.py --long
python3 tools/xbox/tests/rkva_decode_rowlane_model.py --long --contract
python3 tools/xbox/tests/rkva_mq_i8_quality.py
python3 tools/xbox/tests/rkva_mq_i8_lane_model.py
python3 tools/xbox/tests/rkva_kvarn4_mq_partition_model.py
python3 tools/xbox/tests/rkva_kvarn4_mq_split_model.py
```

| Modelo | Saída |
| --- | --- |
| Rowlane, arredondamentos separados | `PASS cases=65 mode_runs=130 worst_out=0.000127561606 worst_lse=1.40178712e-05` |
| Rowlane, contração FMA | Mesmos máximos globais; `PASS cases=65 mode_runs=130` |
| INT8 qualidade | `10 gated cases (dc exact, 8 queries each), PASS`; added/ref 0,062–0,076 |
| INT8 lanes/prefix | `8 cases, PASS`; erro do caminho correto ~1,00× álgebra; controle negativo detectado |
| MQ partição | `ok (MQ2 and MQ4 verified)`; 1860 configurações com representações distintas tratadas por query |
| MQ split | `passed counts=[1,7,8,9,127,128,129,255,256,257,513] max_scaled_O=3.481e-07 max_LSE=5.133e-07` |

As linhas `dc=False ... info` da qualidade são controles sem correção DC;
seus avisos não são falhas do caminho aceito, cujo gate exige `dc=True`.
O modelo rowlane usa dados sintéticos; seu máximo global ~1,28e-4 não é o
~1e-5 do oráculo reportado pelo operador. Os testes usam métricas/tolerâncias
do oráculo existente e incluem caudas, máscaras, splits e rewind/reappend.

## Commits e handoff

- `69c4eb1a8`: shader rowlane e payload (cherry-pick).
- `11d12b64c`: seleção HELLO e preservação de identidade bulk (cherry-pick).
- `0dcb53b49`: modelos/auditoria rowlane (cherry-pick).
- `be2ece5a7`: probes de teto wave/SM6.5/SM6.6.
- `aa9988179`: rowlane default-on, opt-out e remoção do spin.
- `3f155bcbc`: único caminho INT8 prefix, modelos atualizados, sem hooks rejeitados.
- `330dda25b`: comentários e proveniência shader alinhados ao default-on;
  payloads inalterados.
- Commit documental seguinte: política, histórico e este relatório com evidências.

Claude fará os builds host/UWP e a validação final no console antes do merge.
Esta entrega não os qualifica. Validar ausência/presença do opt-out no novo
HELLO, INT8 OFF/ON, buckets/counters ordinary e bulk, e fallback quando PSO/shape
não qualifica. Não houve push, deploy nem alteração de flags no console.
