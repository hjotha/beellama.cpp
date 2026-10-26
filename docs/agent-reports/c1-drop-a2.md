# C1 — remover a variante A2 medida 1,67% mais lenta

Data: 2026-10-26. Clone: `/home/hjotha/beellama-agents/drop-a2`, base
`integ/rkva-xbox-agents-20261026`. Diretório remoto: `C:\Users\hjotha\build\oc-drop-a2-20261026`.

A variante A2 de elisão de rescale foi removida por completo por ter sido medida
bit-exata, mas 1,67% mais lenta no `pass1`. A variante INT8 do B1
(`attn_kvarn4_mq2_i8`, `RKVA_MQ_I8`, `rkva-mq-i8.flag`) foi preservada.

## Superfície removida

* Guarda e os dois blocos condicionais no shader compartilhado
  `tools/xbox/shaders/rkva/rkva_range_attn_kvarn4_mq.hlsl`.
* Wrapper HLSL, header DXIL gerado, alvo do script de compilação, teste de
  exatidão, runner A/B e documento da variante.
* PSO, status, seleção em `attend_range`, campos de perfil (`Sample` e
  `BulkSegmentProfile`), contadores e chave JSON do engine/listener.
* Flag LocalState lida em `remote-attn.cpp`.
* Referências em relatórios B1/B2 e no doc INT8, sem alterar a variante INT8.

## Regeneração DXIL na 193

Wrapper local adaptado do modelo, apontando só para o diretório remoto deste C1
e usando o DXC `C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64\dxc.exe`:

```bash
DXC=/tmp/opencode/dxc193-drop-a2.py \
  bash tools/xbox/scripts/compile-rkva-range-shaders.sh \
  attn_kvarn4_mq2 attn_kvarn4_mq4 attn_kvarn4_mq2_i8
```

Saída literal:

```text
/home/hjotha/beellama-agents/drop-a2/tools/xbox/shaders/generated/rkva_range_attn_kvarn4_mq2_dxil.h 118452 bytes
/home/hjotha/beellama-agents/drop-a2/tools/xbox/shaders/generated/rkva_range_attn_kvarn4_mq4_dxil.h 236896 bytes
/home/hjotha/beellama-agents/drop-a2/tools/xbox/shaders/generated/rkva_range_attn_kvarn4_mq2_i8_dxil.h 143620 bytes
```

Comparação do payload (array `g_rkva_range_*`), antes via `git show HEAD:...` e
depois do arquivo regenerado:

| header | header antes | header depois | payload antes | payload depois | idêntico |
| --- | --- | --- | --- | --- | --- |
| `rkva_range_attn_kvarn4_mq2_dxil.h` | `756ee6cae33211a34cfd0bf68cd970b391e55804f6a895511dc5c851b5891d5d` | `3e7bd600a6d617b6e9d8f7e4d6130919b4c83db130e088194bff652e6b361c47` | `9d81c3beba4e36756c89c6bc70c5807cedb990be67fba683fc714ff7a1c5b431` | `9d81c3beba4e36756c89c6bc70c5807cedb990be67fba683fc714ff7a1c5b431` | sim |
| `rkva_range_attn_kvarn4_mq4_dxil.h` | `4cfa5603ed8b46d5bc5ac5fa1cda7358aa9c5c4e4c62b424eab01af416fa1101` | `09c2272b86e92901e21c45adcc864b892b4b57d3d5b04aba1d28bb44fc75d9c9` | `ef2878458bd37b6f702fec19f36e674a836803d300915cf2604bfd801dfb48bb` | `ef2878458bd37b6f702fec19f36e674a836803d300915cf2604bfd801dfb48bb` | sim |
| `rkva_range_attn_kvarn4_mq2_i8_dxil.h` | `8c5ea1cee99bb64713a988179647299faf367c1814a92cd785f4fd152fc31821` | `0a1a990f7ad9cf4b46891e5cc2d09266bb49535be7ee0f0eb48c2fa0431ca8d1` | `e0e47250c2c9b6e4fbff01fdfb9860391aafc1a916ab55e05e99122e4af2eabc` | `e0e47250c2c9b6e4fbff01fdfb9860391aafc1a916ab55e05e99122e4af2eabc` | sim |

Para os três headers, a única linha diferente foi a linha 3:

```text
differing_lines=[3] extra_lines=0
- // Source SHA256: d4cd61359acd9522cff6cd808bda05caf10a871c6a6edbd17b27c29451d5cab6
+ // Source SHA256: b2f4b7ec9ce4d8c375cc20a79a27e266dd03f7ead8092826a64c323adc9f0b5b
- // Source SHA256: b97a05122e3edd3994bd59b052f3607cdd024133b46c2f7b9a78468c5d544d06
+ // Source SHA256: a0a3f188588e4f91dc533042d8c87b8e200f44a2a04c1921fd94597ba216cbe1
```

## Validação

Perfil CPU (o mesmo binário cobre nomes de kernel, buckets e JSON):

```bash
g++ -std=c++17 -Wall -Wextra -Werror -O1 \
  tools/xbox/tests/rkva_range_profile_cpu.cpp \
  -o /tmp/opencode/rkva_range_profile_cpu
/tmp/opencode/rkva_range_profile_cpu | tail -1
```

```text
rkva_range_profile_cpu: ok
```

Sweep e sintaxe:

```bash
python3 tools/xbox/tests/rkva_pass1_sweep.py --self-test
bash -n tools/xbox/scripts/compile-rkva-range-shaders.sh
python3 -m py_compile tools/xbox/tests/rkva_pass1_sweep.py
```

```text
self-test: OK (4 shapes, 3 fail-closed paths, aggregation)
syntax-ok
```

Qualidade INT8 do B1:

```bash
python3 tools/xbox/tests/rkva_mq_i8_quality.py
```

```text
rkva_mq_i8_quality: 10 gated cases (dc exact, 8 queries each), PASS
```

Busca de referências com a mesma semântica do grep pedido (grafia com classes
para não reintroduzir os termos no próprio relatório):

```bash
git grep -iE 'rescale[_-]skip|mq2[_-]skip|RESCALE[_-]SKIP' | cut -d: -f1 | sort -u
```

```text
docs/agent-reports/a2-pass1-kernel.md
```

Ou seja, só o relatório histórico do A2 ainda menciona a variante removida.
