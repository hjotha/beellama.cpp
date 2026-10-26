# Auditoria de latência RKVA — ferramenta de evidências (CPU)

Data: 2026-10-26. Clone: `/home/hjotha/beellama-agents/latency-audit`.
Branch: `perf/rkva-latency-audit-20261026` (base `4bf192bb2`, exceção Bee sem
worktree). Esta é a revisão dos blockers do pré-review Astra sobre o commit
`ce23a5ac7`.

Escopo: **só ferramenta e evidência CPU**. Nada foi otimizado; nenhum flag,
shader, tier, script de build ou serviço mudou. Não houve GPU/CUDA runtime,
Xbox, SSH, LAN, `systemctl`, build completo nem profiling. **Ninguém mediu o
Xbox neste commit**: `--execute` segue proposto para o root, depois do
`build-j10` e com console ocioso.

## Entregues

* `tools/xbox/scripts/rkva-latency-audit.py` — CLI com três modos.
* `tools/xbox/tests/test_rkva_latency_audit.py` — 54 testes CPU, com doubles de
  socket/`/proc`/runner; nenhum hardware.
* `docs/agent-reports/rkva-latency-audit.md` — este relatório.

Nenhum outro arquivo do repositório foi alterado; scratch em `/tmp/opencode/`.

## Correções desta revisão

1. **IDs de ciclo do probe**: `cycle_ids_unique` agora compara os `cycle_id`
   reais do header, não as durações. A sequência enviada e a recebida têm de
   bater 1:1 e em ordem (`verify_cycle_sequence`); durações idênticas com IDs
   distintos são válidas. ID repetido de verdade falha.
2. **Trace offline**: a correlação usa chave composta
   `(session, origin, cycle)`. `origin` vem de metadados opcionais
   (`pid/process/origin/arm/run_id/trace_id/instance/host`) quando existem.
   Reuso de ID em **janelas independentes** vira `unknown`, não falha; reuso
   sobreposto ou IDs misturados no mesmo grupo falha. Sem `session`/`origin`,
   o escopo é `unknown` — nunca `ok` falso.
3. **PING sem body**: o header é validado por inteiro antes de qualquer leitura
   de body. `payload_bytes != 0` em PING falha sem `recv` de body; o único op
   com body (`GET_STATS`) tem teto de 1 MiB aplicado ao header antes da leitura.
   `session` é validado no request (o `rkva_response` não carrega session).
4. **Parent/child**: `job_finish`, `rpc_total` e `out_decode` publicam
   `residual_total_us` (resto não coberto), nunca `exposed_total_us`. Espera
   exposta só existe para spans de espera (`rpc_header_wait`, `range_wait`, ...).
   Comparação CPU/GPU com contagem, janela ou op diferentes é `failed`
   (exit 2), não um número aproximado.
5. **Rede**: fatos locais (`r8152` USB, 1000 Mb/s, MTU 1500, `rx-usecs=15000`)
   são reportados sem conclusão causal. O tool e o JSON dizem explicitamente
   que esses dados não atribuem latência RKVA.

## CLI

```text
$ python3 tools/xbox/scripts/rkva-latency-audit.py --help
usage: rkva-latency-audit.py [-h] [--trace FILE] [--min-position MIN_POSITION]
                             [--max-position MAX_POSITION]
                             [--context-cap CONTEXT_CAP] [--cpu-samples FILE]
                             [--gpu-samples FILE] [--execute] [--empty-rtt]
                             [--host HOST:PORT] [--reps REPS] [--warmup WARMUP]
                             [--timeout TIMEOUT] [--include-stats] [--allow-busy]
                             [--network-info] [--out FILE] [--strict]
```

* `--trace FILE` (offline): eventos `nq == 1` com `first_pos` em
  `[--min-position, --max-position]` (default `130560..160000`). `--context-cap`
  acima de 160000 é recusado.
* `--execute --empty-rtt --host H:P`: uma conexão, somente `RKVA_PING` (op 9,
  payload 0, `RKVA_SESSION_NONE`), sem HELLO/sessão/engine. `--reps` ≤ 1000,
  `--warmup` ≤ 20, `--timeout` ≤ 3 s; `--include-stats` = um `GET_STATS`.
* `--network-info`: rota/interface local, sysfs e `ethtool -i/-k/-c` (show).
  Pode rodar sozinho; **nunca** consulta o host remoto.
* `--out FILE`: único caminho que cria arquivo. Sem `--out`, nada é escrito.
* `--strict`: eleva escopo desconhecido a exit 2.
* `--allow-busy`: existe, mas é **off** por default e aparece no JSON.

## Gates

| Situação | Comportamento |
| --- | --- |
| `end_us < begin_us`, NaN/inf em `gpu_us`/amostra | exit 1 |
| `max_position`/`context_cap` > 160000 | recusa, exit 1 |
| PING com `payload_bytes != 0` | falha antes de ler body |
| `GET_STATS` declarando body > 1 MiB | falha antes de ler body |
| sequência de `cycle_id` fora de ordem/repetida | falha, socket fechado |
| `(session, origin, cycle)` duplicado em janelas sobrepostas | `failed`, exit 2 |
| mesmo ID em janelas independentes | `unknown`, exit 0 (2 com `--strict`) |
| `cycle` ausente ou sem `session`/`origin` | `unknown`, nada inventado |
| `(session, graph_run, layer)` em posição diferente | spans rejeitados, `partial` |
| CPU/GPU com contagem, janela ou op diferentes | `failed`, exit 2 |
| processo `llama-server`/`llama-bench`/`rkva*` bench/test ativo | probe recusa antes do socket |
| `--network-info` sem privilégio | `unavailable` com motivo; nunca 0 falso |

O guard compara apenas o basename do executável/script em `/proc/<pid>/cmdline`
(e `comm`), excluindo o próprio pid e ancestrais. Argumentos como
`--trace analyze-rkva-...` não disparam o guard (teste dedicado).

`--execute` não usa `RANGE_PROFILE` (op 20): ele exige HELLO + engine, o que
criaria GPU e invalidaria a medida vazia. `PING` é servido antes de qualquer
engine (`case RKVA_PING: return RKVA_OK;` em
`tools/xbox/uwp/rkva/remote-attn.cpp`). O accept de uma conexão nova zera
engine/sessões do servidor; por isso o guard e a exigência de console ocioso.

## Evidência offline: trace real (somente leitura)

Arquivo:
`/home/hjotha/.codex/artifacts/beellama-xbox-rkva-20261026/wf4-cached-p102400-long-r1-trace.3091043.jsonl`
(443,608,988 bytes). Comando:

```bash
nice -n 19 python3 tools/xbox/scripts/rkva-latency-audit.py \
  --trace /home/hjotha/.codex/artifacts/beellama-xbox-rkva-20261026/\
wf4-cached-p102400-long-r1-trace.3091043.jsonl \
  --min-position 130560 --max-position 130600 \
  --out /tmp/opencode/real2.json
```

Saída (exit 0):

```text
scope=ok window=130560..130600 events=114524 tokens=82 layer_groups=1312
  envelope_ms              p50=  52.463 p95=  61.683 total= 4401.229
  attributed_ms            p50=  51.717 p95=  61.021 total= 4342.941
  unattributed_gap_ms      p50=   0.668 p95=   0.832 total=   58.288
  rpc_total_ms_container   p50=  20.792 p95=  21.490 total= 1713.139
  peer_gpu_ms              p50=  11.851 p95=  12.009 total=  972.565
correlation: scope=unknown valid=1312 absent=0 duplicate_ids=656 mixed=0 reused_windows=1312 ambiguous=0
```

`malformed_lines=0`, `identity_conflicts=0`, `events_without_identity=0`.
Fases principais no mesmo recorte (coluna `resid/exp` = residual para
containers, exposto para waits):

| fase | n | p50 µs | p95 µs | por token | resid/exp por token | classe |
| --- | --- | --- | --- | --- | --- | --- |
| `job_finish` | 1312 | 1324 | 1491 | 21,642 ms | 0,622 ms (residual) | medido (container) |
| `rpc_total` | 1312 | 1285 | 1451 | 20,892 ms | 1,882 ms (residual) | medido (container) |
| `rpc_header_wait` | 1312 | 1164 | 1321 | 18,950 ms | 6,423 ms (exposto) | medido (wait) |
| `range_wait` | 1312 | 419 | 593 | 6,796 ms | 4,810 ms (exposto) | medido (wait) |
| `out_decode` | 1312 | 31 | 53 | 0,664 ms | 0,115 ms (residual) | medido (container) |
| `cuda_local_span_gpu` | 5084 | 845 | 1628 | 39,936 ms (soma) | 26,678 ms | medido |
| `cuda_graph_host` | 7626 | 66 | 1308 | 28,675 ms (soma) | 0,962 ms | medido |

Leitura correta:

* Containers aparecem no relatório, **nunca** somados com filhos.
  `parent_child.job_finish.rpc_total` tem `overlap_ratio = 0,9713`; somar os dois
  inflaria ~21 ms/token. Container não é espera: seu resto é `residual`.
* `attributed_ms` é a **união** dos spans de fronteira (wall coberto), não a
  soma de trabalhos; `envelope - attributed` = `unattributed_gap` (≈0,67 ms/token).
* Espera exposta é wall range de span de espera: `rpc_header_wait` 6,423 ms/token
  e `range_wait` 4,810 ms/token. O valor antigo de ~6,9 ms/token é de outro
  trace/janela: magnitude compatível, **não** o mesmo número.
* `rpc_total` (20,9 ms/token) não é espera exposta, apesar de conter o header
  wait; usá-lo como wait é exatamente o erro evitado.
* `cuda_local_span_gpu` p50 = 845 µs por span. A hipótese de ~823 µs/camada é da
  mesma ordem, mas há ~62 spans/token (não 16), então o total de 39,9 ms/token
  **não** é 16 × 823 µs. Nenhuma atribuição CPU/GPU/rede é derivada disso.

### Correlação: por que `unknown` (e não falha) neste arquivo

Há 1312 grupos e 656 `cycle_id` repetidos, sempre em janelas independentes
(duas metades do trace, separadas por ~229 s): `reused_windows=1312`,
`ambiguous=0`. O arquivo não carrega `pid`/`origin`/`arm`; a chave composta é
`(session=0, origin=(), cycle)`. A ferramenta não trata isso como falha de
transferência: marca `unknown` e mantém a atribuição por
`(session, graph_run, layer)`, com zero conflito de identidade. Exit 0; com
`--strict`, exit 2. Em trace com IDs únicos a correlação fica `ok`; IDs
duplicados em janelas sobrepostas falham; sem `session`/`origin`, `unknown`.

## Evidência local de rede (leitura, não causal)

```bash
python3 tools/xbox/scripts/rkva-latency-audit.py --network-info
```

```text
target=None interface=enp196s0f4u1u4 route_source=default_route
driver=r8152 bus=usb id={'vendor': '0bda', 'product': '8153'} mtu=1500 speed_mbps=1000 duplex=full operstate=up
offloads: rx-checksumming=on, tx-checksumming=on, tcp-segmentation-offload=on, generic-segmentation-offload=on, generic-receive-offload=on, large-receive-offload=off [fixed]
coalescing: adaptive-rx=None, adaptive-tx=None, rx-usecs=15000, tx-usecs=0, rx-frames=n/a, tx-frames=n/a
```

Comandos: `ethtool -i`, `ethtool -k`, `ethtool -c` (show). Sem
`ethtool -s/-K/-C/-G`, `ip set`, `sysctl`, IRQ ou MTU write. Sem endereços
locais, MAC ou IPv6 de privacidade. São fatos do **host local**, não do Xbox e
**não provam causalidade** de latência RKVA; nenhuma conclusão causal é
derivada.

## Testes CPU (reais)

```bash
$ nice -n 19 python3 -m unittest tools/xbox/tests/test_rkva_latency_audit.py
Ran 54 tests in 0.078s
OK

$ python3 -m py_compile tools/xbox/scripts/rkva-latency-audit.py \
      tools/xbox/tests/test_rkva_latency_audit.py
compile-ok
```

Novos testes desta revisão: durações idênticas com IDs únicos passam; ID
repetido falha; sequência enviada/recebida em ordem; header com
`payload_bytes=0xFFFFFFFF` falha antes de qualquer body (`recv_calls=1`);
mutações de magic/version/op/cycle/status/payload falham; session do request
validada antes do envio; EOF/timeout fecham o socket; `GET_STATS` acima de 1 MiB
falha antes do body; containers publicam residual e não entram em
`exposed_wait`; reuso de ID em janelas independentes é `unknown`; reuso
sobreposto é `failed`; origens distintas (session/`origin`) resolvem `ok`; sem
origem é `unknown`; `origin` conflitante no mesmo grupo falha; comparação com
contagem/janela/op diferentes é `failed`; comparação sem amostras é `unknown`
(exit 2 com `--strict`); e os 39 testes anteriores continuam válidos.

Flake8 não está instalado no host (`flake8 not installed`); validei
`py_compile` e limite de 125 colunas.

## Riscos residuais / revisão Astra

1. O trace não declara parentesco; aninhamento é contenção estrita de wall
   clock. Spans locais dentro de uma espera reduzem a parte exposta (desejado),
   mas não são "filhos lógicos" declarados.
2. `gpu_us` é reportado pelo peer; só é comparável com CPU quando contagem e
   janela coincidem — a ferramenta não exporta essas amostras sozinha.
3. `--execute` **não foi rodado**; o root deve rodar com console ocioso, pois o
   accept zera engine/sessões.
4. `RANGE_PROFILE` não é usado para RTT (exige HELLO/engine); declarado como
   unsupported para medida vazia, não como ping falso.
5. O recorte é uma janela de 82 tokens a partir de 130560; não é o teste
   completo do usuário (≤160000) e não autoriza conclusão de tier.

## Próximos passos (root, após `build-j10`)

1. Rodar `--execute --empty-rtt` com `--out` próprio e comparar com a janela do
   trace correspondente.
2. Exportar CPU engine e GPU com a mesma contagem/janela e usar
   `--cpu-samples/--gpu-samples`; sem congruência o resultado é `failed`.
3. Só depois de 3 formatos de KV e QA avaliar mudança de overhead de engine;
   esta ferramenta não propõe nem aplica fix.
