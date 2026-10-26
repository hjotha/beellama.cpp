# Auditoria de latência RKVA — ferramenta de evidências (CPU)

Data: 2026-10-26. Clone: `/home/hjotha/beellama-agents/latency-audit`.
Branch: `perf/rkva-latency-audit-20261026` (base `4bf192bb2`). Exceção Bee:
edição direta no checkout, sem worktree.

Esta entrega é **só ferramenta e evidência**. Nada foi otimizado, nenhum flag,
shader, tier, script de build ou serviço mudou. Não houve GPU/CUDA runtime,
Xbox, SSH, LAN, `systemctl`, build completo nem profiling. A leitura do trace e
do link local foi feita na CPU local, com `nice -n 19`; o trace e o sqlite do
`nsys` foram tratados como somente leitura (o sqlite não foi necessário e não
foi aberto).

## Entregues

* `tools/xbox/scripts/rkva-latency-audit.py` — CLI única, três modos.
* `tools/xbox/tests/test_rkva_latency_audit.py` — 39 testes CPU com doubles de
  socket, `/proc` falso e runner falso; nenhum contato com hardware.
* `docs/agent-reports/rkva-latency-audit.md` — este relatório.

Nenhum outro arquivo do repositório foi alterado. Scratch de execução ficou em
`/tmp/opencode/`.

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

* `--trace FILE` (offline): atribui eventos `nq == 1` com `first_pos` na janela
  `[--min-position, --max-position]`, default `130560..160000`. `--context-cap`
  é limitado a **160000**; acima disso a ferramenta recusa em vez de analisar um
  tier que o usuário não mediu.
* `--execute --empty-rtt --host H:P`: uma conexão TCP, somente `RKVA_PING`
  (op 9, payload vazio, `session_id = RKVA_SESSION_NONE`), sem HELLO, sem sessão
  e sem engine. `--reps` ≤ 1000, `--warmup` ≤ 20, `--timeout` ≤ 3 s.
  `--include-stats` envia um único `GET_STATS` (op 10, leitura de controle).
* `--network-info`: rota/interface local, sysfs e `ethtool -i/-k/-c` (somente
  leitura). Pode rodar sozinho; **nunca** consulta o host remoto.
* `--out FILE`: único caminho que cria arquivo. Sem `--out`, nada é escrito.
* `--strict`: eleva escopo desconhecido a exit 2.
* `--allow-busy`: existe, mas é **off** por default e reportado no JSON.

## Gates implementados

| Situação | Comportamento |
| --- | --- |
| `end_us < begin_us`, NaN/inf em `gpu_us`/amostra | `TraceFormatError` → exit 1 |
| `max_position > 160000` ou `context_cap > 160000` | recusa → exit 1 |
| `cycle_id` duplicado entre grupos de layer | correlação = `failed`, exit 2; grupos **não** são unidos |
| `cycle_id` ausente | correlação = `unknown`; nada é inventado (exit 0, ou 2 com `--strict`) |
| mesmo `(session, graph_run, layer)` em posição diferente | conflito de identidade, spans rejeitados, escopo `partial` |
| CPU engine vs GPU com contagem ou janela diferente | `comparable: false`, escopo `unknown`, motivo explícito |
| processo `llama-server`/`llama-bench`/`rkva*` bench/test ativo | probe recusa antes de abrir socket |
| `--network-info` sem privilégio | `unavailable` com motivo; nunca 0 falso |

O guard de processo compara apenas o basename do executável/script em
`/proc/<pid>/cmdline` (e `comm`), excluindo o próprio pid e ancestrais. Argumentos
como `--trace analyze-rkva-...` não disparam o guard; há teste dedicado para isso.

`--execute` não implementa RTT via `RANGE_PROFILE` de propósito: op 20 exige
HELLO + engine, o que criaria GPU e destruiria a medida "vazia". `PING` é servido
antes de qualquer engine em `tools/xbox/uwp/rkva/remote-attn.cpp` (case
`RKVA_PING: return RKVA_OK;`). A consequência operacional é real e está no
relatório: **um novo accept zera engine/sessões do servidor**, por isso o guard e
a exigência de console ocioso.

## Evidência offline: trace real (somente leitura)

Arquivo:
`/home/hjotha/.codex/artifacts/beellama-xbox-rkva-20261026/wf4-cached-p102400-long-r1-trace.3091043.jsonl`
(443,608,988 bytes). Comando:

```bash
nice -n 19 python3 tools/xbox/scripts/rkva-latency-audit.py \
  --trace /home/hjotha/.codex/artifacts/beellama-xbox-rkva-20261026/\
wf4-cached-p102400-long-r1-trace.3091043.jsonl \
  --min-position 130560 --max-position 130600 \
  --out /tmp/opencode/real-130560-130600.json
```

Saída resumida (exit 2 por IDs de correlação duplicados, ver abaixo):

```text
scope=ok window=130560..130600 events=114524 tokens=82 layer_groups=1312
per token (82 tokens, layers/token 16..16):
  envelope_ms              p50=  52.463 p95=  61.683 total= 4401.229
  attributed_ms            p50=  51.717 p95=  61.021 total= 4342.941
  unattributed_gap_ms      p50=   0.668 p95=   0.832 total=   58.288
  rpc_total_ms_container   p50=  20.792 p95=  21.490 total= 1713.139
  peer_gpu_ms              p50=  11.851 p95=  12.009 total=  972.565
correlation: scope=failed valid=1312 absent=0 duplicate_ids=656 mixed=0
```

`malformed_lines=0`, `identity_conflicts=0`, `events_without_identity=0`.
As fases mais relevantes no mesmo recorte:

| fase | n | p50 µs | p95 µs | por token | exposto/token | classe |
| --- | --- | --- | --- | --- | --- | --- |
| `job_finish` | 1312 | 1324 | 1491 | 21,642 ms | 0,622 ms | medido (container) |
| `rpc_total` | 1312 | 1285 | 1451 | 20,892 ms | 1,882 ms | medido (container) |
| `rpc_header_wait` | 1312 | 1164 | 1321 | 18,950 ms | 6,423 ms | medido (wait) |
| `range_wait` | 1312 | 419 | 593 | 6,796 ms | 4,810 ms | medido (wait) |
| `out_decode` | 1312 | 31 | 53 | 0,664 ms | 0,115 ms | medido (container) |
| `cuda_local_span_gpu` | 5084 | 845 | 1628 | 39,936 ms (soma) | 26,678 ms | medido |
| `cuda_graph_host` | 7626 | 66 | 1308 | 28,675 ms (soma) | 0,962 ms | medido |

Leitura correta desses números:

* `job_finish`/`rpc_total`/`out_decode` são containers: aparecem no relatório,
  **nunca** são somados com os filhos. `parent_child.job_finish.rpc_total` tem
  `overlap_ratio = 0,9713`; somar os dois inflaria ~21 ms/token.
* `attributed_ms` é a **união** dos spans de fronteira (wall coberto), não a soma
  de trabalhos. `envelope - attributed = unattributed_gap` (≈0,67 ms/token aqui).
* Espera exposta é wall range de span específico. Neste recorte,
  `rpc_header_wait` exposto = 6,423 ms/token, `range_wait` = 4,810 ms/token.
  O valor antigo de ~6,9 ms/token é de outro trace/janela: a magnitude é
  compatível, mas **não** é o mesmo número e não foi usada como prova.
* `rpc_total` (20,9 ms/token) **não** é espera exposta, apesar de conter o header
  wait; usá-lo como "wait" foi justamente o erro evitado.
* `cuda_local_span_gpu` p50 = 845 µs por span. A hipótese registrada de
  ~823 µs/camada é da mesma ordem, porém há ~62 spans por token (não 16), então
  o total de 39,9 ms/token **não** é 16 × 823 µs. Nenhuma atribuição CPU/GPU/rede
  é derivada disso.

### Correlação: por que exit 2 neste arquivo

O trace real reutiliza `cycle_id`: 1312 grupos de layer, 656 ids, cada id
aparecendo em dois grupos (`duplicate_ids=656`, `ambiguous_groups=1312`). A
ferramenta se recusa a uni-los; a análise por `(session, graph_run, layer)` segue
válida e não houve conflito de identidade. Isso é o gate "ID duplicado ⇒ fail",
não um erro de leitura. Em trace com ids únicos a correlação fica `ok`; sem ids,
fica `unknown` (testado).

## Evidência local de rede (leitura, sem console)

```bash
python3 tools/xbox/scripts/rkva-latency-audit.py --network-info
```

```text
target=None interface=enp196s0f4u1u4 route_source=default_route
driver=r8152 bus=usb id={'vendor': '0bda', 'product': '8153'} mtu=1500 speed_mbps=1000 duplex=full operstate=up
offloads: rx-checksumming=on, tx-checksumming=on, tcp-segmentation-offload=on, generic-segmentation-offload=on, generic-receive-offload=on, large-receive-offload=off [fixed]
coalescing: adaptive-rx=None, adaptive-tx=None, rx-usecs=15000, tx-usecs=0, rx-frames=n/a, tx-frames=n/a
```

Comandos executados: `ethtool -i`, `ethtool -k`, `ethtool -c` (todos show). Não
houve `ethtool -s/-K/-C/-G`, `ip set`, `sysctl`, IRQ ou MTU write. O relatório
não inclui endereços locais, MAC nem IPv6 de privacidade; só o IP alvo quando
`--host` é passado. Esta medição é do **host local**, não do Xbox.

## Testes CPU (reais)

```bash
$ nice -n 19 python3 -m unittest tools/xbox/tests/test_rkva_latency_audit.py
Ran 39 tests in 0.060s
OK

$ python3 -m py_compile tools/xbox/scripts/rkva-latency-audit.py \
      tools/xbox/tests/test_rkva_latency_audit.py
compile-ok
```

Cobertura relevante: trace golden com pai/filho sobrepostos; janela e contagem
incompatíveis rejeitadas; correlação válida/ausente/duplicada; percentis e
contagem de amostras; timeout com socket fechado; guard sem falso positivo para
a própria CLI; rede só com argv read-only; cap de contexto 160000; parse de CLI e
`ast.parse` da fonte.

Flake8 não está instalado neste host (`flake8 not installed`); a validação de
estilo foi limite de 125 colunas (nenhuma linha acima) e `py_compile`.

## Limitações / pontos para revisão Astra

1. O trace não tem campo de parentesco; o aninhamento é por contenção estrita de
   wall clock. Spans locais que caem dentro de uma espera reduzem a parte
   exposta (é o comportamento desejado), mas não são "filhos lógicos" declarados.
2. `gpu_us` é reportado pelo peer; só é comparável com CPU quando contagem e
   janela coincidem — a ferramenta não exporta essas amostras sozinha.
3. `--execute` **não foi rodado** por mim (sem LAN/hardware). O root deve rodar
   depois de `build-j10`, com console ocioso; o accept zera engine/sessões.
4. `rkva_range_profile` (op 20) não é usado para RTT porque exige HELLO/engine;
   isso está declarado como unsupported para medida vazia, não como ping falso.
5. O recorte acima é uma janela de 82 tokens a partir de 130560; não é o teste
   completo do usuário (≤160000) e não autoriza conclusão de tier.

## Próximos passos (dependentes de root)

1. Rodar `--execute --empty-rtt` no host com console ocioso e `--out` próprio;
   comparar RTT com a janela do trace correspondente.
2. Exportar CPU engine e GPU com a mesma contagem/janela e usar
   `--cpu-samples/--gpu-samples`; se não bater, o resultado fica `unknown`.
3. Só depois de 3 formatos de KV e QA, avaliar qualquer mudança de overhead de
   engine; esta ferramenta não propõe nem aplica fix.
