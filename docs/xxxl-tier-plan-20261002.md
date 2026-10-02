# Plano — perfil adaptativo XXXL (2026-10-02)

## Objetivo

Hoje o XXL de 131072 usa a Radeon desde o token 0, porque o placement misto é
decidido pela capacidade do perfil (`docs/xxl-prefill-bottleneck-20261002.md`,
§10–11). A configuração anterior (`ctx-size-xxl = 102400`,
`spec-draft-n-max-xxl = 0`, sem `remote-attn`) cabia inteira na RTX 4070.

Adicionar um sexto perfil adaptativo, `xxxlong` (sufixo de preset `-xxxl`),
para que:

| Perfil | Contexto | MTP | Atenção |
|---|---:|---|---|
| S/M/L/XL | inalterados | inalterados | 100% CUDA |
| XXL | 102400 | off | 100% CUDA KVarN4 |
| XXXL | 131072 | off | mista: CUDA KVarN4 + Radeon Q4 (`remote-attn-min-context = 131072`) |

A Radeon só entra quando prompt + reserva de saída > 102400.

## Regras

- Branch `feat/adaptive-xxxlong-tier` no checkout principal
  `/home/hjotha/beellama.cpp` (exceção beellama: sem worktree).
- Não alterar `tools/server/server-task.cpp` (mudança local do guard de 1%,
  não commitada).
- Não alterar `/home/hjotha/router-production.ini`, units systemd nem a produção.
- Não commitar nem dar push: o revisor commita após validar.
- Comportamento com XXXL desligado (`ctx-size-xxxl = 0`, o padrão) deve ser
  idêntico ao atual.

## Mudanças (espelhar exatamente o padrão de `xxlong`)

1. **`common/common.h`**
   - Campos novos ao lado dos de `xxlong` (linhas ~596-632):
     `ctx_size_xxxlong`, `xxxlong_max_tokens`, `batch_size_xxxlong`,
     `ubatch_size_xxxlong`, `spec_draft_n_max_xxxlong`,
     `cache_type_k/v_xxxlong`, `cache_kvarn_bits_k/v_xxxlong`, `kvarn_xxxlong`,
     `spec_draft_type_k/v_xxxlong`, `spec_draft_kvarn_bits_k/v_xxxlong`,
     `spec_draft_kvarn_xxxlong`.
   - `COMMON_CONTEXT_PROFILE_XXXLONG = 5` **no fim do enum** (linha ~1240): os
     valores existentes não podem mudar, porque são serializados nos snapshots
     de slot.
   - Declarar `common_context_xxxlong_limit`.
2. **`common/arg.cpp`**
   - Opções novas espelhando as de `xxlong` (linhas ~2453-2625):
     `--ctx-size-xxxl/--ctx-size-xxxlong`, `--xxxl-max-tokens/--xxxlong-max-tokens`,
     `--batch-size-xxxl`, `--ubatch-size-xxxl`, `--spec-draft-n-max-xxxl`,
     `--spec-draft-type-k/v-xxxl`, `--cache-type-k/v-xxxl`, cada uma com o env
     `LLAMA_ARG_*_XXXLONG`.
   - Normalizadores `common_params_xxxlong_kvarn_normalize` e
     `common_params_spec_draft_xxxlong_kvarn_normalize` (como 1717/1810),
     chamados junto aos de `xxlong` (~1891-1894).
3. **`common/common.cpp`**
   - `common_context_xxxlong_limit`.
   - `common_context_profile_for_budget` (~3012-3040): depois de decidir que o
     orçamento passa do XL, escolher XXL se `budget <= xxlong_limit`, senão
     XXXL se `ctx_size_xxxlong > 0`. Com XXXL desligado, manter o
     comportamento atual (XXL acima do XL).
   - `common_context_adaptive_error` (~3085-3200): validações espelhadas
     (não negativo; `--xxxlong-max-tokens` exige `--ctx-size-xxxlong`;
     `0 < limit <= ctx_size_xxxlong`; não menor que o teto anterior (XXL, ou
     XL/long); XXXL exige XXL configurado). `max_adaptive_ctx` passa a
     considerar XXXL primeiro.
4. **`tools/server/server-context.cpp`** — todo `switch`/`if` sobre
   `COMMON_CONTEXT_PROFILE_XXLONG` ganha o caso XXXL:
   - membros `adaptive_*_xxxlong` (~3173-3196) e sua inicialização a partir dos
     params (~4572-4632);
   - `adaptive_max_ctx()` (~3198) considera XXXL primeiro;
   - `adaptive_draft_n_for_profile`, `draft_n_max` (std::max inclui XXXL),
     `apply_profile_params`, seleção de `draft_kv`;
   - `adaptive_status_profile_name` → `"xxxlong"`;
   - validação do perfil no snapshot de slot (~2221) aceita XXXL;
   - `raw_ctx` do snapshot (~7707) mapeia XXXL → `ctx_size_xxxlong`;
   - `max` de contexto em ~3754 inclui `ctx_size_xxxlong`;
   - hook de teste `adaptive_test_profile_matches` (~200-216): `find("xlong")`
     e `find("xxlong")` também casam substrings mais longas; testar
     `"xxxlong"` antes e fazer XXL/XL não casarem com o perfil maior.
5. **`tools/server/server-models.cpp`** (~2796): incluir
   `LLAMA_ARG_CTX_SIZE_XXXLONG` no cálculo do `n_ctx` do preset.
6. **Outros**: procurar (`rg -n -i 'xxlong|XXLONG'`) qualquer outro uso —
   README/docs de flags do servidor, `tools/server/README*.md`, a lista de chaves
   do preset — e espelhar.

## Testes a adicionar

- `tests/test-arg-parser.cpp` (perto de 839-910): um perfil de seis níveis
  com `-xxxl`; asserções dos campos; seleção de perfil por orçamento nos
  limites (XL máx, XXL máx, XXL máx + 1 → XXXL, XXXL máx); erros de validação
  (xxxl menor que xxl, max-tokens sem ctx, XXXL sem XXL).
- Rodar e registrar a saída: `test-arg-parser` e qualquer teste adaptativo
  existente que compile no build (`test-adaptive-dm`, se aplicável).

## Build (o revisor executa ou autoriza)

`cmake --build /home/hjotha/beellama.cpp/build-optimized --target llama-server test-arg-parser -j 8`

## Validação em hardware (executada pelo revisor, em janela de manutenção)

Router temporário na porta 18130, INI copiado com XXL = 102400/MTP0 e XXXL =
131072/MTP0, `TMPDIR=/var/tmp/beellama-spool`, slot-save isolado, telemetria e
aborto por fence:

1. **XXL só CUDA**: prompt frio de ~90k tokens. Esperado: perfil `xxlong`,
   `backend=off` (sem Vulkan), sem OOM, prefill na faixa da CUDA.
2. **XXL → XXXL**: o mesmo prefixo + sufixo levando o orçamento acima de
   102400. Esperado: perfil `xxxlong`, `Vulkan-layers=5/16`, reuso do prefixo
   de ~90k (`cache_n` próximo do prefixo, sem refazer o prompt) via
   handoff/snapshot.
3. **XXXL → XXL**: voltar ao prefixo de ~90k + sufixo curto. Esperado:
   restauração com conversão Q4 → KVarN, sem refazer o prompt.
4. **Disco**: com `slot-save-auto` no diretório isolado, reiniciar o router e
   restaurar o snapshot XXXL (`cache_source=disk`).
5. Sem `restore failed`, `tmpfs`, `ErrorDeviceLost` nem alertas de kernel.

## Produção (depois da validação)

- INI: `ctx-size-xxl = 102400`, `spec-draft-n-max-xxl = 0`; bloco novo `*-xxxl`
  com `ctx-size-xxxl = 131072`, `batch/ubatch-size-xxxl = 256`,
  `cache-type-k/v-xxxl = kvarn4`, `spec-draft-type-k/v-xxxl = kvarn4`,
  `spec-draft-n-max-xxxl = 0`; manter `remote-attn-min-context = 131072`.
- Snapshots antigos de `/home/hjotha/llama-slot-cache/` removidos antes da
  troca.
- Restart com a produção ociosa e verificação de `--ctx-size-xxlong 102400` /
  `--ctx-size-xxxlong 131072` em `/v1/models`.

## Rollback

`router-production.ini.backup-xxl-mtp2-20261002` (ou o backup novo) e as
bibliotecas atuais copiadas antes do build.

## Resultado da implementação e validação (2026-10-02)

Implementação feita pelo OpenCode (`opencode/muse-spark-1.3-contributor-free`,
`--variant xhigh`) seguindo este plano; revisão do principal pediu uma correção:
`adaptive_slot_snapshot_carries_mtp()` passou a inferir estado MTP pelo payload
também em XXLONG e XXXLONG. Build `llama-server test-arg-parser test-adaptive-dm`
exit 0; `test-arg-parser` exit 0; `test-adaptive-dm` exit 0; `llama-server --help`
lista as opções `-xxxl`.

Validação em hardware (router temporário 18130, INI copiado, XXL 102400/MTP0,
XXXL 131072/MTP0, `TMPDIR=/var/tmp/beellama-spool`; artefatos em
`/home/hjotha/beellama-mixed-kv-20261001-130910/xxxl-20261002/`):

Primeira rodada (`run6a/`, `slot-save-max-mb = 4096`):

- S1 e S2 passaram; S3 (XXXL → XXL) e S4 (restart → XXXL) refizeram o prompt
  (`no_common_prefix`). Causa: cada snapshot de ~96–104k tem ~1,9–2,1 GB e cada
  request grava `reason=prompt` e `reason=discard`; com 4096 MB, os snapshots
  do outro perfil eram despejados. Snapshots de layout diferente são ignorados
  (`unified snapshot ignored ... KV/attention/RoPE layout mismatch`); só o slot
  vivo faz handoff XXL → XXXL.

Segunda rodada (`slot-save-max-mb = 16384`), sem alertas de kernel:

| Etapa | Perfil | Reuso | Tempo |
|---|---|---|---:|
| S1 frio 96.000 | xxlong (`backend=off`) | — | 159,9 s |
| S2 → 104.000 | xxxlong (`Vulkan-layers=5/16`) | `mixed_handoff`, 95.872 | 185,7 s (8.128 processados) |
| S3 → 96.000+256 | xxlong | `disk`, 96.000 | 8,6 s |
| S4 restart → 104.000+256 | xxxlong | `disk`, 104.000 | 16,4 s |

Lacuna encontrada: após restart, entrar no XXXL a partir de um snapshot só-XXL em
disco (sem slot vivo) não convertia KVarN → misto e refazia o prompt. Causa: o
índice de disco só aceita layout igual ou Q4 → KVarN
(`auto_index_scan_locked`, `auto_convertible_q4_layout`), e o handoff puro → misto
(`server_mixed_kv_handoff::try_handoff`) só consome estados do cache de prompt em
RAM capturados de um contexto puro vivo. Corrigida na seção seguinte. Referência
de custo frio do XXXL (primeira rodada, S4): 104.256 tokens em 1120,1 s.

## Restauração em etapas de snapshot puro em disco para perfil misto (2026-10-02)

`server_context_impl::adaptive_stage_pure_disk_prefix` (`tools/server/server-context.cpp`),
chamada em `process_single_task` antes da troca de perfil. Quando a tarefa entra num
perfil misto (remote-attn ativo para aquele contexto), não há prefixo residente
(slot vivo ou cache de prompt em RAM) e o disco guarda um snapshot KVarN puro que é
prefixo verificado da requisição (>= 4096 tokens):

1. troca para o perfil puro que comporta o snapshot (`common_context_profile_for_budget`);
2. restaura o snapshot do disco nesse perfil (`auto_index_lookup` +
   `auto_restore_into_slot`, que já valida modelo, layout, tokens e checksum);
3. a troca normal para o perfil misto grava o slot no cache de prompt em RAM e o
   handoff existente converte KVarN → Q4 nas camadas remotas.

Qualquer falha (sem candidato, restauração recusada, checkpoint ausente) segue o
caminho antigo: prompt frio no perfil misto.

Validação (27B real, produção parada, instância temporária, `TMPDIR` em disco):

| Escala | Caminho | Resultado |
|---|---|---|
| XXL 12288 / XXXL 24576, 5 camadas remotas, 9.984 → 16.000 tokens | frio no XXXL | 37,1 s |
| idem | handoff com slot vivo | 20,2 s, 9.856 reaproveitados |
| idem | restart + disco (etapas) | 20,9 s, `mixed_handoff`, 9.984 reaproveitados |
| XXL 102400 / XXXL 131072, 96.000 → 104.000 tokens | handoff com slot vivo | 176,3 s, 95.872 reaproveitados |
| idem | restart + disco (etapas) | 183,4 s, `mixed_handoff`, 96.000 reaproveitados |
| idem (antes da correção) | restart + disco | 1120,1 s, prompt refeito |

Distribuição do primeiro token (top-20): escala pequena, KL frio↔etapas 1,5e-06,
frio↔vivo 6,5e-06, vivo↔etapas 1,3e-05, top-1 igual nos três; escala real, KL
vivo↔etapas 6,1e-07, top-1 igual. No caso real, restaurar do disco levou 2,9 s, o
handoff ~25 s (`convert_ms=18553`) e o restante é prefill de 8.000 tokens no
XXXL. Sem `restore failed`, `tmpfs`, `ErrorDeviceLost` nem alertas de kernel.

Um primeiro teste em escala real usou `OFFSET=50000` e, por limite do corpus
(146.328 tokens), ficou em 96.329 tokens dentro do XXL; foi descartado e repetido
com `OFFSET=30000`.
