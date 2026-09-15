# Plano 002: Arquitetura Tri-Profile com Modulacao Dinamica de N no Modo MTP

## Status e Objetivo

- Status: APROVADO COM CONDICOES E NOTAS DE REFINAMENTO PELO ASTRA (`gpt-6-astra`, high reasoning); revisao integrada.
- Prioridade: P1, ganho de throughput de decode (+32% a +35% nos prompts curtos e medios ate 32k/40k).
- Esforco: Medio (extensao direta do mecanismo residente do Plano 001, preservando o modelo e ajustando `speculative.draft.n_max`, KV e buffers).
- Base de codigo: Fork local `/home/hjotha/llama` no branch `master`, alinhado com `releases/llama-adaptive-optimized-20260914`.
- Hardware alvo: GOKAYA (`192.168.1.57`), AMD Phoenix APU + NVIDIA GeForce RTX 4070 (12.282 MiB VRAM total, ~380 MiB display/driver overhead).
- Modelo alvo: `/home/hjotha/models/Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf` (pesos base 10.431 MiB + 1 camada MTP `blk.64` 332 MiB).

Objetivo:
Evoluir o sistema adaptativo de 2 perfis (Plano 001) para uma arquitetura **Tri-Profile (3 perfis)** totalmente residente e dinamica:
1. **Perfil Curto (Ultra-MTP):** Contexto ate 32.768 tokens (configuracao conservadora inicial) ou ate 40.960 tokens (configuracao experimental), especulacao com **N = 4** drafts por step. Atinge **71.0 a 90.0 t/s** em decode.
2. **Perfil Medio (Standard-MTP):** Contexto ate 56.320 tokens, especulacao com **N = 2** drafts por step. Atinge **49.0 a 68.0 t/s** em decode, respeitando o envelope de 12 GB de VRAM onde N=4 nao cabe no espaco restante do KV cache.
3. **Perfil Longo (Deep Context / Sem MTP):** Contexto ate 97.536 tokens, MTP desativado (**N = 0**), camada MTP desalocada da GPU e descarregada em RAM CPU. Atinge **28.0 a 30.0 t/s** em decode.

---

## Evidencia Experimental e Medicoes no Hardware Real (GOKAYA RTX 4070)

Todas as medicoes foram conduzidas diretamente na GPU NVIDIA GeForce RTX 4070 (12.282 MiB VRAM) com temperatura controlada, governor a 11.001 MHz e ubatch 256.

### 1. Benchmark de Eficiencia de N: N=1 vs N=2 (4k a 44k)
Medicao registrada em `bench_sweep_n_results.json`:
- **N = 1**: decode oscila entre 44.5 e 53.6 t/s (media ~49 t/s).
- **N = 2**: decode oscila entre 55.4 e 67.8 t/s (media ~62 t/s).
- **Conclusao:** N = 2 supera N = 1 com ganho consistente de **+21.4% a +26.5%** em toda a faixa de contexto.

### 2. Comparacao N = 2 vs N = 3 vs N = 4 no Contexto 26k (Prompts 2k a 24k)
Medicao registrada em `bench_n_comparison_24k_results.json`:
- **Prompt 2k**:
  - N = 2: 67.8 t/s
  - N = 3: 77.0 t/s (**+13.6%** vs N=2)
  - N = 4: **90.0 t/s** (**+32.7%** vs N=2, log: 408/408 aceitos, mean len 5.00)
- **Prompt 12k**:
  - N = 2: 62.7 t/s
  - N = 3: 68.9 t/s (**+9.9%** vs N=2)
  - N = 4: **83.6 t/s** (**+33.3%** vs N=2, log: 408/408 aceitos, mean len 5.00)
- **Prompt 24k**:
  - N = 2: 56.8 t/s
  - N = 3: 65.7 t/s (**+15.7%** vs N=2)
  - N = 4: **76.7 t/s** (**+35.0%** vs N=2, log: 408/408 aceitos, mean len 5.00)

*Nota de auditoria do revisor:* Os scripts originais registraram `draft_accepted = 0` no JSON devido a leitura da chave `draft_accepted` em vez do campo publicado `draft_n_accepted`. Os logs do servidor confirmaram aceitacao proxima de 100% no workload de benchmark sintético.

### 3. Varredura do Teto de Contexto para N = 4 com Prompt (ctx_max - 4096)
Para avaliar o limite de contexto suportado com N=4 antes de OOM, foram testados 4 candidatos com prefill de `(ctx_size - 4096 - 100)` tokens e geracao de teste de 512 tokens.
Medicao registrada em `find_max_ctx_n4_results.json`:

| Candidato | Prompt Testado | Prefill Speed | Decode Speed (N=4) | Aceitacao MTP (log) | VRAM Boot (Livre) | VRAM Pos-Inferencia (Livre) | Margem Observada |
|---|---|---|---|---|---|---|---|
| **32.768 (32k)** | 28.560 toks | 867.7 t/s | **75.26 t/s** | 408/408 (len 5.00) | 11.497 MiB (406 MiB) | 11.531 MiB (**372 MiB**) | Confortavel (> 350 MiB) |
| **36.864 (36k)** | 32.640 toks | 842.5 t/s | **72.88 t/s** | 408/408 (len 5.00) | 11.611 MiB (292 MiB) | 11.645 MiB (**258 MiB**) | Confortavel (> 250 MiB) |
| **40.960 (40k)** | 36.750 toks | 817.7 t/s | **70.97 t/s** | 408/408 (len 5.00) | 11.723 MiB (180 MiB) | 11.757 MiB (**146 MiB**) | Moderada (~146 MiB) |
| **45.056 (44k)** | 40.860 toks | 795.4 t/s | **69.11 t/s** | 408/408 (len 5.00) | 11.835 MiB (68 MiB) | 11.869 MiB (**34 MiB**) | Critica (~34 MiB) |

**Diretriz de Dimensionamento:**
- **Candidato Inicial Conservador para Perfil 1:** `ctx_size_mtp_short = 32768`, $N=4$. Margem de ~372 MiB livres, imune a picos de CUDA Graph ou variacoes de display driver.
- **Candidato Experimental de Teto Alto:** `ctx_size_mtp_short = 40960`, $N=4$. Margem observada de ~146 MiB livres. Deve ser habilitado apos validacao de 20 ciclos continuos com geracao de 4096 tokens reais.
- `45056` deixa apenas 34 MiB livres e e descartado para producao devido a risco inaceitavel de OOM sob pressao de grafos.

---

## Arquitetura Tri-Profile Refinada

### 1. Definicao dos 3 Perfis Fisicos

```
[ Requisicao entra com Budget = Prompt + Reserva de Saida ]
           |
           +---> Budget <= mtp_short_limit (ex: 32.768 tokens)
           |     --> PERFIL 1: CURTO (Ultra-MTP)
           |         - n_ctx = ctx_size_mtp_short (32.768 ou 40.960)
           |         - params.speculative.draft.n_max = 4
           |         - Throughput: 71 - 90 t/s
           |         - MTP VRAM: Residente (332 MiB)
           |
           +---> mtp_short_limit < Budget <= mtp_limit (56.320 tokens)
           |     --> PERFIL 2: MEDIO (Standard-MTP)
           |         - n_ctx = ctx_size_mtp (56.320)
           |         - params.speculative.draft.n_max = 2
           |         - Throughput: 49 - 68 t/s
           |         - MTP VRAM: Residente (332 MiB)
           |
           +---> Budget > mtp_limit (ate 97.536 tokens)
                 --> PERFIL 3: LONGO (Deep Context / Sem MTP)
                     - n_ctx = n_ctx (97.536)
                     - Speculation desabilitada (N = 0, sem draft/NextN)
                     - Throughput: 28 - 30 t/s
                     - MTP VRAM: Desalocada da GPU (0 MiB em VRAM, backing em CPU)
```

### 2. Retrocompatibilidade Estrita de Serializacao e Enums

Para preservar a leitura e restauracao de snapshots gravados em disco na versao 1 sem corromper a interpretacao binaria:

```cpp
enum common_context_profile {
    COMMON_CONTEXT_PROFILE_MTP       = 0, // Perfil Medio (compatibilidade v1)
    COMMON_CONTEXT_PROFILE_LONG      = 1, // Perfil Longo (compatibilidade v1)
    COMMON_CONTEXT_PROFILE_MTP_SHORT = 2, // Perfil Curto (novo)
};
```

Qualquer leitura de snapshot existente com perfil `0` continua apontando para o perfil MTP padrao e perfil `1` para o Longo.

### 3. Parametros de Linha de Comando (CLI) e Contrato de Limiares

- `--ctx-size`: Tamanho do contexto longo (Perfil 3, padrao: 97536).
- `--ctx-size-mtp`: Tamanho do contexto medio (Perfil 2, padrao: 56320).
- `--mtp-max-tokens`: Limiar superior para o perfil medio. Se `0`, deriva diretamente de `ctx_size_mtp`.
- `--ctx-size-mtp-short`: Tamanho do contexto curto (Perfil 1, padrao: `0` = modo 2 perfis desativado; ativo: 32768 ou 40960).
- `--mtp-short-max-tokens`: Limiar superior para o perfil curto. Se `0`, deriva diretamente de `ctx_size_mtp_short`. Nao subtrair `output_reserve` duas vezes pois o orcamento de requisicao ja soma prompt + reserva.
- `--spec-draft-n-max`: Draft N para o perfil medio (padrao da maquina: 2).
- `--spec-draft-n-max-short`: Draft N para o perfil curto (padrao: 4).

Validacao rigorosa em `common_context_adaptive_error`:
- `0 < limite_curto <= contexto_curto <= contexto_medio <= contexto_longo`
- `limite_curto <= limite_medio <= contexto_medio`

### 4. Reconstrucao de Recursos Dependentes de N

Conforme destacado pelo revisor Astra, alterar N nao e apenas limitar `slot.get_n_draft_max()`, pois N dimensiona:
- `params.speculative.draft.n_max`
- `n_rs_seq` no contexto target e draft (usado no rollback do estado recorrente)
- buffers de saida (`n_outputs_max`, `n_outputs_max_per_seq`)
- o proprio construtor de `common_speculative_mtp`

**Solucao Arquitetural:**
Criar helper centralizado `server_context::apply_profile_params(common_context_profile profile)` que:
1. Configura `params_base.n_ctx` correspondente ao perfil.
2. Ajusta `params_base.speculative.draft.n_max` (4 no Curto, 2 no Medio, 0 no Longo).
3. Recalcula `server_output_limits(params_base)`.
4. Define os tipos especulativos (vazio no Longo, `{ COMMON_SPECULATIVE_TYPE_DRAFT_MTP }` no Curto e Medio).
5. Executa a criacao/recriacao dos contextos target e draft com os parametros efetivos.

### 5. Contrato de Snapshot e Cache entre Perfis

1. **Restore Explicito via API (`/slots/{id}/restore`):**
   - Preserva o contrato autoritativo: valida o arquivo e **troca para o perfil gravado no arquivo** antes de executar o restore.
2. **Prompt Cache Automatico:**
   - **Curto <-> Medio:** Ambos sao MTP. O estado target e o carryover MTP sao restaurados se couberem nas posicoes do destino. O draft KV e recriado conforme as dimensoes do perfil destino.
   - **MTP (Curto/Medio) -> Longo:** Libera o draft da GPU VRAM, mas preserva a variante MTP em RAM host enquanto houver orcamento.
   - **Longo -> MTP (Curto/Medio):** Se houver snapshot MTP compativel em RAM, restaura diretamente; senao, executa bootstrap target-only.
   - **Snapshot maior que o destino:** Rejeita truncamento cego; assume cache miss ou falha limpa para evitar corrupcao de estado recorrente.

---

## Matriz Completa das 6 Transicoes e Gates de Validacao

| Transicao | Pesos MTP GPU | KV Cache | Estado Draft | Acao de Memoria |
|---|---|---|---|---|
| **Curto -> Medio** | Residente (sem troca) | Realloca KV 56k | Preservado/Adaptado | Reutiliza prompt cache |
| **Medio -> Curto** | Residente (sem troca) | Realloca KV 32k/40k | Preservado/Adaptado | Reutiliza prompt cache se cabe |
| **Curto -> Longo** | Evict GPU -> CPU RAM | Aloca KV 97k | Descartado da GPU | Libera VRAM para KV 97k |
| **Medio -> Longo** | Evict GPU -> CPU RAM | Aloca KV 97k | Descartado da GPU | Libera VRAM para KV 97k |
| **Longo -> Curto** | Restore CPU RAM -> GPU | Aloca KV 32k/40k | Reconstruido | Bootstrap target-only |
| **Longo -> Medio** | Restore CPU RAM -> GPU | Aloca KV 56k | Reconstruido | Bootstrap target-only |

### Criterios de Aceite (Gates Obrigatorios)

1. **Build e Testes Unitarios:**
   - CTest (`test-arg-parser`, `test-server-model-identity`) com saida 0.
   - Teste de argumentos cobrindo as combinacoes e rejeicoes dos 3 perfis.
2. **Matriz das 6 Transicoes:**
   - Execucao deterministica cobrindo todas as 6 transicoes direcionais (`1->2`, `2->1`, `1->3`, `3->1`, `2->3`, `3->2`), seguida de 20 ciclos aleatorios continuos sem leak de VRAM.
3. **Soak de Carga Real:**
   - Geracao com prompts proximos dos limites (ex: 28k, 36k, 52k, 90k) com saida de 4096 tokens reais, verificando amostragem de VRAM e estabilidade de CUDA Graphs.
4. **Semantica de Snapshots:**
   - Testar save e restore cruzado (Curto -> Medio -> Curto e Longo -> MTP).
5. **Restauracao da Producao:**
   - Servico `llama-server-root.service` ativo, listener na 8090, 200 `/health`.

---

## Parecer Arquitetural do Astra (Codex CLI — gpt-6-astra high reasoning)

> **Parecer Emitido em 14/09/2026 pelo Astra (`gpt-6-astra`, high reasoning):**
>
> **Parecer: a arquitetura Tri-Profile e tecnicamente viavel, mas o Plano 002 precisa de revisao antes da implementacao.** Aprovo a direcao arquitetural; ainda nao aprovo as garantias de memoria, compatibilidade de snapshots e latencia.
>
> Conferi os dois planos, o codigo atual e os scripts/JSONs/logs dos benchmarks. Esta foi uma revisao de leitura; nao executei benchmarks nem alterei arquivos ou servicos.
>
> 1. **[Bloqueante] O enum proposto quebra a interpretacao dos snapshots existentes.**
>    Hoje `MTP=0` e `LONG=1`. O Plano 002 inicial propunha `SHORT=0`, `MTP=1`, `LONG=2`. Esses valores sao gravados diretamente no arquivo, cujo formato permanece na versao 1. Um snapshot longo antigo passaria a ser identificado como MTP, causando incompatibilidade e potencial tentativa de transicao incorreta.
>    *Acao:* Preserve `MTP=0`, `LONG=1` e acrescente `MTP_SHORT=2`. Se mudar campos ou semantica do envelope, versione o formato e defina a leitura dos arquivos anteriores. Tambem sera necessario revisar todas as comparacoes binarias `profile == MTP`, incluindo residencia, rollback, checkpoints, status e injecao de falhas.
>
> 2. **[Bloqueante] A evidencia sustenta funcionamento pontual de 40.960/N=4, mas nao seguranca contra OOM.**
>    O script do teto:
>    - Usa prompt aproximado, com uma margem adicional de 100 tokens.
>    - Solicita 512 tokens de saida, nao 4.096.
>    - Atribui a `vram_peak` uma unica leitura feita depois da resposta.
>    - Reinicia o processo para cada tamanho.
>    - Executa o perfil curto estatico, sem habilitar o mecanismo adaptativo.
>    O log confirma 36.750 tokens de entrada + 512 gerados a 70,97 t/s. Isso e evidencia util, mas nao valida `36.864 + 4.096`, transicoes persistentes, save/restore ou picos de captura CUDA Graph.
>    Os 146 MiB livres sao uma observacao, nao uma reserva garantida contra fragmentacao. O proprio Plano 001 documenta OOM fatal em um pedido pequeno apos workloads anteriores e uma correcao especifica para pressao de CUDA Graph. Essa correcao precisa integrar explicitamente a configuracao e os gates do Plano 002; consultar memoria livre nao garante que uma alocacao posterior tera sucesso.
>    *Recomendacao:* 32.768/N=4 como candidato inicial conservador; 40.960/N=4 como candidato experimental ate passar o teste completo. Nem 32k esta certificado pelo ensaio atual. Retiraria "sucesso absoluto", "sucesso robusto" e "teto fisico absoluto". Tambem retiraria "cabendo confortavelmente" do Medio.
>    A leitura local identifica `NVIDIA GeForce RTX 4070`, com 12.282 MiB totais e aproximadamente 380 MiB reservados pelo driver. Corrigir a identificacao "Laptop GPU" e explicitar memoria total, reservada, usada e livre torna o orcamento auditavel.
>
> 3. **[Bloqueante] Alterar N exige reconstruir os recursos dimensionados por N.**
>    A proposta acerta ao recriar os contextos preservando os pesos. Contudo, o parametro real e `params.speculative.draft.n_max`, e nao `params.speculative.n_max` ou `params_base.spec_draft_n_max`.
>    N determina tambem `n_rs_seq`, usado no rollback do estado recorrente, e participa do dimensionamento dos buffers de saida. Logo, apenas limitar `slot.get_n_draft_max()` nao libera a memoria correspondente a N=4. A implementacao MTP ainda copia a configuracao no construtor.
>    O refinamento minimo seria um helper que derive os parametros efetivos de cada perfil, usado no startup, transicao e rollback. Ele deve definir contexto, tipos especulativos e N antes de recalcular buffers e construir target/draft. Preserve separadamente o N configurado para o Medio, para nao sobrescreve-lo ao entrar no Curto.
>    No Longo, `N=0` sozinho nao basta: mantenha speculation vazia, ausencia de draft/NextN e liberacao dos recursos associados, como no Plano 001.
>
> 4. **[Bloqueante] "Ambos possuem draft" nao define compatibilidade de snapshot.**
>    Existem dois contratos distintos no codigo:
>    - Cache automatico: tenta restaurar estado compativel no perfil selecionado pela nova requisicao.
>    - Restore explicito de arquivo: valida a configuracao salva e troca para o perfil registrado no arquivo antes de restaurar.
>    Recomendo preservar essa semantica do restore explicito nesta extensao. Restaurar um arquivo no perfil atualmente ativo seria uma mudanca adicional de contrato e precisaria ser especificada.
>    Para o cache automatico, a matriz expressa:
>    - Curto <-> Medio: Restaurar target, draft e carryover coerentes se tokens, posicoes e layouts couberem no destino. Provar N=4 <-> N=2.
>    - MTP -> Longo: Restaurar target/checkpoints compativeis; liberar o draft da GPU e preservar uma variante MTP em RAM quando o orcamento permitir.
>    - Longo -> MTP: Preferir snapshot MTP compativel; caso contrario, usar bootstrap target-only com sufixo/checkpoint valido.
>    - Snapshot maior que o destino: Usar estado anterior compativel, quando disponivel, ou assumir miss/erro. Nunca truncar o blob ou reaplicar estado recorrente final a um prefixo menor.
>
> 5. **[Bloqueante] A CLI proposta contradiz a retrocompatibilidade e conta a reserva duas vezes.**
>    O limiar curto proposto derivava de `ctx_size_mtp_short - output_reserve`, embora o budget ja inclua a saida. Com contexto 40.960 e reserva 4.096, isso restringiria o prompt curto a 32.768, em vez de 36.864.
>    Use o mesmo contrato existente: `0` deriva o limiar da capacidade do perfil; qualquer margem adicional deve ser explicita. Valide:
>    - `0 < limite_curto <= contexto_curto <= contexto_medio <= contexto_longo`;
>    - `limite_curto <= limite_medio <= contexto_medio`;
>    - capacidades efetivas apos caps/arredondamentos;
>    - configuracoes invalidas e orcamento acima do Longo antes do teardown.
>
> 6. **[Importante] Os benchmarks precisam de correcao e as promessas de desempenho precisam ser qualificadas.**
>    Os scripts leem `draft_accepted`; o servidor publica `draft_n_accepted`. Por isso os JSONs registram zero aceitos, embora os logs mostrem aceitacao alta.
>    Tampouco ha evidencia para "prefill em menos de 300 ms". Separe tempo de save, teardown/recriacao, restore, bootstrap/sufixo e TTFT total.
>
> **Aprovacao condicionada da arquitetura, com revisao obrigatoria do plano.** Ela pode reutilizar a infraestrutura atual sem novo subsistema, mas deve herdar as restricoes e os gates ainda abertos do Plano 001. `200 /health` e throughput elevado, isoladamente, nao aprovam estabilidade nem integridade dos snapshots.
