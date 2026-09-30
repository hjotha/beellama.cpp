# Revisão do custo de verificação especulativa — 2026-09-29

Revisão estática do checkout `feat/dflash2-shadow-observation`, HEAD `3d4128354`
mais alterações locais preservadas. Nenhuma mudança de inferência, compilação,
nova execução de modelo ou operação de serviço nesta revisão. Os números abaixo
são contagens extraídas dos logs já existentes da bateria
`benchmarks/mtp-dflash2-hybrid-active-20260929/paired/paired-v1`.

## Resumo e ordem sugerida

| Prioridade | Trabalho | Natureza |
| --- | --- | --- |
| 1 | Condicionar snapshots à possibilidade de replay e corrigir ownership do clone | Mudança localizada em CPU, sem mudar o cálculo do alvo. |
| 2 | Consolidar leitura de outputs com uma barreira de sincronização | Remover chamadas repetidas preservando a barreira necessária. |
| 3 | Rota greedy do verificador retornando somente IDs | Maior oportunidade de reduzir tráfego na eGPU; completar a infraestrutura backend existente. |
| 4 | Atualizar MTP apenas com o prefixo realmente confirmado | Muda a ordem do pipeline; precisa comparar trabalho poupado com sobreposição perdida. |
| 5 | GDN CUDA com leitura direta dos estados por índices | Otimização maior no forward Qwen; suporte correspondente já existe em CPU/Metal. |

Corrigir a separação das métricas de tempo antes de atribuir ganhos a essas
etapas. Não há porcentagem de ganho demonstrada por esta revisão.

## 1. Snapshots incondicionais

Em `tools/server/server-context.cpp`, `post_decode()` clona o sampler e copia
o loop guard, telemetria, contadores de reasoning/visible, flags de parada e
string de motivo em toda verificação. Esse estado salvo só é consumido no ramo
que restaura checkpoint e reexecuta o prefixo.

O teste da necessidade de checkpoint já existe:
`server_speculative_rollback_requires_checkpoint(type, max_rollback, n_draft)`
em `tools/server/server-task.h`. Usar a possibilidade máxima **antes** de amostrar
para decidir se o snapshot precisa existir. O caminho FULL ou fora da janela
continua salvando/restaurando tudo, incluindo callback/loop guard.

`common_sampler_clone()` copia o vetor completo `cur`, além do estado da cadeia.
No caminho CPU ele foi dimensionado por `n_vocab` em `set_logits()`.
Não basta esperar a rejeição para fazer o snapshot: nesse momento o sampler já
foi alterado; a condição precisa expressar possibilidade de replay antecipadamente.

Na referência MTP n4, cinco repetições por cenário:

| Cenário | Verificações | Replays observados | Linhas verificadas | Linhas descartadas |
| --- | ---: | ---: | ---: | ---: |
| Repetição | 205 | 0 | 955 | 10 (1,05%) |
| Código | 285 | 0 | 1085 | 175 (16,13%) |
| Longo | 255 | 0 | 945 | 50 (5,29%) |

São **745 snapshots de sampler sem replay** nessa bateria. A segurança da
eliminação deve vir da capacidade declarada do contexto, não desse histórico
estatístico. O Qwen35 com MTP n4 configura `n_rs_seq=4`; o caminho normal permite
rollback dentro dessa janela.

### Problema de ownership associado

`common/sampling.cpp: common_sampler_clone()` copia `cur`, porém copia
`cur_p.data` ainda apontando para o vetor do sampler de origem.
`common_sampler_copy()` já reponta esse ponteiro para o destino; o clone deve
fazer o mesmo. Consultar os candidatos do clone antes de um novo `sample()` pode
observar memória do original ou invalidada. O restore atual usa `copy()`, que
reponta o destino: o achado não demonstra que esse foi o motivo das divergências
de texto das baterias anteriores. É uma correção de ownership a validar.

## 2. Leituras de output e sincronizações

`common_sampler_sample()` chama `llama_synchronize()` e em seguida `set_logits()`.
Este consulta os getters de probs/logits/IDs do backend e, quando necessário,
os logits crus; o getter do token pronto é consultado depois. Esses getters
voltam a sincronizar. Na rota CPU comum são seis chamadas por token amostrado;
`llama_context::synchronize()` chama o scheduler mesmo sem novos tokens enfileirados.

Proposta: um acesso interno que sincronize e forneça a visão dos outputs prontos,
sem repetir barreiras para cada getter. Manter a ordem `sample → accept → callback`
por token. Uma visão por bloco precisa preservar a validade até o próximo decode
e tratar callbacks capazes de alterar/enfileirar trabalho; usar fallback quando
o contrato não for garantido.

Essas chamadas repetidas não significam seis execuções do modelo ou seis esperas
longas. A maior parte pode encontrar o dispositivo já pronto; medir seu custo.
Não remover simplesmente a barreira do servidor depois de `llama_decode()`:
o histórico `c7a6b18d40` a ampliou aos sub-batches de prefill por causa de ownership
de KV/estado. A barreira de decode com outputs já existia antes dessa alteração.

## 3. Greedy na GPU, com retorno somente dos IDs

O GGUF alvo usado tem arquitetura `qwen35`, `n_vocab=248320` e embedding 5120,
confirmados lendo os metadados locais. As respostas da referência registram
`generation_settings.backend_sampling=false`.

`src/llama-context.cpp` extrai `n_outputs × n_vocab × sizeof(float)` quando
`needs_raw_logits()` é verdadeiro. Uma rodada completa com quatro propostas e
uma âncora tem cinco linhas: **4.966.400 bytes (4,736 MiB) de logits**. Isso não
inclui features/embeddings necessários ao MTP.

A opção `--backend-sampling` já existe. Contudo, ativá-la não estabelece uma
rota de retorno somente de IDs: `llama_sampler_greedy_backend_apply()` define
`data.sampled` e mantém `data.logits`; `build_sampler()` publica esse tensor e
o contexto copia também os outputs de sampling. Além disso, o helper comum
materializa candidatos antes de consultar o token já escolhido no backend.

Completar uma rota explícita para o verificador greedy:

- Calcular a escolha no backend e devolver IDs para a comparação de prefixo.
- Preservar `accept()` e callbacks sequenciais do servidor.
- Manter scores/candidatos para consumidores que precisam deles, especialmente
  o draft MTP com `p_min` e requisições de probabilidades.
- Elegibilidade pela cadeia efetiva, não apenas `temperature=0`: gramática,
  reasoning budget, penalties dependentes do histórico, logprobs e distribuições
  de rejeição precisam da semântica atual ou de suporte backend específico.
- Backend penalties já rejeita `n_outputs_max_per_seq>1`; não contornar esse
  controle. Verificar também os limites de outputs e suporte do split utilizado.

Usar as APIs/graphs de sampling existentes; não recriar um verificador privado.

## 4. Catch-up MTP depois da aceitação

Após o forward do alvo, o servidor chama `common_speculative_process()` antes
de escolher o prefixo aceito. O `process()` MTP decodifica o batch inteiro com
os hiddens reais do alvo. Só depois `accept()` seleciona o carry correto e o
servidor remove a cauda rejeitada.

Proposta a experimentar: separar captura de hidden states da atualização KV
MTP e realizar esta última apenas para âncora + tokens aceitos. No cenário de
código, a bateria descartou 175 das 1085 linhas verificadas (16,13%). Essa é a
fração de linhas de verificação, **não uma estimativa de ganho em tok/s**.

O catch-up do prefixo aceito continua obrigatório: os hiddens previstos pelo MTP
não substituem os hiddens reais do alvo. Também há uma troca de escalonamento:
o decode MTP atual pode se sobrepor ao sampling CPU; adiá-lo pode perder essa
sobreposição. Preservar prefill, zero aceitos, replay, bootstrap e ownership das
linhas antes de outro decode sobrescrever os outputs.

## 5. GDN CUDA: otimização maior no próprio alvo

`src/models/qwen35.cpp` já tem caminho para ler o estado recorrente diretamente
do cache por índices, evitando um gather por camada. O construtor habilita esse
caminho em CPU/Metal; CUDA continua usando a forma com gather.
`ggml/src/ggml-cuda/ggml-cuda.cu` rejeita explicitamente
`GGML_OP_GATED_DELTA_NET` com `src[6]` porque essa variante não foi implementada.

Implementar a leitura indexada no kernel CUDA poderia reduzir cópias/lançamentos
nas camadas recorrentes, mantendo a janela de rollback e os estados intermediários.
Não basta retirar o bloqueio em `supports_op`: isso faria o kernel interpretar
o layout incorretamente. Este cache de estados recorrentes é distinto do antigo
verificador especulativo ring/tape removido do fork.

## 6. Métrica de verificação incompleta

Em `post_decode()`, `t_verify_start` começa **depois** de `llama_decode(ctx_tgt)`
e `common_speculative_process()`. Portanto `verify_ms` mede sampling/accept e
parte da manutenção posterior, sem o forward principal nem o catch-up MTP.

Isso afeta a interpretação de `MTPPHASE phase=verify`. Também merece correção
no controlador DFlash: `server-adaptive-dm.h` usa
`n * draft_ms / ref_depth + verify_ms + accept_ms` para extrapolar profundidades
ainda sem medição. Nessas estimativas falta o custo dominante do alvo/process.
As comparações por `cycle_ms` realmente medido incluem mais etapas; não confundir
essa falha de extrapolação com a validade de todos os scores do controlador.

Separar forward/sincronização do alvo, atualização do draft, sampling/aceitação,
rollback e emissão. Atribuir por slot/coorte corretamente e preservar a medição
de tempo total por token. Uma instrumentação que introduza barreiras extras deve
ser comparada com a execução normal antes de orientar decisões de performance.
