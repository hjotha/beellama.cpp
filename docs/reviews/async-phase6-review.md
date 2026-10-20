# Review da fila assíncrona de dois slots da Fase 6

Parecer: **homologação pendente**. A separação dos buffers por slot permite preparar um payload enquanto o worker faz I/O, mas há bloqueadores de concorrência, lifetime e publicação de estado. A suíte atual passa com ASan/UBSan; a execução com ThreadSanitizer terminou em exit66, com uma inversão de locks e duas data races reportadas. Um teste CPU com o produtor host e o validator real do Xbox também confirmou rejeição de um payload válido de prefill devido à escolha incorreta de RESET.

Escopo: checkout local BeeLlama.cpp, branch `docs/hybrid-attention-master-review`, base `aaa63758a968577c593654d27b80de748dd15c6a`, incluindo as alterações locais da Fase 6. A admissão remota foi testada pela função C++ compartilhada do XLLAMA, sem conectar ao console. Não houve alteração dos fontes de implementação, commit, deploy ou benchmark GPU.

## P1 Inversão de locks ao retornar do backpressure

Referência: [rkva_async_submit_locked](../../ggml/src/ggml-remote-attn/ggml-remote-attn.cpp#L1434), especialmente linha1449; [controle](../../ggml/src/ggml-remote-attn/ggml-remote-attn.cpp#L944), linhas947–951.

`wait_for(alock,...)` retorna com `async_mutex` adquirido. Em seguida, `ctx_lock.lock()` adquire `ctx->mutex` sem soltar `async_mutex`. Isso cria a ordem `async→ctx`, enquanto controles, estatísticas e submits normais usam `ctx→async`.

Interleaving alcançável:

1. O terceiro submit encontra os dois slots ocupados, libera ctx e espera.
2. Um controle adquire ctx.
3. Timeout ou notificação faz o submit readquirir async.
4. O controle espera async; o submit espera ctx.

O ThreadSanitizer confirmou as duas arestas: `rkva_lock_ctx_drained:951` e `rkva_async_submit_locked:1449`. O diagnóstico é de potencial deadlock; o run observado terminou, sem um travamento permanente.

Correção: nunca adquirir ctx segurando async. Soltar async antes de readquirir ctx e depois readquirir async na ordem canônica, em loop, revalidando free slot, falha, admissão e identidade da operação. Manter um deadline absoluto para que despertares/retries não renovem o timeout.

```cpp
// Esboco de ordem de locks; depende da barreira de admissao descrita abaixo.
for (;;) {
    // Entrada: ctx_lock adquirido.
    std::unique_lock<std::mutex> alock(ctx->async_mutex);
    if (!accepting || submitted_epoch != current_epoch) return cancelled;
    if (auto * slot = find_free_slot()) {
        reserve(slot);
        break;
    }
    alock.unlock();
    ctx_lock.unlock();
    {
        std::unique_lock<std::mutex> wait_lock(ctx->async_mutex);
        wait_until(deadline, wait_lock, capacity_or_cancel_predicate);
    } // async liberado antes de readquirir ctx
    ctx_lock.lock();
    // Repetir admissao e validacao; nenhuma observacao anterior e uma reserva.
}
```

Reter ctx durante I/O, na versão anterior, era uma serialização onerosa. Essa retenção, isoladamente, não demonstrava uma inversão. A versão atual introduz uma inversão verificável no retorno da espera.

## P1 Uma falha global permite reciclar um slot ainda RUNNING

Referências: [condição do wait](../../ggml/src/ggml-remote-attn/ggml-remote-attn.cpp#L1002) e reset incondicional na linha1026.

O wait termina não apenas em `DONE_OK/DONE_ERR`, mas também quando observa `ctx->failed`. Uma falha preparando outro slot, um timeout de backpressure ou uma falha de transporte pode marcar esse flag enquanto o worker ainda usa o slot em RUNNING.

O caminho então chama `slot->reset()`: limpa payload/reply, altera tamanhos dos vetores e zera output/nq/stride. O worker pode estar em send, recv ou decode, acessando esses mesmos campos. Preservar a capacidade dos vetores não torna essa modificação concorrente segura.

Falha global deve solicitar cancelamento, não declarar que o worker já aposentou o job. **Somente após a confirmação de retirement do proprietário ativo podem os buffers ser reciclados.** Essa regra também vale para um slot PREPARING: seu produtor precisa parar antes de qualquer reset por outro thread.

No estado saudável, o reset de um slot DONE após consumo preserva corretamente suas capacidades e os vetores de dois slots são independentes. O problema é a autorização de reset no caminho de falha, não `vector::clear()` em si.

O teste de timeout libera os tensores em `tests/test-remote-attn-range.cpp:1628–1634` sem drain explícito antes. Mesmo que um determinado fechamento de socket impeça a escrita nesse run, o teste deve aguardar retirement antes de liberar o grafo.

## P1 Drain concorrente com um wait pode perder a completion ou causar ABA

Referências: [binding por ponteiro](../../ggml/src/ggml-remote-attn/ggml-remote-attn.cpp#L985), [drain](../../ggml/src/ggml-remote-attn/ggml-remote-attn.cpp#L920) e descarte de DONE pelo controle na linha958.

O wait captura um `slot*` uma única vez. Drain e controles podem resetar esse mesmo slot quando o worker publica DONE:

1. Wait0 dorme aguardando RUNNING0; um controle também espera o job.
2. Worker publica DONE0 e notifica.
3. Drain ganha o mutex e reseta o slot para FREE.
4. Wait0 acorda e não encontra DONE, stop ou failed; volta a esperar indefinidamente.

Se o slot for reutilizado, o wait antigo pode observar a conclusão do novo job. Não há verificação de ticket no binding ou no predicado, e a confirmação final não repete sequer a comparação `slot.node==source`.

O teste de invalidação atual chama wait somente depois que invalidate já retornou. Isso verifica ausência do slot, mas não cobre um wait previamente bloqueado.

É necessário um handle `{epoch,ticket}` associado à execução, com completion ou cancelamento observável até o consumidor reconhecê-lo. Um wait que detecta mudança de identidade deve retornar cancelado sem resetar o job novo. Um terminal drain só pode apagar o estado de consumidores que o orquestrador já tornou quiescentes, ou precisa manter registros/tombstones de cancelamento.

## P1 Flags e fechamento do socket não seguem uma disciplina única

Referências: [stream_failure](../../ggml/src/ggml-remote-attn/ggml-remote-attn.cpp#L647), [worker](../../ggml/src/ggml-remote-attn/ggml-remote-attn.cpp#L885), [falha do exec](../../ggml/src/ggml-remote-attn/ggml-remote-attn.cpp#L3255).

`failed` e `connected` são bool comuns. Há escrita sem mutex em `rkva_transact`, leitura sem mutex no worker, leitura sob async no wait e escritas sob ctx em outros caminhos. Mutexes diferentes não criam exclusão entre esses acessos.

O ThreadSanitizer encontrou uma corrida entre a escrita de `connected=false` no caminho de timeout e a leitura do worker. Também reportou `close()` do produtor concorrente com `recv()` do worker.

O produtor ainda fecha o socket em `ggml_remote_attn_exec` quando uma preparação ou espera falha. Isso viola o ownership anunciado enquanto há I/O ativo. Fechar o descriptor em outro thread não prova que o worker já parou ou que os buffers podem ser liberados.

Centralizar a publicação de falha/cancelamento, as notificações e a aposentadoria do canal. Flags atômicas podem ser parte da solução, mas não resolvem sozinhas o lifetime do fd, dos slots ou dos outputs. Uma solicitação de cancelamento pode interromper I/O por mecanismo adequado; close/reconnect e reuso de descriptor precisam ocorrer após a confirmação do proprietário ativo.

Não identifiquei um ciclo adicional introduzido por stats_mutex isoladamente: no caminho examinado, a transação libera stats antes de tentar ctx, e o commit segue ctx→stats. A aresta incompatível demonstrada é async→ctx.

## P1 O stream também pode ser usado por uma execução síncrona

O worker chama `rkva_transact` sem ctx. Um nó síncrono entra por `ggml_remote_attn_exec`, adquire apenas ctx e também pode chamar `rkva_transact`. Esse mutex não exclui o worker que está fazendo I/O sem ele.

Isso permite disputa por cycle_id, resposta recebida pelo leitor errado, interleaving de frames e close concorrente. Os controles que usam `rkva_lock_ctx_drained` evitam o caso normal dessa sobreposição; os caminhos síncronos de execução não aplicam essa barreira.

Preferir um proprietário de I/O para todos os comandos, incluindo controle. Um handoff síncrono também é possível, mas exige fechar a admissão e comprovar quiescência do worker antes de tocar o canal. Não corrigir apenas adicionando um mutex de I/O aguardado enquanto ctx permanece adquirido: o worker pode precisar de ctx para concluir o commit anterior.

## P1 O ACK publica linhas de outro job e o RESET pode ficar incorreto

Referências: [RESET durante preparo](../../ggml/src/ggml-remote-attn/ggml-remote-attn.cpp#L1344), [commit do ACK](../../ggml/src/ggml-remote-attn/ggml-remote-attn.cpp#L840) e [avanço do produtor](../../ggml/src/ggml-remote-attn/rkva-kvarn-producer.h#L268).

A preparação avança o produtor antes do ACK. Com dois jobs da mesma camada, `owner->rows()` representa o fim tentativo dos dois jobs. O ACK de A faz `cached.rows=owner->rows()` e publica também as linhas de B ainda não confirmadas.

O problema tem um efeito adicional na primeira dupla:

1. Camada vazia; A128 foi preparado/enviado, com ACK pendente.
2. Owner tem128 linhas, mas cached.rows continua0.
3. Preparar B8, posições relativas128..135, escolhe RESET por `cached.rows==0`.
4. O RESET live136 contém tail original `[8,136)`.
5. A primeira query de B tem prefixo129 e precisa de tail `[1,129)`.

O preflight real Xbox rejeita essa cobertura. Isso foi confirmado em teste CPU com o produtor host e a função de admissão do XLLAMA:

```text
B after A128: reset=1 live=136 append_begin=0 append_rows=0 Xbox_admission=0
B after A128: reset=0 live=136 append_begin=128 append_rows=8 Xbox_admission=1
```

Não foi uma medição no console. O teste executou as funções reais de produção de payload e de validação de cobertura, sem GPU ou rede externa.

Cada job precisa congelar seu intervalo:

```cpp
slot.session_id = submitted_session;
slot.start_rows = owner->rows();
slot.end_rows = live;
// Bootstrap usa o inicio da transacao, nao o ultimo ACK observado.
bool reset = first == 0 && slot.start_rows == 0;

// Apos o ACK, sob a disciplina de estado/ordem definida:
if (cached.rows != slot.start_rows) return protocol_error;
cached.rows = slot.end_rows;
```

Isso requer separar fronteira tentativamente preparada e fronteira confirmada. Também reservar geração global considerando todos os segmentos pendentes; a checagem individual contra overflow antes de cada preparo não reserva os incrementos dos outros jobs. Um erro com mutação remota incerta continua exigindo fail-closed, sem repetição automática de append.

## P1 O estado anterior à espera não é revalidado

Referências: [validação da operação](../../ggml/src/ggml-remote-attn/ggml-remote-attn.cpp#L1504), leitura de posições nas linhas1534–1548 e [reserva depois da espera](../../ggml/src/ggml-remote-attn/ggml-remote-attn.cpp#L1468).

O submit libera ctx para esperar capacidade. Nesse intervalo, outro submit pode sobrescrever `ctx->pos_scratch`, ou um controle pode mudar sessão/configuração/epoch. Quando o submit retorna, apenas failed/stop são rechecados. Ele recebe a epoch corrente e copia o scratch global, embora seu input e seus parâmetros tenham sido validados antes da espera.

Capturar identidade imutável da operação antes de liberar ctx e revalidá-la depois. Se epoch/session/configuração mudaram, cancelar o submit antigo. Ler as posições do próprio node novamente para scratch exclusivo depois de reservar o slot, ou preservar uma cópia exclusiva durante a espera. Não atribuir a epoch nova a um trabalho validado sob a anterior.

## P2 A barreira de controle ainda não representa cancelamento de epoch

`invalidate_range()` atualmente chama `rkva_lock_ctx_drained()` e só então incrementa range_epoch. Não marca o backend falho nesse método. Assim, aguarda os jobs terminarem e invalida a cobertura depois; não aborta antecipadamente trabalho em voo como descrito no resumo.

O check de epoch em `rkva_finish_native_bulk_job` acontece depois de decodificar para output. Ele não é um mecanismo de prevenção de acesso a tensor antigo. O estado atual evita uma troca normal de epoch durante RUNNING drenando primeiro, mas não oferece a semântica de cancelamento antecipado anunciada.

Drain também não fecha a admissão. Para uma fronteira de controle, a sequência deve ser: bloquear novas reservas, invalidar/cancelar os trabalhos abrangidos conforme a política, aguardar a aposentadoria de produtores e worker, resolver consumidores/leases, executar o controle e só então reabrir admissão. As APIs devem distinguir synchronize que preserva completion, terminal drain e cancelamento.

## P2 Backpressure e a integração com o executor GGML

DONE continua ocupando um slot. Se um executor único executar `submit A; submit B; submit C; wait A; wait B; wait C`, o terceiro submit bloqueia mesmo depois dos dois RPCs concluírem: os waits que liberariam os slots ainda não foram executados.

O teste de backpressure cria uma thread separada para o terceiro submit e usa o thread principal para consumir A. Esse é um contrato válido de produtor/consumidor independentes, mas não representa o executor sequencial de nós do GGML.

A Fase 7 precisa garantir no máximo duas submissões sem consumo, usar um executor que continue despachando consumidores prontos, ou separar slots de transporte de registros de completion. Nesta última opção, o transporte pode reciclar seus buffers após retirement, mantendo a completion identificada e o destino vivo até o consumidor e o H2D terminarem.

As contagens de in_flight incluem PREPARING, PENDING, RUNNING e DONE. Dois slots ocupados não demonstram dois RPCs wire em voo nem, isoladamente, overlap útil de preparação e I/O. Para prová-lo, registrar handshake de request recebido/worker bloqueado e intervalos de preparo, I/O e consumo.

## P2 O contrato zero-allocation ainda não foi demonstrado

Os buffers próprios por slot e clear sem shrink preservam capacidades. Após warmup adequado dos dois slots, isso pode remover alocações desses vetores para shapes dentro do máximo já observado.

Persistem alocações no caminho: `ids` por chamada em `rkva_run_range`, append_k/append_v em `rkva_kvarn_producer::append`, criação de contexto no produtor CPU e crescimento de estatísticas. O reserve inicial de Q tem256×256 floats, enquanto um ubatch256 com24 heads e D256 usa256×24×256 floats; o crescimento inicial é esperado. O fallback do construtor após falha de reserve também permite alocação posterior.

Usar a formulação precisa “buffers de slot reutilizados” até medir construtores, chamadas ao allocator e crescimento de capacidade depois do warmup. Staging por std::vector também não implica memória pinned ou proteção automática de lifetime GPU. A proteção dos tensores e das cópias depende de eventos e ownership externos ao reset desses vetores.

## Achado adicional anterior à fila

No caminho síncrono de `rkva_run_native_range`, linhas1067–1071, `original_v` é alocado, mas somente keys é lido antes de prepare/append. Falta:

```cpp
ggml_backend_tensor_get(values, original_v.data(), 0,
                        original_v.size() * sizeof(float));
```

O caminho assíncrono lê V corretamente. Esse defeito não foi introduzido pela Fase 6, mas precisa de correção se o caminho síncrono continuar disponível. Testes com V zero não o detectam; usar valores distintos e conferir payload/tail original.

## Validação executada e lacunas da suíte

Resultados deste review:

| Verificação | Resultado |
| --- | --- |
| Recompilação e suíte host com ASan/UBSan | Exit0 |
| Recompilação de ggml-base e teste com ThreadSanitizer | Exit66; três warnings |
| TSan inversão de locks | ctx→async em951, async→ctx em1449 |
| TSan flag de conexão | write3257 versus read885 |
| TSan canal | close371/3258 versus recv439/679 |
| Produtor host + validator Xbox em CPU | RESET B136 rejeitado; append B128→136 aceito |

O TSan instrumentou ggml-base e o teste; as bibliotecas auxiliares CPU foram reutilizadas. Os resultados positivos identificam problemas concretos; essa instrumentação parcial não certifica ausência de outras races. O run ASan não encontrou uma falha de memória nessa agenda observada, o que não elimina as janelas estáticas de retirement apontadas.

Arquivos de execução e harnesses estão em `/tmp/beellama-phase6-review`. Um resumo dos diagnósticos está em [async-phase6-tsan.txt](async-phase6-tsan.txt). Os fontes exatos revisados estão identificados em [async-phase6-sources.json](async-phase6-sources.json).

Lacunas importantes dos testes:

- O loopback bulk interpreta somente o primeiro segmento e não valida a cadeia de append, coverage stage/tail, posições e conteúdo de todos os segmentos. Isso mascara o RESET repetido. Reutilizar a admissão real do Xbox em testes de conformidade.
- Os jobs de quatro queries com dados zero não exercitam selo128, tail wrap nem assinaturas distintas de payload e resposta.
- Faltam wait já bloqueado durante drain/invalidate, falha de preparo enquanto outro slot faz I/O e controle concorrente ao retorno do backpressure.
- Timeout configurado em50 resulta em100 ms por `max(100,atoi(...))`; o teste verifica apenas um limite inferior de35 ms. O parâmetro não é um deadline total para RPC, drain ou aquisição de mutex.
- pause_mode é lido/escrito sem uma disciplina uniforme de mutex; proteger a fixture e usar barreiras de estados, em vez de sleeps, para sincronizar a janela pretendida.
- A liberação de tensores no teste de timeout deve ocorrer somente após retirement/drain, mesmo com backend falho.
- Faltam mistura síncrona/assíncrona, identidade/ticket após reuso, ACK de A após preparar B e três submits no mesmo executor GGML.

## Sequência recomendada para correção e qualificação

1. Fixar a ordem ctx→async e definir um mecanismo único de falha/cancelamento e ownership do canal. Não fechar o socket no produtor enquanto o worker ainda o usa.
2. Separar cancel_requested de retirement. Somente o proprietário ativo publica a saída de RUNNING/PREPARING. Completion permanece identificável até consumo ou cancelamento reconhecido.
3. Acrescentar metadados imutáveis por job: sessão, epoch, ticket, camada, intervalo start/end e gerações esperadas. Corrigir RESET e publicação por ACK.
4. Fechar admissão durante controles e revalidar trabalhos que estavam esperando capacidade. Preservar/relê-los a partir dos próprios inputs, sem depender de scratch global anterior.
5. Definir a política de slots DONE para o executor GGML/Fase7 e garantir lifetimes de saída e de H2D sem reutilizar metadata de outro job.
6. Executar testes determinísticos das janelas acima com ASan/UBSan/TSan e validator real; depois qualificar CUDA/RKVA e rollback. Só então medir overlap e TPS.

Manter as APIs e checkpoints upstream de speculative decoding. Não adicionar estado privado de verificador para compensar erros de ordenação da fila. Os problemas observados pertencem ao transporte, à publicação de cache e à gestão de lifetime.
