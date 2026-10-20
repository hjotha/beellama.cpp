# Review das fases 1 e 2 e do tuning de splits da fase 4

Revisão de 20 de outubro de 2026 dos dois checkouts locais. As árvores receberam correções durante a sessão; este relatório distingue o diff inicial do estado final examinado. Os arquivos de patch das fases 1/2 preservam a proposta sobre o diff inicial e não devem ser reaplicados ao checkout que já os incorporou. O patch `xllama-phase4-tests.patch` é incremental sobre a fase 4 examinada.

## Parecer sobre a fase 4

**A nova política é correta no domínio de contexto validado e pode compor um build candidato para qualificação.** Não encontrei truncamento de partials, novo conflito de escrita ou erro de cobertura decorrente da redução de splits. O ganho de desempenho depende de medição no console. Antes de considerar o tuning qualificado, integrar a cobertura numérica da política nova e executar paridade do binário GPU.

### 1 Política e casos de fronteira

Em `uwp/rkva/rkva-range-profile.h`, `kvarn4_mq_split()` calcula:

```text
k0 = clamp(ceil(N / 256), 1, cap)
R  = align_up(ceil(N / k0), 8)
S  = ceil(N / R)
cap = 32 balanced, 64 fine
```

Para N positivo, `R*k0 >= N`, logo `1 <= S <= k0 <= cap`. Recalcular S após alinhar R elimina splits globalmente vazios: `(S-1)*R < N <= S*R`. O último split pode conter menos de R tokens, o que é esperado.

| N remoto | Balanced S × R | Fine S × R | Observação |
| ---: | ---: | ---: | --- |
| 0 | 1 × 0 | 1 × 0 | Resultado da função auxiliar; não autoriza uma query remota sem contexto |
| 1 | 1 × 8 | 1 × 8 | Caminho direto, com máscara da parte excedente |
| 9 | 1 × 16 | 1 × 16 | Alinhamento de oito preservado |
| 129 | 1 × 136 | 1 × 136 | `target_rows` é um alvo, não um tamanho fixo de bloco |
| 256 | 1 × 256 | 1 × 256 | Limite do caminho direto |
| 257 | 2 × 136 | 2 × 136 | Partição não alinhada a 128, legal |
| 6.144 | 24 × 256 | 24 × 256 | Ambos abaixo dos respectivos caps |
| 8.193 | 32 × 264 | 33 × 256 | Fronteira do cap balanced |
| 16.384 | 32 × 512 | 64 × 256 | Valores descritos no pedido, confirmados |
| 16.385 | 32 × 520 | 63 × 264 | S pode diminuir quando N aumenta; isso não é perda de cobertura |
| 34.816 | 32 × 1.088 | 64 × 544 | Capacidade remota máxima deste split posicional |

O alinhamento a oito é suficiente para o shader atual. Cada tile KVarN de oito tokens começa alinhado a oito, e oito divide o grupo de 128; um tile individual não cruza a fronteira de um record. Um split pode começar no meio de um grupo: o shader calcula o endereço/metadados desse grupo e cobre as bordas pelos caminhos residuais. Alinhar splits a 128 seria outra escolha de tuning, não uma correção obrigatória.

O caso zero está tratado sem divisão por zero. Para `0 < N <= 256`, o resultado é `S=1, R=align_up(N,8)`; o dispatch seleciona saída direta e não executa o reducer. A admissão de queries continua responsável por rejeitar ausência de cobertura causal. Não tratar o valor `{1,0}` como permissão de executar atenção vazia no caminho de serving.

A assinatura recebe `uint32_t`, mas o domínio de serving é menor: o descriptor exige `origin+capacity <= INT32_MAX` e o upload limita `live_count` à capacidade. Nesse domínio, as somas de arredondamento e os intervalos de split permanecem representáveis em uint32. Para uma biblioteca que pretendesse aceitar todo `UINT32_MAX`, seria necessário rever as divisões por teto e também `split_end` no HLSL. Isso não constitui um overflow alcançável pelo descriptor válido desta implementação.

### 2 Dispatch e causalidade

Em `uwp/rkva/remote-attn-engine.cpp`, a seleção nova só ocorre após encontrar uma PSO MQ e conservar `q_tile != 0`. Ausência da PSO devolve `q_tile=0` e conserva o split legado. Decode especializado `nq=1` mantém sua política anterior; modo coarse e shapes não qualificadas não passam inadvertidamente a MQ.

Os intervalos do shader são `[s*R, min((s+1)*R, limit_query))`. `limit_query` inclui posição, active_count e máscara. Queries anteriores dentro do mesmo tile podem ter um split terminal vazio mesmo quando o split global contém tokens. O shader publica denominador zero para esses casos; o reducer ignora esses termos. Queries fantasmas de um tile parcial usam `qc=min(qi,n_query-1)` para loads em bounds e não publicam resultados fora de `n_query`.

Pontos examinados no HLSL `rkva_range_attn_kvarn4_mq.hlsl`: limites de split a partir da linha 277, compartilhamento de records a partir da 357, residuais a partir da 422, tail por query a partir da 463 e publicação de partials na redução final. A política altera a quantidade de trabalho por grupo, sem introduzir novos buffers compartilhados ou sincronização entre grupos.

### 3 Reducer e memória

`rkva_range_reduce_opt.hlsl` aceita até 256 splits. Os arrays LDS de máximos, denominadores e pesos têm esse tamanho. Com os novos caps de 32/64, `min(split_k,256)` não trunca dados.

Para S=32/64, apenas as primeiras 32/64 threads carregam max/den e pesos; todas as 128 threads atravessam as barreiras. Os loops só leem entradas `s < splits`, previamente escritas. Os grupos de saída são exclusivos por `(query,head)`, e as lanes escrevem dimensões exclusivas. A barreira UAV e a transição de `range_opt_partials_` para leitura permanecem entre pass1 e reduce. O caminho S=1 escreve O/LSE diretamente e não lê partials antigos.

O engine reserva o scratch otimizado por `256 * 9 * Hq * (D+2) * sizeof(float)`. Para Hq=24,D=256, são 57.065.472 bytes, aproximadamente 54,42 MiB. A nova política usa menos desse espaço, sem alterar a reserva máxima nem exigir aumento do orçamento.

Para um segmento de oito queries:

| N e modo | S anterior → novo | Grupos pass1 anteriores → novos | Partials anteriores → novos |
| --- | ---: | ---: | ---: |
| 6.144 balanced MQ4 | 48 → 24 | 384 → 192 | 9,0703 → 4,5352 MiB |
| 6.144 fine MQ2 | 192 → 24 | 3.072 → 384 | 36,2813 → 4,5352 MiB |
| 16.384 balanced MQ4 | 128 → 32 | 1.024 → 256 | 24,1875 → 6,0469 MiB |
| 16.384 fine MQ2 | 256 → 64 | 4.096 → 1.024 | 48,3750 → 12,0938 MiB |

Esses são tamanhos lógicos de partials escritos e depois lidos, não medições de GDDR6. A redução de 75% dos bytes de partials em 16k não significa 75% menos tempo de atenção. O buffer residente continua reservado para o máximo antigo.

O comentário novo nos testes que chama o fine legado em 16k de “128 splits” está incorreto: eram 256, com 64 tokens por split. O patch de testes corrige esse comentário e remove a sugestão de que um teste puramente CPU tenha medido tráfego GDDR6.

### 4 Risco numérico e de desempenho

Softmax continua usando máximos, denominadores e numeradores F32 com rescale estável. Não há uma instabilidade nova por usar 512 tokens em vez de 128 por split. Entretanto, a ordem das operações muda: há somas locais mais longas e menos termos no reducer. O erro líquido não é necessariamente monotônico, e identidade bit a bit de O/LSE ou da trajetória MTP não é garantida.

Em N<=256 haverá somente `Hkv*ceil(nq/Q_TILE)` grupos. Com Hkv=4 e as variantes efetivamente selecionadas pelo engine, isso corresponde a 4–20 grupos para nq=2..9. Pode subutilizar as 52 CUs, embora elimine transforms e redução. Em N=6k/16k há mais grupos, mas isso não estabelece ocupação efetiva. Medir separadamente nq=2,4,8,9, early overflow e late prefill. Não inserir um threshold de retorno à política antiga sem evidência de regressão.

Em 16k, a redução de splits também reduz bastante a repetição de WHT por split. Essa é uma razão técnica para testar a proposta; não dispensa timestamps do pass1/reduce nem a latência do envelope completo.

### 5 Lacuna de teste e patch incremental

**P1 de qualificação:** `tests/rkva_range_kvarn4_opt_model.py` define `kvarn4_mq_split()`, mas o `run_optimized()` do diff original escolhe apenas `kvarn4_decode_split()` ou `kvarn4_split()`. Executar esse oracle sem alteração não testa a nova partição.

O patch [xllama-phase4-tests.patch](xllama-phase4-tests.patch):

- Exaure N=0..131072 para modos balanced/fine e tiles 2/4, com os caps próprios e cobertura/alinhamento.
- Acrescenta 257, 8193 e 16385, incluindo a mudança não monotônica de S.
- Acrescenta `mq_tile` opcional à emulação F32 para selecionar a política nova.
- Acrescenta um teste de O/LSE contra o oracle F64 independente, com opção `--long` para 6k/16k/35k.
- Mantém asserts ativos no teste CPU de perfil mesmo quando compilado com NDEBUG.
- Corrige os comentários e a identificação do teste de tráfego.

O oracle adicional exercita a matemática da partição e do reducer usando a emulação por query. Ele não emula o compartilhamento LDS entre queries do MQ4. Esse comportamento continua exigindo o modelo de partição causal existente e, principalmente, paridade do shader compilado no console.

Aplicação, a partir da raiz do XLLAMA, após revisar o patch:

```bash
git apply --check /home/hjotha/beellama.cpp/docs/reviews/hybrid-attention-phase12/xllama-phase4-tests.patch
git apply /home/hjotha/beellama.cpp/docs/reviews/hybrid-attention-phase12/xllama-phase4-tests.patch
```

Esses comandos foram validados para o snapshot examinado, mas o patch não foi aplicado ao repositório nesta revisão. Revalidar `--check` se a árvore continuar recebendo alterações.

### 6 Critérios antes de promover o build

Gerar um candidato com identificador novo e registrar commit/diff, MSIX e hashes das PSOs efetivas. O tuning é C++/parâmetros de dispatch: não exige mudar o HLSL, mas precisa provar que o DXIL embarcado corresponde ao caminho qualificado. O MQ2/MQ4 selecionado deve ser registrado, não inferido do nome do modo.

No console, comparar O/LSE para N=255/256/257,6144,16384/16385,34816 e nq=2/3/4/5/8/9, incluindo tiles parciais e prefixes distintos ao redor das fronteiras de 128. Manter tail128, stage e records iguais entre baseline e candidato. Comparar a mesma representação e wire para isolar o tuning; uma mudança simultânea de rounding FP8 dificulta atribuir o resultado.

Registrar erro máximo, NMSE e erro LSE, além de aceitação e tokens aceitos/s em MTP. Um probe antigo que exige igualdade de todos os bytes de atenção entre agendas diferentes pode precisar usar a tolerância numérica aprovada, pois a partição mudou a ordem FP. Records e estado persistente não devem mudar seus contratos.

Comparar pelo menos os pontos 102,4k e 112,6k, early overflow e decode. Exigir ausência de regressão relevante no decode e medir pass1, reduce, cópias, fence e envelope. A lacuna de timestamps bulk F16/FP8 das fases anteriores ainda existe no código examinado; ausência de amostras não significa ausência de custo GPU.

## Consolidação das fases 1 e 2

### Diferenças entre o diff inicial e o estado atual

| Tema | Diff inicial recebido | Estado atual examinado |
| --- | --- | --- |
| SplitZip OOB | Bounds melhorados, mas cast de header desalinhado; variantes host/Xbox diferentes | Header lido com memcpy, tamanho calculado em uint64, `escapes<=n` e consumo exato de escapes |
| Desativação | Host rejeitava SplitZip, Xbox anunciava/aceitava e contornava o preflight de finitude | Capability removida do anúncio, parser/validator/engine rejeitam antes de mutação |
| FP8 explícito | `WIRE_STRICT=0` ainda permitia fallback | Somente auto permite escolha alternativa; FP8 explícito exige as capacidades |
| Rounding | Saturação finita corrigida, mas soma0x80 mantinha half-up e duplo arredondamento | RNE direto escalar e correção dos midpoints no SIMD; capability RNE adicional |
| Pool | Move-back do worker corretamente preservava job/payload/reply | Correção mantida; Q scratch também passou a ser reutilizado |
| Cancelamento | Worker podia sobreviver ao grafo que possui seu output cru | Synchronize espera sem consumir; drain terminal em erro e antes de destruir o scheduler |
| Cópia D3D12 | Cópia suprimida no bulk GPU | Guard correto, mantidas barreiras e pack final; CPU fallback conserva readback necessário |

No diff inicial, o encoder do Xbox saturava corretamente os 256 padrões finitos que antes viravam Inf; o host também saturava FLT_MAX e rejeitava Inf/NaN nos testes. Os helpers preservavam especiais de forma coerente, enquanto os chamadores barravam especiais na admissão/validação. Isso era diferente de afirmar RNE: o probe encontrou 124 divergências contra RNE ao enumerar os padrões F16 finitos. O estado atual resolve essa diferença; a capability nova evita selecionar um servidor FP8 antigo como se tivesse o novo contrato.

Para o decoder atualizado, a igualdade entre `n` e o campo uint32 do header limita n antes de calcular os tamanhos. Assim, a expressão uint64 não transborda no domínio x64, e os bounds das seções e escapes fecham a leitura original fora do buffer quando `in_bytes` descreve realmente a memória disponível. Como em qualquer API de ponteiros crus, a função não pode descobrir se o chamador mentiu sobre o tamanho da alocação. Persistem permissividades de canonicalização de bits reservados/padding, que não são o OOB original e não precisam tornar um codec arquivado um protocolo ativo.

O move-back do job antes de publicar DONE, sob `async_mutex`, estabelece happens-before com wait/drain. Não encontrei corrida nova no fluxo serial válido submit→wait. Antes da correção havia destruição/recriação, não vazamento do unique_ptr. Preservar as capacidades não elimina todo o heap: K/V originais, ids, temporários de append e crescimento de estatísticas ainda podem alocar.

O risco de lifetime por abort antes do wait era anterior ao move-back. O remoto não pertence ao scheduler; o synchronize vazio e o drain tardio do backend não protegiam a saída na destruição/reutilização de tensores. O estado atual acrescenta a barreira terminal nos pontos relevantes. Foi validado o código e a preservação de completion em testes CPU; não foi executado um modelo completo com cancelamento e TSan. Esse stress continua sendo um gate para aumentar concorrência.

A supressão de CopyBufferRegion é segura nos três caminhos examinados: com `bulk_gpu_query_`, `bulk_recording_` conserva o resultado em GPU e finish retorna antes de Map; o pack lê `range_output_` depois de transição válida. O fallback CPU mantém a flag falsa e conserva a cópia. A alocação/swap de `bulk_f16_stage_out_` permanece redundante e pode ser removida depois como limpeza isolada.

### Patches anteriores

Os seguintes patches foram preparados e testados contra cópias temporárias do diff inicial. As correções já estão presentes no checkout atualizado observado durante esta sessão; não os reaplicar:

- [BeeLlama fases1/2](beellama-phase12-fixes.patch) e [XLLAMA fases1/2](xllama-phase12-fixes.patch).
- [BeeLlama FP8 RNE](beellama-fp8-rne.patch) e [XLLAMA FP8 RNE](xllama-fp8-rne.patch), originalmente dependentes do primeiro par.

O teste inicial de reciclagem, com três respostas zero, não provaria que o job deixou de ser recriado. A versão atual melhora o teste de payload e completion, mas ainda é recomendável medir construtores/crescimento de capacidade e exercitar erro remoto, truncamento, cancelamento pendente, reset/rewind e teardown com barreiras determinísticas. Não usar apenas TPS como prova de ausência de alocações ou races.

## Verificações executadas

| Verificação | Resultado |
| --- | --- |
| Testes CPU Xbox do diff inicial, ASan/UBSan | Passaram; não cobriam input desalinhado nem RNE direto |
| Probe de input SplitZip desalinhado no diff inicial | UBSan identificou acesso desalinhado ao header |
| Enumeração dos padrões F16 finitos no diff inicial | Zero conversões finito→especial após clamp; 124 diferenças frente a RNE |
| Codec corrigido, SIMD explícito AVX2/F16C e escalar | Sem divergências no conjunto F16 finito, midpoints FP8 e vizinhos F32 de ambos os sinais; extremos e não finitos em lanes/tail verificados |
| Teste host atual recompilado isoladamente, ggml-base e teste com ASan/UBSan | Exit0; preservação de completion em synchronize e drain terminal exercitados |
| Syntax check do llama-context.cpp atual | Exit0 |
| Testes atuais `rkva_range_profile_cpu.cpp` e `rkva_kvarn_bulk_cpu.cpp`, ASan/UBSan | Passaram |
| Varredura exaustiva da política N=0..131072, modos1/2 e tiles2/4 | Cobertura, alinhamento, caps e último split global não vazio preservados |
| Oracle de split adicional com casos curtos | Passou; maior erro O normalizado 3,481e-7 e LSE absoluto 5,133e-7 |
| Oracle adicional com 6144/16384/16385/34816 | Passou; maior erro O normalizado 3,481e-7 e LSE absoluto 9,187e-7 |
| Patch fase4 | `git apply --check`, syntax C++ e py_compile aprovados; validado em cópias temporárias |

O primeiro run ASan da suíte host também encontrou aproximadamente 48,3 MB não liberados em `test_native_query_prefix_chunks()`, por omissão de `ggml_backend_free(cpu/backend)`, anterior a este diff. A limpeza foi incluída no patch e o run atual passou com detecção de leaks habilitada. Não confundir esse vazamento de fixture com o churn anterior do job.

Os testes adicionais e os binários de validação foram executados em `/tmp/beellama-phase12-review`; não houve build UWP, deploy, benchmark físico ou execução GPU no console. Os resultados de oracle são corretude CPU emulada, não evidência de ganho de TPS.
