# O que falta para investigar o caso do Xbox

**O executor lógico 128 não cobre o caminho de duas waves64 nem sua mesclagem.** A ausência de divergência MQ2/GQA2 nesta matriz não valida o ramo executado no console. Implementar duas waves lógicas com barreiras do workgroup exigiria outra análise de controles divergentes; uma substituição direta pode criar deadlock ou mudar o problema.

Para aproximar o diagnóstico do caso real, a captura mínima deve identificar a primeira operação de atenção relevante e preservar, no mesmo instante:

1. Commit, revisão do aplicativo, variante de shader/PSO, hash do DXIL, GPU/driver e wave size observada. Registrar os flags MQ2/GQA2 e a política de split usada.
2. Geometria exata do dispatch e cbuffer: `n_query`, `active_count`, `n_head`, `n_head_kv`, `head_dim`, `kq_scale`, `origin`, `split_k`, `split_rows`, `direct_output`, `records_bytes` e `precision_tail_tokens`.
3. Queries F32 originais e rotacionadas, posições absolutas e os bytes exatos dos registros KVarN4, stage e tail, com offsets/tamanhos/hashes. Não substituir por uma requantização ou por tensores de outra etapa.
4. Parciais depois da atenção e antes do reducer; depois, O e LSE do reducer. Comparar baseline e candidato com entradas idênticas. Se possível, identificar camada e passo da primeira divergência.
5. Identificadores/hashes do modelo e configuração de execução/sampling, além dos critérios reais do gate. Os 16 passos e o top1 informado não bastam para localizar a causa dentro da atenção.

As linhas KV ativas da fixture sintética não identificam a divisão local/remota do serving nem o contexto total. Os dados acima devem permanecer privados e conter somente o necessário para o diagnóstico. Nenhuma captura, modificação ou implantação no Xbox foi feita por este pacote.
