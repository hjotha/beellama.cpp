# Experimento Q4 na Radeon com atenção remota

## Hipótese

Medir como os caches globais `q4_0`, `q5_0`, `q6_0` e `q8_0` com atenção
nativa no Vulkan da Radeon se comparam a `kvarn4`, usando os pesos do modelo na
CUDA. A primeira matriz muda o tipo global de K e V junto com o posicionamento
remoto; ela não compara um formato diferente em cada dispositivo. Tipos
adicionais são selecionáveis no harness; erros de carregamento e fallbacks
ficam registrados sem exigir recompilação do harness.

## Configuração controlada

`tools/bench-radeon-q4.py` inicia e encerra um `llama-server` isolado por caso,
no `127.0.0.1:18091` por padrão. O processo usa um grupo próprio e a limpeza
envia `SIGTERM`, com `SIGKILL` apenas se o prazo configurado expirar. O modelo e
o binário são identificados por tamanho e SHA-256; bibliotecas `libllama` e
`libggml` encontradas por `ldd` e `/proc/<pid>/maps` também são registradas.
Cada request HTTP grava payload, corpo de resposta, duração, memória/clocks da
Radeon e memória disponível do sistema antes/depois em
`requests.jsonl`; as saídas resumidas ficam em `results.jsonl`, e o log completo
do servidor fica em um arquivo por caso. Cada caso usa um diretório isolado em
`--slot-save-path` para permitir apagar o slot entre prompts; autosave não é
ativado.

Valores padrão da matriz:

- Rotas: `cuda-only` e `remote-vulkan`.
- K/V global: `kvarn4` e `q4_0` por padrão. `--cache-types
  kvarn4,q4_0,q5_0,q6_0,q8_0` inclui as variantes adicionais.
- Atenção Vulkan: primeiras 1 e 16 camadas full-attention, conforme confirmado
  pelo log de placement. A rota local usa `--remote-attn vulkan:0`.
- Prompt inicial exato: 8.192 e 16.384 IDs, incluindo BOS, obtidos de
  `/tokenize` e enviados a `/completion` como IDs. O prompt de continuação
  acrescenta 32 IDs tokenizados separadamente de um trecho independente que
  começa com `\n\nUSER: Please analyze...`; isso evita que a continuação coincida
  com tokens gerados do restante do corpus. `prompts.json` preserva os IDs de
  ambos os requests, e o resultado registra o prefixo esperado e `cache_n` real.
- Capacidade reservada e ocupação observada são valores diferentes. Para
  verificar encaixe, usar `--ctx-size 204800` com request de 8.192 IDs e
  registrar separadamente `cache_n + prompt_n + predicted_n`; o prompt não
  ocupa automaticamente toda a capacidade configurada.
- Decode: 48 tokens no request inicial e 32 na continuação; `temperature=0`,
  seed 42, `ignore_eos=true`, `cache_prompt=true`, um slot e
  `--spec-type none`.
- Contexto suficiente para prompt + sufixo + saída, calculado uma vez para toda
  a matriz; batch/ubatch 256; `--flash-attn on`; sem warmup nem context shift.
- `--kv-tail-tokens 0`. KVarN ainda mantém seu sufixo intrínseco exato de 128
  tokens. O tipo de tail fica explicitamente em F16, embora a comparação atual
  não configure tail adicional.

A ordem dos tipos KV alterna entre repetições. Para cada tipo KV, a rota
CUDA-only serve de controle para as rotas Vulkan. Assim, o efeito de mover
atenção pode ser comparado mantendo o formato fixo. A comparação entre
`kvarn4` e `q4_0` ainda muda K e V globalmente; não deve ser descrita como um
teste de formatos mistos por dispositivo. MTP está desabilitado em todos os
casos. Os pesos são pedidos na CUDA (`--device CUDA0`, `-ngl 99`); o log precisa
confirmar o placement efetivo e a quantidade Vulkan antes do caso ser aceito.

## Evidência de aceitação

Um caso só fica `complete` se a resposta confirmar o número solicitado de
tokens preditos, `cache_n + prompt_n` cobrir exatamente a entrada, o request
inicial frio reportar `cache_n=0`, a continuação reportar `cache_n>0` e o prompt
não estiver truncado. Para `remote-vulkan`, o log deve conter
`Vulkan-layers=N/...`; erros de startup, timeout, contagem ou rota ficam
registrados como falha e o executor prossegue para o próximo caso.

O resultado útil é a matriz pareada em prefill/decode nas cargas de 8k/16k,
com os logs e o estado observado dos clocks/memória da Radeon. Requests e
respostas geradas são preservados para auditoria. Saída textual igual entre
formatos é um dado de smoke, não uma medida de qualidade suficiente por si só.
Também não se extrapola um resultado de 2k para contexto longo.

## Cache misto estático: implementação integrada (stages 1–4)

O caminho opt-in usa `kvarn4` nas camadas locais CUDA e Q4/Q5/Q6/Q8 nas camadas
full-attention selecionadas para Vulkan. `--remote-attn-cache-type-k TYPE` e
`--remote-attn-cache-type-v TYPE` configuram o par remoto; sem ambos, o caminho
existente permanece inalterado. Seleção zero de camadas continua puramente
KVarN e não cria o cache companheiro.

O desenho mantém um único mapa de células, posições e sequências. O cache KVarN
é dono desse metadado e das camadas locais; um cache padrão companheiro possui
somente os payloads Qx das camadas Vulkan. A graph seleciona o formato por
camada. O Qx usa o domínio Hadamard padrão correspondente às rotações do cache
quantizado comum; o companheiro tem metadados próprios de rotação, sem herdar os
valores zerados do metadado KVarN. K/V novo das camadas Vulkan vai diretamente
das projeções para Qx. A Radeon executa a atenção e devolve apenas a ativação de
saída à CUDA. O KV histórico não retorna à CUDA por token nem é convertido de
volta durante a atenção.

O KVarN mantém seu sufixo intrínseco de 128 posições. Esse sufixo não é uma
cauda Qx: a graph seleciona máscara e acesso por camada, e Qx atende o histórico
visível pelo Flash Attention padrão. O caminho misto estático exige tail
adicional zero, sem SWA, prefill migration, cache paginado/unificado ou múltiplos
slots; MTP estático é coberto separadamente. O estado misto tem descritores por
camada e suporta save/restore host em contexto atual e processo novo. O spool
usa `TMPDIR` em filesystem de disco e rejeita tmpfs/ramfs no Linux.

As verificações de igualdade numérica nos prefixos 127/128/129/385 passaram
para padrão Q8 versus misto Q8 com todas as oito camadas remotas, incluindo
tokens, texto e probabilidades top-16. O teste com uma camada Q8 remota também
concluiu. Os controles de save/restore Q4 e Q8 em processo novo passaram em A/C
com Qwen3.5-4B-MTP. Esses resultados validam implementação/estado nesse modelo,
não qualidade ou TPS do 27B. Zero camadas, tipo efetivamente alocado, contagem
de camadas e placement seguem exigindo confirmação no log.

O planner de capacidade também foi exercitado com Qwen3.5-4B: com capacidade
65.536 e 385 tokens ocupados, escolheu zero camadas remotas no orçamento normal;
com uma reserva CUDA intencionalmente apertada, escolheu uma camada Q8 remota e
sete camadas KVarN locais. São verificações do planner e alocação em 4B, não
placement validado no 27B ou a 200K de ocupação.

Na reserva `ctx-size=204800`, o log all-16 registrou buffers KV aproximados de
3.600 MiB em Q4, 4.400 MiB em Q5, 5.200 MiB em Q6 e 6.800 MiB em Q8, além de
126,27 MiB para compute Vulkan. São alocações de capacidade, não prova de que
cada request ocupou 200k posições. Como limite inferior de tráfego, se as
204.800 posições estivessem ocupadas e o decode target-only, sem MTP,
sustentasse 20 tok/s, uma leitura completa do KV por token implicaria cerca de
70,31 / 85,94 / 101,56 /
132,81 GiB/s para Q4/Q5/Q6/Q8. É o payload em bytes por token multiplicado por
tokens/s — um limite inferior teórico, não largura de banda DRAM medida; exclui
rereads e todo outro tráfego. Requests de 8k medem compute e transferências
combinados. É necessária ocupação longa para avaliar um possível limite de
largura de banda. Os snapshots guardam GTT, memória do
sistema disponível/swap e, se acessível, métricas `nvidia-smi`; falha de
alocação ou limite de memória conta como falha real do caso.

O budget Vulkan observado da Radeon é aproximadamente 8.091 MiB, usando RAM
de sistema (15.158 MiB físicos), não VRAM dedicada. Por isso, a telemetria GTT,
memória disponível e buffers Vulkan é necessária junto da capacidade de
contexto solicitada.

## Histórico: evidência preliminar e limites da matriz global

Smokes pareados anteriores, sem tail adicional, usaram 2.048 tokens de prompt e
32 de decode, com as 16 camadas Vulkan. O Q4 registrou 227,2008 tok/s de prefill
e 25,0677 tok/s de decode; o KVarN4 registrou 83,9216 e 13,0559 tok/s. Ambos
produziram o mesmo texto nesse smoke. Esses números justificam executar os
casos longos, mas não estabelecem o resultado de 8k/16k.

As medições diretas completas Q4/Q5/Q6/Q8, com throughput, ocupação e memória,
estão em [radeon-kv-results-20261001.md](radeon-kv-results-20261001.md). A série
all-16 configurou capacidade 204.800 mas os requests mediram 8K/16K processados;
isso não é evidência de inferência do modelo com 200K posições ocupadas.

O teste remoto KVarN4 de 8k não completou dentro do limite de 360 segundos:
7424 tokens processados e 20,68 tokens/s acumulados; o processo foi encerrado.
Esse é um timeout, não um resultado concluído de prefill/decode. O timeout do
harness é configurável e sempre limitado.

Um diagnóstico anterior com tail explícito de 1.024 tokens mediu Q4 a 56,815
tok/s de prefill e 5,868 tok/s de decode em 2.048 tokens. Não misturar esse
resultado com a matriz atual: no código Vulkan, o despacho de attention com
tail segue uma shader de cauda que percorre todo o KV e não usa o caminho
normal otimizado de Flash Attention/split-K. A matriz atual usa tail adicional
zero. Uma otimização futura pode calcular o corpo quantizado e a cauda exata em
parciais separadas e combiná-las com online softmax; isso fica fora deste
experimento.

## Handoff adaptativo: implementação integrada e limites stage 5

A matriz Q4/Q5/Q6/Q8 global continua sendo um controle de kernel/formato. O
serving também tem agora uma troca unidirecional suportada de um prefixo puro
KVarN4 para um destino denso misto: KVarN4 fica nas camadas CUDA e Q4 ou Q8
armazena a camada full-attention escolhida para Vulkan. O caminho exige os
mesmos bits KVarN na origem e destino e tail adicional zero. Não faz repatriação
ou oscilação de camada.

O handoff seleciona o checkpoint host compatível mais longo, restaura uma origem
KVarN híbrida em CPU e materializa a camada Vulkan em janelas limitadas. K/V
novo vai diretamente das projeções ainda não quantizadas para Qx; a Radeon
calcula a atenção e retorna a ativação à CUDA. Posições/células compartilhadas e
estado recorrente (RS) são transferidos e validados antes de publicar o destino.
Não se monta um prefixo inteiro F32 nem um prefixo inteiro Qx. O target preserva
o prefixo reutilizado. Com MTP ativo, o draft é recriado pelo bootstrap existente;
não se afirma que o cache/carry completo do draft foi transferido.

No Linux, antes de criar a origem CPU o glue consulta `MemAvailable` e recusa a
troca se não houver espaço para a origem comprimida, a reserva estimada de RS e
256 MiB de headroom. GTT não é somado separadamente, pois a RAM compartilhada da
Radeon já afeta `MemAvailable`. Isso é uma pré-verificação de alocação, não um
limite de pico de RAM: o snapshot host já residente (incluindo páginas mmap
reclaimáveis), a origem KVarN CPU e o buffer temporário de RS ainda usam memória
durante a troca. O scratch informado pelo conversor não representa sozinho o
pico de RAM.

Evidência real stage 5 com Qwen3.5-4B-MTP e controles 20 W/2700 MHz; cada caso
abaixo terminou com `exit 0` e handoff confirmado:

| Caso | Prefixo de origem | Reuso no handoff | Qx remoto | Continuação observada |
|---|---:|---:|---|---|
| `stage5-B-q4_0-r4` e `stage5-B-q4_0-r5` | 300 | 256 | Q4 | `cache_n=640`, `prompt_n=60` |
| `stage5-B-q8_0` | 300 | 256 | Q8 | `cache_n=640`, `prompt_n=60` |
| `stage5-M2-q4-r2` | 300 | 256 | Q4, MTP ativo | handoff 10/10 aceitos; continuação 19/5 |
| `stage5-M2-q8` | 300 | 256 | Q8, MTP ativo | handoff 10/10 aceitos; continuação 19/5 |
| `stage5-B-large-q4` | 4.096 | 3.968 | Q4 | prompt de 5.000; continuação `cache_n=4992`, `prompt_n=8` |

No caso B-large-Q4, o handoff informou 38.449.152 bytes de origem comprimida,
1.347.584 bytes de scratch, 144,511 ms de conversão e 52.691.540 bytes de
transferência RS. O caso B-Q4 repetido com a versão que inclui a pré-verificação
RAM também passou. O teste CPU `stage5-state-file-stream.json` terminou com
`exit 0`; save/restore misto Q4/Q8 em outro processo passou nos casos A/C. O
runner M2 preserva agora os deltas de log retornados por cada completion; o
fake test cobre gate válido e ausência real dos marcadores.

Nenhum resultado stage 5 é TPS do modelo completo 27B nem teste de contexto
ocupado longo. Os prompts observados foram 300, 700, 4.096 e 5.000 tokens. A
matriz all-16 anterior alocou capacidade 204.800 e processou 8.192 tokens; isso
não demonstra 200K ocupados. A rodada posterior com Qwen3.8-27B foi executada pelo OpenCode e revista pelo
principal; os resultados atuais estão em `radeon-kv-results-20261001.md`.
Houve ocupação real de 98304 tokens e duas tentativas de 200000 que atingiram
o limite de 600 s, sem decode. Capacidade maior foi alocada, mas o objetivo de
200k frios sem forte penalidade não foi atingido.

## Conversão e qualidade após KVarN→Qx

A matriz global mede Q4/Q5/Q6/Q8 usados diretamente como cache. O handoff
stage 5 já converte um prefixo KVarN4 para Q4/Q8 em blocos e continua a
inferência no destino; isso comprova o caminho funcional no modelo 4B, não a
qualidade do prefixo convertido. O conversor antigo `state_convert_q4` continua
fazendo somente Q4→KVarN. O caminho do handoff é separado e precisa preservar o
domínio rotacionado e o tail efetivo. Q8 direto não prova a qualidade de Q8
convertido de KVarN; mais bits podem reduzir erro adicional de conversão, mas
não recuperam precisão perdida na quantização original KVarN4.

Validação realizada e trabalho futuro:

1. Foram comparadas distribuições completas de 248320 IDs em três pontos de
   continuação do 27B, para Q4/Q5/Q6/Q8 versus KVarN4, com histórico/cadência
   iguais. Isso mede a continuação mista completa, não somente requantização.
2. Save/restore, boundaries, corrupção e MTP após handoff passaram nos casos
   documentados. O draft usa bootstrap, sem transferência integral de carry.
3. Há dados de throughput/memória de 8k/16k nos quatro formatos e ocupação real
   de 98304 com Q4. As duas tentativas de 200k frios terminaram por timeout.
4. Permanecem futuras a avaliação de corpus/perplexidade mais ampla e a
   otimização da atenção remota para completar 200k em tempo aceitável.

## Estado da implementação ao fechar esta rodada

Cache misto, conversão em janelas, checkpoints/snapshots e handoff entre
perfis estão integrados. O auto-placement consulta o CUDA proprietário real
e limita a RAM UMA com MemAvailable; o fitter de `ctx-size=0` está conectado
ao modelo de custos mistos e passou em execução curta 4B/27B. Não representa
certificação de desempenho com toda a capacidade preenchida.

Foi testado `GGML_KVARN_WINDOW_CHUNK=4096`: no controle com a mesma capacidade
32768 e prompt 16384, uma camada Q4 remota passou de 377,69 para 623,71 t/s de
prefill. Esse ajuste não resolveu o caso de 200k com nove camadas remotas.
Ele foi aplicado aos testes, não à configuração de produção. O relatório de
resultados contém os parâmetros, limitações e caminhos dos artefatos.
