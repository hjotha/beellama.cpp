# Reprodução local: shaders MQ2/GQA2 em CPU

Este é um diagnóstico com dados sintéticos, não uma correção de produção nem a reprodução do gate original do modelo no Xbox.

## Dependências

Ambiente usado: Linux x86-64, CPU AMD EPYC 9V74, Mesa Lavapipe 25.0.7, LLVM 19.1.7, DXC 1.9.2609.5, g++ 14.2.0, Python 3.12.14 e NumPy 2.3.5. O executor requer headers e loader Vulkan. Os scripts não instalam componentes, não baixam modelos e não modificam repositórios.

Fontes fixadas do fork beellama.cpp:
- Baseline MQ2/cap64: `e8a1380e2485ae27220001337ed5982122fde803`.
- MQ2/cap32: `f3fb8a933c9bf800434f03125cea1c9abf99f157`.
- MQ2/cap128: `4a9eea29ef49ecaab3e72a476a1cd1a576a79efa`.
- GQA2: `582cc8d8e70216e0ba697e735fbe3c6817704f9a`.

Os três commits MQ2 têm HLSL idêntico; os caps alteram o particionamento no host. Os hashes estão nos metadados do pacote.

## Compilação e execução

```sh
python build_portable.py --mq-repo /caminho/mq2 --gqa-repo /caminho/gqa2 \
  --dxc /caminho/dxc --vulkan-include /caminho/include
export VK_DRIVER_FILES=/caminho/lvp_icd.json
export LD_LIBRARY_PATH=/caminho/bibliotecas
python long_fixtures.py fixture2048 --active 2048 --windows 2047,2048,2048
REPLAY_THREADS=4 python run_case.py fixture2048 64 mq_cap_source-3barrier mq64
REPLAY_THREADS=4 python run_case.py fixture2048 64 mq_gqa2-3barrier gqa64
```

`build_portable.py` verifica hashes dos HLSL e gera wrappers com os caminhos recebidos. Não depende dos caminhos originais do workspace. Configure o loader Vulkan para encontrar o ICD CPU e suas bibliotecas. Cada execução exige uma pasta de saída nova e salva comando, tempo, pico de memória, hashes e comparação com o oracle FP64.

`run_case.py` usa `GALLIVM_PERF=nopt`, `LP_NATIVE_VECTOR_WIDTH=128` e cache desativado. `REPLAY_THREADS` seleciona de uma a quatro threads de trabalho. Esses controles são locais ao processo. Nesse Lavapipe, largura de vetor de 128 bits significa subgroup **nativo de 4 lanes**, não wave128.

## O que é emulado

O adaptador `logical128-3barrier.hlsl` implementa uma wave **lógica de 128 lanes** com memória compartilhada e barreiras de um workgroup de 128 threads. A soma preserva uma árvore binária explícita. Os wrappers incluem o corpo original do shader; somente as operações de subgroup são adaptadas. O reducer usa a fonte original com `RKVA_REDUCE_WHT=1`.

As variantes `logical128-*` não são o shader original inalterado nem reproduzem a ordem das operações de uma wave64 do Xbox. A wave lógica 128 usa uma única wave por workgroup e não exercita a divisão e a mesclagem entre duas waves64; um defeito específico desse caminho continua possível. A via nativa original foi testada separadamente e acionou corretamente seu guard com wave8. Não remova esse guard. A configuração `LP_NATIVE_VECTOR_WIDTH=2048` anunciou 64 lanes, mas falhou nos testes de soma e índices; não é suporte funcional a wave64.

Antes de usar outro driver ou CPU, execute e confira os smokes nativo e lógico. A variante de três barreiras foi comparada bit a bit com a implementação anterior do adaptador em fixtures de 256 e 2048 linhas KV.

## Escopo e limites

Os buffers são sintéticos e usam o layout real de registros KVarN4, stage, tail, queries, positions e cbuffer. Os números 102400, 112640 e 130048 indicam **linhas KV ativas da fixture**, não contexto total do modelo nem a divisão local/remota do serving. A geometria exata do teste original, seus tensores e logits continuam ausentes.

Canários verificam sobrescrita no final das alocações, não ausência de todas as leituras fora da faixa. Tempos e memória incluem execução/compilação do executor CPU e não medem desempenho do Xbox. Igualdade neste diagnóstico não certifica DirectX, DXIL ou o driver do console. Nenhuma correção deve ser inferida sem evidência causal no caso real.


## Candidato isolado

O patch de portabilidade para NaN não foi aplicado à fonte original e **não foi executado na matriz longa de 12 testes**. Seus controles limitaram-se aos parciais finitos de 2048 linhas, às saídas diretas de 256 linhas e ao caso negativo de NaN. Consulte RESULTADOS.md para separar a falha original do resultado desse candidato; ele não é um fix validado do Xbox.
