# MQ2 rowlane / GQA2: validação CPU portátil

Este diretório é um experimento de correção, não uma alteração de shader de
produção. MQ2 rowlane é uma **proposta em Python/FP32**, comparada com um oráculo
FP64 independente. O GQA2 é validado separadamente por preprocessamento,
aritmética de layout e funções de política C++. Nenhum desses testes compila
HLSL/DXIL ou demonstra ganho de velocidade no Xbox.

## Para o agente local testar

Use Linux ou WSL, Python 3.11+, NumPy, `g++` com C++17, `patch` e um
preprocessador C (`cpp`, `clang` ou `gcc`). Não precisa de console, modelo,
serviço RPC, rede durante os testes, objetos Git antigos ou outro checkout.
NumPy 2.3.5 e Python 3.12 foram usados nos resultados preservados. Para instalar
a mesma versão do NumPy em um ambiente virtual já autorizado:

```sh
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r requirements.txt
sh run_all.sh
```

Se o ambiente já tiver NumPy, basta `sh run_all.sh`. A instalação de dependências
pode precisar de rede; a execução dos testes não. `PYTHON` e `CXX` podem indicar
outros executáveis compatíveis. `CPP` seleciona o preprocessador. Não use
`python -O`, `NDEBUG` nem relaxe os limites numéricos.

O script pode ser chamado de qualquer diretório. Gera tudo em `.build/`, que é
ignorado pelo Git, e não modifica os resultados preservados. Uma execução
completa leva vários minutos nesta classe de CPU. Resultado esperado:

- seis testes de contrato/layout GQA2;
- 5.598.720 combinações de política e quatro combinações de flags;
- 895.320 seletores, 329.855.172 classificações e 179.064 verificações de
  queries fantasma no trecho HLSL executado como C++;
- 40 cenários numéricos, 104 execuções e 36 comparações bit a bit entre
  dois e seis heads locais;
- controles negativos detectados, sem relaxar os limites OUT/LSE.

Leia `.build/results/summary.json` e `.build/results/manifest.json`. Os JSONs
individuais e logs detalhados ficam no mesmo diretório. Para diagnosticar falha,
comece pelo primeiro comando que falhou e pelo log correspondente. O script
termina imediatamente em erro e remove o resumo agregado antigo antes de rodar.

Ao relatar ao mantenedor, inclua comando, versões, resumo, primeiro erro e se
houve diferença numérica. Não copie logs com caminhos pessoais sem sanitizá-los.
Uma falha não autoriza reduzir tolerâncias nem aplicar mudanças ao runtime.

## Evidências históricas e nova reprodução

- `results/historical-20261106/`: resultados compactos do experimento original,
  antes de sua adaptação para distribuição portátil.
- `results/portable-validation-20261106/`: resultado compacto de uma nova
  execução do pacote portátil; consulte seu registro de validação.
- `.build/results/`: somente a execução local mais recente. Esses arquivos
  nunca substituem os snapshots anteriores.
- `provenance.json`: commits de origem, SHA-256 dos arquivos originais, fixtures
  e alterações de empacotamento. Os modelos base coincidem com o checkout
  `ff008e07eea8e471ebe247ad3ef5dfce35b828bf`.

O snapshot principal preservado é
`4bad0205c1658ac1aa67b358b89884c2e9cddf4c`. O branch histórico GQA2 termina em
`582cc8d8e70216e0ba697e735fbe3c6817704f9a`. A base histórica indicada pelo teste
original era `8bad8e84e67e9fcd43d12f637962b87685e0fecc`; este pacote **não** a
consulta por `git show`. Usa o arquivo principal preservado para a comparação
de tokens. Isso não equivale a uma validação da integração com o `main` atual.

## Fixtures e patch

`fixtures/main_mq.hlsl` é uma cópia imutável do shader principal histórico.
`prepare.py` aplica `patches/gqa2-fixture-only.patch` **apenas** à árvore
descartável `.build/branch/` e confere os hashes do resultado. O patch contém
somente shader MQ, wrapper GQA2 e teste histórico de layout. Ele não implementa
MQ2 rowlane. Não aplique esse patch à árvore de produção.

Os trechos `row_kind` e seleção de prefixo são conferidos contra o shader
reconstruído e executados como C++ com primitivos equivalentes; apenas a
anotação `[unroll]` é removida do trecho de seleção. A política GQA2 é um trecho
independente do branch histórico, com constantes do host explícitas. Os hunks
completos antigos de engine/profile **não aplicam limpos** ao main inspecionado.
Não foram incluídos nem usados como teste de integração.

O arquivo de nome `rkva_kvarn_serial_prefix_probe.py` contém somente a função
`compare()` original. O cliente de rede e o restante do probe foram excluídos.
Os outros modelos são dependências da proposta MQ2. O caminho padrão opcional
de fixtures Bee foi tornado relativo ao pacote. Não há fixtures binários Bee
incluídos e todos os resultados aqui usam registros sintéticos Bee-like.

## O que foi medido

Os resultados históricos tiveram erro absoluto máximo OUT de
`0.000182720825` e LSE de `1.45762484e-05`. O gate permanece:

- OUT: `abs(erro) <= 3e-4 + 3e-4 * abs(referência)`;
- LSE: `abs(erro) <= 3e-4 + 3e-5 * abs(referência)`;
- OUT deve ser finito e a máscara de finitude de LSE deve coincidir.

O caso com maior erro absoluto OUT consumiu 3,009% do orçamento OUT e 1,777%
do orçamento LSE; essas proporções são desse replay, não o máximo global de
todos os orçamentos. A mutação que compartilha `other` da primeira slice de K
entre duas slices é rejeitada com erros OUT de 14,43 e LSE de 2,75.

Há cobertura numérica até 1.025 linhas, nq 2/3/5/9, caudas 0/128, origens
0/72.704, splits 448/512, split forçado 211 e splits vazios, rewind/reappend,
ordem invertida e primeira query vazia. Geometria, mas **não atenção numérica
longa**, cobre 28.672 linhas. Com origem 72.704 e tail128, os splits 448/512
produzem 24.320/28.416 linhas compartilhadas em grupos de 128. A redução do
resíduo geométrico não demonstra qual política será mais rápida no hardware.

Os arrays `groupshared` observados no GQA2 somam 12.384 bytes com seis heads
locais e 4.128 bytes com dois. Menor uso de memória compartilhada não prova
ganho de performance, pois a duplicação de trabalho e a ocupação precisam de
medição real.

## Limites e próximo passo

Não foram executados DXC, Windows/UWP, PSO, GPU/Xbox, paridade de logits/KLD,
estado recorrente ou testes com modelo real. Cabeçalhos DXIL GQA2 não estavam
presentes no patch preservado; `GeneratedDxilTest` não é executado.
`--long` e `--contract` existem no modelo numérico, mas não fazem parte da suíte
padrão nem dos resultados aqui preservados. A multiplicação/acumulação QK usa
a aproximação FP32 fundida definida no modelo; PV padrão não usa a opção de
contração. O runtime CPU não é uma estimativa de velocidade de GPU.

Antes de promover qualquer ideia a produção, faça uma integração separada
contra o main desejado, compile HLSL/DXIL, confira recursos/PSOs e execute a
qualificação numérica e de desempenho de hardware autorizada. Este pacote não
executa nenhuma dessas etapas nem altera qualquer shader ativo.
