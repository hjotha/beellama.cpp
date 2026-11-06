# WHT ping-pong: experimento CPU reproduzível

Protótipo isolado de `tools/xbox/shaders/rkva/rkva_range_reduce_opt.hlsl`.
Não altera o shader de produção. O patch é material de revisão, não é aplicado
pelo runner. Não há compilação HLSL/DXIL, execução GPU/Xbox ou benchmark neste pacote.

## Objetivo e limites

O candidato usa dois bancos LDS de 256 floats para conservar a ordem aritmética
da WHT e reduzir as barreiras do laço de **14 para 7**. Contando a barreira de
inicialização da WHT, são **15 para 8**. Custa **1 KiB adicional de LDS**:
WHT de 1.024 para 2.048 bytes; declarações compartilhadas de 4.104 para 5.128 bytes.
Menos barreiras não demonstra menor latência; a ocupação pode piorar.

Não usa intrínsecos Wave. Mantém 128 lanes participantes, strides de 1 a 64,
normalização por 1/√128, H2 e normalização por 1/√2 separadas. Preserva todo o
código fora da alteração WHT delimitada, incluindo o ramo sem WHT.

O teste extrai o corpo WHT da fonte HLSL e substitui as barreiras por suspensões
em corrotinas C++20. É um **modelo CPU da fonte**, não um compilador HLSL. Usa
float32 estrito com subnormais graduais, sem FMA/reassociação; a política de
denormais e a compilação real do console permanecem sem verificação. NaNs são
comparados por classificação, não por payload ou sinal. O modelo independente
do reducer em torno da WHT é simplificado, não valida o kernel inteiro no GPU.

## Executar localmente

Requisitos: Bash, Python 3, NumPy, GNU `patch`, `cmp` e g++ com C++20,
AddressSanitizer e UndefinedBehaviorSanitizer. Validado com g++ 14.2.0,
Python 3.12.14 e NumPy 2.3.5 em Linux. Em Windows, use WSL/Linux para esta suíte.
Não é necessário DXC, acesso ao console ou ao repositório de origem.

A partir deste diretório:

```sh
bash run_all.sh
```

O comando pode ser chamado por caminho a partir de outro diretório. Todas as
saídas são gravadas em `.build/`, ignorado pelo Git. O runner valida o SHA256 do
fixture, regenera e compara candidato/patch, extrai os kernels CPU, compila e
executa a versão otimizada, verifica estrutura/layout e repete com ASan/UBSan.
Compara também os resultados JSON otimizado/sanitizado. Qualquer falha encerra
a execução com status diferente de zero. Resultado esperado: `PASS_CPU_MODEL_ONLY`.

`PYTHON` e `CXX` podem indicar o executável desejado. Exemplo:

```sh
PYTHON=python3 CXX=g++ bash run_all.sh
```

LeakSanitizer fica desativado por padrão; **não há qualificação de vazamentos**
no resultado publicado. Para habilitá-lo em um ambiente que o suporte:

```sh
WHT_DETECT_LEAKS=1 bash run_all.sh
```

`results.json` é o resumo compacto da execução publicada, não é sobrescrito pelo
runner. Resultados detalhados da nova execução ficam em `.build/results.json`,
`.build/cpu_results.json`, `.build/structure_results.json` e
`.build/sanitized_results.json`; hashes em `.build/source_hashes.json`.
Os contadores dependentes de distribuições aleatórias podem mudar entre
bibliotecas padrão C++; as comparações baseline/candidato/oráculo continuam
sendo feitas dentro da mesma execução. Não use igualdade com o JSON publicado
como substituto das verificações.

## Cobertura da execução publicada

- 2.469 vetores, 632.064 elementos; bases positivas/negativas, aleatórios,
  expoentes amplos, ±zero, subnormais, extremos, Inf e NaN.
- Oráculo independente float32 por pares/blocos: 429.856 comparações finitas e
  57.315 de infinitos bit a bit; 144.893 NaNs por classificação.
- 94 casos do modelo do reducer, incluindo 60 poison e 12 vazios.
- 260 layouts de índices, incluindo 130 com número ímpar de cabeças e splits
  0–257. Não testa o padding do codec bulk.
- 2.048 escalonamentos do modelo de duas wave64 com memória versionada;
  bancos disjuntos, um escritor por destino e rendezvous entre etapas.
- 64 vetores contra matriz H256/16 float64; erro máximo ≈ 1,034e-6.
- Controles negativos detectam variante in-place sem barreira pré-escrita,
  normalizações fundidas e banco final errado: 1.536 vetores em cada caso.
  Remover barreiras do modelo de waves falha em 1.479/2.048 escalonamentos.
- ASan e UBSan sem erros; JSON otimizado e sanitizado idênticos.

## Proveniência

`baseline.hlsl` é o fixture imutável do snapshot informado como
`4bad0205c1658ac1aa67b358b89884c2e9cddf4c`.
`.gitattributes` preserva finais de linha LF para manter os hashes do fixture
e do patch também em checkouts Windows.
Esse identificador de revisão não foi verificado independentemente; o SHA256
completo identifica os bytes realmente testados. `source_manifest.json` registra
os hashes da fonte, candidato e patch e o hash do script de compilação examinado.
`source_hashes.json` identifica os arquivos deste pacote. O runner não depende
de caminhos absolutos nem precisa buscar esse snapshot.

- Baseline: `3513aa4a0dc38597cc0f690e768ad2837873543d1cbfc64a645baa1ec6a586fe`
- Candidato: `d31a509a8bc1966d5b47158c9a31145ef116c5246c22f3fd9078cbc8a546fd82`
- Patch: `0d5bed237343fa451cb9192cba6b399c0eae4b5f30f0b423afe6bd6df5082d85`

## Próximas verificações do agente local

1. Execute esta suíte e registre `.build/results.json` e o compilador usado.
2. Antes de integrar, compare a fonte de produção com o fixture e revise
   `candidate.patch`. Não aplique o patch cegamente em uma fonte diferente.
3. Em uma cópia de teste separada, compile baseline e candidato com o DXC,
   perfil, macros e flags reais. O alvo `reduce_kvarn4` no script examinado não
   passava `-Gis`. Inspecione o DXIL para confirmar as duas normalizações,
   as barreiras de grupo e os recursos; o teste CPU não garante a decisão do
   otimizador. Se mudar `precise` ou `-Gis`, revise e valide a nova alteração.
4. Só com o fluxo local/console autorizado, teste poison, casos vazios, qualidade
   e resultados GPU antes de medir latência. Compare ocupação/registradores/LDS
   e use as mesmas entradas e condições para baseline e candidato.
5. Não conclua ganho de desempenho nem prontidão para produção apenas a partir
   desta suíte. Nenhuma etapa GPU/console dos itens acima foi executada aqui.
