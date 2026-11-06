# Q_rot reuse no decode KVarN4: teste local isolado

Data: 2026-11-06. Resultado: **PASS na validação CPU de matemática/layout**. Não é compilação HLSL, qualificação D3D12/Xbox ou benchmark. A execução deste experimento não acessa máquinas remotas nem altera console, configuração de produção ou credenciais.

## Resultado principal

- **44 casos completos de atenção**: OUT e LSE bit a bit iguais entre recomputar Q_rot e usar Q_rot fornecido, no modelo CPU FP32 de rowlane. Ambos foram comparados com o oráculo FP64 independente, dentro dos limites existentes: `|dOUT| <= 3e-4 + 3e-4|OUT|` e `|dLSE| <= 3e-4 + 3e-5|LSE|`.
- Maiores diferenças contra FP64: OUT **0,00006185070402509041**, LSE **0,000009597882320022677**. As diferenças A/B entre candidato e baseline foram **zero bits**, não esses valores.
- **7 grupos de testes de WHT**: rotina CPU real extraída do main, transliteração do ping-pong decode e transliteração do bulk com dois heads por grupo coincidiram bit a bit para entradas finitas testadas. Inclui todos os **63.488 códigos F16 finitos**, **248 E5M2 finitos** e **254 E4M3FN finitos**, 2.049 heads F32 aleatórios, 256 vetores de base e 17 heads com zeros signed/cancelamentos/diferentes magnitudes. Os códigos são embaralhados e agrupados em vetores, não uma prova exaustiva de todas as combinações de 256 elementos.
- **30 casos de layout** F16/E5M2/E4M3FN verificam Q original, Q_rot, posições, bounds e sentinela; incluem padding do segundo head ímpar. O consumidor usa os endereços exatos do protótipo, com as duas waves lendo as mesmas quatro dimensões por lane.
- **16 verificações de partição de linhas** em 4.096/8.192/16.384/28.672 rows, origins 0/102400 e tail 0/128. Essas profundidades tiveram somente verificação de cobertura/layout, **não OUT/LSE numérico** neste artefato.
- **5 controles negativos** foram detectados: colapsar as duas normalizações para 1/16 (3.281 elementos FP32 diferentes), inverter lanes, ler o plano original como rotacionado, usar Q original no corpo (diferença OUT 8,82346), usar Q_rot na cauda original (diferença OUT 17,33855).

## O protótipo

`qrot-reuse-opt-in.patch` adiciona `RKVA_DECODE_QROT_REUSE`, default **0**, somente no shader decode serial-prefix já especializado. Quando 1, cada lane lê `Load4` do plano Q_rot em:

`n_query * n_head * 256 * 4 + ((qi * n_head + kv_head * 6 + g) * 256 + lane * 4) * 4`.

O caminho antigo permanece sob a opção desligada. A barreira inicial conservadora foi mantida. **Todo o código após o carregamento/initialização de Q é byte-idêntico**, incluindo caminhos record/residual/stage, cauda original, checagens de finitude, merge, WHT de saída e partials. A fonte original da cauda continua no plano zero. Não foi alterado o host, PSO, build script, split policy, dispatch ou reducer. Não há ativação automática.

`test_structure.py` passou para `RKVA_DECODE_ROWLANE=0` e `1`: opção desligada produz saída do preprocessador C **exatamente igual** ao baseline, e opção ligada remove apenas o WHT inicial e lê Q_rot. Combinações sem DECODE_OPT ou SERIAL_PREFIX são rejeitadas pelo guard. **Preprocessar não compila HLSL**.

## Tradeoff estático de acesso a Q

Cada grupo baseline lê 128 threads × 6 heads × 2 floats = **1.536 floats (6 KiB)** de Q original para o WHT inicial. O candidato lê 128 threads × 6 heads × 4 floats = **3.072 floats (12 KiB)** de Q_rot: as duas waves leem os mesmos elementos, cada uma para seus registradores. Esses números excluem a cauda e descrevem apenas os loads expressos no fonte, **não bytes de DRAM**, cache misses ou tempos medidos.

Portanto, eliminar o WHT/barreiras inicial não garante benefício líquido: há duplicação das leituras globais de Q_rot, possivelmente atendida por cache. Uma futura comparação poderia carregar Q_rot uma vez em LDS, com sincronização, mas esse segundo desenho não foi implementado nem testado. Este artefato mantém somente o menor protótipo de Load4 direto.

## Fontes e equivalência

Base principal: snapshot de main `4bad0205c1658ac1aa67b358b89884c2e9cddf4c`. O hash exato de cada fonte está em `source_manifest.json`. Quatro modelos Python preexistentes foram reutilizados; seus hashes de origem e de publicação também estão registrados no manifest.

Os hashes brutos de decode/rowlane entre os dois snapshots diferem por uma linha vazia final; os conteúdos sem whitespace final são idênticos. O bulk unpack tem diferença real (E4), por isso foi usado o bulk e codec **do main**, com transliterações novas. O helper completo KVarN também difere; `cpu_reference.cpp` contém a extração literal apenas da rotina `butterfly_wht256` **do main**, compilada por g++14.2 com `-O2 -fno-fast-math -ffp-contract=off`.

No modelo rowlane copiado, a única alteração numérica normal é receber Q_rot opcional no ponto onde antes calculava WHT(Q); Q original continua reservado à cauda. Há também um parâmetro deliberadamente incorreto usado só no controle negativo. A importação do probe e os entry points autônomos não utilizados foram removidos do modelo rowlane; nenhum probe ou acesso de rede é executado. `model-adaptation.patch` registra todas essas alterações. O oráculo monta registros/stage/tail de uma fonte canônica independente, incluindo escalas/zeros signed e `other` distinto por slice.

A existência dos dois produtores foi revisada estaticamente no main: `remote-attn-engine.cpp` prepara Q original+Q_rot por CPU, e `rkva-kvarn-bulk-f16.inc` seleciona unpack F16/FP8/E4, faz UAV barrier/transição antes de fornecer o buffer a `attend_range`. A sincronização D3D12 foi **lida, não executada**. O arquivo completo `remote-attn-engine.cpp` não é necessário aos testes e não está incluído; seu hash foi preservado para conferência no repositório de origem. Os testes deste pacote não reexecutam a revisão do host.

## Cobertura numérica detalhada

Para cada combinação de origin 0/102400 e tail 0/128:

- Rows 1,127,128,129,255,256,257,513,1025.
- Alternância entre saída direta (`split_k=1`) e splits de 77 rows, incluindo split final vazio e bordas desalinhadas.
- Codecs alternados CPU-F32, F16, E5M2 e E4M3FN. Não é o produto cartesiano completo de codec × shape.
- Modelo testa PV com e sem contração em casos alternados; não determina qual contração o driver fará.
- Query causalmente vazia e 10 splits de 128.
- Rewind 1025→1018, reappend de nove novas linhas até1027, passagem de seal e ring256.

O unpack/layout inclui `(n_query,n_head)` = (1,1),(1,3),(1,5),(3,1),(3,3),(1,6),(1,24),(2,24),(3,24),(9,24). Shapes fora de nq1/6×KV são apenas testes do produtor/endereçamento, **não ampliação do gate do decode** nem teste do kernel MQ.

Diagnóstico separado com NaN, ±Inf e overflow F32 coincidiu na classificação dos resultados CPU/transliterações. Payloads de NaN não foram comparados. Isso não testa nem modifica admissão no servidor; os códigos não finitos FP8 não participam dos testes de entrada admitida.

## Não testado e próximo passo

Não houve DXC/HLSL/DXIL, PSO, shader model/wave64 em GPU, UWP/MSVC, execução Xbox, validação de barreiras real, MSIX, modelo GGUF, geração de tokens, MTP, KLD, replay de captura real ou medição de desempenho. As duas normalizações são idênticas no modelo estrito CPU; otimização/contração/FTZ do compilador e driver ainda exigem comparação de Q_rot/OUT/LSE reais. Uma prova matemática/layout CPU não assegura bit-identidade no hardware.

O próximo passo técnico, em janela e ambiente apropriados, é compilar **um PSO separado opt-in** e comparar A/A e A/B no console, tanto standalone nq1 quanto segmento bulk nq1, preservando mesmos inputs/flags/splits. Medir pass1+reduce e latência ponta a ponta; não extrapolar tempo CPU ou contagem de operações para ganhos Xbox. Nada desse próximo passo foi iniciado aqui.

## Reproduzir e arquivos

No diretório deste artefato, executar `sh ./run.sh` (shell POSIX, Python3, NumPy, g++ e cpp; Linux/WSL). Tudo que executa é local CPU; nenhuma conexão remota é feita. `prepare.py` foi omitido porque dependia de diretórios externos. Os fixtures exatos e arquivos derivados necessários estão incluídos, e `verify_sources.py` confere SHA256, patch/protótipo e extração CPU antes/depois dos testes.

- `results-summary.json`: resumo compacto da execução reproduzida desta cópia.
- `.build/results.json`: resultados detalhados gerados por uma nova execução.
- `.build/structural-results.json`: checks de fonte/preprocessador gerados.
- `.build/`: binário e saídas locais ignorados pelo Git, não parte do pacote.
- `prototype/`: HLSL candidato e include rowlane.
- `qrot-reuse-opt-in.patch`: mudança mínima proposta.
- `source-main/`, `source_manifest.json`: fontes e proveniência.
- `test_qrot.py`, `test_structure.py`, `verify_sources.py`, `cpu_reference.cpp`, `models/`, `run.sh`: reprodução.
- `README.md`: requisitos e instruções para o agente local.
- `SHA256SUMS`: hashes finais dos artefatos.

Os hashes dos fontes originais foram reverificados após a execução e permanecem inalterados.
