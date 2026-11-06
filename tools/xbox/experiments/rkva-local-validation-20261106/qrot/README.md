# Q_rot reutilizado: experimento CPU autocontido

Proposta **desligada por padrão**, sem ligação com host/PSO e sem alteração de
produção. Os testes validam um modelo CPU, não compilam HLSL e não executam GPU
ou Xbox. Não há resultado de desempenho de hardware.

## Repetir os testes locais

Use Linux ou WSL com shell POSIX, Python 3, NumPy, `g++` e `cpp` disponíveis.
A execução registrada usou Python 3.12.14, NumPy 2.3.5 e GCC 14.2.0.
Nenhum pacote é instalado automaticamente e nenhuma conexão de rede é feita.

No diretório que contém este arquivo:

```sh
sh ./run.sh
```

O script funciona independentemente do diretório atual de quem o chama, verifica
os SHA256 dos fontes/modelos, compila apenas a rotina C++ de referência e executa
os testes estruturais e numéricos. Pode levar alguns minutos. O resultado esperado
é `PASS`, com 44 casos de atenção, 7 grupos de transformações, 30 casos de layout,
16 verificações de cobertura longa e 5 controles negativos.

Binário e resultados novos ficam em `.build/`, ignorado pelo Git. Compare
`.build/results.json` com `results-summary.json`; resultados estruturais ficam
em `.build/structural-results.json`. Não use os módulos em `models/` como
executáveis independentes. Para validar a integridade da cópia publicada:

```sh
sha256sum -c SHA256SUMS
```

`prepare.py` foi omitido: ele dependia de diretórios externos usados na pesquisa.
Os fixtures exatos, protótipo e wrapper C++ necessários já estão incluídos.
`verify_sources.py` confirma a extração literal da rotina CPU, a identidade do
include e a equivalência entre patch e protótipo, sem precisar desses diretórios.

## Escopo e custo estático

- Base dos fixtures HLSL/C++: main `4bad0205c1658ac1aa67b358b89884c2e9cddf4c`; SHA256 completos estão em
  `source_manifest.json`. Os modelos preexistentes têm seus hashes de origem e
  de publicação registrados. `model-adaptation.patch` documenta a adaptação.
- `qrot-reuse-opt-in.patch` altera somente o carregamento inicial de Q no decode
  especializado serial-prefix, protegido por `RKVA_DECODE_QROT_REUSE=0`.
- O candidato usa o plano Q_rot já produzido, preservando Q original para a cauda.
  Não há ligação de PSO, seleção pelo host, política de split ou dispatch nova.
- A contagem de loads expressos no fonte **dobra de 6 KiB para 12 KiB por grupo**:
  duas waves leem os mesmos elementos em registradores. Não é tráfego DRAM medido.
  Cache e benefício líquido da remoção do WHT/barreiras iniciais são desconhecidos.
- A igualdade bit a bit observada é do modelo CPU FP32 e da rotina C++ estrita.
  Não estabelece igualdade sob compilador HLSL, FTZ, contração ou driver.

## Roteiro para o agente local

1. Execute esta suíte primeiro e preserve seus resultados. Não aplique o patch a
   um checkout de produção como parte desse teste CPU.
2. Em um checkout experimental separado do repositório, confira se o fonte atual
   coincide com o SHA de `source_manifest.json`; se não coincidir, revise o diff
   antes de portar. No diretório raiz desse checkout, use
   `git apply --check <caminho>/qrot-reuse-opt-in.patch` antes de aplicar.
3. Em uma futura sessão de hardware autorizada, compile um **PSO separado opt-in**
   com `RKVA_DECODE_QROT_REUSE=1`, `RKVA_DECODE_OPT=1` e `RKVA_SERIAL_PREFIX=1`,
   preservando o PSO baseline e o gate existente nq1/6×KV. Essa integração ainda
   precisa ser implementada e revisada; o patch sozinho não ativa o caminho.
4. Compare A/A e A/B no mesmo hardware e com inputs, flags e splits idênticos,
   tanto standalone nq1 quanto segmento bulk nq1. Capture Q_rot, OUT e LSE;
   verifique tail/origin, codecs, causalmente vazio, rewind/reappend e limites.
5. Somente depois da validação funcional, meça pass1+reduce e latência ponta a
   ponta. Não conclua ganho por tempo CPU ou contagem de operações.

Nenhum desses passos de integração/console foi executado por este pacote.
`REPORT.md` contém cobertura, tolerâncias, diferenças observadas e limitações.
