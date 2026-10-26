# INT8 RKVA — diagnóstico Codex, 2026-10-26

**O defeito do INT8 no console continua aberto. Não encontrei uma causa concreta
que explique o pacote 299, e não apresento esta entrega como correção validada.**
Encontrei e corrigi dois defeitos do bisect, acrescentei um interpretador do
DXIL e deixei um braço INT8 sem shuffles pronto para o próximo teste do Claude.
O header INT8 entregue é explicitamente **diagnóstico**, não o build padrão.

Trabalho somente no clone `/home/hjotha/beellama-agents/int8-codex`, branch
`fix/rkva-int8-records-codex-20261026`, base `fdd01db05`. Sem Xbox, GPU,
llama-server, systemctl, builds CUDA/MSVC/UWP, push, merge ou outros clones.
DXC executado somente em `C:\Users\hjotha\build\oc-int8-codex-20261026\`,
com o wrapper local já configurado para esse diretório.

## Defeitos comprovados e correções

1. **O bisect F32 omitia `kq_scale`.** O caminho INT8 aplica `kq_scale * ak`,
   enquanto `RKVA_MQ_I8_F32_PROBE=1` decodificava K com `ak` e reduzia QK sem
   o fator de atenção. Com `kq_scale=1/16`, seus scores de registros eram
   **16 vezes** o correto. Acrescentei `val *= kq_scale` antes da redução.
   Isso invalida a interpretação anterior de uma reprovação desse bisect como
   prova de defeito no mapeamento/PV. **Não explica a falha do INT8**, que já
   aplicava o fator.
2. **O script de compilação passava a anotação de proveniência ao DXC.**
   `compile_flags+=("extra: $RKVA_EXTRA_DEFINES")` colocava texto de relatório
   na linha de comando do compilador. Agora a anotação vai somente ao gerador
   do header; os `-D` continuam na chamada real. O build pelo script com o novo
   define foi executado com sucesso.

Prova numérica do primeiro defeito: o interpretador rejeita o DXIL F32 antigo
(`.oc-artifacts/dxc/mq2_i8_probe.txt`). Primeiros quatro scores:

```
antigo:   0.629949  17.053427  48.568752  -0.124924
correto:  0.039372   1.065839   3.035547  -0.007808
```

O DXIL F32 corrigido passa 8.832 comparações de score, erro máximo `2,03e-6`.

## Evidência sobre o defeito original

Recompilei o fonte inicial com o DXC autorizado e obtive exatamente os
141.612 bytes do header original, SHA256
`85b2e376a9ddd8071d77791afe5fac99e942eb3eec3f95a482d3a27aa26e854e`.
Portanto o artefato local inicial não estava desatualizado.

A inspeção e execução CPU do DXIL confirmam, nos casos testados:

- Endereçamento de bytes de K `[dim][pos]`, slices, `scale/zero` por dimensão
  e `other` por posição; queries com 16 dimensões por lane.
- Opcode 163 `Dot4AddI8Packed(acc,a,b)`, com o acumulador correto na cadeia,
  bytes signed e conversão `sitofp`; empacotamento e índices esperados.
  O contrato do opcode está na
  [definição oficial do DXC](https://github.com/microsoft/DirectXShaderCompiler/blob/main/utils/hct/hctdb.py).
- Máximo de quantização por slice, constante mantida parcial por lane,
  redução XOR `{8,4,2,1}` e broadcasts das lanes `{0,16,32,48}`.
- Ramos do caminho de scores uniformes dentro da wave nas fixtures.
  O fast path não usa LDS; a memória compartilhada aparece no merge/WHT.

O novo `tools/xbox/tests/rkva_mq_i8_dxil_model.py` interpreta as instruções SSA
reais do arquivo `-Fc`: loads de buffers brutos, conversões F16/F32, aritmética
F32, inteiros, phis, branches e coletivas. Usa registros do encoder do probe
serial, dois grupos de registros, KV heads 0/3, duas queries, duas waves,
`live=256/384` e fast ranges de 120/248 linhas. Não parte de planos de K
pré-decodificados para executar o shader.

| DXIL interpretado | Scores comparados | Erro máximo contra referência CPU |
|---|---:|---:|
| INT8 original do pacote 299 | 8.832 | `7,87e-6` contra álgebra Q8 |
| INT8 sem shuffles, entregue | 8.832 | `9,30e-6` contra álgebra Q8 |
| F32 com `kq_scale` corrigido | 8.832 | `2,03e-6` contra score F32 |

**Limite importante:** o interpretador termina antes do primeiro dot F32
residual/tail e valida os scores, não os numeradores PV, o merge final ou o
código de máquina gerado pelo driver. Não emula bugs de GPU nem todas as
reassociações permitidas pelo fast math. O sucesso do INT8 original nesse
teste não constitui evidência de correção em hardware.

Também comparei hipóteses numéricas com os dados das fronteiras do probe, seed
20261266. Para `boundary 256 nq=2`, o hardware registra OUT `15,7028`, LSE
`16,0559`; a álgebra Q8 prevê aproximadamente `0,1986 / 0,1109`. Interpretar
Q8 como unsigned prevê `46,93 / 787,63`; tratar nibbles como signed de 4 bits
prevê `46,93 / 878,87`; descartar registros prevê `36,97 / 7,21`.
Essas hipóteses simples não reproduzem o resultado. Experimentos locais em
`.oc-artifacts/fault_sweep.py` e `.oc-artifacts/fault_sweep.log`.

## Braço entregue para o Claude

`RKVA_MQ_I8_MASKED_REDUCE_PROBE=1` elimina **todas** as chamadas
`WaveReadLaneAt` do fast path:

- Dois `WaveActiveMax` com máscara de slice substituem o máximo XOR. As
  quatro cópias de Q/scale nos grupos de linhas são iguais, portanto o máximo
  de cada slice é o mesmo.
- Quatro `WaveActiveSum(i8_row == t ? val : 0)` produzem os quatro scores.
  Todas as lanes executam cada coletiva; exatamente 16 contribuem por linha.
- Permanecem a quantização Q8, os 96 calls estáticos de dp4a, K/V, o online
  softmax, PV, residual/tail, merge e layout `[split][q*h][258]`.

Isto é um bisect das operações de wave e de sua compilação, **não uma causa
raiz demonstrada**. Também altera o código gerado/pressão de registradores;
mesmo um sucesso no console não provaria sozinho um bug específico do driver.
Não medi desempenho desse braço. O ganho anterior de 30,7% não é resultado
atribuível a ele.

O define tem default 0 no fonte. O **header commitado foi gerado com 1** e
registra isso em `Additional compiler options`. Recompilar sem o define
restaura o caminho INT8 original, ainda quebrado no console.

| Payload | Bytes | SHA256 |
|---|---:|---|
| MQ2 exato | 118452 | `9d81c3beba4e36756c89c6bc70c5807cedb990be67fba683fc714ff7a1c5b431` |
| MQ4 exato | 236896 | `ef2878458bd37b6f702fec19f36e674a836803d300915cf2604bfd801dfb48bb` |
| MQ2 INT8 masked, entregue | 140344 | `de271b3c7d28050a09647a974d302eaa46853e24d562bcdcec44889ced9e4bd0` |

MQ2 e MQ4 foram recompilados e comparados byte a byte com os payloads iniciais.
Nos dois headers, somente a linha `Source SHA256` mudou. O header INT8 também
foi comparado com o binário cuja assembly foi interpretada.

## Testes executados

Todos passaram:

```bash
python3 tools/xbox/tests/rkva_mq_i8_quality.py
python3 tools/xbox/tests/rkva_mq_i8_lane_model.py
python3 tools/xbox/tests/rkva_pass1_sweep.py --self-test
g++ -std=c++17 -Wall -Wextra -Werror -O1 \
  tools/xbox/tests/rkva_range_profile_cpu.cpp \
  -o .oc-artifacts/rkva_range_profile_cpu
.oc-artifacts/rkva_range_profile_cpu
```

Quality: 10 casos gated; lane model: 8 casos; sweep: 4 shapes e 3 caminhos
fail-closed; profile CPU: `ok`. Os casos informativos sem correção DC do
quality continuam podendo ultrapassar 1,15x; não pertencem ao gate DC-exact.
Usei `OPENBLAS_NUM_THREADS=1` nos testes NumPy.

Reprodução do teste novo (arquivos DXIL/assembly locais, não commitados):

```bash
python3 tools/xbox/tests/rkva_mq_i8_dxil_model.py \
  .oc-artifacts/dxc/mq2_i8_codex_before.txt
python3 tools/xbox/tests/rkva_mq_i8_dxil_model.py \
  .oc-artifacts/dxc/mq2_i8_masked.txt
python3 tools/xbox/tests/rkva_mq_i8_dxil_model.py \
  .oc-artifacts/dxc/mq2_i8_f32_corrected.txt --f32-probe
```

Para gerar os arquivos, usar o wrapper com `-Fc`, `-T cs_6_4`,
`-E CSRangeAttnKvarn4Mq`, `-D RKVA_MQ_QT=2`, o wrapper HLSL INT8 e `-Fo`;
o wrapper grava a assembly ao lado do binário. Acrescentar o define do braço
masked ou F32 conforme o caso. O teste antigo F32 foi executado e rejeitado
pela comparação numérica, não por erro de parser.

## Próxima execução no console — somente Claude

1. Usar o header INT8 entregue, recompilar o pacote, ativar a flag pelo fluxo
   já controlado pelo Claude e executar o probe inalterado:

   ```bash
   python3 tools/xbox/tests/rkva_kvarn_serial_prefix_probe.py \
     --execute --host 192.168.1.200 --port 50054 --modes 2 --full \
     --seed 20261266 --output serial-i8-masked-codex.json
   ```

   Confirmar bucket `kvarn4_mq2_i8` nos casos MQ2 com tail 128. O braço sem
   fast records deve continuar na tolerância exata. Nos casos com registros,
   procurar erro de quantização Q8, não OUT/LSE de dezenas. A expectativa
   típica é OUT na ordem de `1e-2..1e-1` e LSE `1e-2`, **não `1e-5`**.
   Isso não é um teto universal: a própria álgebra prevê `0,199 / 0,111` para
   boundary 256 nq2 e alguns outros outliers maiores. Comparar por caso com
   o modelo Q8; não relaxar o oráculo original nem usar apenas `success`,
   pois sua tolerância continua sendo a de um kernel exato.

2. Repetir o bisect F32 com a correção de escala antes de interpretar seu
   resultado anterior. Compilar pelo script, com o DXC autorizado:

   ```bash
   DXC="$PWD/.oc-artifacts/dxc/dxc193.py" \
   RKVA_EXTRA_DEFINES='-D RKVA_MQ_I8_F32_PROBE=1' \
     bash tools/xbox/scripts/compile-rkva-range-shaders.sh attn_kvarn4_mq2_i8
   ```

   Recompilar o pacote e executar o mesmo probe, salvando JSON separado.
   Este braço deve ter erro da ordem de F32/OFF, não erro Q8.
   Para reconstruir o header INT8 desta entrega:

   ```bash
   DXC="$PWD/.oc-artifacts/dxc/dxc193.py" \
   RKVA_EXTRA_DEFINES='-D RKVA_MQ_I8_MASKED_REDUCE_PROBE=1' \
     bash tools/xbox/scripts/compile-rkva-range-shaders.sh attn_kvarn4_mq2_i8
   ```

Se o masked INT8 passar no critério Q8, há um candidato a workaround para
qualificar com sweep/rollback/capacity e medir novamente. Se falhar e o F32
corrigido passar, investigar a quantização/dp4a e a compilação desse caminho.
Se ambos falharem, ainda será necessário localizar o primeiro valor divergente
em hardware; não concluir automaticamente que PV ou o layout está errado.
Um braço adicional combinando os dois defines pode então isolar o caminho
F32 sem shuffles. Não foi necessário para as verificações CPU desta entrega.

Somente depois do gate numérico, medir o sweep com as mesmas condições do
OFF e `--expect-mq2-kernel kvarn4_mq2_i8`. Nenhuma alegação nova de desempenho
é feita aqui. A flag permanece opt-in.

## Commits

- `f41ef2d22`: escala do bisect F32, script de defines, braço masked e
  interpretador DXIL.
- `fb68b0e3b`: header INT8 diagnóstico e hashes dos headers exatos, com
  payloads MQ2/MQ4 preservados.

**Pendente:** causa concreta e correção qualificada do defeito original no
Xbox. O resultado do bisect que o Claude estava executando não chegou nesta
sessão; foi enviado aviso sobre a escala ausente, sem acessar o console.
