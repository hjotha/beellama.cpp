# Handover operacional · paridade de prefill Strata × BeeLlama

**Objetivo ativo.** Atingir no BeeLlama o prefill da Strata no **Qwen3.8-Flash-Next Q2_0 completo, 48 camadas, pp28k, A100**, com os mesmos pesos e IDs, DFlash2 comparável e corretude verificada. Medir, integrar em `main` apenas o que passar e registrar as evidências. A referência completa confirmada é **2.597,40 t/s Bee × 4.932,87 t/s Strata**, ou **52,7%**. Faltam cerca de **5,20 s por pedido**. Há candidatos mais rápidos no recorte local; sua combinação ainda precisa de uma A100.

**Para entregar a outra CLI.** “Assuma este objetivo no gokaya. Leia este arquivo e verifique os processos e sessões antes de agir. Continue do primeiro item pendente da fila. Use a medição da Strata na mesma GPU, confira amostras, hashes, dispersão e VRAM, e preserve o estado da produção. Registre cada aceitação, rejeição e bloqueio real junto dos artefatos.”

Estado da entrega conferido às **16:59 CET de 24/11/2026**. A pedido do usuário, a build continua e a execução de novas janelas fica para a próxima CLI. Os PIDs e a disponibilidade de Colab expiram. Antes deste handover, `main` e `origin/main` estavam em `e7e04c4f7` (relatórios). O merge de código que removeu o staging do prefill DFlash2 é `8f9feff1e`. A incorporação de registros em `main` é separada da promoção de binário no serviço.

**Cópia transferível das experiências.** No disco externo do gokaya, `/media/hjotha/GAMES/strata-test/handover-20261124/` contém o README de restauração, `experiments.bundle` com as refs locais, `worktrees.tsv` com as bases dos patches, `patches/`, scripts do voo A100, inventário de artefatos e `SHA256SUMS`. O bundle é incremental e precisa do histórico de `origin/main`. Verificar antes de importar:

```bash
cap=/media/hjotha/GAMES/strata-test/handover-20261124
(cd "$cap" && sha256sum -c SHA256SUMS)
git bundle verify "$cap/experiments.bundle"
git bundle list-heads "$cap/experiments.bundle"
```

Se a outra CLI já usa o gokaya, os worktrees existentes são a primeira opção. Para outro checkout, seguir o README da cápsula: buscar as refs em `refs/remotes/handover/exp/*`, criar uma árvore limpa no commit indicado para cada patch e rodar `git apply --check` antes de aplicá-lo. Pesos, executáveis, conta e chaves são dependências externas; o manifesto aponta os artefatos já instalados. A cápsula privada verificada em 24/11 inclui seis refs experimentais, sete patches de worktrees (incluindo o dump pós-FA), 32 fontes do voo A100 e 69 arquivos de reprodução local. O inventário `artifacts.tsv` localiza os payloads externos. O arquivo `capture-times.md` data a coleta do WIP; mudanças posteriores devem ser examinadas na worktree. Os arquivos de `/tmp` citados são específicos do host. O repositório remoto contém este plano e a história aprovada.

## 1. Primeiros minutos na .57

```bash
cd /home/hjotha/beellama.cpp
hostname
git status -sb
git log -5 --oneline
git worktree list
ps -C ninja -C llama-server -o pid,etime,stat,args
pgrep -c nvcc || true
free -g
df -h /home/hjotha /media/hjotha/GAMES
systemctl is-active llama-server-root.service
curl -fsS --max-time 5 http://127.0.0.1:8090/health
ssh hjotha@192.168.1.157 'env HOME=/home/hjotha/.collabosm/isolated-home /home/hjotha/.collabosm/colab-cli/bin/colab sessions'
```

- **Build e gate SM89 concluídos.** A build `/tmp/opencode/beellama-ada-qsa-radix-main` concluiu o link de `llama-server`, `test-backend-ops`, `topk-parity` e `teacher-logits` após adicionar `-DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath-link,/opt/cuda/lib64"` em `build.sh`. O gate na RTX 4070 (`live-1795537039`) aprovou `test-backend-ops -b CUDA0 -o TOP_K` (539/539), `topk-parity` (`6bd6af6303a917c151fa66b8be872646b00716cca8d14f4269115cd233366f39`), `teacher-logits` (`c2e6ba6657291dc03742d010bade68b5da51904498683726852f18c2d77c49d3`) e `pp28k` vanilla (**12.817,15 → 13.367,27 t/s**, **+4,29%**, sem sobreposição). O cherry-pick `4bfed157e` (`ggml/src/ggml-cuda/top-k.cu`) foi integrado em `main`. O antigo build `/work-cpufb` de 22/11 já foi validado e encerrado.
- **4070 liberada ao entregar.** O `checkpoint-window.sh` terminou. A produção foi restaurada, `systemctl is-active` retornou `active` e `/health` retornou `{"status":"ok"}`. A asserção de igualdade dos 64 tokens falhou **depois de salvar as 12 amostras**. Cada braço reproduziu o próprio hash nas seis medições, porém os dois braços divergiram no 13º token. Resultado: checkpoint padrão 18.267,20 t/s (17.751,22–19.076,11; hash `7b1d91ad0ae9`), desligado 19.113,74 t/s (6.859,49–19.936,87; hash `398521a70475`). Nos quatro processos, os contadores de checkpoint foram **25 / 0 / 0 / 25** (cinco pedidos por processo). As faixas se sobrepõem. O resultado local sem draft não qualifica esta troca como bit-exact. Na A100, manter o pareamento sem checkpoints nos dois engines **e comparar o candidato contra o controle Bee também sem checkpoints**. Os logs e o JSON estão em `/tmp/opencode/strata-prefill-parity/slice4-native/checkpoints-1795535163/`.
- **A100.** A conta Pro do ColabOSM na .157 continuou retornando `Service Unavailable` (saída `5`) nas quatro novas tentativas (`paired-assign-20261124T161649Z.log` standard, `20261124T161740Z` high-mem, `20261124T162119Z` standard e `20261124T162216Z` high-mem). `colab sessions` confirmou zero sessões ativas. Não há resultado novo de A100 para a pilha experimental. Reconsultar antes de criar uma VM.
- **Disco.** A última limpeza moveu 16 arquivos inativos para o externo: 11,56 GiB transferidos, ganho líquido observado de 11,16 GiB na raiz. Houve conferência SHA-256 e links mantêm os caminhos. Os backups, modelos de teste e vários arquivos de voo antigos dependem de `/media/hjotha/GAMES` montado. Manifestos idênticos em `/tmp/opencode/gokaya-offload-20261124.tsv` e `/media/hjotha/GAMES/gokaya-offload-20261124/manifest.tsv`. O OLMoE permaneceu no disco interno porque o smoke usa hardlink.

## 2. Fila de execução

| Ordem | Trabalho e estado | Próxima ação e evidência exigida |
|---|---|---|
| 1 | **A100 · prioridade principal.** Pacotes já compilados na .57, copiados/verificados na .157. | Usar o launcher da seção 4. Rodar baseline histórico, baseline pareado e candidato na **mesma VM**. Uma A100-80G deve gerar seu próprio baseline. Salvar o tar e verificar as amostras antes de qualquer merge. |
| 2 | **Gate Ada independente.** Concluído e promovido (`4bfed157e`). | Em `live-1795537039`: `test-backend-ops -b CUDA0 -o TOP_K` 539/539 OK; `topk-parity` e `teacher-logits` idênticos byte a byte; `pp28k` vanilla 12.817,15 → 13.367,27 t/s (**+4,29%**, faixas 12.787,44–12.872,23 vs 13.344,22–13.403,66). Cherry-pick integrado em `main`. |
| 3 | **Fusão MoE e entradas compactas.** A combinação local passou nos logits e nas transições de modalidade. | Na A100 comparar `stock`, `direct` e `deferred`; depois medir a pilha com as mesmas condições de checkpoints da Strata. Confirmar ativações reais no log, hash de saída, logits e ausência de OOM/fallback. |
| 4 | **Atenção QSA 4×16.** Diagnóstico pós-FA concluído (`postfa-1795537226`); segue experimental fora de `main`. | `post-fa.json` confirmou Q e máscara 17/17 idênticos (`symdiff=0`) em todos os 7 chunks; chunks 0–1 idênticos (`dense-f16`); chunks 2–6 divergem na saída FA (`nmse` 3,27e-6–5,08e-6, `max_abs` até `3,80e-2` no `head=2`); no chunk 6 as 3 queries da cauda `[3968, 3969, 3970]` são bit-exact porque `8×8` e `4×16` formam a mesma cauda. Próximo passo opcional: kernel que compartilhe K/V entre dois subgrupos preservando uniões/fronteiras `8×8`. |
| 5 | **Checkpoints e precisão densa.** Novos cenários e diagnóstico descritos nas seções 4 e 6. | O ensaio local já mostrou divergência no 13º token; na A100, verificar contadores, hashes por braço e efeito das fronteiras de lote. Preservar o controle antigo para medir quanto do gap muda com o protocolo. FP16 denso demanda decisão de qualidade além do throughput. |
| 6 | **Integração e produção.** As otimizações experimentais não estão implantadas pelo simples fato de o relatório estar em `main`. | Consolidar somente os candidatos aprovados, testar a construção combinada, fazer merge `--no-ff` e push. Registrar commit, binários/flags, hardware, amostras e restauração do serviço. Repetir o A/B final do modelo completo. |

O critério de **promoção de uma mudança** é ganho de pelo menos 3% fora da dispersão, com teste real e corretude apropriada. Para uma mudança declarada bit-exact, exigir hash idêntico e checar os logits quando disponível. Um único token greedy igual é insuficiente. Separadamente, **encerrar o objetivo** exige mostrar que o BeeLlama alcançou o prefill da Strata na comparação completa, dentro da dispersão observada. O piso de 3% para promover um patch não define equivalência entre engines.

## 3. Pesos e protocolo

| Item | Valor fixado |
|---|---|
| Modelo | `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, revisão `2a55d75962e22f7a4a1d9963ab6eae3678537831`, diretório `Q2_0`. Shard 1: 37.623.740.192 bytes, SHA `69820c02ec7d0b45ef2ebb19d6620299db749fe2aded7f39f93c6b88b199b720`. Shard 2: 28.800.138.432 bytes, SHA `316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113`. |
| Pesos locais | Completos em `/media/hjotha/GAMES/strata-test/models/Q2_0/`. O antigo `/tmp/opencode/fn40` na .57 contém arquivos esparsos usados para cabeçalhos. Na VM o downloader verifica ambas as cargas em `/content/fn40/real/Q2_0`. |
| Strata e draft | Strata 0.2.20 `a34979fc94f3d126b9bb8210f4ad311345688b52`. DFlash2 pareado em formato GGUF para Bee e DF2P para Strata, em `/tmp/opencode/dflash-paired-24/` na .157. Respectivos SHA `a4972f7ff4375c66688c220e0dbc6e98a821f07ed1d1c4e1a607dc2a649212a4` e `c7d83c0cdfeadc7c692a1587795adb0181cfd94dacd96ac1a91f71ea7ecc62ed`. |
| Entrada | 28.547 IDs reais, SHA-256 de `json.dumps(ids)`: `a8351eb8760a2c97dbd6e28a8422bf5ee93da9eb13e337910583b54240779ef1`. Fonte `/tmp/opencode/qwen4exp-no-output-sm80/{bench.py,prompt-tokens.json}`. O hash de IDs separados por vírgulas dos traces usa outra serialização. |
| Comparação final | Temperatura 0, seed 42, sem reuso de prompt, uma sequência, contexto 32.768, KV 8-bit, 64 tokens de saída. Checar `n_prompt=28547`, reuso zero, 64 saídas e draft realmente usado. Dois warmups de prompt inteiro e pelo menos três medições. Registrar mediana, mínimo/máximo, hash e pico de VRAM. |
| Pareamento DFlash2 | Bee `--spec-type draft-dflash --spec-draft-n-max 5 --spec-draft-ubatch-size 128 --spec-draft-p-min 0 --spec-dm-controller off`. Na Strata, janela seis inclui a âncora e comporta cinco propostas. A receita histórica da Strata usa janela oito. |
| Qualidade extra | Teacher-forced nas 64 posições finais, vocabulário de 248.320 logits por posição; comparar arquivos e NLL/KL/argmax com `dense-quality/compare.py`. O hash da referência local de **quatro** camadas é `c2e6ba6657291dc03742d010bade68b5da51904498683726852f18c2d77c49d3`. Para as 48 camadas criar o controle na própria VM. |

A melhor receita **integrada** de A100 usa `-ngl 99 -c 32768 -b 4096 -ub 4096 -ncmoe 2 --moe-cache-mib 2048 --parallel 1 --fit off -fa on -ctk q8_0 -ctv q8_0 --load-mode mmap --lazy-mode on`, com os parâmetros de draft acima. Ativar `LLAMA_MOE_CACHE_PREFILL=1`, `LLAMA_MOE_MAP_DEVICE=1`, `LLAMA_MMQ_GATE_UP_GLU=1`, `LLAMA_MMQ_GATE_UP_SHARED_Q8=1`, `GGML_CUDA_MMQ_SM80_Q2_DOWN_COMPACT=1`, `GGML_CUDA_MMQ_SM80_Q2_DOWN_K_OUTER=1`, `GGML_CUDA_TOP_K_RADIX_ONE_TILE=1` e `LLAMA_DFLASH2_PREFILL_D2D=1`. A lista exata de argumentos, inclusive modelo do draft, está em `paired-df2.json` e nos drivers que sobrescrevem `-b/-ub`, n-max e draft ubatch. Os novos flags de MoE/PLE/QSA precisam dos respectivos binários experimentais.

## 4. Executar e recolher a A100

O cliente Colab roda no **host .157**. O estado da conta Pro pertence a `/home/hjotha/.collabosm/isolated-home`. Os caminhos locais de upload são resolvidos nessa máquina. A pasta do voo na .157 é `/tmp/opencode/qwen4exp-no-output-sm80`; a cópia dos scripts em desenvolvimento na .57 fica em `/tmp/opencode/strata-prefill-parity`.

```bash
ssh hjotha@192.168.1.157 'bash /tmp/opencode/qwen4exp-no-output-sm80/try-paired-a100.sh standard'
# Para a tentativa seguinte com mais RAM de host:
ssh hjotha@192.168.1.157 'bash /tmp/opencode/qwen4exp-no-output-sm80/try-paired-a100.sh high-mem'
```

Executar **uma tentativa por vez**. `try-paired-a100.sh` usa lock, verifica sessões existentes, detecta alocações tardias e chama `swa-native-paired-flight.sh`. O primeiro comando, se conseguir a GPU, já executa o trabalho. High-mem solicita RAM de host; o script identifica a GPU efetiva pelo modelo e MiB. Saída 5 significa que a tentativa falhou e a listagem subsequente não mostrou a sessão esperada. Saída 3 indica lock ocupado ou sessão existente; examinar o job atual. Uma janela de observação expirada não prova término da VM ou do benchmark.

Na .157 verificar `paired-assign-*.log`, `paired-flight-*.log` e o arquivo `swa-results-a100-{40gb,80gb}-*.tgz`. O pacote com código pré-compilado, fontes da Strata fixadas, modelos verificados e scripts evita compilar CUDA na VM. O wrapper encerra a sessão depois de recolher o arquivo. Se for interrompido, consultar `colab sessions`; acompanhar qualquer medição ativa e, ao fim, baixar os resultados e executar `colab stop -s <sessão>`. Preservar uma VM em uso quando houver outro teste pronto. Parar se for ficar ociosa aguardando compilação longa.

```bash
ssh hjotha@192.168.1.157 'ls -lt /tmp/opencode/qwen4exp-no-output-sm80/paired-flight-*.log /tmp/opencode/qwen4exp-no-output-sm80/swa-results-a100-*.tgz'
# Escolher o NOME exato do novo resultado listado acima.
scp "hjotha@192.168.1.157:/tmp/opencode/qwen4exp-no-output-sm80/$NOME" /tmp/opencode/strata-prefill-parity/
tar -tzf "/tmp/opencode/strata-prefill-parity/$NOME"
```

Os scripts `postfa-window.sh` e `stock-window.sh` dependem de os auxiliares locais estarem nos caminhos descritos na seção 5; a cápsula registra a origem de cada um.

No tar, conferir primeiro `gpu.csv`, `gpu-class.txt`, `weights.log`, os `strata-fresh-*.json`, `fair-moe-pp28k.json`, `combo-moe-pp28k.json` e os logs por braço. Os rótulos `fair` usam Strata com `--prefill 4096 --ple-io mmap` e `ckpt=0` no pedido GEN e os braços Bee `control-default`, `control-nocp`, `candidate-nocp`; os dois últimos usam `--ctx-checkpoints 0`. O controle com checkpoint preserva o custo da receita antiga. Confirmar que os logs têm zero criações nos braços sem checkpoint e que o número de amostras está completo. Comparar hashes **entre os braços do Bee** para atribuir regressões. A concordância entre os dois engines é outra medida. Examinar também `quality-*/` e o microbenchmark `native-micro.log`. `500 Compute error` mais `cudaMalloc failed` significa OOM do grafo; registrar como configuração que não coube.

**Estado do voo:** fontes e arquivos de preparação verificados localmente e na .157. O voo completo com o novo pareamento ainda aguarda a primeira alocação efetiva. O 4×16 com divergência numérica tem pacote SM80 de diagnóstico separado; o candidato principal usa a fusão MoE, token rows Q3_K, PLE e bounds.

## 5. Operar a RTX 4070 e o build

O repositório principal fica em `/home/hjotha/beellama.cpp`; o serviço de produção deve ser conferido pelo executável de `ExecStart` e pelo healthcheck, pois pode servir uma release anterior a `main`. Builds somente no gokaya, no ambiente Noble `/home/hjotha/chroots/noble`, `-j10`, no máximo dois builds CUDA simultâneos e com memória livre suficiente. Os scripts locais usam o toolchain do Noble em `/opt/cuda/bin`. Para reconstrução SM80, a configuração original foi Release, `GGML_CUDA=ON`, `GGML_CUDA_FA=ON`, `GGML_NATIVE=OFF`, `LLAMA_BUILD_TESTS=ON`, `LLAMA_CURL=OFF`, exemplos e UI desligados, `CMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined` e `CMAKE_CUDA_FLAGS=-I/opt/cuda/targets/x86_64-linux/include/cccl`. Criar a worktree fora do checkout principal e montá-la no chroot quando for usar esse método. Para reproduzir um artefato específico, o `build.sh` correspondente registra arquitetura e opções exatas. Para A100 usar `CMAKE_CUDA_ARCHITECTURES=80`; para a build recém-iniciada da 4070, SM89. A limpeza de um worktree com build ativo invalida a medição. Credenciais e chaves permanecem nas máquinas.

Na 4070, usar os scripts de janela: verificam health e contadores de pedidos, param `llama-server-root.service`, garantem exclusividade da GPU e restauram a produção com trap. Usar apenas uma janela por vez. Ao encerrar, exigir `systemctl is-active llama-server-root.service` e `curl -fsS http://127.0.0.1:8090/health` retornando `{"status":"ok"}`. Se o dono anterior terminou e o serviço permaneceu parado, iniciar com `sudo systemctl start llama-server-root.service` e verificar a saúde. A medição local usa o GGUF com camadas 0–3 extraídas do modelo, 2.048 experts, primeira PLE e primeira QSA, pesos residentes na 4070, 28.547 IDs e KV Q8. A tabela PLE real permanece no externo, em mmap. A comparação local usual gera um token e não usa draft.

```bash
# Gate release após o Ninja concluir e quando a janela anterior estiver encerrada.
d=/tmp/opencode/beellama-ada-qsa-radix-main
tail -n 5 "$d/build.log"
ls -lh "$d/build/bin/"{llama-server,test-backend-ops}
# Executar se os probes ainda não existirem.
test -x "$d/build/bin/teacher-logits" && test -x "$d/build/bin/topk-parity" || bash "$d/build-probes.sh"
sudo bash "$d/stock-window.sh"
```

`stock-window.sh` executa o teste de índices, os dois teacher-forced e quatro processos na ordem 0/1/1/0 para `GGML_CUDA_TOP_K_RADIX_TWO_TILES_ADA`. O flag de um tile fica ligado em ambos os braços. Conferir os resultados individuais em `$d/live-*/pp28k-*/result.json`, os hashes no diretório `verify/` e as bibliotecas resolvidas. Promover somente a alteração de `top-k.cu` de `exp/ada-qsa-two-tile-radix-main` se aprovado. O guard exige CC 8.9, k=512 e dois tiles de 4.096; esse ganho específico pertence à Ada.

Para reproduzir o combo experimental usar `/tmp/opencode/strata-prefill-parity/slice4-native/combo-window.sh` com **dois argumentos**, ambos `/tmp/opencode/beellama-ada-qsa-radix-deferred/staged`; por exemplo `sudo bash "$q/slice4-native/combo-window.sh" "$stage" "$stage"`, com `q=/tmp/opencode/strata-prefill-parity`. O script seleciona o core em `slice4-token-flat`, as bibliotecas de cada braço e os flags. O probe do radix possui empates e scores mascarados. No Nsight, `slice4-nsys/gpu-combo-window.sh` perfila ambos os motores. A coleta externa começa no segundo warmup e o relatório recorta apenas o terceiro pedido. Validar **7 chamadas QSA e 21 GDN por engine**; uma coleta anterior perdeu o primeiro chunk da Strata.

## 6. O que já foi aprendido

| Hardware, hipótese | Evidência | Decisão |
|---|---|---|
| A100 completa, D2D + K-outer + radix um tile | Host 2.107,68; D2D 2.444,94; combinado **2.597,40 t/s**, faixa 2.596,68–2.599,76, pico 40.368 MiB. Strata 4.932,87, faixa 4.932,02–4.933,21. | Baseline histórico. A meta exige repetir no protocolo pareado. |
| A100, por que há diferença | Trace de 11,35 s, GPU ativa 9,92 s. Gate/up Q2_0 1.703 ms, down Q2_0 869 ms, densas Q3_K 1.231 ms. Entre primeiro e último kernel: H2D 105 ms, D2H 106 ms. Cinco checkpoints de 152,619 MiB somaram 597,05 ms entre decisão e confirmação. | Medir o controle sem checkpoint; evitar extrapolar os custos PCIe da 4070. Os tempos observados não são ganhos aditivos garantidos. |
| 4070, quatro camadas e mesmo input | PLE compacta + transposição: 11.884,45 → 13.026,38 t/s. Strata: **29.278,97 t/s**. | Ensaio local útil para localizar custos. Uma em quatro camadas usa PLE no recorte, uma em 48 no completo. |
| MoE Q2_0 com aritmética Bee | Três rotas reais; 16,14–17,20 ms → 8,76–9,13 ms no microbenchmark, zero diferenças nos resultados por expert e após redução. Pesos gate/up/down em três ponteiros originais. | Wrapper experimental opt-in; acesso direto aos pesos, sem repack de 707.788.800 bytes. |
| Combo Ada final antes da QSA 4×16 | **18.262,55** controle, **19.393,20** só radix dois tiles, **20.666,67 t/s** com fusão adiada também nas camadas 1 e 3 (+13,16% sobre o controle). Faixas separadas; 64 vetores de logits idênticos. | Qualificado no recorte. Ainda experimental no modelo completo. |
| Onde sobrou tempo no combo local | Prefill do trace 1.381,30 ms Bee × 977,60 ms Strata. MoE 258,03 × 251,14 ms; cópias 35,84 × 27,55 ms; FA/QSA **170,56 × 78,97 ms**; intervalos sem atividade CUDA 90,66 × 4,54 ms. | Atenção e espera do host são alvos. O tempo de densas não pode ser comparado somando apenas nomes explícitos; há kernels genéricos sem atribuição. |
| Token rows Q3_K e bounds | Três H2D de 40 MiB por ubatch local tornaram-se uma cópia de 4.505.600 bytes + índice. Os 12 casos de texto, embeddings, mistos e LoRA em CUDA produziram o mesmo SHA `2765fd0a7affe4c99cc72f09ff69e239f8652f27ec49916414e048a3f00f6667`. Bounds substitui máscara FP16 por dois I32 por token. | Retestar na A100. Algumas rodadas locais tiveram dispersão ou um outlier grande. O primeiro helper de `mlock` fixava páginas privadas; o corrigido usa mmap somente para leitura. |
| Conversão HC compartilhada / quarta chamada QSA esparsa a 16k | +1,65% e +1,04%, respectivamente; ambos bit-exact nos 64 vetores. | Abaixo de 3%; fora da promoção. |
| Nova QSA 4×16 Ada | **19.915,42 → 20.588,58 t/s** (+3,38%, seis amostras por braço). No log, somente os chunks 3–7 mudaram; token 56553 preservado. Argmax igual em **61/64**, KL média 0,00013411, delta médio NLL +0,00012659, delta absoluto máximo de logit 0,61427. | Divergência de logits confirmada. Diagnosticar saída antes do MoE e não promover como bit-exact. |
| FP16/cuBLAS denso | A100: cerca de +6% com alteração de saída. Na 4070: 25 tensores ativados, **57/64** argmax iguais, KL média 0,00036118, delta médio NLL +0,00784. Controle repetido idêntico. | Diagnóstico numérico. Provar qualidade no modelo inteiro antes de qualquer escolha de produção. |
| Experimentos anteriores | Reuse map Q2_0: +0,70% pp28k. CPU fallback keyed IQ3_S: −1,15% prefill, −58,06% decode, hashes diferentes. Variante de tile híbrida: +1,11%. | Já medidos; não repetir sem mudança de hipótese. Fallback preservado em `feat/cpufb-verify`. |

Para a QSA 4×16, as máscaras foram capturadas nos dois engines. No último chunk, Bee reteve em média 2.049,5 células por query, e a união de oito queries cobriu 5.946,0 células. Na Strata os valores foram 2.051 e 5.941,9. A Strata tratou um token no decode; os índices são lógicos antes da paginação e a máscara de Bee é causal. Comparar as distribuições e o trabalho emitido separadamente. O novo arranjo 4×16 reduz a união e muda as partições do Stream-K. A próxima evidência é a primeira diferença da saída de FA por chunk, usando as mesmas queries, máscaras e pesos, antes da amplificação do erro na rede.

## 7. Mapa de código e de evidências

Os nomes abaixo são **locais ao gokaya**, salvo indicação de que estão em `main`. Confirmar com `git branch -a --contains <sha>`. Alguns worktrees têm modificações sem commit; o HEAD sozinho descreve apenas a base nesses casos.

| Recurso | Onde retomar |
|---|---|
| Integração básica MoE, SM80/SM89 | `exp/strata-native-q2` em `d519f6a26`. Receita de build do shim com código Strata/MIT em `tools/strata-raw-q2/README.md` **desse commit**. Inspecionar com `git show d519f6a26:tools/strata-raw-q2/README.md`. Gate estrito de geometria, placement e captura. |
| Execução direta com política normal de CUDA Graphs | `exp/strata-native-q2-graphs`, `e4c15331b`. O caminho só roda quando o backend realmente executa diretamente. A/B local +4,69%; trace por requisição em `slice4-native/1795530451`. |
| Down adiado | `exp/strata-native-q2-deferred`, `5ca5d9e8e`. Nas camadas ímpares o otimizador colocou gate/up/GLU do expert compartilhado antes do down roteado. Manter o down na posição original, após verificar dependências e não sobreposição. Foram 13 fusões adicionais por pedido. |
| Core PLE e token rows | `exp/qwen4exp-token-flat-ple`, `eacaac410`. Flags `LLAMA_PLE_CUDA_DEQUANT`, `LLAMA_PLE_CONV_SHARED_TRANSPOSE` e `LLAMA_QWEN4EXP_TOKEN_ROWS`. O core fica em `slice4-token-flat/`. Teste de transições em `slice4-native/bounds-1795531092/transition-*`. |
| Radix Ada | `5efa09ba3` (original), `4bfed157e` (cherry-pick isolado em `exp/ada-qsa-two-tile-radix-main`, promovido em `main`). O gate release usou `/home/hjotha/worktrees/beellama-ada-qsa-radix-main`; em `/tmp/opencode/beellama-ada-qsa-radix-main/` estão `build.sh`, `build-probes.sh`, `verify.sh`, `pp28k.py`, `stock-window.sh`, o log e a rodada `live-1795537039`. |
| QSA e diagnóstico de precisão | Árvores `beellama-qsa-mma-4x16`, `beellama-qsa-mma-4x16-sm80`, `beellama-qsa-selection-dump`, `beellama-qsa-fa-output-dump` e `beellama-qsa-q8-16k-bucket` sob `/home/hjotha/worktrees/`. Os diretórios homônimos em `/tmp/opencode/` contêm README, stderr de build, probes e `staged/`. No próximo teste, usar **o mesmo** backend diagnóstico para 0 e 1. O pacote SM80 de 4×16 e o dump de FA ainda não qualificaram a rota na A100. |
| Casos com aritmética densa alterada | `diag/sm80-flash-next-dense-cublas`; diagnóstico Ada em `/tmp/opencode/beellama-slice4-dense-cublas-ada/teacher-ab.sh`. A/B sobre `LLAMA_QWEN4EXP_HC_SHARED_BF16` em `/tmp/opencode/strata-prefill-parity/slice4-hc-shared-bf16/`. |
| Evidências A100 históricas | Sob `/tmp/opencode/strata-prefill-parity/`: `d2d-reviewed-results.tgz`, `prefill-d2d-reviewed-1-nsys-4096-1.{log,sqlite}`, `a100-idle-budget.{py,json}`; usar o intervalo do primeiro ao último kernel para excluir upload inicial dos pesos. |
| Modelo local e Nsight | `/media/hjotha/GAMES/flash-next-q2-layers0-3-20261124/manifest.json` contém a geometria e os hashes. `slice4-nsys/live-1795532843/{summary.json,bee-activities.csv,strata-activities.csv}` é o pareamento do combo; `live-1795530891` é a etapa anterior. |
| Ensaios 4070 | Em `slice4-native/`: `combo-1795532659` (A/B, logits), `hc-1795533315`, `qsa16-1795533718`, `qsa4x16-1795534896/{slice-ab.json,quality.json,0.log,1.log}`, `selection-1795534159/{selection.json,paired-selection.json,dense/fp16-vs-mmq.json}`. |
| Build e validação originais | `/tmp/opencode/opencode-session-transcript.md` (ler primeiro para o histórico); depois `codex-opgpu-out2.txt`, `codex-review-mmq-out.txt`, `codex-p1-out.txt` e `codex-expert-major-out.txt`, no mesmo diretório. São transcripts operacionais privados. |

**Diagnóstico de FA concluído (`postfa-1795537226`).** O backend local `/tmp/opencode/beellama-qsa-fa-output-dump/staged` amostrou o resultado F32 depois de MMA e do fixup, antes do próximo consumidor, em `/tmp/opencode/strata-prefill-parity/slice4-native/postfa-1795537226/`. Em todos os 7 chunks (0–6) de `post-fa.json`, `Q` e máscara coincidiram 17/17 (`finite_cell_symdiff = 0`). Os chunks 0–1 (`dense-f16`) ficaram idênticos (`output = 17/17`, `nmse = 0`). Os chunks 2–3 (`dense-f16` vs `sparse-q8-4x16`) e 4–6 (`sparse-q8-8x8` vs `sparse-q8-4x16`) divergiram (`output_nmse` `3,27e-6` a `5,08e-6`, `output_max_abs` até `3,80e-2` no `head=2`). No chunk 6 (`q=3971`), exatamente as 3 queries finais `[3968, 3969, 3970]` foram bit-exact (`output = 3/17`) porque formam a mesma cauda nos dois layouts. Para reprodução futura usar `LLAMA_QSA_FA_Q8_4X16=0/1`, `LLAMA_QSA_FA_Q8_WIDE_16K=0` e prefixos novos com `compare.py`:

```bash
python3 /tmp/opencode/beellama-qsa-fa-output-dump/compare.py "$out/fa0" "$out/fa1"
```

Medir latência em processo separado com o dump desligado; as cópias de observação sincronizam a GPU. Se a qualidade mudar, guardar erro por posição/head, KL, NLL e os hashes antes de decidir o destino do candidato.

A análise de fronteiras em `/tmp/opencode/beellama-qsa-fa-output-dump/boundary-audit.json` mostrou uma razão concreta para diferenças mesmo com os IDs em ordem crescente: 8,45–12,15% dos pares adjacentes de células válidas nos chunks 3–7 mudaram de relação com os limites de 32 linhas. Isso pode mudar o máximo de softmax e o reescalonamento em FP16 antes do merge. É uma hipótese de mecanismo, a ser conferida com o dump pós-FA. Uma direção exata possível é compartilhar a leitura de K/V entre subgrupos mantendo os limites originais dos tiles e as partições ativas da redução.

## 8. Disciplina de integração e pendências herdadas

1. Revalidar `git status`, worktrees, diffs e o commit do binário antes de editar. Implementação em worktree própria criada a partir de `main`. Quando uma branch deriva de outra experimental, selecionar só o patch pretendido. A flag de radix Ada tem um cherry-pick limpo justamente para isso.
2. Os bundles multicamadas combinam objetos compilados de origens identificadas. Alguns executam cubins SM80 na Ada. A correção aritmética, o hash da biblioteca, o dispositivo efetivamente detectado e as mensagens de ativação fazem parte da evidência. Conferir `SHA256SUMS`, `ldd -r`, `cuobjdump` e, depois, o teste no hardware. Não confundir link bem-sucedido com despacho real.
3. Se o teste local e a medição-alvo aprovarem, registrar comando, flags, pesos, hashes do input/output, mediana/faixa, VRAM, SHA de código e GPU. Fazer merge `--no-ff`, push em `origin main`, e remover apenas as worktrees/branches concluídas após preservar o resultado. Branches rejeitadas úteis para diagnóstico ficam identificadas como experimentais. Ao instalar em produção, validar o binário servido e `/health` separadamente.
4. Se a A100 continuar indisponível, avançar nos comparadores, correção numérica de QSA e gate vanilla Ada. Ao terminar uma janela, verificar serviço/health, sessões Colab, builds vivos e disco. Não iniciar outra alocação quando a listagem antiga ou uma falha de rede deixarem o estado da primeira incerto.
5. O backlog anterior inclui contexto de 256k, integração QSA direto à FA, testes antigos e comparação IQ3_S. Reuse map e fallback CPU já foram decididos; os três testes obsoletos foram corrigidos. `test-hybrid-rpc-prefix-state` continua desabilitado pelo contrato antigo de SWA/tail. Decode histórico Q2_0 chegou a 87,3 t/s com DFlash2 no pós-28k, e o registro anterior contém 88,0 t/s na execução com `profit`; a interpretação do controlador e a configuração atual estão no relatório de fechamento. Esses itens não substituem o gate do prefill completo.

**Leitura complementar.** O livro de medições contém os detalhes das famílias MMQ, QSA direto, cache MoE e todos os A/B. O plano original contextualiza o escopo e as decisões anteriores. O fechamento de 22/11 contém os testes e as rejeições da primeira fila. [Livro de medições](../advisor-plans/004-strata-prefill-baseline.md) · [Plano de paridade](PLAN-strata-parity.md) · [Fechamento original](agent-reports/strata-parity-closeout-20261122.md) · [Status original](agent-reports/strata-parity-status-20261122.md)
