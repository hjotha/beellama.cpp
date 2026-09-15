# Plano de Implementação: Phase-Aware NVIDIA VRAM / Memory Clock Governor

**Data:** 2026-09-07  
**Autor:** Antigravity (Gemini) para hjotha  
**Alvo:** `llama-server` (fork `/home/hjotha/llama`) na máquina de produção `.57` (`gokaya` / RTX 4070 12GB eGPU) e GPUs NVIDIA em geral  
**Status:** Proposta de Arquitetura & Plano Técnico para Revisão  

---

## 1. Sumário Executivo & Motivação

### 1.1 A Assimetria Física da Inferência de LLMs
A inferência de modelos de linguagem opera em dois regimes computacionais completamente distintos:

1. **Prefill (Processamento de Prompt):**
   - **Gargalo:** *Compute-bound* (GEMM massivo).
   - O hardware satura as SMs (*Streaming Multiprocessors*) e os Tensor Cores calculando a atenção e projeções para todos os tokens de entrada em lote ($B \times N$).
   - Demanda alta potência (*Power Limit* de 200W na RTX 4070) para que o clock da GPU atinja 2600–2700 MHz.
   - O clock da memória é relevante, mas o gargalo primário é a taxa de cálculo da GPU (TFLOPs).

2. **Decode (Geração de Tokens):**
   - **Gargalo:** *Memory-Bandwidth-bound* puro ($O(1)$ token por passo para batch pequeno/unitário).
   - A intensidade aritmética é de aproximadamente $1\text{ FLOP}/\text{byte}$. Cada token gerado exige transferir todos os parâmetros ativos do modelo e o KV cache da VRAM para a cache/registradores das SMs.
   - As SMs passam grande parte do tempo ociosas aguardando a chegada dos dados da VRAM. Elevar o clock da GPU acima de 1800–2000 MHz não traz ganhos mensuráveis de tokens por segundo (TPS), mas consome calor e energia (por isso 165W no Decode economiza 35W com menos de 1% de perda de TPS).
   - **Contudo, a velocidade da memória (VRAM Bandwidth) dita linearmente a taxa de Decode.** Na RTX 4070 (barramento 192-bit @ 10501 MHz = 21 Gbps GDDR6X), a largura de banda máxima teórica é de **$504\text{ GB/s}$**.

### 1.2 O Problema do Comportamento Padrão do Driver NVIDIA
- **Oscilação e Throttling no Decode:** Quando o *power limit* é reduzido para 165W durante o Decode, o algoritmo de gerenciamento de energia dinâmico da NVIDIA pode manter ou derrubar o clock da VRAM para degraus intermediários (como 10251 MHz ou 5001 MHz) para acomodar picos transitórios na SM, gerando quedas de TPS.
- **Desperdício Térmico e Energético no IDLE:** Se o clock da memória for travado estaticamente em 10501 MHz fora do servidor (ex: via script externo ou watchdog), a GPU não consegue descer para o estado de baixo consumo P8 (405 MHz). O consumo de repouso salta de **$4.7\text{ W}$** para **$14.1\text{ W}$** (+200% de calor basal desnecessário nos módulos GDDR6X).

### 1.3 Objetivo da Nova Feature
Estender o módulo existente `server-gpu-power.{h,cpp}` para criar um **Governor Unificado de Energia e Clock de Memória por Fase**:
- Travar a memória no degrau máximo (**10501 MHz** na RTX 4070) **exclusivamente durante a fase de `DECODE`**, garantindo 100% da largura de banda teórica sem flutuações.
- Permitir configuração opcional de clock para a fase de `PREFILL` (ou deixar o driver gerenciar dinamicamente).
- **Resetar automaticamente o clock de memória no `IDLE`** (`nvmlDeviceResetMemoryLockedClocks`), permitindo que a VRAM desça imediatamente para 405 MHz e o consumo caia para menos de 5W quando o servidor estiver ocioso.
- Manter restauração segura e limpa no shutdown, unload de modelo e transição de sleep.

---

## 2. Investigação Técnica no Hardware Real (.57 / RTX 4070)

Uma checagem direta na máquina de produção `.57` confirmou a viabilidade e identificou os seguintes fatos concretos:

1. **Suporte a Locked Clocks em Runtime:**
   ```text
   $ nvidia-smi -lmci
   Memory Clock Switching Type: Runtime
   ```
2. **Degraus Discretos de Memória Suportados na RTX 4070:**
   - `10501 MHz` (Boost Máximo / P0 / 21 Gbps / ~504 GB/s)
   - `10251 MHz` (P2)
   - `5001 MHz` (P5)
   - `810 MHz`
   - `405 MHz` (Idle P8 / Baixo Consumo)
3. **Consumo Medido em Repouso:**
   - Com memória travada a 10501 MHz: **$14.10\text{ W}$**
   - Com memória resetada (405 MHz): **$4.70\text{ W}$**
   *(Diferença de quase 10W apenas no controlador de memória e chips GDDR6X)*.
4. **Símbolos NVML Disponíveis no Driver (libnvidia-ml.so.1):**
   - `nvmlDeviceSetMemoryLockedClocks(nvmlDevice_t, unsigned int minMemClockMHz, unsigned int maxMemClockMHz)`
   - `nvmlDeviceResetMemoryLockedClocks(nvmlDevice_t)`
   - `nvmlDeviceGetSupportedMemoryClocks(nvmlDevice_t, unsigned int *count, unsigned int *clocksMHz)`
   *Nota da documentação oficial da NVIDIA (`nvml.h`):* O uso de `nvmlDeviceSetApplicationsClocks` é obsoleto (deprecated) e será removido no CUDA 14.0. A NVIDIA recomenda expressamente o uso de `nvmlDeviceSetMemoryLockedClocks`.

---

## 3. Arquitetura Proposta e Integração no LLaMA

A feature será integrada de forma elegante e coesa no código existente, aproveitando a estrutura já testada e validada no commit `afa219d99`.

### 3.1 Nomenclatura e Interface de CLI / ENV
Serão adicionadas flags específicas e opcionais para o controle de clock de memória:

| Argumento CLI | Variável de Ambiente | Descrição | Default |
| :--- | :--- | :--- | :--- |
| `--gpu-mem-clock-decode <MHz>` | `LLAMA_ARG_GPU_MEM_CLOCK_DECODE` | Clock de memória travado (MHz) durante geração de tokens (*Decode*). Ex: `10501`. | `-1` (desabilitado) |
| `--gpu-mem-clock-prefill <MHz>` | `LLAMA_ARG_GPU_MEM_CLOCK_PREFILL` | Clock de memória travado (MHz) durante processamento de prompt (*Prefill*). Ex: `10251` ou `10501`. | `-1` (dinâmico pelo driver) |
| `--gpu-mem-clock-device <N>` | `LLAMA_ARG_GPU_MEM_CLOCK_DEVICE` | Índice do dispositivo NVML (reutiliza `--gpu-power-device` se não especificado). | `0` |

*Compatibilidade:* O usuário pode configurar:
1. Apenas potência (`--gpu-power-prefill 200 --gpu-power-decode 165`);
2. Apenas memória (`--gpu-mem-clock-decode 10501`);
3. Ambos em conjunto (governor unificado completo).

### 3.2 Estruturas de Dados (`server-gpu-power.h`)
```cpp
struct server_gpu_power_config {
    int32_t prefill_w          = -1;
    int32_t decode_w           = -1;
    int32_t mem_clock_prefill  = -1; // em MHz (opcional)
    int32_t mem_clock_decode   = -1; // em MHz (ex: 10501)
    int32_t device             = 0;

    bool power_enabled() const { return prefill_w != -1 && decode_w != -1; }
    bool mem_clock_enabled() const { return mem_clock_decode > 0 || mem_clock_prefill > 0; }
    bool enabled() const { return power_enabled() || mem_clock_enabled(); }
};

struct server_gpu_device_info {
    std::string           name;
    int32_t               device                  = 0;
    uint32_t              original_power_limit_mw = 0;
    uint32_t              min_power_limit_mw      = 0;
    uint32_t              max_power_limit_mw      = 0;
    std::vector<uint32_t> supported_mem_clocks_mhz;
};
```

### 3.3 Extensão da Interface do Backend (`server_gpu_power_backend`)
```cpp
class server_gpu_power_backend {
  public:
    virtual ~server_gpu_power_backend() = default;

    virtual bool init(int32_t device, server_gpu_device_info & info, std::string & error) = 0;
    virtual bool set_power_limit(uint32_t power_limit_mw, std::string & error)                  = 0;
    virtual bool set_memory_locked_clocks(uint32_t min_mhz, uint32_t max_mhz, std::string & error) = 0;
    virtual bool reset_memory_locked_clocks(std::string & error)                                = 0;
    virtual void shutdown()                                                                     = 0;
};
```

### 3.4 Resolução Dinâmica NVML (`server-gpu-power.cpp`)
Adicionar ponteiros de função para os novos símbolos em runtime (via `dlsym` no Linux e `GetProcAddress` no Windows), garantindo que compilações sem CUDA SDK / NVML continuem funcionando perfeitamente:
```cpp
using nvml_device_set_memory_locked_clocks_t = nvml_return_t (*)(nvml_device_t, unsigned int, unsigned int);
using nvml_device_reset_memory_locked_clocks_t = nvml_return_t (*)(nvml_device_t);
using nvml_device_get_supported_memory_clocks_t = nvml_return_t (*)(nvml_device_t, unsigned int *, unsigned int *);
```

### 3.5 Tabela de Transições de Fase e Ações do Governor

| Transição de Fase | Power Limit | Memory Clock Action | Racional de Hardware |
| :--- | :--- | :--- | :--- |
| **`* -> PREFILL`** | Aplica `prefill_w` (200W) | Se `mem_clock_prefill > 0`, trava no valor; senão, chama `reset_memory_locked_clocks()` | Permite que a GPU canalize o budget térmico/elétrico para as SMs durante GEMM sem travar a memória se não solicitado. |
| **`* -> DECODE`** | Aplica `decode_w` (165W) | Trava no `mem_clock_decode` (`nvmlDeviceSetMemoryLockedClocks(10501, 10501)`) | Garante 504 GB/s ininterruptos de largura de banda para o streaming do modelo e KV cache. |
| **`* -> IDLE`** | Mantém (driver controla repouso) | Chama `reset_memory_locked_clocks()` | Permite que a GPU desça a VRAM para 405 MHz (P8), economizando ~10W de consumo basal. |
| **`Sleep / Shutdown`** | Restaura `original_power_limit_mw` | Chama `reset_memory_locked_clocks()` | Deixa a máquina em estado 100% limpo e padrão após parada do serviço. |

### 3.6 Deduplicação e Zero Overhead no Loop Quente
- O método `update(phase)` compara a fase atual e os alvos (`target_power` e `target_mem_clock`) com os últimos valores efetivamente aplicados (`last_applied_mem_clock_`).
- Em chamadas sucessivas na mesma fase (ex: geração de 4096 tokens em sequência, onde cada token passa por `update_slots()`), **nenhuma chamada NVML é feita**.
- Custo por passo de decode: uma checagem simples de inteiros (`phase == last_phase`). Overhead: **< 10 nanossegundos**.

---

## 4. Plano de Testes e Validação

### 4.1 Testes Unitários (`tests/test-server-gpu-power.cpp`)
Expandir o mock `fake_gpu_power_backend` para registrar as chamadas de memória:
- `applied_mem_clocks: std::vector<std::pair<uint32_t, uint32_t>>`
- `reset_mem_calls: int`
- Cenários a cobrir:
  1. *Apenas Decode Configurado:* Verificar que `PREFILL` reseta a memória e `DECODE` trava em 10501 MHz.
  2. *Retorno a IDLE:* Verificar que a transição para `IDLE` dispara `reset_memory_locked_clocks()`.
  3. *Deduplicação:* Chamadas repetidas de `DECODE` não duplicam escritas no backend.
  4. *Validação de Frequência Inválida:* Tentar inicializar com frequência inexistente (ex: 99999 MHz) deve falhar na validação de inicialização.
  5. *Tolerância a Falhas:* Se `nvmlDeviceSetMemoryLockedClocks` retornar erro (ex: falta de permissão), desabilitar os writes de memória com warning nos logs sem derrubar o servidor.

### 4.2 Testes do Argument Parser (`tests/test-arg-parser.cpp`)
- Testar flags `--gpu-mem-clock-decode`, `--gpu-mem-clock-prefill`.
- Testar variáveis de ambiente `LLAMA_ARG_GPU_MEM_CLOCK_DECODE`.
- Rejeição de valores negativos ou zerados.

### 4.3 Validação no Hardware Real (.57 / RTX 4070)
1. **Benchmark de Latência de Troca:**
   - Medir o tempo de chaveamento de clock do driver NVIDIA durante a transição Prefill $\rightarrow$ Decode para garantir que não introduz estalo ou stall na inferência (estimado em < 2 ms).
2. **Benchmark A/B de TPS:**
   - Contexto de 50k tokens com 1024 de saída (Qwen3.8-27B).
   - *Cenário A:* Padrão atual (165W decode sem memory lock).
   - *Cenário B:* 165W decode com `--gpu-mem-clock-decode 10501`.
   - Comparar TPS médio, desvio padrão e estabilidade da geração.
3. **Monitoramento Térmico e de Potência:**
   - Validar com `nvidia-smi` que o repouso após o request retorna para 405 MHz e ~5W de consumo.

---

## 5. Roteiro de Implementação (Passo a Passo para Execução)

Quando este plano for revisado e aprovado, a execução seguirá as regras rígidas do projeto:

1. **Isolamento em Worktree:**
   ```bash
   git worktree add /home/hjotha/worktrees/llama-gpu-mem-governor -b feat/gpu-memory-clock-governor
   ```
2. **Implementação do Código:**
   - `tools/server/server-gpu-power.h` e `server-gpu-power.cpp` (estruturas, backend, NVML bindings).
   - `common/common.h` e `common/arg.cpp` (flags CLI e env vars).
   - `tools/server/server-context.cpp` (passagem de parâmetros na inicialização).
3. **Testes Automatizados:**
   - Atualização e execução de `tests/test-server-gpu-power.cpp` e `tests/test-arg-parser.cpp`.
4. **Build e Teste Real:**
   - Compilação com CMake e Ninja na worktree.
   - Sonda com o binário na `.57` via `llama-server-root.service`.
5. **Merge e Cleanup:**
   - Merge automático para `master`, push para origin e remoção da worktree.
