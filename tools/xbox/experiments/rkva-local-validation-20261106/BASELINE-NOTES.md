# Baseline adicional: resultado histórico, não nova qualificação

Na campanha CPU de 2026-11-06, o checkout anterior
`ff008e07eea8e471ebe247ad3ef5dfce35b828bf` passou os modelos existentes de
partição MQ2/MQ4, split e testes de encadeamento de perfil.

Três compilações C++ com `-Wall -Wextra -Werror` falharam inicialmente por
`head_dim` não usado em um helper SAW inalterado. Mantendo o aviso e usando
somente `-Wno-error=unused-parameter`, os testes funcionais de bulk/FP8,
perfil e cobertura/ring/rewind passaram.

Isso **não é um passe de build estrito**, nem valida novos caminhos E4 do
snapshot mais recente. Estes resultados não substituem os testes próprios dos
candidatos e não foram usados para afrouxar seus gates. O baseline antigo não é
duplicado neste pacote; os candidatos incluem as fixtures necessárias às suas
reproduções portáveis.
