# Rodada interrompida — não usar como benchmark

O modo sync-only chegou ao aquecimento com launched=0/errors=0, mas a primeira
requisição medida ficou sem resposta após o servidor de teste entrar em shutdown.
Os serviços de produção foram observados ativos antes do finally do runner;
artefatos de um benchmark concorrente apareceram no checkout. A rodada foi
interrompida e excluída das comparações.

SIGINT também atingiu o controlador de CPU da versão antiga do runner: ele
executou sua restauração no finally, mas o processo principal registrou broken
pipe ao pedir confirmação. Uma auditoria de leitura comparou as 16 políticas
reais ao snapshot inicial e confirmou igualdade (`restoration-audit.json`).
A produção foi restaurada. O runner revisado usa porta 59589, lease de benchmark,
processos auxiliares em sessões próprias e verifica produção/CPU por requisição.
