# BeeLlama INI presets

Presets store reusable llama.cpp arguments in an INI file. Use them with the
router server when several models need different paths, cache policies, or
speculative settings:

```powershell
llama-server --models-preset .\models.ini
```

The exact preset argument and preset-only defaults are listed in the
[BeeLlama argument reference](beellama-args.md#presets).

## Using Presets with the Server

## File format

- Write option names without leading dashes: `ctx-size`, not `--ctx-size`.
- Put shared values in `[*]`.
- Put each named model in its own section.
- Later command-line arguments override values loaded from the selected preset.
- `load-on-startup` and `stop-timeout` are preset-only keys; they are not CLI
  arguments.

```ini
[*]
mmap       = 1
kv-unified = 1
parallel   = 1

[Qwen-DFlash-KVarN]
model                  = D:/models/qwen.gguf
spec-type              = draft-dflash
spec-draft-model       = D:/models/qwen-dflash.gguf
spec-dm-controller     = profit
cache-type-k            = kvarn4
cache-type-v            = kvarn4
flash-attn              = on
reasoning-loop-guard    = force-close
load-on-startup         = 1
stop-timeout            = 10
```

This example omits `spec-draft-n-max`, so DFlash uses
`dflash.block_size - 1` and the default-on profit controller adapts within that
limit. Add `spec-draft-n-max = N` when a fixed upper bound is required; add
`spec-dm-controller = off` when the resolved or explicit depth must remain
static.

## Router model selection

The section name is the router model name. A shared section can define ordinary
server defaults, while each model section supplies its own model source and
overrides:

```ini
[*]
ctx-size    = 32768
batch-size  = 1024
ubatch-size = 512

[qwen-local]
model        = D:/models/qwen.gguf
cache-type-k = kvarn4
cache-type-v = kvarn4

[gpt-oss-hf]
hf          = ggml-org/gpt-oss-20b-GGUF
temp        = 1.0
top-p       = 1.0
top-k       = 0
```

`load-on-startup = 1` autoloads a section. The total number of startup sections
must not exceed `--models-max` unless that upstream limit is configured as
unlimited.

## Remote presets

A Hugging Face preset repository contains `preset.ini` at its root and points
to the actual model repositories from its named sections. Load it through the
normal `-hf` flow:

```powershell
llama-server -hf user/preset-repository
```

Remote presets can select models and server options. Use only repositories you
trust, and inspect `preset.ini` before deployment.

Do not place `hf-token` in a preset. It is a sensitive option and is omitted
from preset serialization and public router metadata. Supply `HF_TOKEN` in the
router environment (or `--hf-token` at startup); child model processes receive
only the environment value, never a token-bearing argv entry.

## BeeLlama migration rules

New presets must use `draft-dflash`, standard q cache names, or `kvarnN` target
cache names. Do not carry forward TurboQuant/TCQ formats,
`spec-dflash-cross-ctx`, tree-verifier settings, `GGML_DFLASH_*` variables, or
`GGML_CUDA_FA_HALF_QUANTS`. The complete redirect and removal list is in
[Migration from earlier versions](beellama-args.md#migration-from-earlier-versions).

Please make sure to provide the correct `hf-repo` for each child preset. Otherwise, you may get error: `The specified tag is not a valid quantization scheme.`

## System-level config

The system-level config, added in PR [#26118](https://github.com/ggml-org/llama.cpp/pull/26118), allows sharing the same set of options among multiple tools and examples. Unlike the sections above, it is not limited to the server.

These files are loaded on startup if present. A later file overrides an earlier one:
1. System-wide: `/etc/llama.cpp/config.ini` (or `%PROGRAMDATA%\llama.cpp\config.ini` on Windows)
2. User-level: `$XDG_CONFIG_HOME/llama.cpp/config.ini`, `~/.config/llama.cpp/config.ini` by default (or `%APPDATA%\llama.cpp\config.ini` on Windows)

The config file is applied first, then its options are overridden by ENV variables, CLI arguments and model presets (in router mode).

Note:
- Only the `[*]` and default sections are used; options written before any section header belong to "default. Named sections are ignored
- Tool-specific options can be specified, but will be ignored (with a warning) if the example doesn't support it<br/>Example: if you specify `port = 1234`, only `llama-server` will use it, other examples will ignore it
- `model` or `hf-repo` are not recommended to be configured system-level, because it may introduce conflicts<br/>Example: a `hf-repo` in the config file still takes effect when you pass `-m` on the command line, so you may load a different model than expected
