# MQ capture: isolated MSVC object compilation

- Base: `582cc8d8e70216e0ba697e735fbe3c6817704f9a`, dedicated worktree
  `research/rkva-gqa2-capture-20261030`.
- Windows machine: `.193`, isolated directory
  `C:\Users\hjotha\build\gqa2-capture-20261030`. The local `compile.bat`
  imports the VS2022 x64 developer environment, compiles both changed UWP
  translation units with C++17, AVX2 and `WINAPI_FAMILY_APP`, and exited 0.
  It uses no shared build objects, task, signing or package. This is not a
  solution build, link, package, deployment or Xbox validation.
- Original generated DXIL-header file SHA-256, identical locally and on Windows:
  - MQ2: `7119a24ed39ef605a1b47a63157daadc5a45a9e2da8a60cc10b1fe86f4b82857`
  - GQA2: `83ec36606535d716e880dc5bef39a615a60421e5afb41c18efcdd61a6edef71c`
- Final modified inputs, identical locally and on Windows:
  - `remote-attn-engine.cpp`: `c09ebda8697a91ffca78a05947dfbb789de5e57c43c880ebccff1a343b25ee63`
  - `remote-attn-engine.h`: `afb7840cd896d096eddd7815984c25588b545c9f02257bc08ffb7790356d368c`
  - `rkva-kvarn-bulk-f16.inc`: `63b33dcba4a94af0e286b04e530dc34e7132d40b722769a8244570f814fbcb73`
- Final object SHA-256:
  - `engine.obj`: `77dd5a363443bfc441e10d5a063e5706d967406196e05d6af868e471931f8c2d`
  - `server.obj`: `0121b01a39c2e5de1e090fcdd910b35073ff311c2ca8ce3aa15846ac96f708c8`

The project can plausibly be linked in an isolated full source copy with NuGet
dependencies restored locally and `AppxPackageSigningEnabled=false`, but that
has not been tried. Do not use `build-uwp.ps1` for an unsigned build: it also
manages certificates and calls the signing finalizer.
