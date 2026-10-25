# Xbox frontend

Read the repository root AGENTS.md and tools/xbox/README.md first. This is the
Xbox frontend of BeeLlama, not an independent core repository. Edit only on the
root task branch; the root beellama no-worktree exception applies.

Use the root core via the existing CMake llama target or uwp/BeeLlama.props.
Never create a nested llama.cpp checkout, add a core submodule, or patch core
sources during a build. Shared cache/protocol fixes belong in the root.

Keep the existing xllama C++/WinRT namespace, MIT notices and package identity
unless an explicit migration is requested. Signing files, models, NuGet caches,
MSIX files and build outputs stay out of Git. The LAN API remains opt-in.

Host tests do not compile UWP/D3D12 code. Validate Windows edits with MSVC on the
configured build host, and distinguish package compilation from console runtime
validation. Do not deploy or uninstall an existing Xbox app merely to test a
build. Build and deploy commands are in README.md.
