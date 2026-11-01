#!/usr/bin/env python3
"""Static guard: explicit and automatic native saves must journal references
through the one shared publisher, inside the same store-lock critical section."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def window(source: str, needle: str, size: int, label: str) -> str:
    index = source.find(needle)
    require(index >= 0, f"{label} is missing")
    return source[index:index + size]


def main() -> None:
    source = (ROOT / "tools/server/server-context.cpp").read_text(encoding="utf-8")

    explicit = window(source, "void save_native_stream_slot(const server_task & task", 2400,
                      "explicit native save")
    require("publish_native_stream_slot(slot, staging, task.slot_action.filepath, false,"
            in explicit,
            "the explicit native save must use the shared publisher")
    require("server_native_refs::append" not in explicit,
            "the explicit native save must not journal separately from the shared publisher")

    automatic = window(source, "void auto_save_native_stream_slot(server_slot & slot)", 4200,
                       "automatic native save")
    require("publish_native_stream_slot(slot, staging, destination, true, publication, failure)"
            in automatic,
            "the automatic native save must use the shared publisher")
    require("server_native_refs::append" not in automatic,
            "the automatic native save must not journal separately from the shared publisher")

    publisher = window(source, "bool publish_native_stream_slot(server_slot & slot", 9000,
                       "shared native publisher")
    require("const auto registry_lock_held = [&store](const std::string & path) {\n"
            "                return store.owns_exclusive_for(path);"
            in publisher,
            "the journal lock predicate must be the exclusive route-state store lock")
    require("server_native_refs::append(slot_store, registry_entry,\n"
            "                                registry_lock_held, &journal_error)"
            in publisher,
            "the pending journal record must be appended under the store lock")
    require("server_native_refs::append(slot_store, committed, registry_lock_held, &journal_error)"
            in publisher,
            "the committed journal record must be appended under the store lock")
    require("registry_entry.st        = server_native_refs::state::pending;"
            in publisher,
            "the pre-publication record must be pending")
    require("source::auto_save" in publisher and "source::explicit_save" in publisher,
            "the shared publisher must record the automatic/explicit source")

    staged = publisher.find("staged_callback = [&](const char * staged_path, size_t staged_bytes)")
    pending = publisher.find("server_native_refs::append(slot_store, registry_entry,")
    save_call = publisher.find("const size_t nwrite = server_route_state_save(")
    committed = publisher.find("server_native_refs::append(slot_store, committed,")
    require(0 <= staged < pending < save_call < committed,
            "pending must be appended by the staged callback before the save, committed after it")
    require("slot.position_split_p,\n                    staged_callback);" in publisher,
            "the staged callback must be passed to server_route_state_save")

    # Retention must treat the journal as store metadata, not as a snapshot.
    require("server_native_refs::is_journal_file(std::filesystem::path(path).filename().string())"
            in source,
            "the unified snapshot quota planner must skip the reference journal")

    refs = (ROOT / "tools/server/server-native-refs.h").read_text(encoding="utf-8")
    for needle in (
        '"3c1ac4014f51a0ff550c3e1346463623140e22f8c9a21bba80ff66d6b09fc582"',
        '"3c7218bdc40af2312e5dba397b7d2a846d3f4beb717f760353cdcc7b6f627ab1"',
        '"4d5532221ee86514043d1cabb790afb0b09e757ff11b824931b0de3bb0b24fc3"',
        '"674016a7971750acd47f65809f25c08733712390e717792ec41685224bb2378b"',
        '"68b109a4d14bbd26d7937cc9f8268a6bdb706584239531c440762857036d82c5"',
        '"ef1c05ae42b5d79406ba3cc48b15f8107049a5c07e76339b23ad9fd96b972dd7"',
        '"fff30e532caeac47750f17351664f3b392ffdafc0801514d09395c5326bee09a"',
    ):
        require(needle in refs, f"baseline pinned key is missing: {needle}")

    print("PASS: native reference journal is written by the shared publisher under the store lock")


if __name__ == "__main__":
    main()
