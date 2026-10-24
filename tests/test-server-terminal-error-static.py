#!/usr/bin/env python3
"""Every server refusal path must send exactly one terminal error before release."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    source = (ROOT / "tools/server/server-context.cpp").read_text(encoding="utf-8")

    helper_start = source.find("void fail_slot(server_slot & slot")
    require(helper_start >= 0, "fail_slot helper is missing")
    helper = source[helper_start:helper_start + 400]
    send = helper.find("send_error(slot, error, type);")
    release = helper.find("slot.release();")
    require(send >= 0, "fail_slot must send a terminal error")
    require(release > send, "fail_slot must send the terminal error BEFORE release")

    # Refusal branches introduced/fixed for the remote-failure path. Each one
    # must call fail_slot (single terminal result) rather than a bare release.
    messages = (
        "context shift: failed to remove",
        "context reuse: failed to move KV chunk",
        "failed to clear memory before cold prefill",
        "failed to trim speculative suffix after checkpoint restore",
        "failed to remove rejected draft suffix",
        "failed to restore speculative checkpoint transaction",
    )
    for message in messages:
        index = source.find(message)
        require(index >= 0, f"refusal branch not found: {message}")
        window = source[index:index + 700]
        fail = window.find("fail_slot(")
        require(fail >= 0, f"refusal branch must call fail_slot: {message}")
        require(".release()" not in window[:fail],
                f"bare release before fail_slot on refusal branch: {message}")

    require("fail_slot(*child," in source,
            "child prompt-copy refusal must send the child its own terminal error")


if __name__ == "__main__":
    main()
