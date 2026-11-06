"""Reconstruct read-only test fixtures, never production files or Git objects."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
TREE = ROOT / ".build" / "branch"
META = json.loads((ROOT / "provenance.json").read_text())


def verify(path, expected):
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    assert actual == expected, f"fixture hash mismatch: {path.relative_to(ROOT)}"


def main():
    baseline = META["baseline_shader"]
    patch = META["saved_gqa2_branch"]
    verify(ROOT / baseline["bundled_path"], baseline["saved_source_sha256"])
    verify(ROOT / patch["included_subset"], patch["subset_sha256"])
    for fixture in META["extracted_fixtures"]:
        verify(ROOT / fixture["path"], fixture["sha256"])
    if TREE.exists():
        shutil.rmtree(TREE)
    shader = TREE / baseline["repository_path"]
    shader.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / baseline["bundled_path"], shader)
    subprocess.run(
        ["patch", "--batch", "--forward", "--fuzz=0", "-p1", "-i",
         str(ROOT / patch["included_subset"])], cwd=TREE, check=True
    )
    for fixture in META["reconstructed_fixture_files"]:
        verify(TREE / fixture["path"], fixture["sha256"])
    # Verify the executed C++ excerpts still match the actual shader fixture.
    source = shader.read_text()
    a = source.index("uint row_kind(")
    b = source.index("// Original-domain tail", a)
    assert source[a:b] == (ROOT / "fixtures/row_kind_extract.inc").read_text()
    a = source.index("    uint split_begin = split * split_rows;")
    b = source.index("    // Hoisted / pre-computed Q WHT", a)
    assert source[a:b].replace("[unroll] ", "") == (
        ROOT / "fixtures/selector_extract.inc").read_text()
    print("Fixture hashes, reconstruction and exact shader excerpts passed.")


if __name__ == "__main__":
    main()
