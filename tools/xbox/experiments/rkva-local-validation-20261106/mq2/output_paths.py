"""Generated outputs stay inside this portable package."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BUILD = ROOT / ".build"
RESULTS = BUILD / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
