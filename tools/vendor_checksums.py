"""Recompute checksums after an intentional vendor update; no network or builds."""
import json
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from testpacks.prepare import tree_hash

path = ROOT / "vendor/sources.json"
sources = json.loads(path.read_text(encoding="utf-8"))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--check", action="store_true", help="Verify without changing the manifest")
args = parser.parse_args()
mismatches = []
for name, metadata in sources.items():
    actual = tree_hash(ROOT / "vendor" / name)
    if metadata.get("tree_sha256") != actual:
        mismatches.append(name)
    metadata["tree_sha256"] = actual
if args.check:
    if mismatches:
        raise SystemExit("Bundled source mismatch: " + ", ".join(mismatches))
    print("All %d bundled source trees verified" % len(sources))
else:
    path.write_text(json.dumps(sources, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
