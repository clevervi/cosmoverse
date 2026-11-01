import json
import pathlib
import sys

try:
    import yaml
except ImportError:
    yaml = None

ROOT = pathlib.Path(__file__).resolve().parent.parent

REQUIRED = [
    "README.md",
    "LICENSE",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "CODE_OF_CONDUCT.md",
    "CHANGELOG.md",
    "AGENTS.md",
    "CLAUDE.md",
    "docs/MASTER_DOCUMENT.md",
]

missing = [path for path in REQUIRED if not (ROOT / path).is_file()]
if missing:
    print("Missing required files:", ", ".join(missing), file=sys.stderr)
    sys.exit(1)

if yaml is None:
    print("PyYAML not installed; skipping YAML validation.")
else:
    yaml_files = sorted(list((ROOT / ".github").rglob("*.yml")) + list((ROOT / "configs").rglob("*.yaml")))
    for path in yaml_files:
        yaml.safe_load(path.read_text(encoding="utf-8"))
    print(f"YAML OK: {len(yaml_files)} files")

json_files = sorted(list((ROOT / ".github").rglob("*.json")) + list((ROOT / "notebooks").rglob("*.ipynb")))
for path in json_files:
    json.loads(path.read_text(encoding="utf-8"))
print(f"JSON OK: {len(json_files)} files")
