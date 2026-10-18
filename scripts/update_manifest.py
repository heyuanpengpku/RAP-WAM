"""Regenerate release file hashes; this does not run checks or certify validation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rap_wam.check_release import release_files


def main():
    target = ROOT / 'provenance/release_manifest.json'
    manifest = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in release_files(ROOT) if path != target}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Recorded {len(manifest)} source file hashes. Validation records are unchanged.')


if __name__ == '__main__':
    main()
