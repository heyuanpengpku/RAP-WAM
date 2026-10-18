"""Check source dependencies, syntax, private paths and the optional release manifest."""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess


def release_files(root):
    for p in sorted(root.rglob('*')):
        parts = p.relative_to(root).parts
        if parts[0] in {'.venv', 'venv', 'assets', 'data', 'outputs', '.cache'}:
            continue
        if any(x in ['.git', '__pycache__', 'build', 'dist', '.pytest_cache'] or x.endswith('.egg-info') for x in parts):
            continue
        if p.name == '.env' or p.name.startswith('.env.') or p.suffix in {'.pt', '.pth', '.ckpt', '.npz', '.safetensors'}:
            continue
        if p.is_file() and p.name != '.DS_Store':
            yield p


def check(root):
    root = Path(root).resolve()
    if not (root / 'pyproject.toml').exists():
        raise ValueError('Run from the release repository, or pass --root')
    errors = []
    declared = re.search(r'^name\s*=\s*"([^"]+)"', (root / 'pyproject.toml').read_text(), re.M)
    if not declared or declared.group(1) != 'rap_wam':
        errors.append('Project metadata must use the public package name')
    files = list(release_files(root))
    media_file = root / 'provenance/media.json'
    media = {row['path']: row['sha256'] for row in json.loads(media_file.read_text())} if media_file.exists() else {}
    for path in root.rglob('*'):
        if path.relative_to(root).parts[0] not in {'.venv', 'venv', 'assets', 'data', 'outputs', '.cache', '.git'} and path.is_symlink():
            errors.append('Symbolic links cannot be published: ' + str(path.relative_to(root)))
    code_roots = [root, root / 'third_party/wan_official']
    internal = {'rap_wam', 'wan'}
    counts = {'files': len(files), 'python': 0, 'shell': 0}
    private_prefixes = ['/' + v + '/' for v in ['share_data', 'efs-exp', 'Users', 'home', 'root', 'mnt']]
    for p in files:
        rel = str(p.relative_to(root))
        if not rel.isascii():
            errors.append('Non-ASCII filename: ' + rel)
        try:
            text = p.read_text()
        except UnicodeError:
            if media.get(rel) != hashlib.sha256(p.read_bytes()).hexdigest():
                errors.append('Unexpected binary file: ' + rel)
            continue
        if re.search(r'[\u3400-\u9fff\uf900-\ufaff\U00020000-\U000323af]', text):
            errors.append('Non-English text: ' + rel)
        if any(prefix in text for prefix in private_prefixes):
            errors.append('Nonportable private path: ' + rel)
        if re.search(r'-----BEGIN (?:RSA |OPENSSH |EC |ENCRYPTED )?PRIVATE KEY-----', text):
            errors.append('Private key: ' + rel)
        if re.search(r'(?:glpat-[A-Za-z0-9_-]{15,}|ghp_[A-Za-z0-9]{20,})', text):
            errors.append('Access-token pattern: ' + rel)
        if re.search(r'https?://[^\s/@:]+:[^\s/@]+@', text):
            errors.append('Credential-bearing URL: ' + rel)
        legacy = re.compile(r'(?i)(?<![a-z0-9])' + 'h' + 'yp' + r'(?![a-z0-9])|' + 'v' + '58')
        if legacy.search(rel) or legacy.search(text):
            errors.append('Legacy project identifier: ' + rel)
        if p.suffix == '.py':
            counts['python'] += 1
            try:
                tree = ast.parse(text, filename=rel)
            except SyntaxError as e:
                errors.append(str(e)); continue
            for node in ast.walk(tree):
                mods = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module] if isinstance(node, ast.ImportFrom) and node.module and not node.level else []
                for mod in mods:
                    if mod.split('.')[0] in internal:
                        candidates = [r / mod.replace('.', '/') for r in code_roots]
                        if not any(c.is_dir() or c.with_suffix('.py').is_file() for c in candidates):
                            errors.append(f'{rel}:{node.lineno}: missing internal module {mod}')
                if isinstance(node, ast.ImportFrom) and node.level and node.module:
                    parent = p.parent
                    for _ in range(node.level - 1):parent = parent.parent
                    target = parent.joinpath(*node.module.split('.'))
                    if not target.is_dir() and not target.with_suffix('.py').is_file():
                        errors.append(f'{rel}:{node.lineno}: missing relative module')
        elif p.suffix == '.sh':
            counts['shell'] += 1
            x = subprocess.run(['bash', '-n', str(p)], capture_output=True, text=True)
            if x.returncode:errors.append(x.stderr)
    manifest = root / 'provenance/release_manifest.json'
    if manifest.exists():
        expected = json.loads(manifest.read_text())
        actual = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in files if p != manifest}
        if expected != actual:
            errors.append('Release manifest does not match source files')
    return {'counts': counts, 'errors': sorted(set(errors))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='.')
    args = parser.parse_args()
    result = check(args.root)
    print(json.dumps(result, indent=2))
    raise SystemExit(bool(result['errors']))


if __name__ == '__main__':
    main()
