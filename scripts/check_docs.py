"""Check local documentation references and code-block syntax without dependencies."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote, urlsplit


FENCES = re.compile(r"^```([^\n]*)\n(.*?)^```[ \t]*$", re.M | re.S)
LINKS = re.compile(r"!?\[[^\]]*\]\(([^\s)]+)(?:\s+\"[^\"]*\")?\)")


class HTMLReferences(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.anchors = set()
        self.images_without_alt = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.anchors.add(attrs["id"])
        for key in ("href", "src"):
            if key in attrs:
                self.links.append(attrs[key])
        if tag == "img" and not attrs.get("alt", "").strip():
            self.images_without_alt += 1


def anchors(text):
    counts = Counter()
    result = set()
    for title in re.findall(r"^#{1,6}\s+(.+?)\s*#*\s*$", text, re.M):
        # GitHub-style slugs for the plain-English headings used by this repo.
        slug = re.sub(r"[^\w\- ]", "", title.lower()).replace(" ", "-")
        suffix = "" if not counts[slug] else "-" + str(counts[slug])
        result.add(slug + suffix)
        counts[slug] += 1
    parser = HTMLReferences()
    parser.feed(text)
    return result | parser.anchors


def check(root):
    root = Path(root).resolve()
    documents = sorted(p for p in root.rglob("*.md") if not any(
        part in {".git", ".venv", "build", "dist", "__pycache__"}
        or part.endswith(".egg-info") for part in p.relative_to(root).parts)
        and p.relative_to(root).parts[0] not in {"assets", "data", "outputs", ".cache", "venv"})
    errors = []
    counts = {"documents": len(documents), "local_links": 0,
              "shell_blocks": 0, "python_blocks": 0}
    for document in documents:
        name = str(document.relative_to(root))
        text = document.read_text()
        blocks = list(FENCES.finditer(text))
        if len(re.findall(r"^```", text, re.M)) != 2 * len(blocks):
            errors.append(f"{name}: unmatched code fence")
        for block in blocks:
            language, body = block.groups()
            if language.strip() in {"bash", "sh", "shell"}:
                counts["shell_blocks"] += 1
                result = subprocess.run(["bash", "-n"], input=body,
                                        capture_output=True, text=True)
                if result.returncode:
                    errors.append(f"{name}: invalid shell block: {result.stderr.strip()}")
            elif language.strip() == "python":
                counts["python_blocks"] += 1
                try:
                    ast.parse(body)
                except SyntaxError as error:
                    errors.append(f"{name}: invalid Python block: {error.msg}")
        prose = FENCES.sub("", text)
        levels = [len(mark) for mark in re.findall(r"^(#{1,6}) ", prose, re.M)]
        if any(current > previous + 1 for previous, current in zip(levels, levels[1:])):
            errors.append(f"{name}: heading hierarchy skips a level")
        parser = HTMLReferences()
        parser.feed(prose)
        if parser.images_without_alt:
            errors.append(f"{name}: HTML image without a nonempty alt description")
        references = LINKS.findall(prose) + parser.links
        for reference in references:
            parsed = urlsplit(reference)
            if parsed.scheme or parsed.netloc:
                continue
            counts["local_links"] += 1
            target = (document.parent / unquote(parsed.path)).resolve() if parsed.path else document
            try:
                target.relative_to(root)
            except ValueError:
                errors.append(f"{name}: link escapes repository: {reference}")
                continue
            if not target.exists():
                errors.append(f"{name}: missing local target: {reference}")
            elif parsed.fragment and target.is_file() and target.suffix == ".md":
                if unquote(parsed.fragment) not in anchors(target.read_text()):
                    errors.append(f"{name}: missing heading/anchor: {reference}")
    readme = (root / "README.md").read_text()
    if re.findall(r"^# (.+)$", FENCES.sub("", readme), re.M) != ["RAP-WAM"]:
        errors.append("README.md: project heading must use the display name RAP-WAM")
    expected_titles = {
        "docs/README.md": "RAP-WAM Documentation",
        "CONTRIBUTING.md": "Contributing to RAP-WAM",
        "CHANGELOG.md": "RAP-WAM Changelog",
        "examples/README.md": "RAP-WAM Examples",
    }
    for relative, expected in expected_titles.items():
        path = root / relative
        if path.is_file() and not re.search(r"^# " + re.escape(expected) + r"$", path.read_text(), re.M):
            errors.append(f"{relative}: project heading must use {expected}")
    citation = root / "CITATION.cff"
    if citation.is_file() and not re.search(r'^title: [\"\']?RAP-WAM[\"\']?$', citation.read_text(), re.M):
        errors.append("CITATION.cff: software title must use RAP-WAM")
    metadata = root / "pyproject.toml"
    if metadata.is_file():
        text = metadata.read_text()
        if not re.search(r'^name = "rap_wam"$', text, re.M):
            errors.append("pyproject.toml: distribution name must remain rap_wam")
        if not re.search(r'^description = "RAP-WAM:', text, re.M):
            errors.append("pyproject.toml: description must use the RAP-WAM display name")
    citations = [body.strip() for language, body in FENCES.findall(readme)
                 if language.strip() == "bibtex"]
    if citations != [(root / "citation.bib").read_text().strip()]:
        errors.append("README.md: BibTeX must match citation.bib exactly")
    for example in sorted((root / "examples").glob("*.env.example")):
        result = subprocess.run(["bash", "-n", str(example)], capture_output=True, text=True)
        if result.returncode:
            errors.append(f"{example.name}: invalid environment template")
    return {"counts": counts, "errors": sorted(set(errors))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    result = check(args.root)
    print(json.dumps(result, indent=2))
    raise SystemExit(bool(result["errors"]))


if __name__ == "__main__":
    main()
