from __future__ import annotations

import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlparse


ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
REQUIRED_PAGES = {"index.html", "results.html", "methodology.html", "about.html"}
REMOTE_SCHEMES = {"http", "https", "mailto", "tel", "data"}


class AssetParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.references: list[tuple[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        for attribute in ("href", "src"):
            value = values.get(attribute)
            if value:
                self.references.append((attribute, value))


def local_target(page: Path, reference: str) -> Path | None:
    parsed = urlparse(reference)
    if parsed.scheme in REMOTE_SCHEMES or reference.startswith("#"):
        return None
    clean_path = unquote(parsed.path)
    if not clean_path:
        return None
    return (page.parent / clean_path).resolve()


def main() -> int:
    failures: list[str] = []
    pages = {path.name for path in DIST.glob("*.html")}
    missing_pages = sorted(REQUIRED_PAGES - pages)
    if missing_pages:
        failures.append(f"Missing required pages: {', '.join(missing_pages)}")

    for page in sorted(DIST.glob("*.html")):
        parser = AssetParser()
        parser.feed(page.read_text(encoding="utf-8"))
        for attribute, reference in parser.references:
            target = local_target(page, reference)
            if target is not None and not target.exists():
                failures.append(f"{page.name}: missing {attribute} target {reference}")

    for stylesheet in DIST.rglob("*.css"):
        contents = stylesheet.read_text(encoding="utf-8")
        for reference in re.findall(r"url\(['\"]?([^)'\"]+)", contents):
            target = local_target(stylesheet, reference)
            if target is not None and not target.exists():
                failures.append(
                    f"{stylesheet.relative_to(DIST)}: missing CSS target {reference}"
                )

    config_path = DIST / "assets" / "models" / "model-config.json"
    verification_path = DIST / "assets" / "models" / "verification.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        verification = json.loads(verification_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        failures.append(f"Model metadata could not be read: {error}")
    else:
        if len(config.get("members", [])) != 5:
            failures.append("Model configuration must contain five ensemble members")
        for member in config.get("members", []):
            target = DIST / member.get("url", "")
            if not target.is_file():
                failures.append(f"Missing model asset: {member.get('url')}")
        if verification.get("ensemble_class_agreement") != 1.0:
            failures.append("Export verification did not achieve full class agreement")

    runtime_dir = DIST / "assets" / "vendor" / "onnxruntime"
    runtime_files = {
        "ort.wasm.min.js",
        "ort-wasm-simd-threaded.mjs",
        "ort-wasm-simd-threaded.wasm",
    }
    for filename in sorted(runtime_files):
        if not (runtime_dir / filename).is_file():
            failures.append(f"Bundled ONNX browser runtime is missing {filename}")

    if failures:
        print("Static validation failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print(
        "Static validation passed: 4 pages, 5 model members, local runtime, "
        "and all referenced local assets are present."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
