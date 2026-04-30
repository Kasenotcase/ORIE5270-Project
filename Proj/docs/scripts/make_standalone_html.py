#!/usr/bin/env python3
"""Create a shareable single-file copy of the progress report.

Local image references in the source HTML are embedded as base64 data URIs.
Remote scripts such as MathJax CDN are intentionally left as remote URLs.
"""

import base64
import mimetypes
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
SOURCE_HTML = DOCS / "zoom_progress_report.html"
TARGET_HTML = DOCS / "zoom_progress_report_standalone.html"


SRC_RE = re.compile(r'src="([^"]+)"')


def is_remote(src):
    return src.startswith(("http://", "https://", "data:"))


def embed_file(src, html_dir):
    path = (html_dir / src).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Missing local asset referenced by HTML: {src}")

    mime_type, _ = mimetypes.guess_type(path.name)
    if mime_type is None:
        mime_type = "application/octet-stream"

    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build():
    html_dir = SOURCE_HTML.parent
    text = SOURCE_HTML.read_text(encoding="utf-8")
    replacements = {}

    for src in SRC_RE.findall(text):
        if is_remote(src):
            continue
        replacements[src] = embed_file(src, html_dir)

    for src, data_uri in replacements.items():
        text = text.replace(f'src="{src}"', f'src="{data_uri}"')

    text = text.replace(
        "</head>",
        "  <!-- Local image assets are embedded for single-file sharing. "
        "MathJax remains loaded from CDN for formula rendering. -->\n</head>",
        1,
    )
    TARGET_HTML.write_text(text, encoding="utf-8")
    return TARGET_HTML, len(replacements)


if __name__ == "__main__":
    target, embedded_count = build()
    print(target)
    print(f"embedded_local_assets={embedded_count}")
