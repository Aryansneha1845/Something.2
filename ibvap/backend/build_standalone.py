"""Builds a portable single-file web dashboard (HTML + inline CSS + JS)."""
from pathlib import Path

STATIC = Path(__file__).resolve().parent.parent / "frontend"
OUT = Path(__file__).resolve().parent.parent / "standalone_dashboard.html"

css = (STATIC / "style.css").read_text(encoding="utf-8")
js = (STATIC / "app.js").read_text(encoding="utf-8")
html = (STATIC / "index.html").read_text(encoding="utf-8")

# Replace stylesheet link with inline <style>
import re
html = re.sub(r'<link rel="stylesheet"[^>]*>', "", html)
html = html.replace("</head>", f"<style>\n{css}\n</style>\n</head>")

# Replace script tags with inline <script>
html = re.sub(r'<script src="[^"]*"></script>', "", html)
html = html.replace("</body>", f"<script>\n{js}\n</script>\n</body>")

OUT.write_text(html, encoding="utf-8")
print(f"Standalone dashboard generated at: {OUT} ({len(html)} bytes)")
