"""Convert a markdown report in reports/ to PDF (markdown -> HTML -> headless Chromium).
Usage: python3 scripts/make_report_pdf.py reports/PHASE1_REPORT.md
"""
import sys, re
from pathlib import Path
import markdown
import subprocess, tempfile, os

src = Path(sys.argv[1])
body = markdown.markdown(src.read_text(), extensions=["tables", "fenced_code"])
css = """body{font-family:Arial,Helvetica,sans-serif;font-size:10.5pt;line-height:1.4;margin:0;color:#222}
h1{font-size:19pt;border-bottom:2px solid #2a6fbb;padding-bottom:4px} h2{font-size:14pt;color:#2a6fbb;margin-top:18px}
table{border-collapse:collapse;width:100%;font-size:8.8pt;margin:8px 0} th,td{border:1px solid #bbb;padding:3px 5px;text-align:left}
th{background:#eef3fa} code{background:#f2f2f2;padding:1px 3px;font-size:9pt} pre{background:#f2f2f2;padding:8px;font-size:8.8pt;white-space:pre-wrap}
li{margin:2px 0} tr,pre{page-break-inside:avoid}"""
html = f"<html><head><meta charset='utf-8'><style>{css}</style></head><body>{body}</body></html>"
out = src.with_suffix(".pdf")
chrome = os.environ.get("CHROME_BIN", "/opt/pw-browsers/chromium")
with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False) as f:
    f.write(html)
subprocess.run([chrome, "--headless", "--no-sandbox", "--disable-gpu", "--no-pdf-header-footer",
                f"--print-to-pdf={out}", f.name], check=True, capture_output=True)
os.unlink(f.name)
print("wrote", out)
