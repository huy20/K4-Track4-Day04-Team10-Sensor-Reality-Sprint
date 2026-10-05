"""Xuất báo cáo Markdown ra PDF (A4) bằng Chrome/Chromium headless.

Chạy:  pip install markdown && python scripts/build_report_pdf.py [reports/step6_report.md]
PDF được ghi cạnh file .md (cùng tên, đuôi .pdf). Ảnh dùng đường dẫn tương đối như trong .md.
"""
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import markdown

CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
]

CSS = """
@page { size: A4; margin: 14mm 14mm 16mm; }
body { font-family: Arial, "Helvetica Neue", sans-serif; font-size: 9.6pt; line-height: 1.42;
       color: #0b0b0b; }
h1 { font-size: 16pt; margin: 0 0 4px; }
h2 { font-size: 12pt; margin: 14px 0 5px; padding-bottom: 2px; border-bottom: 1px solid #c3c2b7;
     break-after: avoid; }
p, ul, ol { margin: 4px 0; }
li { margin: 1px 0; }
ul, ol { padding-left: 18px; }
blockquote { margin: 6px 0; padding: 4px 10px; border-left: 3px solid #c3c2b7; color: #52514e; }
table { border-collapse: collapse; width: 100%; margin: 6px 0; font-size: 8.8pt;
        break-inside: avoid; }
th, td { border: 1px solid #e1e0d9; padding: 3px 6px; text-align: left; vertical-align: top; }
th { background: #f0efec; }
code { font-family: Menlo, Consolas, monospace; font-size: 8.4pt; background: #f0efec;
       padding: 0 3px; border-radius: 3px; }
img { max-width: 100%; display: block; margin: 6px auto; break-inside: avoid; }
a { color: #1c5cab; text-decoration: none; }
"""


def find_chrome():
    for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser",
                 "msedge"):
        if shutil.which(name):
            return shutil.which(name)
    for path in CHROME_CANDIDATES:
        if Path(path).exists():
            return path
    sys.exit("Không tìm thấy Chrome/Chromium/Edge để in PDF.")


def main():
    md_path = Path(sys.argv[1] if len(sys.argv) > 1 else "reports/step6_report.md").resolve()
    pdf_path = md_path.with_suffix(".pdf")
    text = md_path.read_text(encoding="utf-8")
    # python-markdown cần thụt 4 dấu cách cho list con; file .md viết theo kiểu GitHub (2 dấu cách)
    text = re.sub(r"(?m)^  (?=(?:[-*]|\d+\.) )", "    ", text)
    body = markdown.markdown(text, extensions=["tables"])
    title = md_path.stem
    html = (f'<!doctype html><html lang="vi"><head><meta charset="utf-8">'
            f'<base href="{md_path.parent.as_uri()}/"><title>{title}</title>'
            f"<style>{CSS}</style></head><body>{body}</body></html>")
    with tempfile.TemporaryDirectory() as tmp:
        html_path = Path(tmp) / "report.html"
        html_path.write_text(html, encoding="utf-8")
        subprocess.run([find_chrome(), "--headless", "--disable-gpu", "--no-pdf-header-footer",
                        "--allow-file-access-from-files", f"--print-to-pdf={pdf_path}",
                        html_path.as_uri()], check=True, capture_output=True)
    print("Đã ghi", pdf_path)


if __name__ == "__main__":
    main()
