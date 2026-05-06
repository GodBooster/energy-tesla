#!/usr/local/opt/python@3.13/bin/python3
"""
ocr_pdf_mac.py — OCR a scanned PDF using macOS native Vision framework (via ocrmac)
Renders each page to PNG with pymupdf, then OCRs with macOS Vision.

Usage:
    python3 ocr_pdf_mac.py path/to/file.pdf --out output.txt [--dpi 200] [--pages 1-50]
    python3 ocr_pdf_mac.py path/to/file.pdf --out output.txt --pages 30-60

Install:
    pip3 install --break-system-packages pymupdf ocrmac
"""

import argparse
import sys
import tempfile
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--pages", type=str, default=None, help="e.g. 1-50 or 30 (1-based)")
    args = parser.parse_args()

    try:
        import fitz
        from ocrmac import ocrmac
    except ImportError as e:
        print(f"ERROR: {e}\nInstall: pip3 install --break-system-packages pymupdf ocrmac", file=sys.stderr)
        return 1

    if not args.pdf.exists():
        print(f"ERROR: {args.pdf} not found", file=sys.stderr)
        return 1

    pdf = fitz.open(str(args.pdf))
    total = len(pdf)

    if args.pages:
        if "-" in args.pages:
            a, b = args.pages.split("-")
            start, end = int(a) - 1, int(b)
        else:
            start = int(args.pages) - 1
            end = start + 1
    else:
        start, end = 0, total

    end = min(end, total)
    print(f"OCR: pages {start+1}–{end} of {total} ({args.dpi} dpi)", file=sys.stderr)

    results = []
    for pg_idx in range(start, end):
        page = pdf[pg_idx]
        pix = page.get_pixmap(dpi=args.dpi)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmp = f.name
        pix.save(tmp)
        try:
            annotations = ocrmac.OCR(tmp, language_preference=["en-US"]).recognize()
            text = "\n".join([a[0] for a in annotations])
        except Exception as e:
            text = f"[OCR ERROR: {e}]"
        finally:
            os.unlink(tmp)

        header = f"\n{'='*60}\n=== PAGE {pg_idx+1} ===\n{'='*60}\n"
        results.append(header + text)

        if (pg_idx - start + 1) % 10 == 0:
            print(f"  ... {pg_idx+1}/{end}", file=sys.stderr)

    full = "\n".join(results)

    if args.out:
        args.out.write_text(full, encoding="utf-8")
        print(f"Written {len(full)} chars to {args.out}", file=sys.stderr)
    else:
        print(full)

    return 0


if __name__ == "__main__":
    sys.exit(main())
