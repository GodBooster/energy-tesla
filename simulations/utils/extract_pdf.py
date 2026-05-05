"""
extract_pdf.py — extract text from a PDF using pdfminer.six

Usage:
    python3 extract_pdf.py path/to/file.pdf
    python3 extract_pdf.py path/to/file.pdf --start 1000 --end 5000

Install:
    pip3 install --break-system-packages pdfminer.six
"""

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract text from PDF with pdfminer.six")
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--start", type=int, default=0, help="Character offset to start at")
    parser.add_argument("--end", type=int, default=None, help="Character offset to end at")
    parser.add_argument("--out", type=Path, default=None, help="Write to file instead of stdout")
    args = parser.parse_args()

    try:
        from pdfminer.high_level import extract_text
    except ImportError:
        print("ERROR: pdfminer.six not installed. Run:", file=sys.stderr)
        print("  pip3 install --break-system-packages pdfminer.six", file=sys.stderr)
        return 1

    if not args.pdf.exists():
        print(f"ERROR: {args.pdf} not found", file=sys.stderr)
        return 1

    text = extract_text(str(args.pdf))
    chunk = text[args.start:args.end]

    if args.out:
        args.out.write_text(chunk, encoding="utf-8")
        print(f"Written {len(chunk)} chars to {args.out}")
    else:
        print(chunk)
    return 0


if __name__ == "__main__":
    sys.exit(main())
