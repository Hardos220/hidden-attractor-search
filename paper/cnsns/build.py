"""Build the CNSNS manuscript and its supplement with Tectonic.

    python paper/cnsns/build.py            # supplement -> main -> supplement -> main (cross-references via xr-hyper)
    python paper/cnsns/build.py --main     # main only (uses the existing supplement.aux)

Outputs paper/cnsns/main.pdf and paper/cnsns/supplement.pdf and prints the undefined references, citations and
overfull boxes found in the logs. TECTONIC may point to the executable (default C:/Cursor/tools/tectonic/tectonic.exe).
"""
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TECTONIC = os.environ.get("TECTONIC", "C:/Cursor/tools/tectonic/tectonic.exe")


def run(name):
    cmd = [TECTONIC, "-X", "compile", "--keep-intermediates", "--keep-logs", f"{name}.tex"]
    p = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0:
        print(p.stdout[-3000:], p.stderr[-3000:])
        raise SystemExit(f"tectonic failed on {name}.tex")
    return p


def check(name):
    log = (HERE / f"{name}.log").read_text(encoding="utf-8", errors="replace")
    undef_ref = sorted(set(re.findall(r"Reference `([^']+)' on page \d+ undefined", log)))
    undef_cit = sorted(set(re.findall(r"Citation `([^']+)' on page \d+ undefined", log)))
    overfull = re.findall(r"Overfull \\hbox \(([\d.]+)pt too wide\)[^\n]*lines? (\d+)", log)
    pages = re.search(r"Output written on [^(]*\((\d+) pages", log)
    print(f"{name}: pages {pages.group(1) if pages else '?'}; undefined refs {len(undef_ref)} {undef_ref[:15]}; "
          f"undefined citations {len(undef_cit)} {undef_cit[:15]}; overfull {len(overfull)} "
          f"{[(float(w), l) for w, l in overfull if float(w) > 1][:15]}")


def main():
    names = ["main"] if "--main" in sys.argv else ["supplement", "main", "supplement", "main"]
    for n in names:
        run(n)
    for n in sorted(set(names)):
        check(n)


if __name__ == "__main__":
    main()
