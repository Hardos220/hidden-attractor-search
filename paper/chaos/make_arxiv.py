"""Build the arXiv source package of the manuscript from the flattened build of build_chaos.py.

    python paper/chaos/build_chaos.py        # first: manuscript.tex, manuscript.bbl, manuscript.pdf
    python paper/chaos/make_arxiv.py [--test]

Produces paper/chaos/arxiv/ (staging: manuscript.tex, manuscript.bbl, figures/*.pdf used by the text) and
paper/chaos/arxiv_v1.zip to upload. Differences from the journal build:
  * `\\pdfoutput=1` in the first line, so arXiv runs pdflatex and accepts the PDF figures;
  * the unused red `\\placeholder` macro is dropped;
  * the AIP preprint credit line ("submitted to Chaos ...") is printed under the title block;
  * the bibliography is shipped as the ready .bbl (arXiv does not run BibTeX) with Unicode dashes
    replaced by TeX ones, and both files are checked to be pure ASCII.
With --journal the same package is built for the journal submission instead (paper/chaos/journal_source/ and
paper/chaos/chaos_source.zip): no preprint credit line, and refs.bib is shipped next to the ready .bbl.
With --test the staged source is compiled with Tectonic (XeTeX engine; arXiv uses pdflatex, the sources are
plain ASCII and standard packages, so this is a smoke test, not the final word) and the page count is printed.
"""
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGE = HERE / "arxiv"
ZIP = HERE / "arxiv_v1.zip"
TECTONIC = Path(os.environ.get("TECTONIC", r"C:\Cursor\tools\tectonic\tectonic.exe"))

CREDIT = (r"\noindent{\footnotesize The following article has been submitted to \textit{Chaos: An Interdisciplinary "
          r"Journal of Nonlinear Science}. After it is published, it will be found at \url{https://pubs.aip.org/aip/cha}.}"
          "\n\\par\\medskip\n")


def ascii_check(name, text):
    bad = [(i, ch) for i, ch in enumerate(text) if ord(ch) > 127]
    if bad:
        i, ch = bad[0]
        raise SystemExit(f"{name}: {len(bad)} non-ASCII characters, first U+{ord(ch):04X} at {i}: {text[max(0, i-30):i+30]!r}")


def main(argv):
    journal = "--journal" in argv
    global STAGE, ZIP
    if journal:
        STAGE, ZIP = HERE / "journal_source", HERE / "chaos_source.zip"
    tex = (HERE / "manuscript.tex").read_text(encoding="utf-8")
    bbl = (HERE / "manuscript.bbl").read_text(encoding="utf-8")
    if "\\begin{quotation}" not in tex:
        raise SystemExit("manuscript.tex has no lead paragraph (quotation); run build_chaos.py first")

    tex = re.sub(r"% text the author must supply[^\n]*\n\\newcommand\{\\placeholder\}[^\n]*\n", "", tex)
    if "\\placeholder{" in tex:
        raise SystemExit("manuscript.tex still uses \\placeholder{}: fill the fields first")
    tex = "\\pdfoutput=1\n" + tex
    if not journal:
        tex = tex.replace("\\begin{quotation}", CREDIT + "\\begin{quotation}", 1)
    dashes = lambda t: t.replace("\u2013", "--").replace("\u2014", "---").replace("\u2019", "'")
    bbl = dashes(bbl)
    ascii_check("manuscript.tex", tex)
    ascii_check("manuscript.bbl", bbl)
    bib = None
    if journal:
        bib = dashes((HERE / "refs.bib").read_text(encoding="utf-8"))
        ascii_check("refs.bib", bib)

    figs = sorted(set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{(figures/[^}]+)\}", tex)))
    missing = [f for f in figs if not (HERE / f).exists()]
    if missing:
        raise SystemExit(f"missing figures: {missing}")

    if STAGE.exists():
        shutil.rmtree(STAGE)
    (STAGE / "figures").mkdir(parents=True)
    (STAGE / "manuscript.tex").write_text(tex, encoding="ascii", newline="\n")
    (STAGE / "manuscript.bbl").write_text(bbl, encoding="ascii", newline="\n")
    if bib is not None:
        (STAGE / "refs.bib").write_text(bib, encoding="ascii", newline="\n")
    for f in figs:
        shutil.copy2(HERE / f, STAGE / f)

    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(STAGE / "manuscript.tex", "manuscript.tex")
        z.write(STAGE / "manuscript.bbl", "manuscript.bbl")
        if bib is not None:
            z.write(STAGE / "refs.bib", "refs.bib")
        for f in figs:
            z.write(STAGE / f, f)
    print(f"staged {len(figs)} figures; {ZIP.name}: {ZIP.stat().st_size / 1e6:.1f} MB")

    if "--test" in argv:
        out = STAGE / "_test"
        out.mkdir(exist_ok=True)
        # XeTeX has no \pdfoutput primitive, and Tectonic would run BibTeX (no .bib here) and empty the .bbl:
        # the smoke test compiles a copy without the first line and with the .bbl read directly
        test_tex = tex.split("\n", 1)[1].replace("\\bibliography{refs}", "\\input{manuscript.bbl}")
        (out / "manuscript.tex").write_text(test_tex, encoding="ascii", newline="\n")
        shutil.copy2(STAGE / "manuscript.bbl", out / "manuscript.bbl")
        shutil.copytree(STAGE / "figures", out / "figures", dirs_exist_ok=True)
        r = subprocess.run([str(TECTONIC), "--keep-logs", "-o", str(out), "manuscript.tex"], cwd=out,
                           capture_output=True, text=True)
        log = (out / "manuscript.log").read_text(encoding="utf-8", errors="replace") if (out / "manuscript.log").exists() else ""
        if r.returncode != 0:
            print(r.stderr[-2000:])
            raise SystemExit("test compile failed")
        try:
            import fitz
            print("test compile: pages", fitz.open(str(out / "manuscript.pdf")).page_count)
        except ImportError:
            print("test compile: ok")
        for key in ("undefined", "Overfull"):
            print(f"  {key}: {log.count(key)}")
        shutil.rmtree(out)


if __name__ == "__main__":
    main(sys.argv[1:])
