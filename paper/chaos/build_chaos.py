"""Build the Chaos (AIP) submission from paper/chaos/.

1. main.tex + sections/*.tex -> manuscript.tex (one flattened file, as AIP prefers);
2. cited keys -> refs.bib from data/bibliography.json (the same verified records as the Russian article,
   with the OVERRIDES of paper/build_article.py), journal names abbreviated, TeX-escaped;
3. compile with Tectonic -> manuscript.pdf, then report undefined references, overfull boxes and missing glyphs.

    python paper/chaos/build_chaos.py [--no-pdf]
"""
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "paper"))
from build_article import fmt_author, load_bib  # noqa: E402

TECTONIC = Path(os.environ.get("TECTONIC", r"C:\Cursor\tools\tectonic\tectonic.exe"))   # or any tectonic on PATH

JOURNAL_ABBR = {
    "Chaos, Solitons &amp; Fractals": "Chaos Solitons Fractals",
    "Chaos, Solitons & Fractals": "Chaos Solitons Fractals",
    "Chaos: An Interdisciplinary Journal of Nonlinear Science": "Chaos",
    "Chinese Physics B": "Chin. Phys. B",
    "Communications in Nonlinear Science and Numerical Simulation": "Commun. Nonlinear Sci. Numer. Simul.",
    "Engineering Applications of Artificial Intelligence": "Eng. Appl. Artif. Intell.",
    "Entropy": "Entropy",
    "Evolutionary Computation": "Evol. Comput.",
    "Frontiers in Robotics and AI": "Front. Robot. AI",
    "IEEE Transactions on Evolutionary Computation": "IEEE Trans. Evol. Comput.",
    "IEEJ Transactions on Electrical and Electronic Engineering": "IEEJ Trans. Electr. Electron. Eng.",
    "IFAC Proceedings Volumes": "IFAC Proc. Vol.",
    "IFAC-PapersOnLine": "IFAC-PapersOnLine",
    "International Journal of Bifurcation and Chaos": "Int. J. Bifurcation Chaos",
    "Journal of Computer and Systems Sciences International": "J. Comput. Syst. Sci. Int.",
    "Journal of Open Source Software": "J. Open Source Softw.",
    "Journal of Physics A: Mathematical and Theoretical": "J. Phys. A: Math. Theor.",
    "Journal of Statistical Physics": "J. Stat. Phys.",
    "Meccanica": "Meccanica",
    "Methodology And Computing In Applied Probability": "Methodol. Comput. Appl. Probab.",
    "Nature Communications": "Nat. Commun.",
    "Nature Physics": "Nat. Phys.",
    "Neural Networks": "Neural Netw.",
    "New Journal of Physics": "New J. Phys.",
    "Nonlinear Dynamics": "Nonlinear Dyn.",
    "Numerische Mathematik": "Numer. Math.",
    "Physica D: Nonlinear Phenomena": "Physica D",
    "Physical Review A": "Phys. Rev. A",
    "Physical Review E": "Phys. Rev. E",
    "Physical Review Letters": "Phys. Rev. Lett.",
    "Physical Review Research": "Phys. Rev. Res.",
    "Physics Letters A": "Phys. Lett. A",
    "Physics Reports": "Phys. Rep.",
    "Probabilistic Engineering Mechanics": "Probab. Eng. Mech.",
    "Proceedings of the National Academy of Sciences": "Proc. Natl. Acad. Sci. U.S.A.",
    "SIAM Journal on Applied Dynamical Systems": "SIAM J. Appl. Dyn. Syst.",
    "Science": "Science",
    "Scientific Reports": "Sci. Rep.",
    "Statistics and Computing": "Stat. Comput.",
    "Stochastic Analysis and Applications": "Stoch. Anal. Appl.",
    "Structural Safety": "Struct. Saf.",
    "The European Physical Journal Plus": "Eur. Phys. J. Plus",
    "The European Physical Journal Special Topics": "Eur. Phys. J. Spec. Top.",
    "Operations Research": "Oper. Res.",
    "Journal of Statistical Mechanics: Theory and Experiment": "J. Stat. Mech.: Theory Exp.",
    "Journal of Computational Physics": "J. Comput. Phys.",
    "Journal of the Atmospheric Sciences": "J. Atmos. Sci.",
    "Technometrics": "Technometrics",
    "Artificial Intelligence": "Artif. Intell.",
    "Nonlinearity": "Nonlinearity",
    "Japan Journal of Industrial and Applied Mathematics": "Jpn. J. Ind. Appl. Math.",
    "Neural Computation": "Neural Comput.",
    "Cognitive Computation": "Cogn. Comput.",
}
# records whose container is not a journal (Crossref metadata of the DOIs)
BOOKS = {
    "Hsu1987": {"publisher": "Springer-Verlag", "address": "New York"},
    "LaiTel2011": {"publisher": "Springer", "address": "New York", "series": "Applied Mathematical Sciences", "volume": "173"},
}
CHAPTERS = {
    "Zelinka2021Chapter": {"booktitle": "Chaotic Systems with Multistability and Hidden Attractors",
                           "series": "Emergence, Complexity and Computation", "publisher": "Springer", "address": "Cham"},
}
# Crossref returned the names of this record with CJK characters mixed in
AUTHOR_FIX = {"Cao2025AL": ["Cao X.W.", "Ru X.L.", "Yan G."]}
TEX_CHARS = {"á": "{\\'a}", "ä": '{\\"a}', "é": "{\\'e}", "ö": '{\\"o}', "š": "{\\v{s}}", "ü": '{\\"u}', "ó": "{\\'o}",
             "í": "{\\'i}", "ñ": "{\\~n}", "ç": "{\\c{c}}", "ć": "{\\'c}", "č": "{\\v{c}}", "ł": "{\\l}", "ř": "{\\v{r}}",
             "ø": "{\\o}",
             "ʼ": "'", "’": "'", "‘": "`", "“": "``", "”": "''", "–": "--", "—": "---", "×": "$\\times$"}


def tex(s):
    s = re.sub(r"\s+", " ", str(s)).strip().replace("&amp;", "&")
    s = re.sub(r"([&%#_$])", r"\\\1", s)
    s = "".join(TEX_CHARS.get(c, c) for c in s)
    bad = sorted({c for c in s if ord(c) > 127})
    if bad:
        raise ValueError(f"no TeX form for {''.join(bad)!r} in {s!r}")
    return s


def bib_author(a):
    """'Surname I.O.' (fmt_author of the Russian build) -> 'Surname, I. O.'; names without initials stay whole."""
    s = fmt_author(a)
    m = re.match(r"^(.*?)\s+((?:[A-ZА-ЯЁ][a-zа-яё]?\.-?)+)$", s)
    if not m:
        return "{" + tex(s) + "}"
    sur, ini = m.group(1), m.group(2)
    ini = re.sub(r"\.(?=[A-ZА-ЯЁ])", ". ", ini)
    return f"{tex(sur)}, {tex(ini)}"


def entry(key, r):
    title = (r.get("title") or "").rstrip(".")
    title = title.replace("“", "‘").replace("”", "’").replace('"', "'")    # the style quotes the title itself
    f = {"author": " and ".join(bib_author(a) for a in AUTHOR_FIX.get(key, r.get("authors") or [])),
         "title": "{" + tex(title).replace("‘", "`") + "}", "year": str(r.get("year"))}
    clean = lambda x: re.sub(r"\s+", " ", str(x)).strip() if x not in (None, "") else None  # noqa: E731
    vol, iss, pages = clean(r.get("volume")), clean(r.get("issue")), clean(r.get("pages"))
    if vol and " " in vol and not iss:          # "32 11" = volume 32, issue 11
        vol, iss = vol.split(" ", 1)
    doi = r.get("doi")
    if doi and str(doi).lower().startswith("10.48550/arxiv"):
        doi = None
    venue = (r.get("venue") or "").strip()
    if key in BOOKS:
        kind = "book"
        f.update(BOOKS[key])
    elif key in CHAPTERS:
        kind = "incollection"
        f.update(CHAPTERS[key])
    elif r.get("type") == "book":                    # venue "Address: Publisher"
        kind = "book"
        address, _, publisher = venue.partition(": ")
        f.update({"publisher": tex(publisher or address), "address": tex(address) if publisher else ""})
    elif r.get("type") == "proceedings-article":
        kind = "inproceedings"
        f["booktitle"] = tex(venue)
    elif venue and not venue.lower().startswith("arxiv"):
        kind = "article"
        if venue not in JOURNAL_ABBR:
            raise KeyError(f"{key}: no abbreviation for journal {venue!r}")
        f["journal"] = JOURNAL_ABBR[venue]
        if vol and not vol.startswith("abs/"):
            f["volume"] = vol
        if iss:
            f["number"] = iss.lstrip("0") or iss
    else:
        kind = "misc"
        f["eprint"] = r["arxiv"]
        f["archiveprefix"] = "arXiv"
    if pages:
        f["pages"] = pages.replace(" ", "").replace("-", "--")
    if doi:
        f["doi"] = str(doi).lower().replace("https://doi.org/", "")
    body = ",\n".join(f"  {k} = {{{v}}}" for k, v in f.items() if v)     # title is double-braced: case kept
    return f"@{kind}{{{key},\n{body}\n}}\n"


def flatten(path):
    text = path.read_text(encoding="utf-8")
    return re.sub(r"\\input\{([^}]+)\}", lambda m: flatten(HERE / (m.group(1) + ".tex")), text)


def cited_keys(text):
    keys = []
    for m in re.finditer(r"\\(?:cite|onlinecite|citep|citet)\{([^}]+)\}", text):
        for k in m.group(1).split(","):
            k = k.strip()
            if k and k not in keys:
                keys.append(k)
    return keys


def initial_condition_36():
    p = HERE / "_build" / "suptitles.txt"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.startswith("e40:fig_lorenz_transient"):
                m = re.search(r"initial condition \(([^)]+)\)", line)
                if m:
                    return "$(" + m.group(1) + ")$"
    return "[see figure script]"


def main(argv):
    text = flatten(HERE / "main.tex")
    text = text.replace("%%IC36%%", initial_condition_36())
    text = "%% Flattened by build_chaos.py from main.tex and sections/*.tex; edit the sources, not this file.\n" + text
    (HERE / "manuscript.tex").write_text(text, encoding="utf-8")
    keys = cited_keys(text)
    bib = load_bib()
    missing = [k for k in keys if k not in bib]
    if missing:
        raise SystemExit(f"keys not in data/bibliography.json: {missing}")
    (HERE / "refs.bib").write_text("".join(entry(k, bib[k]) + "\n" for k in keys), encoding="utf-8")
    figs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text)
    absent = [f for f in figs if not (HERE / f).exists()]
    print(f"manuscript.tex: {len(text)} chars, {len(figs)} figures, {len(keys)} references")
    if absent:
        print("missing figure files:", absent)
    if "--no-pdf" in argv:
        return
    r = subprocess.run([str(TECTONIC), "--keep-logs", "--keep-intermediates", "manuscript.tex"], cwd=HERE,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    log = (HERE / "manuscript.log").read_text(encoding="utf-8", errors="replace") if (HERE / "manuscript.log").exists() else ""
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:])
        raise SystemExit("tectonic failed")
    for pat, label in ((r"LaTeX Warning: (?:Reference|Citation) [^\n]*undefined[^\n]*", "undefined"),
                       (r"Overfull \\hbox \(([\d.]+)pt too wide\)[^\n]*lines? (\d+)", "overfull"),
                       (r"Missing character: There is no [^\n]*", "glyph")):
        hits = re.findall(pat, log)
        if label == "overfull":
            hits = [f"{float(w):.1f}pt at line {ln}" for w, ln in hits if float(w) > 1.0]
        print(f"{label}: {len(hits)}", *sorted(set(hits))[:30], sep="\n  ")
    try:
        import fitz
        doc = fitz.open(HERE / "manuscript.pdf")
        print(f"manuscript.pdf: {doc.page_count} pages")
    except Exception as e:  # noqa: BLE001
        print("page count unavailable:", e)


if __name__ == "__main__":
    main(sys.argv[1:])
