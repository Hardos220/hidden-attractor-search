"""Сборка статьи в HTML.

Источник: paper/src/*.html (склеиваются в порядке имён). Разметка:
  {{cite:Key1,Key2}}          ссылка на литературу -> [1, 2]; нумерация по первому упоминанию,
                              записи берутся из data/bibliography.json (проверенные DOI)
  <figure data-fig="NAME">    рисунок results/figures/NAME.png; в начало <figcaption> ставится «Рис. N.»
  {{fig:NAME}}                номер рисунка
  <table data-tab="NAME">     таблица; в начало <caption> ставится «Таблица N.»
  {{tab:NAME}}                номер таблицы
  {{references}}              список литературы (стиль «Известий вузов. ПНД»)
  {{references_en}}           References (латиница)
  {{toc}}                     оглавление по заголовкам h2
Формулы — MathJax (\\( \\), $ $, $$ $$, \\label/\\eqref).

Выход: paper/article.html (рисунки в paper/figures/) и paper/article_standalone.html
(рисунки внутри файла, уменьшенные до 1700 px по ширине).
"""
import base64
import io
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "paper" / "src"
FIGSRC = ROOT / "results" / "figures"
OUT = ROOT / "paper"

# Ручные уточнения записей (заглавные буквы в названиях из Crossref, препринты без издания)
OVERRIDES = {
    "LeonovKuznetsov2013": {"title": "Hidden attractors in dynamical systems. From hidden oscillations in Hilbert–Kolmogorov, "
                                     "Aizerman, and Kalman problems to hidden chaotic attractor in Chua circuits"},
    "Molaie2013": {"title": "Simple chaotic flows with one stable equilibrium"},
    "Alexander1992Riddled": {"title": "Riddled basins"},
    "Kuznetsov2020Izv": {"title": "Теория скрытых колебаний и устойчивость систем управления"},
    "Kuznetsov2020JCSSI": {"title": "Theory of hidden oscillations and stability of control systems"},
    "Haerter2026BBT": {"arxiv": "2607.14762"},
    "KongLai2026": {"arxiv": "2609.01914"},
    "DNNBasins2021": {"arxiv": "2109.06564"},
    "MouretClune2015": {"arxiv": "1504.04909"},
    "Andrievsky2017Aircraft": {"arxiv": "1711.10302"},
    "Datseris2026GlobalCont": {"arxiv": "2607.09332", "doi": None},
    "Hsu1987": {"type": "book", "authors": ["Hsu C.S."], "venue": "New York: Springer-Verlag",
                "title": "Cell-to-cell mapping: A method of global analysis for nonlinear systems",
                "doi": "10.1007/978-1-4757-3892-6"},
    "LaiTel2011": {"venue": "New York: Springer (Applied Mathematical Sciences. Vol. 173)", "volume": None},
    "Rohm2022Unseen": {"authors": ["Gauthier D.J.", "Fischer I.", "Röhm A."],
                       "venue": "Chaos: An Interdisciplinary Journal of Nonlinear Science"},
}
# Русскоязычная запись -> её англоязычная версия для References
EN_ALT = {"Kuznetsov2020Izv": "Kuznetsov2020JCSSI"}

INITIAL_RE = re.compile(r"^(?:[A-ZА-ЯЁ][a-zа-яё]?\.)+(?:-(?:[A-ZА-ЯЁ]\.)+)?$")
PARTICLES = {"van", "der", "de", "von", "da", "di", "la", "le", "dos", "del"}


def load_bib():
    recs = json.loads((ROOT / "data" / "bibliography.json").read_text(encoding="utf-8-sig"))
    bib = {}
    for r in recs:
        r = dict(r)
        r.update(OVERRIDES.get(r["key"], {}))
        bib[r["key"]] = r
    return bib


def fmt_author(a):
    a = re.sub(r"\s+", " ", a.replace(",", " ")).strip()
    toks = a.split(" ")
    j = len(toks)
    while j > 0 and INITIAL_RE.match(toks[j - 1]):
        j -= 1
    if 0 < j < len(toks):
        surname, inits = " ".join(toks[:j]), "".join(toks[j:])
    else:
        k = len(toks) - 1
        while k > 0 and toks[k - 1].lower() in PARTICLES:
            k -= 1
        surname, inits = " ".join(toks[k:]), ""
        for g in toks[:k]:
            if INITIAL_RE.match(g):
                inits += g
            elif "-" in g:
                inits += "-".join(p[0] + "." for p in g.split("-") if p)
            else:
                inits += g[0] + "."
    if surname.isupper() and len(surname) > 1:
        surname = " ".join("-".join(q.capitalize() for q in p.split("-")) for p in surname.split(" "))
    return f"{surname} {inits}".strip()


def esc(s):
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def fmt_ref(r):
    au = ", ".join(fmt_author(a) for a in (r.get("authors") or []))
    title = re.sub(r"\s+", " ", (r.get("title") or "")).strip().rstrip(".")
    venue = re.sub(r"\s+", " ", (r.get("venue") or "")).strip()
    year = r.get("year")
    clean = lambda x: re.sub(r"\s+", "", str(x)) if x not in (None, "") else None  # noqa: E731
    vol, iss, pages = clean(r.get("volume")), clean(r.get("issue")), clean(r.get("pages"))
    is_ru = bool(re.search("[А-Яа-яЁё]", title + venue))
    V, N, P = ("Т.", "№", "С.") if is_ru else ("Vol.", "no.", "P.")
    if iss:
        iss = iss.lstrip("0") or iss
    if pages:
        pages = pages.replace("-", "–")
    journal = bool(venue) and venue.lower() not in ("arxiv.org", "arxiv preprint", "arxiv")
    if r.get("type") == "book":
        s = f"{esc(au)} {esc(title)}. {esc(venue)}, {year}."
    elif not journal and r.get("arxiv"):
        s = f"{esc(au)} {esc(title)} // arXiv preprint. {year}. arXiv:{r['arxiv']}."
    else:
        s = f"{esc(au)} {esc(title)} // {esc(venue)}. {year}."
        parts = []
        if vol and not str(vol).startswith("abs/"):
            parts.append(f"{V} {vol}")
        if iss:
            parts.append(f"{N} {iss}")
        if parts:
            s += " " + ", ".join(parts) + "."
        if pages:
            s += f" {P} {pages}."
    doi = r.get("doi")
    if doi and not str(doi).lower().startswith("10.48550/arxiv"):
        s += f' DOI: <a href="https://doi.org/{doi}">{esc(doi)}</a>.'
    return s


def compress(nums):
    nums = sorted(set(nums))
    out, i = [], 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        a, b = nums[i], nums[j]
        link = lambda n: f'<a href="#ref-{n}">{n}</a>'  # noqa: E731
        if b - a >= 2:
            out.append(f"{link(a)}–{link(b)}")
        elif b == a + 1:
            out += [link(a), link(b)]
        else:
            out.append(link(a))
        i = j + 1
    return ", ".join(out)


def downscale_png(path, max_w=1700):
    """Уменьшенная копия рисунка для встраивания: WebP (качество 90), при отсутствии — PNG."""
    from PIL import Image
    im = Image.open(path).convert("RGB")
    if im.width > max_w:
        h = int(im.height * max_w / im.width)
        im = im.resize((max_w, h), Image.LANCZOS)
    buf = io.BytesIO()
    try:
        im.save(buf, format="WEBP", quality=90, method=6)
        return buf.getvalue(), "image/webp"
    except (OSError, KeyError, ValueError):
        buf = io.BytesIO()
        im.save(buf, format="PNG", optimize=True)
        return buf.getvalue(), "image/png"


CSS = """
:root{--bg:#fbfaf7;--paper:#ffffff;--fg:#1d1d1b;--muted:#5c5b57;--rule:#dedbd2;--accent:#2a5fb0;--box:#f3f1ea;--fig:#fcfcfb}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#17171a;--paper:#1f1f23;--fg:#e9e7e1;--muted:#a9a69d;--rule:#3a3a40;--accent:#8fb4ff;--box:#26262b;--fig:#fcfcfb}}
:root[data-theme="dark"]{--bg:#17171a;--paper:#1f1f23;--fg:#e9e7e1;--muted:#a9a69d;--rule:#3a3a40;--accent:#8fb4ff;--box:#26262b;--fig:#fcfcfb}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--fg);font-family:"PT Serif",Georgia,"Times New Roman",serif;font-size:17px;line-height:1.62}
main{max-width:900px;margin:0 auto;padding:24px 16px 64px;background:var(--paper)}
h1,h2,h3,h4{font-family:"PT Sans","Segoe UI",Arial,sans-serif;line-height:1.25;margin:1.6em 0 .6em}
h1{font-size:1.75em;margin-top:.4em}
h2{font-size:1.32em;border-bottom:1px solid var(--rule);padding-bottom:.25em}
h3{font-size:1.1em}
p{margin:.55em 0;text-align:justify;hyphens:auto}
a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}
.meta{font-family:"PT Sans",Arial,sans-serif;font-size:.9em;color:var(--muted)}
.authors{font-family:"PT Sans",Arial,sans-serif;font-size:1.02em;margin:.3em 0}
.abstract{background:var(--box);border-left:3px solid var(--accent);padding:12px 16px;margin:1em 0;font-size:.95em}
.abstract p{margin:.35em 0}
.kw{font-size:.93em}
.toc{font-family:"PT Sans",Arial,sans-serif;font-size:.92em;background:var(--box);padding:10px 16px;margin:1em 0}
.toc ol{margin:.3em 0;padding-left:1.4em}
figure{margin:1.6em 0;text-align:center}
figure img{max-width:100%;height:auto;background:var(--fig);border:1px solid var(--rule);border-radius:4px;padding:4px}
figcaption{font-size:.88em;color:var(--muted);text-align:justify;margin-top:.45em;line-height:1.45}
figcaption b{color:var(--fg)}
.tablewrap{overflow-x:auto;margin:1.2em 0}
table{border-collapse:collapse;font-family:"PT Sans",Arial,sans-serif;font-size:.86em;min-width:60%;margin:0 auto}
caption{caption-side:top;text-align:left;font-size:.95em;color:var(--muted);padding:.3em 0}
caption b{color:var(--fg)}
th,td{border-bottom:1px solid var(--rule);padding:4px 8px;text-align:left;vertical-align:top}
th{border-bottom:2px solid var(--fg);font-weight:600}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
.ci{font-size:.8em;color:var(--muted);white-space:nowrap}
tr.grp td{border-top:2px solid var(--rule)}
.refs li{margin:.25em 0;font-size:.9em;text-align:left;word-break:break-word}
.note{font-size:.9em;color:var(--muted)}
.stmt{background:var(--box);padding:8px 14px;margin:.9em 0;border-radius:3px}
.mjx-chtml{font-size:105% !important}
mjx-container[display="true"]{overflow-x:auto;overflow-y:hidden;max-width:100%}
.theme-toggle{position:fixed;right:12px;top:10px;font:13px "PT Sans",Arial,sans-serif;background:var(--box);color:var(--fg);border:1px solid var(--rule);border-radius:4px;padding:3px 8px;cursor:pointer}
@media (max-width:600px){body{font-size:16px}p{text-align:left}main{padding:16px 16px 48px}}
@media print{.theme-toggle,.toc{display:none}body{background:#fff}main{max-width:none;padding:0}figure{break-inside:avoid}}
"""

TEMPLATE = """<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Скрытые аттракторы как редкие события</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=PT+Sans:wght@400;700&family=PT+Serif:ital,wght@0,400;0,700;1,400&display=swap" rel="stylesheet">
<style>{css}</style>
<script>
window.MathJax = {{tex: {{inlineMath: [['$','$'], ['\\\\(','\\\\)']], displayMath: [['$$','$$'], ['\\\\[','\\\\]']], tags: 'ams'}},
                  chtml: {{scale: 1.0}}}};
</script>
<script async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-mml-chtml.js"></script>
</head>
<body>
<button class="theme-toggle" onclick="(function(){{var r=document.documentElement;var d=r.getAttribute('data-theme');r.setAttribute('data-theme',d==='dark'?'light':'dark');}})()">тема</button>
<main>
{body}
</main>
</body>
</html>
"""


def build(standalone=False):
    bib = load_bib()
    files = sorted(SRC.glob("*.html"))
    body = "\n".join(p.read_text(encoding="utf-8") for p in files)
    warnings = []
    # формулы с \label — в нумеруемое окружение equation (при tags: 'ams' $$...$$ не нумеруется)
    body = re.sub(r"\$\$((?:(?!\$\$).)*?\\label\{[^}]+\}(?:(?!\$\$).)*?)\$\$",
                  r"\\begin{equation}\1\\end{equation}", body, flags=re.S)
    # --- рисунки
    fignum = {}
    for name in re.findall(r'<figure data-fig="([^"]+)"', body):
        fignum.setdefault(name, len(fignum) + 1)
    figdir = OUT / "figures"
    figdir.mkdir(exist_ok=True)

    def fig_block(m):
        name, rest = m.group(1), m.group(2)
        n = fignum[name]
        src = FIGSRC / f"{name}.png"
        if not src.exists():
            warnings.append(f"нет файла рисунка {src.name}")
            img = f'<div class="note">[нет файла {name}.png]</div>'
        elif standalone:
            raw, mime = downscale_png(src)
            data = base64.b64encode(raw).decode("ascii")
            img = f'<img src="data:{mime};base64,{data}" alt="Рис. {n}">'
        else:
            shutil.copy2(src, figdir / src.name)
            img = f'<a href="figures/{name}.png"><img src="figures/{name}.png" alt="Рис. {n}" loading="lazy"></a>'
        rest = re.sub(r"<figcaption>", f"<figcaption><b>Рис. {n}.</b> ", rest, count=1)
        return f'<figure id="fig-{name}">{img}{rest}</figure>'

    body = re.sub(r'<figure data-fig="([^"]+)">(.*?)</figure>', fig_block, body, flags=re.S)

    def fig_ref(m):
        names = [x.strip() for x in m.group(1).split(",")]
        out = []
        for nm in names:
            if nm not in fignum:
                warnings.append(f"ссылка на неизвестный рисунок {nm}")
                out.append("??")
            else:
                out.append(f'<a href="#fig-{nm}">{fignum[nm]}</a>')
        return ", ".join(out)

    body = re.sub(r"\{\{fig:([^}]+)\}\}", fig_ref, body)
    # --- таблицы
    tabnum = {}
    for name in re.findall(r'<table data-tab="([^"]+)"', body):
        tabnum.setdefault(name, len(tabnum) + 1)

    def tab_block(m):
        name, rest = m.group(1), m.group(2)
        n = tabnum[name]
        rest = re.sub(r"<caption>", f"<caption><b>Таблица {n}.</b> ", rest, count=1)
        return f'<div class="tablewrap"><table id="tab-{name}">{rest}</table></div>'

    body = re.sub(r'<table data-tab="([^"]+)">(.*?)</table>', tab_block, body, flags=re.S)
    # доверительные интервалы в ячейках таблиц — мельче и без переноса
    body = re.sub(r"(<td[^>]*>)(.*?)(</td>)",
                  lambda m: m.group(1) + re.sub(r"\[(\d[\d.]*), (\d[\d.]*)\]", r'<span class="ci">[\1, \2]</span>', m.group(2)) + m.group(3),
                  body, flags=re.S)
    body = re.sub(r"\{\{tab:([^}]+)\}\}",
                  lambda m: f'<a href="#tab-{m.group(1)}">{tabnum.get(m.group(1), "??")}</a>', body)
    for nm in re.findall(r'href="#tab-([^"]+)">\?\?', body):
        warnings.append(f"ссылка на неизвестную таблицу {nm}")
    # --- литература
    order = []
    for m in re.finditer(r"\{\{cite:([^}]+)\}\}", body):
        for k in m.group(1).split(","):
            k = k.strip()
            if k not in bib:
                warnings.append(f"нет в библиографии: {k}")
            elif k not in order:
                order.append(k)
    num = {k: i + 1 for i, k in enumerate(order)}
    body = re.sub(r"\{\{cite:([^}]+)\}\}",
                  lambda m: "[" + compress([num[k.strip()] for k in m.group(1).split(",") if k.strip() in num]) + "]", body)
    ref_ru = "\n".join(f'<li id="ref-{num[k]}">{fmt_ref(bib[k])}</li>' for k in order)
    ref_en = "\n".join(f'<li>{fmt_ref(bib[EN_ALT.get(k, k)])}</li>' for k in order)
    body = body.replace("{{references}}", f'<ol class="refs">\n{ref_ru}\n</ol>')
    body = body.replace("{{references_en}}", f'<ol class="refs">\n{ref_en}\n</ol>')
    # --- оглавление
    heads = re.findall(r'<h2 id="([^"]+)">(.*?)</h2>', body)
    toc = "<nav class=\"toc\"><b>Содержание</b><ol>" + "".join(f'<li><a href="#{i}">{t}</a></li>' for i, t in heads) + "</ol></nav>"
    body = body.replace("{{toc}}", toc)
    left = re.findall(r"\{\{[^}]+\}\}", body)
    if left:
        warnings.append(f"неразобранные метки: {sorted(set(left))[:10]}")
    html = TEMPLATE.format(css=CSS, body=body)
    out = OUT / ("article_standalone.html" if standalone else "article.html")
    out.write_text(html, encoding="utf-8")
    return out, len(fignum), len(tabnum), len(order), warnings


def main():
    for sa in (False, True):
        out, nf, nt, nr, w = build(standalone=sa)
        print(f"{out.name}: рисунков {nf}, таблиц {nt}, источников {nr}, размер {out.stat().st_size / 1e6:.1f} МБ")
        for x in dict.fromkeys(w):
            print("  предупреждение:", x)
    # контроль: все ли подписи авторов разобраны
    bib = load_bib()
    odd = [(k, a, fmt_author(a)) for k, r in bib.items() for a in (r.get("authors") or []) if not re.search(r"\.$", fmt_author(a))]
    for k, a, f in odd[:10]:
        print("  автор без инициалов:", k, a, "->", f)


if __name__ == "__main__":
    sys.exit(main())
