"""Сборка библиотеки литературы проекта docs/01_bibliografiya.md
из data/bibliography.json (проверенные метаданные) и data/bib_annotations.json (заметки).

Как пополнять библиотеку: новые работы записать кандидатами в data/bib_candidates_<дата>.json (key, group, doi или
arxiv, title), проверить `python src/tools/verify_bib.py data/bib_candidates_<дата>.json` (режим слияния), добавить
заметки (note, use, read, pdf, found, при необходимости group) в data/bib_annotations.json и пересобрать этот файл:
`python src/tools/build_bib_md.py`.
"""
import html
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BIB = ROOT / "data" / "bibliography.json"
ANN = ROOT / "data" / "bib_annotations.json"
OUT = ROOT / "docs" / "01_bibliografiya.md"


def clean(s):
    """Нормализация строк из Crossref: HTML-сущности, переносы строк, лишние пробелы."""
    if not s:
        return s
    s = html.unescape(str(s))
    s = re.sub(r"\s+", " ", s).strip()
    return s


def input_titles():
    """Исходные (человекочитаемые) названия из файлов кандидатов — на случай ВЕРХНЕГО РЕГИСТРА в Crossref."""
    titles = {}
    for f in sorted((ROOT / "data").glob("bib_candidates*.json")):
        try:
            for c in json.loads(f.read_text(encoding="utf-8-sig")):
                titles[c["key"]] = c["title"]
        except Exception:
            pass
    return titles


def fmt_author(a):
    a = clean(a)
    if a and a.upper() == a:
        # ФАМИЛИЯ И.О. -> Фамилия И.О.
        a = " ".join(w.capitalize() if (len(w) > 2 and "." not in w) else w for w in a.split(" "))
    return a


def fmt_authors(authors):
    if not authors:
        return "—"
    authors = [fmt_author(a) for a in authors]
    if len(authors) > 6:
        return ", ".join(authors[:3]) + " и др."
    return ", ".join(authors)


def fmt_cite(r, titles=None):
    title = clean(r.get("title", ""))
    if title and title.upper() == title and titles and r.get("key") in titles:
        title = titles[r["key"]]
    parts = [fmt_authors(r.get("authors")), f"({r.get('year', '—')}).", title.rstrip(".") + "."]
    venue = clean(r.get("venue") or "")
    vol = r.get("volume")
    issue = r.get("issue")
    pages = r.get("pages")
    v = venue
    if vol:
        v += f" {vol}"
        if issue:
            v += f"({issue})"
    if pages:
        v += f": {pages}"
    if v:
        parts.append(v + ".")
    ids = []
    if r.get("doi"):
        ids.append(f"DOI: {r['doi']}")
    if r.get("arxiv"):
        ids.append(f"arXiv: {r['arxiv']}")
    if ids:
        parts.append(" ".join(ids) + ".")
    return " ".join(parts)


def main():
    bib = {r["key"]: r for r in json.loads(BIB.read_text(encoding="utf-8-sig"))}
    ann = json.loads(ANN.read_text(encoding="utf-8-sig"))
    groups = ann.pop("_groups")
    titles = input_titles()
    lines = []
    lines.append("# Библиотека литературы проекта\n")
    nver = sum(1 for r in bib.values() if r.get("verified"))
    lines.append(f"Всего записей: {len(bib)}, из них проверено по DOI/arXiv/названию: {nver}. "
                 "Источник метаданных указан в квадратных скобках (crossref или semanticscholar). "
                 "Записи, помеченные «не проверено», в статью не попадают до ручной проверки.\n")
    lines.append("Ядро библиографии (★) — источники, на которые опирается постановка задачи и введение статьи; "
                 "остальные — вспомогательные.\n")
    lines.append("У каждой записи: *Содержание* — о чём работа; *Использование* — чем она полезна нам; *Прочитано* — "
                 "целиком, ключевые разделы или только аннотация; *PDF* — полный текст в literature/ (папка не в git); "
                 "*Найдено* — когда и в связи с чем. Как пополнять библиотеку, написано в src/tools/build_bib_md.py.\n")
    read_name = {"full": "целиком", "partial": "ключевые разделы", "abstract": "только аннотация",
                 "none": "не читалась, известна по ссылкам"}
    for g, gname in groups.items():
        recs = [(k, bib[k]) for k in ann if k in bib and (ann[k].get("group") or bib[k].get("group")) == g]
        recs.sort(key=lambda kr: (not ann[kr[0]].get("core", False), kr[1].get("year") or 0, kr[0]))
        if not recs:
            continue
        lines.append(f"\n## {gname}\n")
        for k, r in recs:
            a = ann[k]
            star = "★ " if a.get("core") else ""
            ver = f"[{r.get('source')}]" if r.get("verified") else "[НЕ ПРОВЕРЕНО]"
            cites = f", цитирований по S2: {r['citations']}" if r.get("citations") is not None else ""
            lines.append(f"- {star}**{k}**. {fmt_cite(r, titles)} {ver}{cites}")
            lines.append(f"  - *Содержание:* {a['note']}")
            lines.append(f"  - *Использование в работе:* {a['use']}")
            extra = []
            if a.get("read"):
                extra.append(f"*Прочитано:* {read_name.get(a['read'], a['read'])}")
            if a.get("pdf"):
                extra.append(f"*PDF:* {a['pdf']}")
            if a.get("found"):
                extra.append(f"*Найдено:* {a['found']}")
            if extra:
                lines.append("  - " + "; ".join(extra))
    shown = {k for k in ann if k in bib and (ann[k].get("group") or bib[k].get("group")) in groups}
    missing = [k for k in bib if k not in shown]
    if missing:
        lines.append("\n## Записи без аннотации\n")
        for k in missing:
            lines.append(f"- {k}. {fmt_cite(bib[k], titles)}")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"written {OUT} ({len(bib)} records, {nver} verified)")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
