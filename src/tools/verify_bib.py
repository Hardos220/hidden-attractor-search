"""Верификация библиографии по DOI (Crossref) и arXiv/названию (Semantic Scholar).

Вход:  data/bib_candidates.json  — список кандидатов (key, group, doi, arxiv, title).
Выход: data/bibliography.json    — проверенные записи с метаданными и статусом,
       data/bib_report.txt       — краткий отчёт.

Правило из методики academic-research-skills: ни одной непроверенной ссылки.
Запись считается проверенной, если найдена по DOI в Crossref или по arXiv-id /
названию в Semantic Scholar и сходство названий >= 0.75.
"""
import json
import sys
import time
import difflib
import urllib.parse
import urllib.request
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data" / "bib_candidates.json"
OUT = ROOT / "data" / "bibliography.json"
REP = ROOT / "data" / "bib_report.txt"

UA = "hidden-attractor-search/0.1 (academic bibliography verification script)"
S2 = "https://api.semanticscholar.org/graph/v1/paper/"
S2_FIELDS = "title,authors,year,venue,journal,externalIds,citationCount,abstract"


def get_json(url, tries=5, pause=1.2):
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=40) as r:
                data = json.load(r)
            time.sleep(pause)
            return data
        except urllib.error.HTTPError as e:
            if e.code == 404:
                time.sleep(pause)
                return None
            if e.code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            time.sleep(pause)
            if attempt == tries - 1:
                return None
        except Exception:
            time.sleep(2 * (attempt + 1))
    return None


def sim(a, b):
    a = (a or "").lower().strip().rstrip(".")
    b = (b or "").lower().strip().rstrip(".")
    return difflib.SequenceMatcher(None, a, b).ratio()


def crossref(doi):
    d = get_json("https://api.crossref.org/works/" + urllib.parse.quote(doi, safe=""))
    if not d or "message" not in d:
        return None
    m = d["message"]
    title = (m.get("title") or [""])[0]
    authors = []
    for a in m.get("author", []):
        fam = a.get("family", "")
        giv = a.get("given", "")
        authors.append((fam + " " + "".join(x[0] + "." for x in giv.replace("-", " ").split() if x)).strip())
    year = None
    for k in ("published-print", "published-online", "issued", "created"):
        parts = (m.get(k) or {}).get("date-parts")
        if parts and parts[0] and parts[0][0]:
            year = parts[0][0]
            break
    return {
        "title": title,
        "authors": authors,
        "year": year,
        "venue": (m.get("container-title") or [""])[0],
        "volume": m.get("volume"),
        "issue": m.get("issue"),
        "pages": m.get("page") or m.get("article-number"),
        "doi": m.get("DOI"),
        "type": m.get("type"),
        "source": "crossref",
    }


def s2_norm(p):
    if not p:
        return None
    j = p.get("journal") or {}
    ext = p.get("externalIds") or {}
    return {
        "title": p.get("title"),
        "authors": [a.get("name") for a in p.get("authors", [])],
        "year": p.get("year"),
        "venue": p.get("venue") or j.get("name"),
        "volume": j.get("volume"),
        "issue": None,
        "pages": j.get("pages"),
        "doi": ext.get("DOI"),
        "arxiv": ext.get("ArXiv"),
        "citations": p.get("citationCount"),
        "abstract": p.get("abstract"),
        "source": "semanticscholar",
    }


def s2_by_id(ident):
    return s2_norm(get_json(S2 + ident + "?fields=" + S2_FIELDS))


def s2_search(title):
    q = urllib.parse.quote(title)
    d = get_json(S2 + "search?query=" + q + "&limit=3&fields=" + S2_FIELDS)
    if not d or not d.get("data"):
        return None
    best = max(d["data"], key=lambda p: sim(p.get("title"), title))
    return s2_norm(best)


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else SRC
    cands = json.loads(src.read_text(encoding="utf-8-sig"))
    # режим слияния: уже проверенные записи из OUT сохраняются, новые добавляются/заменяются по key
    prev = []
    if OUT.exists() and src != SRC:
        prev = json.loads(OUT.read_text(encoding="utf-8-sig"))
    new_keys = {c["key"] for c in cands}
    out = [r for r in prev if r.get("key") not in new_keys]
    lines = []
    for c in cands:
        rec = dict(c)
        found = None
        if c.get("doi"):
            found = crossref(c["doi"])
        if (not found or sim(found["title"], c["title"]) < 0.6) and c.get("arxiv"):
            f2 = s2_by_id("arXiv:" + c["arxiv"])
            if f2 and sim(f2["title"], c["title"]) >= sim((found or {}).get("title"), c["title"]):
                found = f2
        if (not found or sim(found["title"], c["title"]) < 0.6) and c.get("doi"):
            f2 = s2_by_id("DOI:" + c["doi"])
            if f2 and sim(f2["title"], c["title"]) >= sim((found or {}).get("title"), c["title"]):
                found = f2
        if not found or sim(found["title"], c["title"]) < 0.6:
            f2 = s2_search(c["title"])
            if f2 and sim(f2["title"], c["title"]) >= sim((found or {}).get("title"), c["title"]):
                found = f2
        if (not found or sim(found["title"], c["title"]) < 0.6) and c.get("manual"):
            # ручная верификация: метаданные сняты напрямую со страницы arXiv/издателя
            found = dict(c["manual"])
            found["source"] = "manual:" + c.get("manual_note", "page checked")
        if found:
            s = sim(found["title"], c["title"])
            rec["title_input"] = c["title"]
            rec.update({k: v for k, v in found.items() if v is not None})
            rec["title_similarity"] = round(s, 3)
            rec["verified"] = bool(s >= 0.75)
        else:
            rec["verified"] = False
            rec["title_similarity"] = 0.0
        # S2 abstract/citations for DOI-found records too (cheap enrichment)
        if rec["verified"] and found.get("source") == "crossref":
            ident = "DOI:" + rec["doi"]
            extra = s2_by_id(ident)
            if extra:
                if extra.get("citations") is not None:
                    rec["citations"] = extra["citations"]
                if extra.get("abstract"):
                    rec["abstract"] = extra["abstract"]
                if extra.get("arxiv"):
                    rec["arxiv"] = extra["arxiv"]
        status = "OK " if rec["verified"] else "FAIL"
        line = f"{status} {c['key']:32s} sim={rec['title_similarity']:.2f} src={rec.get('source','-'):15s} doi={rec.get('doi')}  {rec.get('year')}  {str(rec.get('venue'))[:40]}"
        print(line, flush=True)
        lines.append(line)
        out.append(rec)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    nok = sum(1 for r in out if r["verified"])
    summary = f"\nVERIFIED {nok}/{len(out)}"
    print(summary)
    lines.append(summary)
    REP.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
