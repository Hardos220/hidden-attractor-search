"""Deposit a snapshot of the article repository in Zenodo and obtain its DOI.

    python src/tools/zenodo_deposit.py reserve  --zip hidden-attractor-search-v1.0.0.zip
    python src/tools/zenodo_deposit.py publish  --id <deposition id>

`reserve` creates a new deposition with a pre-reserved DOI, uploads the archive, sets the metadata of
.zenodo.json from the article repository and prints the DOI (10.5281/zenodo.N) without publishing, so that
the DOI can be written into the manuscript and the README first. `publish` makes the record public; after
that the files cannot be changed (a new version can be added).

The personal access token (scopes deposit:write and deposit:actions, created at
https://zenodo.org/account/settings/applications/tokens/new/) is read from the environment variable
ZENODO_TOKEN or from the file given by --token-file; it is never printed. With --sandbox the request goes
to https://sandbox.zenodo.org (separate account and token) for a dry run.
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO = Path(os.environ.get("ARTICLE_REPO", r"C:\Cursor\hidden-attractor-search"))


def api(base, token, method, path, data=None, raw=None, ctype="application/json"):
    url = base + path + ("&" if "?" in path else "?") + "access_token=" + urllib.parse.quote(token)
    body = raw if raw is not None else (json.dumps(data).encode() if data is not None else None)
    req = urllib.request.Request(url, data=body, method=method)
    if body is not None:
        req.add_header("Content-Type", ctype)
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            txt = r.read().decode()
            return json.loads(txt) if txt else {}
    except urllib.error.HTTPError as e:
        msg = e.read().decode(errors="replace")
        raise SystemExit(f"Zenodo HTTP {e.code} on {method} {path}: {msg[:800]}")


def token_of(args):
    if args.token_file:
        return Path(args.token_file).read_text(encoding="utf-8").strip()
    t = os.environ.get("ZENODO_TOKEN", "").strip()
    if not t:
        raise SystemExit("no token: set ZENODO_TOKEN or pass --token-file")
    return t


def cmd_reserve(args):
    base = "https://sandbox.zenodo.org/api" if args.sandbox else "https://zenodo.org/api"
    token = token_of(args)
    meta = json.loads((REPO / ".zenodo.json").read_text(encoding="utf-8"))
    meta.setdefault("version", args.version)
    meta["prereserve_doi"] = True
    dep = api(base, token, "POST", "/deposit/depositions", data={"metadata": meta})
    dep_id = dep["id"]
    doi = dep["metadata"]["prereserve_doi"]["doi"]
    print(f"deposition id: {dep_id}")
    print(f"reserved DOI:  {doi}")
    zpath = Path(args.zip)
    bucket = dep["links"]["bucket"]
    with open(zpath, "rb") as f:
        raw = f.read()
    url = bucket + "/" + urllib.parse.quote(zpath.name) + "?access_token=" + urllib.parse.quote(token)
    req = urllib.request.Request(url, data=raw, method="PUT")
    req.add_header("Content-Type", "application/octet-stream")
    with urllib.request.urlopen(req, timeout=1800) as r:
        info = json.loads(r.read().decode())
    print(f"uploaded: {info.get('key')} ({info.get('size', 0) / 1e6:.1f} MB)")
    state = {"id": dep_id, "doi": doi, "sandbox": bool(args.sandbox), "zip": str(zpath), "version": args.version}
    out = ROOT / "results" / "zenodo_deposit.json"
    out.write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(f"state written to {out}; publish with: python src/tools/zenodo_deposit.py publish --id {dep_id}"
          + (" --sandbox" if args.sandbox else ""))


def cmd_upload(args):
    """Replace the files of an unpublished deposition by the given archive (after the DOI was written into it)."""
    base = "https://sandbox.zenodo.org/api" if args.sandbox else "https://zenodo.org/api"
    token = token_of(args)
    dep = api(base, token, "GET", f"/deposit/depositions/{args.id}")
    for f in dep.get("files", []):
        api(base, token, "DELETE", f"/deposit/depositions/{args.id}/files/{f['id']}")
        print(f"deleted: {f['filename']}")
    zpath = Path(args.zip)
    url = dep["links"]["bucket"] + "/" + urllib.parse.quote(zpath.name) + "?access_token=" + urllib.parse.quote(token)
    req = urllib.request.Request(url, data=zpath.read_bytes(), method="PUT")
    req.add_header("Content-Type", "application/octet-stream")
    with urllib.request.urlopen(req, timeout=1800) as r:
        info = json.loads(r.read().decode())
    print(f"uploaded: {info.get('key')} ({info.get('size', 0) / 1e6:.1f} MB)")


def cmd_publish(args):
    base = "https://sandbox.zenodo.org/api" if args.sandbox else "https://zenodo.org/api"
    token = token_of(args)
    rec = api(base, token, "POST", f"/deposit/depositions/{args.id}/actions/publish")
    print(f"published: DOI {rec.get('doi')}  record {rec.get('links', {}).get('record_html') or rec.get('links', {}).get('html')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("reserve")
    r.add_argument("--zip", required=True)
    r.add_argument("--version", default="1.0.0")
    u = sub.add_parser("upload")
    u.add_argument("--id", required=True)
    u.add_argument("--zip", required=True)
    p = sub.add_parser("publish")
    p.add_argument("--id", required=True)
    for s in (r, u, p):
        s.add_argument("--sandbox", action="store_true")
        s.add_argument("--token-file")
    args = ap.parse_args()
    {"reserve": cmd_reserve, "upload": cmd_upload, "publish": cmd_publish}[args.cmd](args)


if __name__ == "__main__":
    main()
