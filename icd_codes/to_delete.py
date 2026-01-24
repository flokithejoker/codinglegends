import re, zipfile, tempfile, argparse
from pathlib import Path
from datetime import datetime
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

CMS = "https://www.cms.gov/medicare/coding-billing/icd-10-codes"
ROOT = Path.home() / "ICD CODING" / "icd_codes"
YEAR_RE = re.compile(r"20\d{2}")
MULT = 3

def hb(n):
    for u in ["B","KB","MB","GB","TB"]:
        if n < 1024: return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}PB"

def main(force=False):
    s = requests.Session()
    soup = BeautifulSoup(s.get(CMS, timeout=60).text, "html.parser")
    urls = [urljoin("https://www.cms.gov", a["href"]) for a in soup.find_all("a", href=True)
            if a["href"].lower().endswith((".zip", ".pdf"))]

    yrs = [int(y) for u in urls for y in YEAR_RE.findall(u)]
    y = max(yy for yy in yrs if 2015 <= yy <= datetime.now().year + 1)
    items = [u for u in urls if str(y) in u]
    out = ROOT / str(y)

    print("Destination:", out)
    print("Latest year:", y, "| items:", len(items))

    if out.exists() and any(out.iterdir()) and not force:
        print("Already downloaded (folder non-empty). Use --force to redo.")
        return

    total = 0; unk = 0
    for u in items:
        try:
            cl = s.head(u, allow_redirects=True, timeout=30).headers.get("Content-Length")
            total += int(cl) if cl and cl.isdigit() else 0
            unk += 0 if (cl and cl.isdigit()) else 1
        except Exception:
            unk += 1

    if total:
        print("Rough size:")
        print("  download ~", hb(total))
        print("  extracted+overhead ~", hb(total * MULT), f"(≈{MULT}×)")
        print("  combined ~", hb(total * (MULT + 1)))
    if unk:
        print(f"(note: {unk} item(s) had unknown size)")

    if input("Download + extract now? [y/N] ").strip().lower() not in ("y","yes"):
        return

    out.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for i, u in enumerate(items, 1):
            name = Path(u.split("?")[0]).name
            print(f"[{i}/{len(items)}] {name}")
            data = s.get(u, timeout=180).content
            if name.lower().endswith(".pdf"):
                (out / name).write_bytes(data)
            else:
                zp = td / name
                zp.write_bytes(data)
                zipfile.ZipFile(zp).extractall(out)

    print("Done:", out)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true")
    main(p.parse_args().force)