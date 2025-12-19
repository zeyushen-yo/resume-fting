#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import requests

try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:
    BeautifulSoup = None  # fallback to regex if bs4 unavailable


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Augment pairs JSONL with full job descriptions")
    p.add_argument("--input", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--timeout", type=int, default=20)
    return p.parse_args()


def greenhouse_board_slug(company: str) -> str:
    return re.sub(r"[^a-z0-9]", "", company.strip().lower())


def guess_greenhouse_url(company: str, gh_id: str) -> str:
    slug = greenhouse_board_slug(company)
    return f"https://boards.greenhouse.io/{slug}/jobs/{gh_id}"


def html_to_text(html: str) -> str:
    if not html:
        return ""
    if BeautifulSoup is not None:
        soup = BeautifulSoup(html, "html.parser")
        # Remove script/style
        for tag in soup(["script", "style", "noscript"]):
            tag.extract()
        text = soup.get_text("\n")
        return re.sub(r"\n\s*\n+", "\n\n", text).strip()
    # fallback: strip tags
    txt = re.sub(r"<script[\s\S]*?</script>", " ", html, flags=re.I)
    txt = re.sub(r"<style[\s\S]*?</style>", " ", txt, flags=re.I)
    txt = re.sub(r"<[^>]+>", " ", txt)
    txt = re.sub(r"\s+", " ", txt)
    return txt.strip()


def fetch_job_description(url: str, company: Optional[str], timeout: int) -> str:
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
    })

    def try_fetch(target_url: str) -> str:
        try:
            resp = session.get(target_url, timeout=timeout, allow_redirects=True)
            if resp.ok and resp.text:
                txt = html_to_text(resp.text)
                if len(txt.split()) > 50:
                    return txt
        except Exception:
            return ""
        return ""

    # Prefer Greenhouse boards URL when gh_jid is present
    m = re.search(r"gh_jid=(\d+)", url)
    if m and company:
        gh_id = m.group(1)
        gh_url = guess_greenhouse_url(company, gh_id)
        txt = try_fetch(gh_url)
        if txt:
            return txt

    # Fallback to original URL
    txt = try_fetch(url)
    if txt:
        return txt

    # Last attempt: if gh_jid present but company not known, try a generic boards URL
    if m:
        gh_id = m.group(1)
        generic = f"https://boards.greenhouse.io/jobs/{gh_id}"
        txt = try_fetch(generic)
        if txt:
            return txt
    return ""


def augment_file(input_path: Path, output_path: Path, timeout: int = 20) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen: Dict[str, str] = {}
    written = 0
    processed = 0
    added_jd = 0
    with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                row: Dict[str, Any] = json.loads(line)
            except Exception:
                continue
            processed += 1
            if row.get("job_description"):
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
                if processed % 50 == 0:
                    print(f"[{input_path.name}] processed={processed} added_jd={added_jd} (already had JD)")
                    sys.stdout.flush()
                continue
            js = row.get("job_source") or {}
            url = js.get("url") or ""
            company = js.get("company") or ""
            jd_text = ""
            key = f"{company}|{url}"
            if key in seen:
                jd_text = seen[key]
            else:
                jd_text = fetch_job_description(url, company, timeout)
                seen[key] = jd_text
            if jd_text:
                row["job_description"] = jd_text
                added_jd += 1
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
            if processed % 50 == 0:
                print(f"[{input_path.name}] processed={processed} added_jd={added_jd}")
                sys.stdout.flush()
    print(f"Wrote {written} lines ➜ {output_path}")


def main() -> None:
    args = parse_args()
    augment_file(Path(args.input), Path(args.output), timeout=args.timeout)


if __name__ == "__main__":
    main()


