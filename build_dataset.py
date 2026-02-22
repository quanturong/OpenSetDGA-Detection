import argparse
import csv
from datetime import datetime
import glob
import io
import json
import os
import random
import re
import shutil
import string
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import pandas as pd
import requests
from urllib.parse import urlparse, quote_plus

try:
    import tldextract

    _EXTRACT = tldextract.TLDExtract(cache_dir=".tldextract_cache", suffix_list_urls=None)
except Exception:
    _EXTRACT = None

DOMAIN_RE = re.compile(
    r"^(?=.{1,253}$)(?!-)[a-z0-9-]{1,63}(?<!-)(\.(?!-)[a-z0-9-]{1,63}(?<!-))+$"
)

DEFAULT_OOD_FEEDS = [
    "https://urlhaus.abuse.ch/downloads/text_online/",
    "https://openphish.com/feed.txt",
    "https://raw.githubusercontent.com/hagezi/dns-blocklists/main/domains/multi.txt",
    "https://phishing.army/download/phishing_army_blocklist.txt",
    "https://feodotracker.abuse.ch/downloads/domainblocklist.txt",
    "https://raw.githubusercontent.com/stamparm/blackbook/master/blackbook.txt",
]


def extract_domain(s: str) -> Optional[str]:
    if not isinstance(s, str):
        return None
    s = s.strip().lower()
    if not s:
        return None
    if "://" in s:
        try:
            u = urlparse(s)
            s = (u.hostname or "").lower()
        except Exception:
            return None
    s = s.strip().rstrip(".")
    if s.startswith("*."):
        s = s[2:]
    return s or None


def is_valid_domain(d: str) -> bool:
    if not d or "." not in d:
        return False
    return bool(DOMAIN_RE.match(d))


def etld_plus_one(d: str) -> Optional[str]:
    """Normalize to eTLD+1 when possible."""
    if not d:
        return None
    if _EXTRACT is None:
        parts = d.split(".")
        return ".".join(parts[-2:]) if len(parts) >= 2 else None

    ext = _EXTRACT(d)
    if not ext.domain or not ext.suffix:
        return None
    return f"{ext.domain}.{ext.suffix}"


def norm_family(name: str) -> str:
    if not isinstance(name, str):
        return "unknown"
    name = name.strip().lower()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^a-z0-9_+\-]", "", name)
    return name or "unknown"


def download_file(url: str, out_path: str, timeout: int = 60) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout, headers={"User-Agent": "dataset-builder/1.0"}) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def try_git_clone(repo_url: str, dest_dir: str) -> bool:
    try:
        if os.path.isdir(dest_dir) and os.listdir(dest_dir):
            return True
        os.makedirs(os.path.dirname(dest_dir), exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", repo_url, dest_dir], check=True)
        return True
    except Exception:
        return False


def ensure_chrmor_25dga(cache_dir: str) -> str:
    """
    Gets chrmor/DGA_domains_dataset and returns path to dga_domains_full.csv
    Prefer ZIP download; fallback to git clone.
    """
    os.makedirs(cache_dir, exist_ok=True)
    zip_path = os.path.join(cache_dir, "chrmor_25dga.zip")
    repo_dir = os.path.join(cache_dir, "chrmor_25dga_repo")
    csv_path = os.path.join(repo_dir, "dga_domains_full.csv")

    if os.path.isfile(csv_path):
        return csv_path

    zip_url = "https://github.com/chrmor/DGA_domains_dataset/archive/refs/heads/master.zip"
    try:
        if not os.path.isfile(zip_path):
            print(f"[DL] {zip_url}")
            download_file(zip_url, zip_path)
        if os.path.isdir(repo_dir):
            shutil.rmtree(repo_dir, ignore_errors=True)
        os.makedirs(repo_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(repo_dir)
        candidates = glob.glob(os.path.join(repo_dir, "**", "dga_domains_full.csv"), recursive=True)
        if not candidates:
            raise FileNotFoundError("dga_domains_full.csv not found inside zip")
        return candidates[0]
    except Exception as e:
        print(f"[WARN] ZIP method failed: {e}")

    ok = try_git_clone("https://github.com/chrmor/DGA_domains_dataset", repo_dir)
    if not ok:
        raise RuntimeError("Cannot obtain chrmor/DGA_domains_dataset (zip + git clone failed).")
    if not os.path.isfile(csv_path):
        candidates = glob.glob(os.path.join(repo_dir, "**", "dga_domains_full.csv"), recursive=True)
        if not candidates:
            raise FileNotFoundError("dga_domains_full.csv not found after git clone")
        return candidates[0]
    return csv_path


def ensure_tranco_top1m(cache_dir: str) -> str:
    """
    Downloads Tranco top-1m CSV zip and returns extracted CSV path.
    """
    os.makedirs(cache_dir, exist_ok=True)
    zip_url = "https://tranco-list.eu/top-1m.csv.zip"
    zip_path = os.path.join(cache_dir, "tranco_top-1m.csv.zip")
    csv_path = os.path.join(cache_dir, "tranco_top-1m.csv")

    if os.path.isfile(csv_path):
        return csv_path

    if not os.path.isfile(zip_path):
        print(f"[DL] {zip_url}")
        download_file(zip_url, zip_path)

    with zipfile.ZipFile(zip_path, "r") as z:
        names = z.namelist()
        target = None
        for n in names:
            if n.endswith(".csv"):
                target = n
                break
        if target is None:
            raise FileNotFoundError("No CSV in Tranco zip.")
        with z.open(target) as f_in, open(csv_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    return csv_path


def load_tranco(csv_path: str, n: int, seed: int = 42) -> pd.DataFrame:
    """
    Tranco format commonly: rank,domain
    We'll stratify sample by rank deciles to reduce 'too top-heavy' bias.
    """
    df = pd.read_csv(csv_path, header=None, names=["rank", "domain"])
    df["domain"] = df["domain"].astype(str)

    df = df.dropna(subset=["domain"]).copy()
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    df = df.dropna(subset=["rank"]).copy()
    df["decile"] = pd.qcut(df["rank"], 10, labels=False, duplicates="drop")

    rng = random.Random(seed)
    selected_idx = set()
    per = max(1, n // df["decile"].nunique())
    for d in sorted(df["decile"].unique()):
        chunk = df[df["decile"] == d]
        take = min(per, len(chunk))
        if take <= 0:
            continue
        idx = rng.sample(list(chunk.index), k=take)
        selected_idx.update(idx)

    out = df.loc[list(selected_idx), ["domain"]].copy() if selected_idx else pd.DataFrame(columns=["domain"])
    if len(out) < n:
        remain = n - len(out)
        remaining_idx = df.index.difference(list(selected_idx))
        if len(remaining_idx) > 0:
            rest = df.loc[remaining_idx, ["domain"]].sample(
                n=min(remain, len(remaining_idx)),
                random_state=seed,
            )
            out = pd.concat([out, rest], ignore_index=True)

    out = out.head(n).copy()
    out["family"] = "benign"
    out["source"] = "tranco"
    out["label"] = "benign"
    return out


def load_chrmor_25dga(dga_csv: str) -> pd.DataFrame:
    """
    chrmor/DGA_domains_dataset (dga_domains_full.csv) thường có format:
      label,family,domain
    ví dụ: dga,gozi,mortiscontrastatim.com
    """
    df = pd.read_csv(dga_csv, dtype=str)
    cols_norm = {str(c).strip().lower(): c for c in df.columns}

    if {"label", "family", "domain"}.issubset(set(cols_norm.keys())):
        out = pd.DataFrame({
            "label": df[cols_norm["label"]].astype(str).str.strip().str.lower(),
            "family": df[cols_norm["family"]].astype(str),
            "domain": df[cols_norm["domain"]].astype(str),
            "source": "chrmor_25dga"
        })
    else:
        df = pd.read_csv(dga_csv, header=None, dtype=str)
        if df.shape[1] < 3:
            raise ValueError(f"Unexpected CSV shape {df.shape}. Expected at least 3 columns: label,family,domain")

        out = pd.DataFrame({
            "label": df.iloc[:, 0].astype(str).str.strip().str.lower(),
            "family": df.iloc[:, 1].astype(str),
            "domain": df.iloc[:, 2].astype(str),
            "source": "chrmor_25dga"
        })

    out = out[~(
        out["label"].str.lower().eq("label")
        & out["family"].str.lower().eq("family")
        & out["domain"].str.lower().eq("domain")
    )].copy()

    benign_tokens = {"benign", "legit", "legitimate", "normal", "alexa"}
    out["label"] = out["label"].apply(lambda x: "benign" if x in benign_tokens else "dga")

    out["family"] = out["family"].map(norm_family)
    out.loc[out["label"] == "benign", "family"] = out.loc[out["label"] == "benign", "family"].replace({"unknown": "benign"})

    return out[["domain", "family", "source", "label"]]


def load_360netlab_suspicious(cache_dir: str, max_rows: int, seed: int = 42) -> pd.DataFrame:
    """
    Optional OOD malicious-ish: 360netlab/DGA repo (no family labels).
    We'll download as zip and parse all *.txt/*.csv to extract first token per line.
    """
    os.makedirs(cache_dir, exist_ok=True)
    zip_url = "https://github.com/360netlab/DGA/archive/refs/heads/master.zip"
    zip_path = os.path.join(cache_dir, "360netlab_dga.zip")
    repo_dir = os.path.join(cache_dir, "360netlab_dga_repo")

    if not os.path.isfile(zip_path):
        print(f"[DL] {zip_url}")
        try:
            download_file(zip_url, zip_path)
        except Exception as e:
            print(f"[WARN] 360netlab download failed: {e}")
            return pd.DataFrame(columns=["domain", "family", "source", "label"])

    if os.path.isdir(repo_dir):
        shutil.rmtree(repo_dir, ignore_errors=True)
    os.makedirs(repo_dir, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(repo_dir)
    except Exception as e:
        print(f"[WARN] 360netlab unzip failed: {e}")
        return pd.DataFrame(columns=["domain", "family", "source", "label"])

    patterns = [
        os.path.join(repo_dir, "**", "*.txt"),
        os.path.join(repo_dir, "**", "*.csv"),
        os.path.join(repo_dir, "**", "*.json"),
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(p, recursive=True))

    def extract_candidates_from_line(line: str):
        if not line:
            return []
        line = line.strip()
        if not line or line.startswith("#"):
            return []

        pieces = [line]
        pieces.extend(re.split(r"[\s,;|\t]+", line))

        out = []
        for p in pieces:
            p = p.strip().strip("\"'()[]{}<>")
            if not p:
                continue
            d = extract_domain(p)
            if d and is_valid_domain(d):
                out.append(d)
        return out

    rows = set()
    for f in files:
        low = f.lower()
        try:
            with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                if low.endswith(".json"):
                    try:
                        obj = json.load(fh)
                    except Exception:
                        fh.seek(0)
                        obj = None

                    if obj is not None:
                        stack = [obj]
                        while stack:
                            cur = stack.pop()
                            if isinstance(cur, dict):
                                stack.extend(cur.values())
                            elif isinstance(cur, list):
                                stack.extend(cur)
                            elif isinstance(cur, str):
                                for d in extract_candidates_from_line(cur):
                                    rows.add(d)
                        continue

                for line in fh:
                    for d in extract_candidates_from_line(line):
                        rows.add(d)
        except Exception:
            continue

    rows = list(rows)
    rng = random.Random(seed)
    if len(rows) > max_rows:
        rows = rng.sample(rows, k=max_rows)

    out = pd.DataFrame({"domain": rows})
    out["family"] = "ood_suspicious"
    out["source"] = "360netlab"
    out["label"] = "ood"
    return out


def load_public_ood_feeds(feed_urls: list, max_rows: int, seed: int = 42, timeout: int = 30) -> pd.DataFrame:
    """
    Optional public OOD feed ingestion.
    Supports text/csv/json-like payloads and extracts domain candidates broadly.
    """
    if not feed_urls or max_rows <= 0:
        return pd.DataFrame(columns=["domain", "family", "source", "label"])

    def extract_candidates_from_text(text: str):
        if not isinstance(text, str) or not text.strip():
            return []
        out = []
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            pieces = [line]
            pieces.extend(re.split(r"[\s,;|\t]+", line))
            for p in pieces:
                p = p.strip().strip("\"'()[]{}<>")
                if not p:
                    continue
                d = extract_domain(p)
                if d and is_valid_domain(d):
                    out.append(d)
        return out

    def feed_source_name(url: str) -> str:
        p = urlparse(url)
        host = (p.netloc or "public_feed").lower()
        path_parts = [x for x in (p.path or "").split("/") if x]
        if host == "raw.githubusercontent.com" and len(path_parts) >= 2:
            return f"{host}:{path_parts[0]}/{path_parts[1]}"
        if path_parts:
            return f"{host}:{path_parts[-1].lower()}"
        return host

    rows = []
    for url in feed_urls:
        try:
            r = requests.get(url, timeout=timeout, headers={"User-Agent": "dataset-builder/1.0"})
            if r.status_code != 200:
                print(f"[WARN] Feed unavailable ({r.status_code}): {url}")
                continue

            source_name = feed_source_name(url)
            text = r.text

            parsed_json = None
            try:
                parsed_json = r.json()
            except Exception:
                parsed_json = None

            if parsed_json is not None:
                stack = [parsed_json]
                while stack:
                    cur = stack.pop()
                    if isinstance(cur, dict):
                        stack.extend(cur.values())
                    elif isinstance(cur, list):
                        stack.extend(cur)
                    elif isinstance(cur, str):
                        for d in extract_candidates_from_text(cur):
                            rows.append((d, source_name))
            else:
                for d in extract_candidates_from_text(text):
                    rows.append((d, source_name))
        except Exception as e:
            print(f"[WARN] Feed load failed: {url} ({e})")

    seen = set()
    dedup_rows = []
    for d, s in rows:
        key = (d, s)
        if key in seen:
            continue
        seen.add(key)
        dedup_rows.append((d, s))

    rng = random.Random(seed)
    if len(dedup_rows) > max_rows:
        dedup_rows = rng.sample(dedup_rows, k=max_rows)

    out = pd.DataFrame(dedup_rows, columns=["domain", "source"]) if dedup_rows else pd.DataFrame(columns=["domain", "source"])
    out["family"] = "ood_public_feed"
    out["label"] = "ood"
    return out[["domain", "family", "source", "label"]]


def _load_json_dict(path: str) -> Dict:
    if not path or not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _save_json_dict(path: str, obj: Dict) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, path)


def _load_success_queries(path: str) -> list:
    if not path or not os.path.isfile(path):
        return []
    out = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                q = line.strip()
                if q:
                    out.append(q)
    except Exception:
        return []
    return list(dict.fromkeys(out))


def _save_success_queries(path: str, queries: list) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    unique = list(dict.fromkeys([str(x).strip() for x in queries if str(x).strip()]))
    with open(path, "w", encoding="utf-8") as f:
        for q in unique:
            f.write(q + "\n")


def build_crt_query_seeds(tranco_csv: str, max_seeds: int = 5000, seed: int = 42) -> list:
    if max_seeds <= 0:
        return []
    try:
        df = pd.read_csv(tranco_csv, header=None, names=["rank", "domain"], dtype=str)
        df = df.dropna(subset=["domain"]).copy()
        df["domain"] = df["domain"].map(extract_domain)
        df = df[df["domain"].notna()].copy()
        domains = df["domain"].drop_duplicates().tolist()
        rng = random.Random(seed)
        if len(domains) > max_seeds:
            domains = rng.sample(domains, k=max_seeds)
        return domains
    except Exception:
        return []


def crawl_crtsh_ood(
    n: int,
    seed: int = 42,
    max_attempts: int = 1200,
    request_timeout: float = 30.0,
    sleep_s: float = 0.25,
    query_seeds: Optional[list] = None,
    fail_stop: int = 200,
    cache_path: Optional[str] = None,
    success_queries_path: Optional[str] = None,
    backoff_base: float = 1.0,
    backoff_max: float = 30.0,
) -> Tuple[pd.DataFrame, Dict]:
    rng = random.Random(seed)
    default_query_seeds = [
        "test.com", "app.com", "api.com", "mail.com", "shop.com", "cloud.com", "online.com",
        "service.com", "secure.com", "update.com", "cdn.com", "host.com", "login.com",
        "verify.com", "portal.com", "account.com", "global.com", "network.com", "world.com",
        "group.com", "support.com", "system.com", "office.com", "tech.com", "digital.com",
    ]
    successful_prev = _load_success_queries(success_queries_path or "")
    query_pool = list(dict.fromkeys(successful_prev + (query_seeds or []) + default_query_seeds))
    if not query_pool:
        query_pool = default_query_seeds
    head = successful_prev[:]
    tail = [q for q in query_pool if q not in set(head)]
    rng.shuffle(tail)
    query_pool = head + tail

    qcache = _load_json_dict(cache_path or "")

    got = set()
    fail_streak = 0
    attempts = 0
    success_queries_run = []
    fail_reason_counts: Dict[str, int] = {}
    cache_hits = 0

    def mark_fail(reason: str):
        nonlocal fail_streak
        fail_streak += 1
        fail_reason_counts[reason] = fail_reason_counts.get(reason, 0) + 1

    for q in query_pool:
        if attempts >= max_attempts:
            break
        if len(got) >= n:
            break

        attempts += 1

        cached = qcache.get(q)
        if isinstance(cached, dict) and cached.get("status") == "ok" and isinstance(cached.get("domains"), list):
            cache_hits += 1
            before = len(got)
            for d in cached.get("domains", []):
                if d and is_valid_domain(d):
                    got.add(d)
                    if len(got) >= n:
                        break
            if len(got) > before:
                fail_streak = 0
                success_queries_run.append(q)
                continue

        url = f"https://crt.sh/?q={quote_plus(q)}&output=json"

        try:
            r = requests.get(url, timeout=request_timeout, headers={"User-Agent": "dataset-builder/1.0"})
            sc = int(r.status_code)
            txt = r.text or ""

            if sc == 429:
                mark_fail("status_429")
                delay = min(backoff_max, max(3.0, backoff_base * (2 ** min(fail_streak, 8)))) + rng.uniform(0, 1.2)
                time.sleep(delay)
                qcache[q] = {"status": "fail", "reason": "status_429", "code": sc, "ts": int(time.time())}
                if fail_streak >= fail_stop:
                    break
                continue

            if 500 <= sc <= 599:
                mark_fail(f"status_{sc}")
                delay = min(backoff_max, backoff_base * (2 ** min(fail_streak, 8))) + rng.uniform(0, 0.8)
                time.sleep(delay)
                qcache[q] = {"status": "fail", "reason": f"status_{sc}", "code": sc, "ts": int(time.time())}
                if fail_streak >= fail_stop:
                    break
                continue

            if sc != 200:
                mark_fail(f"status_{sc}")
                time.sleep(sleep_s)
                qcache[q] = {"status": "fail", "reason": f"status_{sc}", "code": sc, "ts": int(time.time())}
                if fail_streak >= fail_stop:
                    break
                continue

            if "Unsupported use" in txt:
                mark_fail("unsupported_use")
                time.sleep(sleep_s)
                qcache[q] = {"status": "fail", "reason": "unsupported_use", "code": sc, "ts": int(time.time())}
                if fail_streak >= fail_stop:
                    break
                continue

            try:
                data = r.json()
            except Exception:
                mark_fail("non_json")
                time.sleep(sleep_s)
                qcache[q] = {
                    "status": "fail",
                    "reason": "non_json",
                    "code": sc,
                    "snippet": txt[:120],
                    "ts": int(time.time()),
                }
                if fail_streak >= fail_stop:
                    break
                continue

            if isinstance(data, dict):
                data = [data]
            if not isinstance(data, list):
                mark_fail("non_json_list")
                time.sleep(sleep_s)
                qcache[q] = {"status": "fail", "reason": "non_json_list", "code": sc, "ts": int(time.time())}
                if fail_streak >= fail_stop:
                    break
                continue

            before = len(got)
            q_domains = set()
            for item in data:
                if not isinstance(item, dict):
                    continue
                nv = item.get("name_value")
                if not nv:
                    continue
                for name in str(nv).splitlines():
                    d = extract_domain(name)
                    if d and is_valid_domain(d):
                        q_domains.add(d)
                        got.add(d)
                        if len(got) >= n:
                            break
                if len(got) >= n:
                    break

            if len(got) > before:
                fail_streak = 0
                success_queries_run.append(q)
                qcache[q] = {
                    "status": "ok",
                    "domains": sorted(list(q_domains))[:50000],
                    "count": len(q_domains),
                    "ts": int(time.time()),
                }
            else:
                mark_fail("no_domains")
                qcache[q] = {
                    "status": "fail",
                    "reason": "no_domains",
                    "code": sc,
                    "ts": int(time.time()),
                }
                if fail_streak >= fail_stop:
                    break
        except requests.exceptions.Timeout:
            mark_fail("timeout")
            delay = min(backoff_max, backoff_base * (2 ** min(fail_streak, 8))) + rng.uniform(0, 0.8)
            time.sleep(delay)
            qcache[q] = {"status": "fail", "reason": "timeout", "ts": int(time.time())}
            if fail_streak >= fail_stop:
                break
            continue
        except requests.exceptions.RequestException as e:
            mark_fail("request_exception")
            delay = min(backoff_max, backoff_base * (2 ** min(fail_streak, 8))) + rng.uniform(0, 0.8)
            time.sleep(delay)
            qcache[q] = {
                "status": "fail",
                "reason": "request_exception",
                "detail": type(e).__name__,
                "ts": int(time.time()),
            }
            if fail_streak >= fail_stop:
                break
            continue
        except Exception as e:
            mark_fail("unexpected_exception")
            time.sleep(sleep_s)
            qcache[q] = {
                "status": "fail",
                "reason": "unexpected_exception",
                "detail": type(e).__name__,
                "ts": int(time.time()),
            }
            if fail_streak >= fail_stop:
                break
            continue

        time.sleep(sleep_s)

    _save_json_dict(cache_path or "", qcache)
    merged_success = list(dict.fromkeys(_load_success_queries(success_queries_path or "") + success_queries_run))
    _save_success_queries(success_queries_path or "", merged_success)

    out = pd.DataFrame({"domain": list(got)})
    out["family"] = "ood_ct"
    out["source"] = "crtsh"
    out["label"] = "ood"
    stats = {
        "attempts": attempts,
        "cache_hits": cache_hits,
        "fail_streak": fail_streak,
        "fail_stop_hit": fail_streak >= fail_stop,
        "fail_reason_counts": fail_reason_counts,
        "success_queries_run": len(success_queries_run),
    }
    return out, stats


def build_tranco_tail_pool(tranco_csv: str, tail_start_ratio: float = 0.7) -> pd.DataFrame:
    """Build a clean, deduplicated OOD fallback pool from Tranco tail."""
    tr_tail = pd.read_csv(tranco_csv, header=None, names=["rank", "domain"])
    tr_tail["rank"] = pd.to_numeric(tr_tail["rank"], errors="coerce")
    tr_tail = tr_tail.dropna(subset=["rank"]).sort_values("rank")
    start_idx = int(len(tr_tail) * tail_start_ratio)
    tail = tr_tail.iloc[start_idx:][["domain"]].copy()
    tail["family"] = "ood_tranco_tail"
    tail["source"] = "tranco_tail"
    tail["label"] = "ood"
    tail = clean_domains(tail)
    return tail.drop_duplicates(subset=["domain"], keep="first")


def clean_domains(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df.columns = [str(c).strip().lower() for c in df.columns]

    if "domain" not in df.columns:
        raise ValueError("Input df must have column 'domain'")
    if "family" not in df.columns:
        df["family"] = "unknown"
    if "source" not in df.columns:
        df["source"] = "unknown"
    if "label" not in df.columns:
        df["label"] = "unknown"

    df["domain"] = df["domain"].map(extract_domain)
    df = df[df["domain"].notna()].copy()

    df["domain"] = df["domain"].map(etld_plus_one)
    df = df[df["domain"].notna()].copy()

    df = df[df["domain"].map(is_valid_domain)].copy()
    if "family" not in df.columns:
        df["family"] = "unknown"
    df["family"] = df["family"].fillna("unknown").map(norm_family)
    if "source" not in df.columns:
        df["source"] = "unknown"
    if "label" not in df.columns:
        df["label"] = "unknown"

    return df.drop_duplicates(subset=["domain", "family", "source"], keep="first")


@dataclass
class SplitConfig:
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1


def split_known(df_known: pd.DataFrame, seed: int, cfg: SplitConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assert abs((cfg.train_ratio + cfg.val_ratio + cfg.test_ratio) - 1.0) < 1e-6
    df = df_known.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(df)
    n_train = int(n * cfg.train_ratio)
    n_val = int(n * cfg.val_ratio)
    train = df.iloc[:n_train].copy()
    val = df.iloc[n_train:n_train + n_val].copy()
    test = df.iloc[n_train + n_val:].copy()
    return train, val, test


def add_open_set_metadata(df: pd.DataFrame, split_hint: str, force_unknown: bool = False) -> pd.DataFrame:
    """
    Add standardized metadata columns for open-set experiments.
    - class_label: benign / family_name / unknown
    - split_hint: known / unknown
    """
    out = df.copy()
    if force_unknown:
        out["class_label"] = "unknown"
    else:
        out["class_label"] = out.apply(
            lambda r: "benign" if str(r.get("label", "")).lower() == "benign" else str(r.get("family", "unknown")),
            axis=1,
        )
    out["split_hint"] = split_hint
    return out


def sample_exact(df: pd.DataFrame, n: int, seed: int, name: str, strict: bool = False) -> pd.DataFrame:
    """Sample exactly n rows when available; otherwise return all and optionally fail in strict mode."""
    if n <= 0:
        return df.head(0).copy()
    if len(df) >= n:
        return df.sample(n=n, random_state=seed).copy()
    msg = f"[WARN] Not enough {name}: requested={n:,}, available={len(df):,}."
    if strict:
        raise RuntimeError(msg)
    print(msg)
    return df.copy()


def dedupe_dga_by_domain(df_dga: pd.DataFrame) -> pd.DataFrame:
    if len(df_dga) == 0:
        return df_dga.copy()
    tmp = df_dga.copy()
    tmp["_fam_freq"] = tmp.groupby(["domain", "family"])["domain"].transform("size")
    tmp = tmp.sort_values(["domain", "_fam_freq", "family"], ascending=[True, False, True])
    out = tmp.drop_duplicates(subset=["domain"], keep="first").drop(columns=["_fam_freq"])
    dropped = len(df_dga) - len(out)
    if dropped > 0:
        print(f"[INFO] DGA domain-level dedup removed {dropped:,} conflicting/duplicate rows")
    return out


def split_source_cap(df: pd.DataFrame, source_prefix: str, max_count: int, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if len(df) == 0 or max_count < 0 or not source_prefix:
        return df, df.head(0).copy()
    src_col = df["source"].astype(str).str.lower()
    src_mask = src_col.str.startswith(str(source_prefix).lower())
    src_n = int(src_mask.sum())
    if src_n <= max_count:
        return df, df.head(0).copy()

    df_src = df[src_mask]
    df_other = df[~src_mask]
    if max_count <= 0:
        print(f"[INFO] Source cap removed all rows from '{source_prefix}' ({src_n:,})")
        return df_other.copy(), df_src.copy()

    keep_src = df_src.sample(n=max_count, random_state=seed)
    overflow = df_src.drop(index=keep_src.index)
    print(f"[INFO] Source cap applied to '{source_prefix}': {src_n:,} -> {max_count:,}")
    return pd.concat([df_other, keep_src], ignore_index=True), overflow.copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="dataset_out")
    ap.add_argument("--cache", default="dataset_cache")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--benign", type=int, default=150_000, help="Benign from Tranco")
    ap.add_argument("--known_dga", type=int, default=120_000, help="Known DGA (from labeled families)")
    ap.add_argument("--unknown_family", type=int, default=20_000, help="Hold-out families (ground-truth unknown DGA)")
    ap.add_argument("--ood", type=int, default=10_000, help="OOD in-the-wild from 360netlab + tranco tail")

    ap.add_argument("--holdout_families", type=int, default=5, help="How many families to hold out as 'unknown DGA'")
    ap.add_argument("--use_360netlab_fill", action="store_true", help="Use 360netlab as extra OOD fill if needed")
    ap.add_argument("--oversample_factor", type=float, default=1.35, help="Oversampling factor before clean/dedup to better hit final targets")
    ap.add_argument("--strict_targets", action="store_true", help="Fail if requested targets cannot be met")
    ap.add_argument("--disable_default_ood_feeds", action="store_true", help="Disable built-in public OOD feed URLs")
    ap.add_argument(
        "--ood_feed_url",
        action="append",
        default=[],
        help="Additional public OOD feed URL (can pass multiple times).",
    )
    ap.add_argument("--crt_max_attempts", type=int, default=1200, help="Max attempts for crt.sh crawling")
    ap.add_argument("--crt_timeout", type=float, default=30.0, help="Request timeout (seconds) for crt.sh")
    ap.add_argument("--crt_sleep", type=float, default=0.25, help="Sleep between crt.sh requests")
    ap.add_argument("--crt_seed_count", type=int, default=5000, help="How many Tranco domains to use as CRT query seeds")
    ap.add_argument("--crt_fail_stop", type=int, default=200, help="Stop CRT early after this many consecutive failures")
    ap.add_argument("--crt_backoff_base", type=float, default=1.0, help="Base seconds for exponential backoff")
    ap.add_argument("--crt_backoff_max", type=float, default=30.0, help="Max seconds for exponential backoff")
    ap.add_argument("--crt_cache_file", default="crtsh_query_cache.json", help="Disk cache file for CRT query results")
    ap.add_argument("--crt_success_file", default="crtsh_success_queries.txt", help="Disk file of successful CRT queries")
    ap.add_argument(
        "--ood_min_non_tail_ratio",
        type=float,
        default=0.0,
        help="Preferred minimum ratio of OOD from non-tail sources (crt/feed/360). Range [0,1].",
    )
    ap.add_argument(
        "--ood_min_non_tail_count",
        type=int,
        default=0,
        help="Preferred minimum absolute count from non-tail sources (crt/feed/360).",
    )
    ap.add_argument(
        "--ood_max_tail_count",
        type=int,
        default=-1,
        help="Preferred maximum count from tranco_tail in final OOD. -1 means no cap.",
    )
    ap.add_argument(
        "--ood_max_tail_ratio",
        type=float,
        default=0.35,
        help="Preferred maximum ratio from tranco_tail in [0,1].",
    )
    ap.add_argument(
        "--enforce_ood_mix",
        action="store_true",
        help="Fail run if requested OOD source-mix constraints cannot be satisfied.",
    )
    ap.add_argument(
        "--ood_cap_source",
        default="raw.githubusercontent.com",
        help="Source prefix to cap for reducing source skew (set empty to disable).",
    )
    ap.add_argument(
        "--ood_cap_source_ratio",
        type=float,
        default=0.4,
        help="Max ratio of OOD from capped source in [0,1].",
    )
    ap.add_argument(
        "--ood_cap_source_count",
        type=int,
        default=-1,
        help="Optional absolute max count for capped source; -1 to ignore.",
    )

    args = ap.parse_args()
    args.ood_min_non_tail_ratio = max(0.0, min(1.0, args.ood_min_non_tail_ratio))
    args.ood_min_non_tail_count = max(0, args.ood_min_non_tail_count)
    args.ood_max_tail_ratio = max(0.0, min(1.0, args.ood_max_tail_ratio))
    args.ood_cap_source_ratio = max(0.0, min(1.0, args.ood_cap_source_ratio))
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.cache, exist_ok=True)

    run_name_base = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_name = run_name_base
    run_idx = 1
    run_outdir = os.path.join(args.outdir, run_name)
    while os.path.exists(run_outdir):
        run_name = f"{run_name_base}_{run_idx:02d}"
        run_outdir = os.path.join(args.outdir, run_name)
        run_idx += 1
    os.makedirs(run_outdir, exist_ok=True)
    print(f"[RUN] output directory: {run_outdir}")

    tranco_csv = ensure_tranco_top1m(args.cache)
    benign_raw_n = int(max(args.benign, args.benign * args.oversample_factor))
    benign_raw_n = min(1_000_000, benign_raw_n)
    df_benign = load_tranco(tranco_csv, n=benign_raw_n, seed=args.seed)

    dga_csv = ensure_chrmor_25dga(args.cache)
    df_all = load_chrmor_25dga(dga_csv)

    df_dga = df_all[df_all["label"] == "dga"].copy()

    df_benign = clean_domains(df_benign)
    df_dga = clean_domains(df_dga)
    df_dga = dedupe_dga_by_domain(df_dga)

    families = sorted(df_dga["family"].unique().tolist())
    rng = random.Random(args.seed)
    rng.shuffle(families)
    holdout = set(families[: min(args.holdout_families, len(families))])
    known_fams = [f for f in families if f not in holdout]

    df_unknown_pool = df_dga[df_dga["family"].isin(holdout)].copy()
    df_known_pool = df_dga[df_dga["family"].isin(known_fams)].copy()

    df_unknown = sample_exact(
        df_unknown_pool,
        n=args.unknown_family,
        seed=args.seed,
        name="unknown_family",
        strict=args.strict_targets,
    )
    df_unknown["split"] = "test_unknown_family"

    df_known_dga = sample_exact(
        df_known_pool,
        n=args.known_dga,
        seed=args.seed,
        name="known_dga",
        strict=args.strict_targets,
    )
    df_known_dga["split"] = "known_pool"

    dga_domains = set(df_known_dga["domain"].tolist())
    benign_pool = df_benign[~df_benign["domain"].isin(dga_domains)].copy()
    df_benign_final = sample_exact(
        benign_pool,
        n=args.benign,
        seed=args.seed,
        name="benign",
        strict=args.strict_targets,
    )
    df_benign_final["split"] = "known_pool"

    df_known = pd.concat([df_benign_final, df_known_dga], ignore_index=True)
    df_known = df_known.drop_duplicates(subset=["domain", "label", "family"], keep="first")

    train, val, test_known = split_known(df_known, seed=args.seed, cfg=SplitConfig())
    train["split"] = "train"
    val["split"] = "val"
    test_known["split"] = "test_known"

    train = add_open_set_metadata(train, split_hint="known")
    val = add_open_set_metadata(val, split_hint="known")
    test_known = add_open_set_metadata(test_known, split_hint="known")
    df_unknown = add_open_set_metadata(df_unknown, split_hint="unknown", force_unknown=True)

    protected_domains = set(df_known["domain"].tolist()) | set(df_unknown["domain"].tolist())

    crt_query_seeds = build_crt_query_seeds(
        tranco_csv=tranco_csv,
        max_seeds=args.crt_seed_count,
        seed=args.seed,
    )
    print(f"[INFO] CRT query seeds: {len(crt_query_seeds):,}")

    crt_cache_path = args.crt_cache_file
    if not os.path.isabs(crt_cache_path):
        crt_cache_path = os.path.join(args.cache, crt_cache_path)
    crt_success_path = args.crt_success_file
    if not os.path.isabs(crt_success_path):
        crt_success_path = os.path.join(args.cache, crt_success_path)

    df_ood, crt_stats = crawl_crtsh_ood(
        n=args.ood,
        seed=args.seed,
        max_attempts=args.crt_max_attempts,
        request_timeout=args.crt_timeout,
        sleep_s=args.crt_sleep,
        query_seeds=crt_query_seeds,
        fail_stop=args.crt_fail_stop,
        cache_path=crt_cache_path,
        success_queries_path=crt_success_path,
        backoff_base=args.crt_backoff_base,
        backoff_max=args.crt_backoff_max,
    )
    print(
        f"[INFO] CRT stats: attempts={crt_stats.get('attempts', 0):,}, "
        f"cache_hits={crt_stats.get('cache_hits', 0):,}, "
        f"success_queries_run={crt_stats.get('success_queries_run', 0):,}"
    )
    if crt_stats.get("fail_reason_counts"):
        print("[INFO] CRT fail reasons:")
        for reason, cnt in sorted(crt_stats["fail_reason_counts"].items(), key=lambda x: (-x[1], x[0])):
            print(f"  - {reason}: {cnt:,}")

    df_ood = clean_domains(df_ood)
    df_ood = df_ood[~df_ood["domain"].isin(protected_domains)].copy()

    non_tail_parts = [df_ood.copy()]

    feed_urls = []
    if not args.disable_default_ood_feeds:
        feed_urls.extend(DEFAULT_OOD_FEEDS)
    if args.ood_feed_url:
        feed_urls.extend(args.ood_feed_url)
    if feed_urls:
        df_feed = load_public_ood_feeds(
            feed_urls=feed_urls,
            max_rows=max(args.ood * 3, args.ood),
            seed=args.seed,
        )
        df_feed = clean_domains(df_feed)
        non_tail_parts.append(df_feed)

    if args.use_360netlab_fill:
        df_360 = load_360netlab_suspicious(
            cache_dir=args.cache,
            max_rows=max(args.ood * 3, args.ood),
            seed=args.seed,
        )
        df_360 = clean_domains(df_360)
        non_tail_parts.append(df_360)

    df_non_tail = pd.concat(non_tail_parts, ignore_index=True) if non_tail_parts else pd.DataFrame(columns=["domain", "family", "source", "label"])
    df_non_tail = clean_domains(df_non_tail)
    df_non_tail = df_non_tail[~df_non_tail["domain"].isin(protected_domains)].copy()
    df_non_tail = df_non_tail.drop_duplicates(subset=["domain"], keep="first")

    capped_overflow = df_non_tail.head(0).copy()
    if args.ood_cap_source:
        cap_from_ratio = int(args.ood * args.ood_cap_source_ratio)
        cap_candidates = [cap_from_ratio]
        if args.ood_cap_source_count >= 0:
            cap_candidates.append(args.ood_cap_source_count)
        source_cap = min(cap_candidates) if cap_candidates else -1
        df_non_tail, capped_overflow = split_source_cap(
            df_non_tail,
            source_prefix=args.ood_cap_source,
            max_count=source_cap,
            seed=args.seed,
        )
        if args.enforce_ood_mix and source_cap >= 0:
            capped_now = int(
                df_non_tail["source"].astype(str).str.lower().str.startswith(args.ood_cap_source.lower()).sum()
            )
            if capped_now > source_cap:
                raise RuntimeError(
                    f"[WARN] Source cap not satisfied for {args.ood_cap_source}: got={capped_now:,}, cap={source_cap:,}"
                )

    prefer_non_tail = max(int(args.ood * args.ood_min_non_tail_ratio), args.ood_min_non_tail_count)
    prefer_non_tail = min(args.ood, max(0, prefer_non_tail))

    if len(df_non_tail) >= args.ood:
        df_ood = df_non_tail.sample(n=args.ood, random_state=args.seed).copy()
    else:
        df_ood = df_non_tail.copy()

    if len(df_ood) < prefer_non_tail:
        msg = (
            f"[WARN] Non-tail OOD below preferred threshold: "
            f"preferred={prefer_non_tail:,}, available={len(df_ood):,}"
        )
        if args.enforce_ood_mix:
            raise RuntimeError(msg)
        print(msg)

    shortfall = max(0, args.ood - len(df_ood))
    if shortfall > 0:
        print(f"[INFO] OOD shortfall after non-tail={shortfall:,}; filling from tranco_tail")
        tail_pool = build_tranco_tail_pool(tranco_csv)
        blocked = protected_domains | set(df_ood["domain"].tolist())
        tail_pool = tail_pool[~tail_pool["domain"].isin(blocked)].copy()
        tail_pool = tail_pool.drop_duplicates(subset=["domain"], keep="first")

        max_tail_from_ratio = int(args.ood * args.ood_max_tail_ratio)
        max_tail = max_tail_from_ratio
        if args.ood_max_tail_count >= 0:
            max_tail = min(max_tail, args.ood_max_tail_count)
        max_tail = min(args.ood, max(0, max_tail))
        cur_tail = int((df_ood["source"] == "tranco_tail").sum()) if len(df_ood) else 0
        allowed_tail_add = max(0, max_tail - cur_tail)
        need_tail = shortfall
        if need_tail > allowed_tail_add:
            msg = (
                f"[WARN] Tail cap blocks full top-up: need={need_tail:,}, allowed_tail_add={allowed_tail_add:,}"
            )
            if args.enforce_ood_mix:
                raise RuntimeError(msg)
            print(msg)
            need_tail = allowed_tail_add

        if need_tail > 0 and len(tail_pool) > 0:
            take = min(need_tail, len(tail_pool))
            df_tail = tail_pool.sample(n=take, random_state=args.seed).copy()
            df_ood = pd.concat([df_ood, df_tail], ignore_index=True)

        remain = max(0, args.ood - len(df_ood))
        if remain > 0 and len(capped_overflow) > 0:
            blocked2 = protected_domains | set(df_ood["domain"].tolist())
            overflow_pool = capped_overflow[~capped_overflow["domain"].isin(blocked2)].copy()
            overflow_pool = overflow_pool.drop_duplicates(subset=["domain"], keep="first")
            if len(overflow_pool) > 0:
                take2 = min(remain, len(overflow_pool))
                df_ood = pd.concat(
                    [df_ood, overflow_pool.sample(n=take2, random_state=args.seed)],
                    ignore_index=True,
                )
                remain = max(0, args.ood - len(df_ood))

        if remain > 0 and not args.enforce_ood_mix:
            print(f"[WARN] Remaining OOD shortfall after tail cap/pool: {remain:,}")

    if len(df_ood) > args.ood:
        df_ood = df_ood.sample(n=args.ood, random_state=args.seed).copy()

    df_ood = add_open_set_metadata(df_ood, split_hint="unknown", force_unknown=True)

    df_unknown_ood = df_ood.copy()
    df_unknown_ood["split"] = "test_unknown_ood"

    known_benign_count = int((df_known["label"] == "benign").sum())
    known_dga_count = int((df_known["label"] == "dga").sum())
    unknown_family_count = int(len(df_unknown))
    unknown_ood_count = int(len(df_unknown_ood))

    checks = [
        ("benign", args.benign, known_benign_count),
        ("known_dga", args.known_dga, known_dga_count),
        ("unknown_family", args.unknown_family, unknown_family_count),
        ("unknown_ood", args.ood, unknown_ood_count),
    ]
    for name, requested, got in checks:
        if got < requested:
            msg = f"[WARN] {name} shortfall: requested={requested:,}, got={got:,}"
            if args.strict_targets:
                raise RuntimeError(msg)
            print(msg)

    def save(df: pd.DataFrame, name: str, subdir: str) -> None:
        out_subdir = os.path.join(run_outdir, subdir)
        os.makedirs(out_subdir, exist_ok=True)
        path = os.path.join(out_subdir, name)
        df.to_csv(path, index=False)
        print(f"[OK] {os.path.join(subdir, name)}: {len(df):,} rows")

    save(train, "train.csv", "known")
    save(val, "val.csv", "known")
    save(test_known, "test_known.csv", "known")
    save(df_unknown, "test_unknown_family.csv", "unknown_family")
    save(df_unknown_ood, "test_unknown_ood.csv", "unknown_ood")
    save(df_unknown_ood, "test_ood.csv", "unknown_ood")

    print("\nOOD source breakdown:")
    if len(df_unknown_ood) == 0:
        print("  (empty)")
    else:
        src_counts = df_unknown_ood["source"].fillna("unknown").value_counts()
        for src, cnt in src_counts.items():
            print(f"  - {src}: {cnt:,}")
    if len(df_unknown_ood) < args.ood:
        print(f"[WARN] OOD shortfall: requested={args.ood:,}, got={len(df_unknown_ood):,}")

    print("\nHold-out families (unknown DGA ground truth):")
    print(sorted(list(holdout)))


if __name__ == "__main__":
    main()
