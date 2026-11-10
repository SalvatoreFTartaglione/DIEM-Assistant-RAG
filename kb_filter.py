"""
Knowledge-base deduplication module for the DIEM Agentic RAG project.

Reads diem_knowledge_base.json and applies type-specific deduplication:

HTML documents:
1. URL normalization dedup  (keep first per normalized source URL)
2. Content-hash dedup        (keep first per identical text)

PDF documents (three passes):
1. Year dedup        — group URLs that differ only by the year segment,
                        keep the most recent version.
2. Content-hash dedup — drop PDFs with byte-identical extracted text.
3. Cross-domain dedup — when the same /rescue/ path appears on multiple
                        domains, keep the corsi.unisa.it copy.

This module is the single point of deduplication in the pipeline.
extractor.py intentionally performs NO deduplication, so that both
cross-domain copies of a PDF survive into the knowledge base and the
PDF Pass 2 / Pass 3 logic below can operate on the full set.
"""

import hashlib
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse


RAW_DOCUMENTS_PATH = "diem_knowledge_base.json"


# ---------------------------------------------------------------------------
# URL Normalization  (replicated from extractor.py to keep this module
# independent — importing extractor would trigger its module-level setup).
# ---------------------------------------------------------------------------

def _normalize_url(url: str) -> str:
    """Lowercase scheme/host and strip trailing slash."""
    url = url.strip().rstrip("/")
    p   = urlparse(url)
    return p._replace(scheme="https", netloc=p.netloc.lower()).geturl()


# ---------------------------------------------------------------------------
# HTML Deduplication
# ---------------------------------------------------------------------------
def dedupe_html_documents(documents: List[Dict]) -> Tuple[List[Dict], Dict]:
    """
    Deduplicate HTML documents using two passes:
    1. Normalized-URL dedup — keep the first document per normalized source.
    2. Content-hash dedup    — keep the first document per identical text.
                            Documents carrying an 'easycourse_category'
                            metadata field are exempt from this pass: their
                            source URL is the canonical identifier and their
                            text content may legitimately be near-identical
                            across different calendar pages.
    Returns the deduplicated list and a small report dict.
    """
    before = len(documents)
    unique: List[Dict] = []
    # Pass 1 — normalized URL (EasyCourse documents are exempt)
    by_url: Dict[str, Dict] = {}
    for doc in documents:
        easycourse_category = doc.get("metadata", {}).get("easycourse_category")
        if easycourse_category:
            unique.append(doc)
            continue
        key = _normalize_url(doc.get("source", ""))
        if key not in by_url:
            by_url[key] = doc
    after_url = len(by_url) + len(unique)

    # Pass 2 — content hash (EasyCourse documents are exempt)
    seen_hashes: Set[str] = set()
    for doc in by_url.values():
        easycourse_category = doc.get("metadata", {}).get("easycourse_category")
        if easycourse_category:
            unique.append(doc)
            continue
        text_hash = hashlib.sha256(doc.get("text", "").encode("utf-8")).hexdigest()
        if text_hash not in seen_hashes:
            seen_hashes.add(text_hash)
            unique.append(doc)

    report = {
        "html_before":           before,
        "html_after_url":        after_url,
        "html_url_removed":      before - after_url,
        "html_after_content":    len(unique),
        "html_content_removed":  after_url - len(unique),
    }
    return unique, report


# ---------------------------------------------------------------------------
# PDF Year Extraction
# ---------------------------------------------------------------------------

_YEAR_PATTERN = re.compile(r"(?<=/)(199\d|20\d{2})(?=[/_\-]|$)")


def _extract_year_and_key(url: str) -> Tuple[Optional[int], str]:
    """
    Extract the year from a PDF source URL and return a normalized group key
    obtained by replacing the year segment with a fixed placeholder.
    """
    match = _YEAR_PATTERN.search(url)
    if not match:
        return None, url

    year = int(match.group(0))
    group_key = _YEAR_PATTERN.sub("__YEAR__", url)
    return year, group_key


# ---------------------------------------------------------------------------
# PDF Deduplication  (three passes)
# ---------------------------------------------------------------------------

def filter_pdf_documents(pdf_docs: List[Dict]) -> Tuple[List[Dict], Dict]:
    """
    Deduplicate PDF documents through three passes:

      Pass 1 — Year:        keep the most recent version of each PDF group
                            identified by URL structure (year-insensitive key).
      Pass 2 — Content:     drop PDFs whose extracted text is byte-identical.
      Pass 3 — Cross-domain: when the same /rescue/ path appears on multiple
                            domains, keep the corsi.unisa.it copy.

    Returns the deduplicated PDF list and a report dict.
    """
    # ── Pass 1: Year-based dedup ──────────────────────────────────────────
    # The grouping key is the URL with the year segment replaced by a
    # placeholder.  (The legacy per-page "pdf_page" component has been
    # removed: extractor.py emits one markdown blob per PDF, never per page,
    # so pdf_page was always 0 — dead logic.)
    groups: Dict[str, Dict] = {}
    group_sources: Dict[str, Set[str]] = defaultdict(set)

    for doc in pdf_docs:
        url = doc["source"]
        year, group_key = _extract_year_and_key(url)
        group_sources[group_key].add(url)

        if group_key not in groups:
            groups[group_key] = {"best_year": year, "best_doc": doc}
        else:
            existing_year = groups[group_key]["best_year"]
            if year is not None and (existing_year is None or year > existing_year):
                groups[group_key]["best_year"] = year
                groups[group_key]["best_doc"] = doc

    retained_pdfs = [g["best_doc"] for g in groups.values()]
    duplicates_removed = len(pdf_docs) - len(retained_pdfs)

    multi_source_groups = [
        {"key": key, "sources": sorted(sources)}
        for key, sources in group_sources.items()
        if len(sources) > 1
    ]

    # ── Pass 2: Content-hash dedup ────────────────────────────────────────
    seen_hashes: Dict[str, Dict] = {}
    for doc in retained_pdfs:
        text_hash = hashlib.sha256(doc["text"].encode("utf-8")).hexdigest()
        if text_hash not in seen_hashes:
            seen_hashes[text_hash] = doc
        else:
            # Tie-break deterministically by lexicographic source URL.
            if doc["source"] < seen_hashes[text_hash]["source"]:
                seen_hashes[text_hash] = doc

    deduped_pdfs = list(seen_hashes.values())
    content_duplicates_removed = len(retained_pdfs) - len(deduped_pdfs)

    # ── Pass 3: Cross-domain dedup ────────────────────────────────────────
    # Two PDFs sharing the same path after "/rescue/" are the same file
    # served from different domains.  Prefer the corsi.unisa.it copy.
    _RESCUE_PATTERN = re.compile(r"/rescue/(.+)")

    path_groups: Dict[str, Dict] = {}
    no_rescue: List[Dict] = []

    for doc in deduped_pdfs:
        url = doc["source"]
        m = _RESCUE_PATTERN.search(url)
        if not m:
            no_rescue.append(doc)
            continue
        rescue_path = m.group(1).lower()
        if rescue_path not in path_groups:
            path_groups[rescue_path] = doc
        else:
            # Give precedence to the corsi.unisa.it copy.
            if "corsi.unisa.it" in url:
                path_groups[rescue_path] = doc

    cross_domain_deduped = list(path_groups.values()) + no_rescue
    cross_domain_removed = len(deduped_pdfs) - len(cross_domain_deduped)

    report = {
        "pdf_before":                  len(pdf_docs),
        "pdf_after_year":              len(retained_pdfs),
        "year_removed":                duplicates_removed,
        "content_removed":             content_duplicates_removed,
        "cross_domain_removed":        cross_domain_removed,
        "pdf_after":                   len(cross_domain_deduped),
        "groups_with_duplicates":      multi_source_groups,
    }
    return cross_domain_deduped, report


# ---------------------------------------------------------------------------
# Top-level Filter
# ---------------------------------------------------------------------------

def filter_documents(documents: List[Dict]) -> Tuple[List[Dict], Dict]:
    """
    Split the knowledge base into HTML and PDF documents, deduplicate each
    type with its dedicated logic, and recombine.
    """
    html_docs: List[Dict] = []
    pdf_docs: List[Dict] = []

    for doc in documents:
        if doc.get("type") == "pdf":
            pdf_docs.append(doc)
        else:
            html_docs.append(doc)

    deduped_html, html_report = dedupe_html_documents(html_docs)
    deduped_pdf, pdf_report   = filter_pdf_documents(pdf_docs)

    report = {**html_report, **pdf_report}
    return deduped_html + deduped_pdf, report


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def run() -> None:
    """Execute the full deduplication pipeline."""
    print("=== DIEM Agentic RAG: KB Deduplication Started ===\n")

    raw_path = Path(RAW_DOCUMENTS_PATH)
    full_backup_path = raw_path.parent / "raw_documents_full.json"

    print(f"[Filter] Loading documents from {raw_path} ...")
    with open(raw_path, "r", encoding="utf-8") as fh:
        documents: List[Dict] = json.load(fh)
    print(f"[Filter] {len(documents)} documents loaded.")

    shutil.copy2(raw_path, full_backup_path)
    print(f"[Filter] Full backup saved to {full_backup_path}")

    print("[Filter] Applying deduplication filters ...")
    filtered_documents, report = filter_documents(documents)

    # Report PDF groups where older versions were discarded.
    if report.get("groups_with_duplicates"):
        print("\n[Filter] PDF groups where older versions were discarded:")
        for group in report["groups_with_duplicates"]:
            urls = group["sources"]
            years = []
            for u in urls:
                m = _YEAR_PATTERN.search(u)
                years.append(int(m.group(0)) if m else None)
            best_year = max((y for y in years if y is not None), default=None)
            kept_url = next(
                u for u, y in zip(urls, years)
                if y == best_year or best_year is None
            )
            discarded = [u for u in urls if u != kept_url]
            print(f"[Filter]   Kept      : {kept_url} (year: {best_year})")
            for d in discarded:
                print(f"[Filter]   Discarded : {d}")
    else:
        print("[Filter] No duplicate PDF year-groups found.")

    with open(raw_path, "w", encoding="utf-8") as fh:
        json.dump(filtered_documents, fh, ensure_ascii=False, indent=2)
    print(f"\n[Filter] Filtered documents saved to {raw_path}")

    print("\n[Filter] Summary:")
    print(f"[Filter]   HTML documents before        : {report['html_before']}")
    print(f"[Filter]   HTML removed (URL)           : {report['html_url_removed']}")
    print(f"[Filter]   HTML removed (content)       : {report['html_content_removed']}")
    print(f"[Filter]   HTML documents after         : {report['html_after_content']}")
    print("[Filter]   ---")
    print(f"[Filter]   PDF documents before         : {report['pdf_before']}")
    print(f"[Filter]   PDF removed (year)           : {report['year_removed']}")
    print(f"[Filter]   PDF removed (content)        : {report['content_removed']}")
    print(f"[Filter]   PDF removed (cross-domain)   : {report['cross_domain_removed']}")
    print(f"[Filter]   PDF documents after          : {report['pdf_after']}")
    print("[Filter]   ---")
    print(f"[Filter]   Total documents after filter : {len(filtered_documents)}")
    print("\n=== KB Deduplication Completed ===")


if __name__ == "__main__":
    run()