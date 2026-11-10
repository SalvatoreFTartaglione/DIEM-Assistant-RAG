"""
extractor.py

Web crawler that builds the DIEM knowledge base (diem_knowledge_base.json).

Crawling phases:
    1    www.diem.unisa.it   recursive crawl (RecursiveUrlLoader, depth 4) that
                             also discovers course, structure and PDF links
    1.5  /home/bandi         calls for applications and PhD calls, year by year
    1.6  DIEM structures     labs and facilities discovered in phase 1
    2    docenti.unisa.it    professor pages (home, teaching for every academic
                             year, CV, research sections)
    3    corsi.unisa.it      degree-program sites discovered in phase 1 (depth 3)
    5    easycourse.unisa.it lesson timetables and exam calendars
    4    PDFs                every PDF queued by the previous phases, converted
                             to Markdown with pymupdf4llm (runs last)

HTML pages are cleaned with BeautifulSoup (header/nav/footer removal) and
trafilatura (main-content extraction to Markdown). Each record carries its
source URL, type (html/pdf) and contextual metadata (professor, academic year,
call category, ...) later used as retrieval filters.

No deduplication happens here: run kb_filter.py afterwards.
"""

import json
import os
import re
import ssl
import sys
import tempfile
import time
from datetime import datetime
from urllib.parse import parse_qs, urljoin, urlparse

import pymupdf4llm
import requests
import trafilatura
import urllib3
from bs4 import BeautifulSoup
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain_core.documents import Document
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---------------------------------------------------------------------------
# CONFIGURATION & GLOBAL STATE
# ---------------------------------------------------------------------------
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
OUTPUT_PATH = "diem_knowledge_base.json"
LOG_PATH = "crawler_log.txt"
MAX_DEPTH_DIEM = 4
MAX_DEPTH_COURSES = 3

FORBIDDEN_EXTENSIONS = (".css", ".js", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".woff", ".ttf", ".json")

# ── CALLS CONFIGURATION ────────────────────────────────────
MIN_YEAR = 2020
DIEM_STRUTTURA   = "300638"
DOTTORATO_KEYWORD = "INGEGNERIA DELL'INFORMAZIONE"

BANDI_CON_STRUTTURA = [
    ("139", "Incarichi di Insegnamento",          "struttura"),
    ("504", "Collaborazioni con il Dipartimento", "struttura"),
    ("316", "Personale Tecnico Amministrativo",   "struttura"),
    ("67",  "Assegni di Ricerca",                 "struttura"),
    ("292", "Borse di Ricerca",                   "struttura"),
    ("293", "Borse e Premi",                      "cdsStruttura"),
    ("505", "Altri bandi",                        "struttura"),
]

BASE_BANDI = "https://www.diem.unisa.it/home/bandi"

# ── METADATA FLAGS ───────────────────────────────────────────
# True  = save all metadata (staff_id, bando_label, year, etc.)
# False = save ONLY source and type (minimal configuration)
ENABLE_METADATA = True


def _build_meta(source: str, doc_type: str = "html", **extras) -> dict:
    """
    Build the metadata dict for a Document respecting ENABLE_METADATA.
    If False, the **extras are ignored: only source and type remain.
    Empty/None values in extras are discarded.
    """
    meta = {"source": source, "type": doc_type}
    if ENABLE_METADATA:
        meta.update({k: v for k, v in extras.items() if v not in (None, "")})
    return meta


PROCESSED_URLS = set()
DISCOVERED_COURSES = set()
DISCOVERED_STRUCTURES = set()
# PDF url -> metadata to attach to the document (e.g. bando_label, anno)
DISCOVERED_PDF = {}


# ---------------------------------------------------------------------------
# HTTP SETUP
# ---------------------------------------------------------------------------
# Some university hosts serve incomplete certificate chains, so TLS
# verification is disabled for this crawler only. The patches below are
# applied explicitly by configure_http() (called from main), never at import.

def _build_retry_adapter() -> HTTPAdapter:
    retry = Retry(
        total=3,
        backoff_factor=2,        # waits 2s, then 4s between retries
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    return HTTPAdapter(max_retries=retry)

_original_session_init = requests.Session.__init__
original_get = requests.get


def _patched_session_init(self, *args, **kwargs):
    # Patching Session.__init__ (instead of only requests.get) makes EVERY
    # session, including RecursiveUrlLoader's internal one, inherit the retry
    # adapter and verify=False.
    _original_session_init(self, *args, **kwargs)
    adapter = _build_retry_adapter()
    self.mount("https://", adapter)
    self.mount("http://", adapter)
    self.verify = False


def robust_get(*args, **kwargs):
    """requests.get with default timeout, User-Agent and explicit retries on network errors."""
    kwargs["verify"] = False
    kwargs.setdefault("timeout", 30)
    headers = kwargs.get("headers") or {}
    headers.setdefault("User-Agent", "Mozilla/5.0")
    kwargs["headers"] = headers
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return original_get(*args, **kwargs)
        except requests.exceptions.RequestException as e:
            url = args[0] if args else kwargs.get("url", "Unknown URL")
            if attempt < max_retries - 1:
                print(f"      [RETRY {attempt+1}/{max_retries}] {url} - retrying in 3s")
                time.sleep(3)
            else:
                raise e


def configure_http() -> None:
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    ssl._create_default_https_context = ssl._create_unverified_context
    requests.Session.__init__ = _patched_session_init
    requests.get = robust_get


# ---------------------------------------------------------------------------
# DATA PROCESSING & EXPORT
# ---------------------------------------------------------------------------
def _normalize_url(url: str) -> str:
    url = url.strip().rstrip("/")
    parsed = urlparse(url)
    normalized = parsed._replace(scheme="https", netloc=parsed.netloc.lower())
    return normalized.geturl()

MIN_CONTENT_CHARS = 150

def export_to_json(documents: list[Document], filename: str):
    print(f"\n=== EXPORTING DATA: saving to {filename} ===")
    dataset_finale = []
    skipped_short = 0

    for doc in documents:
        text = doc.page_content
        if len(text) < MIN_CONTENT_CHARS:
            skipped_short += 1
            continue
        if not doc.metadata.get("type"):
            print(f"  [WARN] missing type on: {doc.metadata.get('source', 'unknown')}")

        doc_dict = {
            "text": text,
            "source": doc.metadata.get("source", ""),
            "type": doc.metadata.get("type", "unknown")
        }

        if ENABLE_METADATA:
            doc_dict["metadata"] = {k: v for k, v in doc.metadata.items() if k not in ["source", "type"]}

        dataset_finale.append(doc_dict)

    if skipped_short:
        print(f"  [FILTER] Skipped {skipped_short} documents shorter than {MIN_CONTENT_CHARS} chars.")

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(dataset_finale, f, indent=4, ensure_ascii=False)

    print(f"  Data saved to '{filename}' with {len(dataset_finale)} entries.")

def print_summary(documents: list[Document]):
    print(f"\n=== TOTAL: {len(documents)} unique documents collected ===")
    if not documents:
        print("No documents collected! Check your connection and URLs.")
        return

    dettaglio_tipi = {}
    for doc in documents:
        tipo = doc.metadata.get("type", "unknown")
        dettaglio_tipi[tipo] = dettaglio_tipi.get(tipo, 0) + 1

    for tipo, quantita in dettaglio_tipi.items():
        print(f"      - {tipo}: {quantita}")
    print("\nINGESTION PHASE COMPLETED!")


# ---------------------------------------------------------------------------
# EXTRACTION HELPERS
# ---------------------------------------------------------------------------
def extract_diem_prof_ids(staff_page_url: str) -> list[str]:
    r = requests.get(staff_page_url, verify=False)
    if r.status_code != 200:
        return []
    soup = BeautifulSoup(r.text, "html.parser")
    staff_ids = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "rubrica.unisa.it/persone" in href and "matricola=" in href:
            qs = parse_qs(urlparse(href).query)
            if "matricola" in qs:
                staff_ids.add(qs["matricola"][0].zfill(6))
    return list(staff_ids)

def clean_html(raw_html: str, url: str = "") -> str:
    if not raw_html:
        return ""
    soup = BeautifulSoup(raw_html, "html.parser")
    for tag in soup.find_all(["header", "nav"]):
        tag.decompose()
    footers = soup.find_all(
        lambda t: t.name in ("footer",) or
                (t.name == "div" and t.get("id") in ("footer", "subfooter")) or
                (t.name == "div" and "footer" in " ".join(t.get("class", [])))
    )
    for footer in footers:
        for sibling in footer.find_next_siblings():
            sibling.decompose()
        footer.decompose()

    result = trafilatura.extract(
        str(soup), url=url, output_format="markdown", include_tables=True,
        include_links=False, include_images=False, include_comments=False,
        favor_precision=True, deduplicate=True
    )
    return result.strip() if result else ""


def print_website_metadata(raw_html: str, url: str) -> dict:
    """
    RecursiveUrlLoader metadata extractor. Besides returning the metadata, it
    applies the discard rules (sitemaps, other departments, static assets,
    duplicates) and queues PDF / course / structure links for later phases.
    Discarded pages are marked with a "discard_*" source.
    """
    if "sitemap" in url.lower():
        print(f"      [DISCARD - Sitemap] {url}")
        return {"source": "discard_sitemap"}

    parsed_url = urlparse(url)
    qs = parse_qs(parsed_url.query)

    for key, values in qs.items():
        if "struttura" in key.lower() and DIEM_STRUTTURA not in values:
            return {"source": "discard_wrong_department"}

    url_base = url.split("?")[0].lower()
    if url_base.endswith(FORBIDDEN_EXTENSIONS):
        return {"source": "discard_extension"}

    url_pulito = _normalize_url(url)
    if url_pulito in PROCESSED_URLS:
        print(f"      [DISCARD - Duplicate] {url}")
        return {"source": "discard_duplicate"}

    PROCESSED_URLS.add(url_pulito)


    soup = BeautifulSoup(raw_html, "html.parser")
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        full_url = urljoin(url, href).split("#")[0]
        if "unisa.it" not in urlparse(full_url).netloc.lower():
            continue
        qs_anno = parse_qs(urlparse(full_url).query).get("anno", [None])[0]
        if qs_anno is not None:
            try:
                if int(qs_anno) < MIN_YEAR:
                    continue
            except ValueError:
                pass

        url_no_query = full_url.split("?")[0]
        if url_no_query.lower().endswith(".pdf"):
            if full_url not in DISCOVERED_PDF:
                DISCOVERED_PDF[full_url] = {"type": "pdf"}
                print(f"      [QUEUED PDF] {full_url}")
        elif "corsi.unisa.it" in full_url:
            DISCOVERED_COURSES.add(full_url)
            print(f"      [QUEUED COURSE] {full_url}")
        elif "diem.unisa.it/dipartimento/strutture" in full_url and "id=" in full_url:
            DISCOVERED_STRUCTURES.add(full_url)

    return {"source": url}


# ---------------------------------------------------------------------------
# CRAWLING PHASES
# ---------------------------------------------------------------------------
def crawl_diem_base() -> list[Document]:
    print("\n=== PHASE 1: Crawl www.diem.unisa.it ===")
    docs_collected = []
    print("  Crawling seed: https://www.diem.unisa.it/")
    loader = RecursiveUrlLoader(
        url="https://www.diem.unisa.it/",
        max_depth=MAX_DEPTH_DIEM,
        extractor=clean_html,
        metadata_extractor=print_website_metadata,
        exclude_dirs=["https://www.diem.unisa.it/en", "https://www.diem.unisa.it/en/", "https://www.diem.unisa.it/home/bandi"],
        prevent_outside=True
    )
    for doc in loader.load():
        source_url = doc.metadata.get("source", "").lower()
        if source_url.startswith("discard"): continue
        doc.metadata["type"] = "html"
        docs_collected.append(doc)
        print(f"      [SUCCESS Base DIEM] {doc.metadata.get('source')}")

    print(f"  -> Extracted {len(docs_collected)} DIEM documents")
    return docs_collected


def crawl_bandi() -> list[Document]:
    print("\n=== PHASE 1.5: Bandi DIEM ===")
    docs_collected = []
    current_year   = datetime.now().year
    anni = list(range(MIN_YEAR, current_year + 1))

    # ── helper: save an HTML page as Document ──────────
    def _save_html(url: str, label: str, anno: int) -> BeautifulSoup | None:
        """
        Download the URL, save it as a Document and return
        the BeautifulSoup for further navigation.
        Returns None if the page is empty or in error.
        """
        try:
            r = requests.get(url, verify=False, timeout=15)
            if r.status_code != 200:
                print(f"      [HTTP {r.status_code}] {url}")
                return None
            content = clean_html(r.text, url=url)
            if not content or len(content.strip()) < 120:
                print(f"      [SKIP - Empty] {url}")
                return None
            docs_collected.append(Document(
                page_content=content,
                metadata=_build_meta(url, "html", bando_label=label, anno=anno)
            ))
            print(f"      [HTML] {label} {anno} — {url}")
            return BeautifulSoup(r.text, "html.parser")
        except Exception as e:
            print(f"      [ERROR] {url} - {e}")
            return None

    # ── helper: queue all PDFs found on the page ──────
    def _queue_pdfs(soup: BeautifulSoup, page_url: str,
                    label: str, anno: int) -> None:
        """
            Find all <a href="...pdf"> links on the page
            and add them to DISCOVERED_PDF with correct metadata.
            Handles both absolute and relative URLs.
            """
        for a in soup.find_all("a", href=True):
            raw_href = a["href"].strip()
            full_url = urljoin("https://www.diem.unisa.it", raw_href)
            if not full_url.lower().endswith(".pdf"):
                continue
            # Skip external links (pica.cineca, etc.)
            if "diem.unisa.it" not in full_url:
                continue
            if full_url not in DISCOVERED_PDF:
                DISCOVERED_PDF[full_url] = ( {"bando_label": label, "anno": anno, "type": "pdf"} if ENABLE_METADATA
                                            else {"type": "pdf"})
                print(f"         [PDF QUEUED] {full_url}")

    # ── helper: collect announcement links (?bando=) and idConcorso ──
    def _find_bando_links(soup: BeautifulSoup, page_url: str) -> tuple[set, set]:
        """
        Return two sets:
        - bando_links: URLs with ?bando= (but without idConcorso)
        - concorso_links: URLs with ?idConcorso=
        """
        bando_links    = set()
        concorso_links = set()
        for a in soup.find_all("a", href=True):
            full = urljoin(page_url, a["href"]).split("#")[0]
            if "diem.unisa.it" not in full:
                continue
            if "idConcorso=" in full:
                concorso_links.add(full)
            elif "bando=" in full and "modulo=" in full and "anno=" in full:
                bando_links.add(full)
        return bando_links, concorso_links

    # ── crawl a single year for structured announcements ─────
    def _crawl_anno_struttura(modulo: str, anno: int, label: str, struttura_param: str) -> None:
        # Level 1 — listing page
        url_listing = (f"{BASE_BANDI}?anno={anno}"f"&{struttura_param}={DIEM_STRUTTURA}&modulo={modulo}")
        soup_l1 = _save_html(url_listing, label, anno)
        if soup_l1 is None:
            return
        _queue_pdfs(soup_l1, url_listing, label, anno)

        bando_links, concorso_links = _find_bando_links(soup_l1, url_listing)

        # Level 2 — single announcement pages (?bando=XXXX)
        for burl in bando_links:
            soup_l2 = _save_html(burl, label, anno)
            if soup_l2 is None:
                continue
            _queue_pdfs(soup_l2, burl, label, anno)
                # Also collect any idConcorso present
            _, cl2 = _find_bando_links(soup_l2, burl)
            concorso_links.update(cl2)
            time.sleep(0.3)

        # Level 3 — single competition pages (?idConcorso=XXXX)
        for curl in concorso_links:
            soup_l3 = _save_html(curl, label, anno)
            if soup_l3 is None:
                continue
            _queue_pdfs(soup_l3, curl, label, anno)
            time.sleep(0.3)

    # ── crawl PhD (no structure, keyword filter) ──
    def _crawl_anno_dottorati(anno: int) -> None:
        label = "Dottorati di Ricerca"
        url_listing = f"{BASE_BANDI}?anno={anno}&modulo=226"
        soup_l1 = _save_html(url_listing, label, anno)
        if soup_l1 is None:
            return
        _queue_pdfs(soup_l1, url_listing, label, anno)

        # Find announcement/competition links whose row text contains the keyword
        kw = DOTTORATO_KEYWORD.upper().replace("'", "").replace("\u2019", "").replace(" ", "")
        diem_links = set()
        for a in soup_l1.find_all("a", href=True):
            href = a["href"]
            if "idConcorso=" not in href and "bando=" not in href:
                continue
            container = (a.find_parent("tr") or a.find_parent("li") or a.parent)
            row_text = (container.get_text(separator=" ", strip=True)
                        .upper()
                        .replace("'", "").replace("\u2019", "").replace(" ", ""))
            if kw in row_text:
                full = urljoin(url_listing, href).split("#")[0]
                diem_links.add(full)
                print(f"         [PHD LINK] {full}")

        if not diem_links:
            print(f"      [PHD {anno}] No DIEM links found")
            return

        # Level 2 — DIEM announcement/competition pages
        label_diem = "Dottorati di Ricerca - DIEM"
        for durl in diem_links:
            soup_l2 = _save_html(durl, label_diem, anno)
            if soup_l2 is None:
                continue
            _queue_pdfs(soup_l2, durl, label_diem, anno)
            # Go one level deeper if the page links to single competitions (idConcorso)
            _, cl2 = _find_bando_links(soup_l2, durl)
            for curl in cl2:
                soup_l3 = _save_html(curl, label_diem, anno)
                if soup_l3:
                    _queue_pdfs(soup_l3, curl, label_diem, anno)
                time.sleep(0.3)
            time.sleep(0.3)

    # ── main execution ─────────────────────────────────
    for modulo, label, struttura_param in BANDI_CON_STRUTTURA:
        print(f"\n  [{label}]")
        for anno in reversed(anni):
            _crawl_anno_struttura(modulo, anno, label, struttura_param)
            time.sleep(0.3)

    print("\n  [PhD Research]")
    for anno in reversed(anni):
        _crawl_anno_dottorati(anno)
        time.sleep(0.3)

    print(f"\n  -> {len(docs_collected)} HTML announcements | " f"{len(DISCOVERED_PDF)} PDFs queued")
    return docs_collected


def crawl_structures() -> list[Document]:
    print(f"\n=== PHASE 1.6: DIEM Structures ({len(DISCOVERED_STRUCTURES)} pages) ===")
    docs_collected = []
    for url in DISCOVERED_STRUCTURES:
        try:
            r = requests.get(url, verify=False, timeout=15)
            if r.status_code != 200: continue
            content = clean_html(r.text, url=url)
            if not content or not content.strip(): continue
            docs_collected.append(Document(
                page_content=content,
                metadata={"source": url, "type": "html"}
            ))
            print(f"      [SUCCESS] {url}")
        except Exception as e:
            print(f"      [SKIPPED] {url} - {e}")
        time.sleep(0.3)
    return docs_collected


def crawl_professor() -> list[Document]:
    print("\n=== PHASE 2: DIEM Professors on docenti.unisa.it ===")
    docs_collected   = []
    staff_name_cache = {}   # staff_id -> "Nome Cognome"
    current_year     = datetime.now().year

    staff_ids = extract_diem_prof_ids("https://www.diem.unisa.it/dipartimento/personale")
    if not staff_ids:
        return docs_collected

    staff_urls = []
    for staff_id in staff_ids:
        staff_urls.extend([
            f"https://docenti.unisa.it/{staff_id}/home",
            f"https://docenti.unisa.it/{staff_id}/didattica",
            f"https://docenti.unisa.it/{staff_id}/curriculum",
            f"https://docenti.unisa.it/{staff_id}/ricerca/progetti",
            f"https://docenti.unisa.it/{staff_id}/ricerca/pubblicazioni",
            f"https://docenti.unisa.it/{staff_id}/ricerca/premi",
            f"https://docenti.unisa.it/{staff_id}/ricerca/spin-off"
        ])

    for url in staff_urls:
        try:
            r = requests.get(url, verify=False, timeout=15)
            if r.status_code != 200:
                print(f"      [SKIPPED - HTTP {r.status_code}] {url}")
                continue

            staff_id = url.split("docenti.unisa.it/")[1].split("/")[0]
            soup     = BeautifulSoup(r.text, "html.parser")

            # --- HOME ---
            if url.endswith("/home"):
                if not soup.find("table"):
                    print(f"      [SKIPPED - No home table found] {url}")
                    continue
                content = clean_html(r.text, url=url)
                if not content.strip():
                    print(f"      [SKIPPED - Empty content] {url}")
                    continue

                match = re.search(
                    r"#+ ([A-ZÀÈÉÌÒÙA-Za-zàèéìòù]+"
                    r"(?:\s+[A-ZÀÈÉÌÒÙA-Za-zàèéìòù]+)+)"
                    r"(?:\s*\||\n|$)",
                    content
                )
                if match:
                    prof_name = match.group(1).strip().title()
                    staff_name_cache[staff_id] = prof_name
                    print(f"      [NAME] {staff_id} → {prof_name}")
                else:
                    print(f"      [WARN] Name not found for staff_id={staff_id}")

                docs_collected.append(Document(
                    page_content=content,
                    metadata=_build_meta(url, "html",
                         staff_id=staff_id,
                         professor_name=staff_name_cache.get(staff_id, ""))
                ))
                print(f"      [SUCCESS] {url}")

            # --- DIDATTICA ---
            elif url.endswith("/didattica"):
                course_links = set()

                # The default teaching page only shows the current year: iterate over all years
                for anno_iter in range(MIN_YEAR, current_year + 1):
                    didattica_url = f"{url}?anno={anno_iter}"
                    try:
                        rd = requests.get(didattica_url, verify=False, timeout=15)
                        if rd.status_code != 200:
                            continue
                        soup_did = BeautifulSoup(rd.text, "html.parser")
                        for a in soup_did.find_all("a", href=True):
                            full = urljoin(didattica_url, a["href"]).split("#")[0]
                            if "didattica?anno=" in full and "id=" in full:
                                course_links.add(full)
                        time.sleep(0.3)
                    except Exception:
                        continue

                if not course_links:
                    print(f"      [SKIPPED - No valid courses found] {url}")
                else:
                    print(f"      [TEACHING] {staff_id} -> {len(course_links)} courses found ({MIN_YEAR}–{current_year})")

                for course_url in course_links:
                    try:
                        anno_corso = int(parse_qs(
                            urlparse(course_url).query
                        ).get("anno", ["0"])[0])
                        rc = requests.get(course_url, verify=False, timeout=15)
                        if rc.status_code == 200:
                            course_content = clean_html(rc.text, url=course_url)
                            if course_content.strip():
                                docs_collected.append(Document(
                                    page_content=course_content,
                                    metadata=_build_meta(course_url, "html",
                                        staff_id=staff_id,
                                        anno_accademico=f"{anno_corso}/{anno_corso + 1}",
                                        professor_name=staff_name_cache.get(staff_id, ""))
                                ))
                                print(f"         [SUCCESS - COURSE] {course_url}")
                            else:
                                print(f"         [SKIPPED - Empty] {course_url}")
                        else:
                            print(f"         [SKIPPED - HTTP {rc.status_code}] {course_url}")
                    except Exception as e:
                        print(f"         [SKIPPED - Error] {course_url} - {e}")

            # --- CURRICULUM & RICERCA ---
            else:
                content = clean_html(r.text, url=url)
                if len(content.strip()) >= 120:
                    meta = _build_meta(url, "html",
                        staff_id=staff_id,
                        professor_name=staff_name_cache.get(staff_id, ""))
                    if ENABLE_METADATA and "/ricerca/" in url:
                        meta["sezione_ricerca"] = url.split("/ricerca/")[1]
                    docs_collected.append(Document(page_content=content, metadata=meta))
                    print(f"      [SUCCESS] {url}")
                else:
                    print(f"      [SKIPPED - Too short] {url}")

        except Exception as e:
            print(f"      [SKIPPED - Error] {url} - {e}")

        time.sleep(0.3)

    print(f"  -> Extracted {len(docs_collected)} professor documents")
    print(f"  -> Professor names cached: {len(staff_name_cache)}")
    return docs_collected


def crawl_courses() -> list[Document]:
    print("\n=== PHASE 3: DIEM courses ===")
    docs_collected = []
    course_seeds = list(DISCOVERED_COURSES)

    if course_seeds:
        print(f"  -> Found {len(course_seeds)} course links during Phase 1!")
        for url in course_seeds:
            course_loader = RecursiveUrlLoader(
                url=url, max_depth=MAX_DEPTH_COURSES, extractor=clean_html,
                metadata_extractor=print_website_metadata, prevent_outside=True
            )
            for doc in course_loader.load():
                source_url = doc.metadata.get("source", "").lower()

                # Discard reasons are already logged by print_website_metadata.
                if source_url.startswith("discard"):
                    continue

                if "/en/" in source_url or source_url.endswith("/en"):
                    print(f"      [SKIPPED - English] {source_url}")
                    continue

                doc.metadata["type"] = "html"
                # Year taken from the URL query string, when present
                qs_anno = parse_qs(urlparse(source_url).query).get("anno", [None])[0]
                if qs_anno and ENABLE_METADATA:
                    try:
                        doc.metadata["anno"] = int(qs_anno)
                    except ValueError:
                        pass

                docs_collected.append(doc)
                print(f"      [SUCCESS] {source_url}")

        print(f"  -> Courses extracted successfully. Total: {len(docs_collected)}")
    else:
        print("  -> No course links found.")

    return docs_collected


def download_pdfs() -> list[Document]:
    print(f"\n=== PHASE 4: Download {len(DISCOVERED_PDF)} intercepted PDFs ===")
    docs_collected = []

    for pdf_url, extra_metadata in DISCOVERED_PDF.items():
        try:
            r = requests.get(pdf_url, verify=False, timeout=20)
            if r.status_code != 200:
                print(f"      [SKIPPED - HTTP {r.status_code}] {pdf_url}")
                continue

            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(r.content)
                tmp_path = tmp.name

            try:
                md = pymupdf4llm.to_markdown(tmp_path)
            finally:
                os.unlink(tmp_path)  # always remove the temp file, even if conversion fails

            if md:
                md = re.sub(r"\*\*==> picture \[.*?\] intentionally omitted <==\*\*\s*", "", md)
                md = md.replace('\u2028', '\n').replace('\u2029', '\n\n')

            if not md or not md.strip():
                print(f"      [SKIPPED - Empty/Image Only] {pdf_url}")
                continue

            doc_metadata = {"source": pdf_url}
            doc_metadata.update(extra_metadata)

            docs_collected.append(Document(
                page_content=md,
                metadata=doc_metadata
            ))

            # If it has bando_label, print it to show it was captured
            bando_label = doc_metadata.get("bando_label", "")
            if bando_label:
                print(f"      [OK PDF BANDO] {bando_label} - {pdf_url}")
            else:
                print(f"      [OK PDF] {pdf_url}")

        except Exception as e:
            print(f"      [SKIPPED] {pdf_url} - {e}")

    return docs_collected


EASYCOURSE_INFO_CODES = ("06127", "IE127", "06227", "IE227")  # Computer Engineering BSc + MSc course codes


def crawl_easycourse() -> list[Document]:
    """Crawl EasyCourse lesson timetables and EasyTest exam calendars (Computer Engineering BSc + MSc only)."""
    print("\n=== PHASE 5: EasyCourse Timetables & Exam Calendars ===")
    docs_collected = []
    current = datetime.now().year
    ay = f"{current}-{current + 1}" if datetime.now().month >= 9 else f"{current - 1}-{current}"

    # --- Target 1: lesson timetables (unchanged behaviour) ---
    lessons_url = (
        "https://easycourse.unisa.it/EasyCourse/Orario/"
        f"Dipartimento_di_Ingegneria_dellInformazione_ed_Elettrica_e_Matematica_Applicata/{ay}/1337/ttCdlHtml.html"
    )
    try:
        print(f"  Crawling EasyCourse (lessons): {lessons_url}")
        r = requests.get(lessons_url, verify=False, timeout=15)
        if r.status_code == 200:
            content = clean_html(r.text, url=lessons_url)
            if content and content.strip():
                docs_collected.append(Document(
                    page_content=content,
                    metadata={"source": lessons_url, "type": "html", "easycourse_category": "orario_lezioni"}
                ))
                print(f"      [SUCCESS] {lessons_url}")
            else:
                print(f"      [SKIPPED - Empty Content] {lessons_url}")
        else:
            print(f"      [SKIPPED - HTTP {r.status_code}] {lessons_url}")
    except Exception as e:
        print(f"      [SKIPPED - Error] {lessons_url} - {e}")

    # --- Targets 2..5: exam calendars, Computer Engineering only ---
    # The '#fragment' anchors are never sent to the server, so the index page is fetched once.
    # The real exam tables live in the linked 'Curricula/...html' sub-pages; links are filtered by course code.
    exams_index_url = (
        "https://easycourse.unisa.it/EasyTest/Calendario/"
        "Facolta_di_Ingegneria_-_Esami/1354/ttCdlHtml.html"
    )
    code_pattern = re.compile(
        r"_(?:" + "|".join(EASYCOURSE_INFO_CODES) + r")(?:_\d+)?\.html$",
        re.IGNORECASE,
    )
    try:
        print(f"  Fetching EasyTest exam index: {exams_index_url}")
        r = requests.get(exams_index_url, verify=False, timeout=15)
        if r.status_code != 200:
            print(f"      [SKIPPED - HTTP {r.status_code}] {exams_index_url}")
            return docs_collected

        soup = BeautifulSoup(r.text, "html.parser")
        exam_links = set()
        for a in soup.find_all("a", href=True):
            full_url = urljoin(exams_index_url, a["href"].strip()).split("#")[0]
            if "/Curricula/" in full_url and code_pattern.search(full_url):
                exam_links.add(full_url)

        print(f"      [INFO] {len(exam_links)} Computer Engineering exam pages found")
        for exam_url in sorted(exam_links):
            try:
                rc = requests.get(exam_url, verify=False, timeout=15)
                if rc.status_code != 200:
                    print(f"      [SKIPPED - HTTP {rc.status_code}] {exam_url}")
                    continue
                content = clean_html(rc.text, url=exam_url)
                if content and content.strip():
                    docs_collected.append(Document(
                        page_content=content,
                        metadata={"source": exam_url, "type": "html", "easycourse_category": "calendario_esami"}
                    ))
                    print(f"      [SUCCESS] {exam_url}")
                else:
                    print(f"      [SKIPPED - Empty Content] {exam_url}")
            except Exception as e:
                print(f"      [SKIPPED - Error] {exam_url} - {e}")
            time.sleep(0.3)
    except Exception as e:
        print(f"      [SKIPPED - Error] {exams_index_url} - {e}")

    return docs_collected


# ---------------------------------------------------------------------------
# LOGGER
# ---------------------------------------------------------------------------

class Logger:
    """Tee stdout to the terminal and to a UTF-8 log file."""

    def __init__(self, filename: str = LOG_PATH):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------
def main():
    start_time = time.time()
    configure_http()
    logger = Logger(LOG_PATH)
    sys.stdout = logger

    try:
        all_documents: list[Document] = []
        all_documents.extend(crawl_diem_base())
        all_documents.extend(crawl_structures())
        all_documents.extend(crawl_professor())
        all_documents.extend(crawl_courses())
        all_documents.extend(crawl_bandi())
        all_documents.extend(crawl_easycourse())
        all_documents.extend(download_pdfs())   # last: consumes the PDFs queued by every phase

        # Deduplication is intentionally delegated to kb_filter.py.
        export_to_json(all_documents, OUTPUT_PATH)
        print_summary(all_documents)
        elapsed_time = time.time() - start_time
        print(f"\n=== TOTAL ELAPSED TIME: {elapsed_time:.2f} seconds => {elapsed_time / 60:.2f} minutes")
    finally:
        sys.stdout = logger.terminal
        logger.close()


if __name__ == "__main__":
    main()