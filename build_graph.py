#!/usr/bin/env python3
"""
build_graph.py
----------------
Pulls professor/publication data from three sources:
  1. Google Scholar   (via `scholarly` - unofficial scraper, fragile, see notes below)
  2. Semantic Scholar  (official Academic Graph API - primary source of truth)
  3. arXiv             (official API - catches new preprints fastest)

Then AUTO-DISCOVERS additional collaborators: rather than hand-typing 100+ names
(which risks the same kind of wrong-person mismatch you already caught with
Golubov), the script extracts real co-authors from each seed professor's actual
paper list and expands the graph itself, up to --max-nodes total.

Writes data/graph.json in the exact shape cmp_network.html expects.

USAGE
  python build_graph.py                          # seeds only, no expansion
  python build_graph.py --expand --max-nodes 130  # seeds + auto-discovered collaborators
  python build_graph.py --no-scholar              # skip Google Scholar (faster, no CAPTCHA risk)
  python build_graph.py --seed seeds.json --expand --max-nodes 100

NOTES ON GOOGLE SCHOLAR
  `scholarly` scrapes public Google Scholar pages. Google does not offer an API and
  actively rate-limits/CAPTCHAs scrapers, especially from cloud/datacenter IPs (which is
  exactly what GitHub Actions runners are). Practical implications:
    - Running this from a residential/personal IP works far more reliably than CI.
    - If you must run it in CI, route through a proxy pool (scholarly supports
      ProxyGenerator with free or paid proxies - see fetch_google_scholar() below).
    - Treat Scholar as an ENRICHMENT source (citation counts, Scholar profile links),
      not the backbone. Semantic Scholar + arXiv should carry the graph even if
      Scholar fails or gets blocked on a given run.
    - This script fails soft on Scholar: if a lookup errors out, that professor
      just won't have Scholar-sourced fields, and the run continues.

NOTES ON --expand
  Auto-discovered collaborators are found via NAME SEARCH (we don't know their
  IDs ahead of time), so they carry the same disambiguation risk common names
  have throughout this script - a coauthor called "J. Kim" might resolve to the
  wrong J. Kim. Each auto-discovered node gets "_autoDiscovered": true in the
  intermediate data so you can spot-check/fix them by adding an explicit
  semanticScholarId to DEFAULT_SEEDS once you notice a bad match, same as
  we did for Golubov/Meyer.
  Expansion is also SLOW and API-hungry: ~4 requests + delays per node, so 130
  nodes is roughly 10-15 minutes and a few hundred API calls. Get a free
  Semantic Scholar API key (semanticscholar.org/product/api) if you hit 429s.
"""

import json
import re
import time
import argparse
import sys
from collections import Counter
from pathlib import Path
from datetime import datetime, timezone

import requests

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1"
OPENALEX_API = "https://api.openalex.org"
ARXIV_API = "https://export.arxiv.org/api/query"

# Field-of-study -> legend color bucket used by the frontend
FIELD_KEYWORDS = {
    "sc":   ["superconduct", "josephson", "cooper pair", "andreev"],
    "topo": ["topological", "berry curvature", "chern", "weyl", "dirac semimetal",
             "moire", "moiré", "twisted bilayer"],
    "mag":  ["magnon", "spintronic", "skyrmion", "spin caloritronic", "antiferromagnet",
             "spin liquid", "quantum magnet"],
}

REQUEST_DELAY = 1.2  # seconds between API calls, be polite to free tiers

# ---------------------------------------------------------------------------
# SEED LIST — verified via web search (institution + identity cross-checked,
# not just name-matched). semanticScholarId is set where confirmed; None
# entries fall back to name search (usually fine for distinctive names, but
# check the printed h-index against what you'd expect - a wildly low number
# almost always means a mismatch, same pattern as the earlier Golubov bug).
#
# HOW TO FIND/FIX A semanticScholarId:
#   1. https://www.semanticscholar.org -> search name -> open correct profile
#   2. URL is semanticscholar.org/author/Some-Name/1234567 -> copy the number
# ---------------------------------------------------------------------------

DEFAULT_SEEDS = [
    # --- from your PhD advisor list / earlier sessions ---
    {"id": "nganguli",  "name": "Nirmal Ganguli",   "inst": "IISER Bhopal",
     "semanticScholarId": "2230898382"},
    {"id": "jmeyer",    "name": "Julia S. Meyer",   "inst": "CEA Grenoble / Neel Institute",
     "semanticScholarId": None},  # TODO: name search finds nothing - look up manually
    {"id": "jrobinson", "name": "Jason W. A. Robinson", "inst": "University of Cambridge",
     "semanticScholarId": "2238234588"},
    {"id": "nbirge",    "name": "Norman O. Birge",  "inst": "Michigan State University",
     "semanticScholarId": "5019407"},
    {"id": "agolubov",  "name": "Alexander A. Golubov", "inst": "University of Twente",
     "semanticScholarId": None},  # TODO: search matched wrong Golubov (h=1) - fix manually
    {"id": "meschrig",  "name": "Matthias Eschrig", "inst": "Royal Holloway, University of London",
     "semanticScholarId": "2341330952"},
    {"id": "jlinder",   "name": "Jacob Linder",     "inst": "NTNU Trondheim",
     "semanticScholarId": "4866030"},

    # --- newly requested, identity/institution verified via web search ---
    {"id": "vmadhavan", "name": "Vidya Madhavan",   "inst": "UIUC",
     "semanticScholarId": None},  # STM/quantum materials; distinctive name, search should work
    {"id": "pjherrero",  "name": "Pablo Jarillo-Herrero", "inst": "MIT",
     "semanticScholarId": "1412823938"},  # confirmed via semanticscholar.org profile URL
    {"id": "rpsingh",   "name": "Ravi Prakash Singh", "inst": "IISER Bhopal",
     "semanticScholarId": None},  # quantum materials/crystal growth, confirmed via Google Scholar
    {"id": "mscheurer", "name": "Mathias S. Scheurer", "inst": "University of Stuttgart",
     "semanticScholarId": None},  # note: goes by "Mathias" not "Matthias" in his own byline
    {"id": "ssachdev",  "name": "Subir Sachdev",    "inst": "Harvard University",
     "semanticScholarId": None},
    {"id": "tsenthil",  "name": "T. Senthil",       "inst": "MIT",
     "semanticScholarId": None},
    {"id": "mueda",     "name": "Masahito Ueda",    "inst": "University of Tokyo / RIKEN",
     "semanticScholarId": None},  # best-guess match for "Miguel M Ueda" - VERIFY this is who you meant
    {"id": "kwatanabe", "name": "Kenji Watanabe",   "inst": "NIMS Japan",
     "semanticScholarId": None},  # hBN crystal growth - extremely high edge count expected
    {"id": "kfmak",     "name": "Kin Fai Mak",      "inst": "Cornell University",
     "semanticScholarId": None},
    {"id": "ayazdani",  "name": "Ali Yazdani",      "inst": "Princeton University",
     "semanticScholarId": None},

    # --- add new professors here, same shape ---
    # {"id": "shortname", "name": "Full Name", "inst": "Institution",
    #  "semanticScholarId": "1234567"},   # or None to fall back to name search
]


def slugify(name: str) -> str:
    """Turn a discovered coauthor's name into a short node id."""
    s = re.sub(r"[^a-z0-9]", "", name.lower())
    return s[:16] or "unknown"


# ---------------------------------------------------------------------------
# SOURCE 1: Semantic Scholar  (primary backbone)
# ---------------------------------------------------------------------------

def fetch_semantic_scholar(name: str, known_id: str | None = None) -> dict | None:
    """
    Fetch an author's profile + paper list.
    If known_id is given, skip search entirely and go straight to that author ID -
    this is the fix for common-name collisions (search picking the wrong "A. Golubov").
    """
    try:
        if known_id:
            author_id = known_id
            r = requests.get(
                f"{SEMANTIC_SCHOLAR_API}/author/{author_id}",
                params={"fields": "name,affiliations,hIndex,paperCount,citationCount"},
                timeout=15,
            )
            r.raise_for_status()
            author = r.json()
        else:
            r = requests.get(
                f"{SEMANTIC_SCHOLAR_API}/author/search",
                params={"query": name, "fields": "name,affiliations,hIndex,paperCount,citationCount"},
                timeout=15,
            )
            r.raise_for_status()
            results = r.json().get("data", [])
            if not results:
                return None
            author = results[0]  # best match; ambiguous for common names - use known_id instead
            author_id = author["authorId"]

        time.sleep(REQUEST_DELAY)
        r2 = requests.get(
            f"{SEMANTIC_SCHOLAR_API}/author/{author_id}/papers",
            params={"fields": "title,year,venue,externalIds,authors", "limit": 100},
            timeout=15,
        )
        r2.raise_for_status()
        papers = r2.json().get("data", [])

        return {
            "semanticScholarId": author_id,
            "hIndex": author.get("hIndex"),
            "paperCount": author.get("paperCount"),
            "citationCount": author.get("citationCount"),
            "affiliations": author.get("affiliations", []),
            "papers": papers,
        }
    except requests.RequestException as e:
        print(f"  [semantic scholar] failed for {name}: {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# SOURCE 2: arXiv  (freshest preprints)
# ---------------------------------------------------------------------------

def fetch_arxiv_recent(name: str, max_results: int = 25) -> list[dict]:
    """Pull recent arXiv listings for an author name."""
    import xml.etree.ElementTree as ET

    query = f'au:"{name}"'
    try:
        r = requests.get(
            ARXIV_API,
            params={
                "search_query": query,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
                "max_results": max_results,
            },
            timeout=15,
        )
        r.raise_for_status()
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(r.content)
        entries = []
        for entry in root.findall("atom:entry", ns):
            entries.append({
                "title": entry.findtext("atom:title", default="", namespaces=ns).strip(),
                "published": entry.findtext("atom:published", default="", namespaces=ns),
                "arxiv_id": entry.findtext("atom:id", default="", namespaces=ns).split("/abs/")[-1],
                "authors": [a.findtext("atom:name", namespaces=ns)
                            for a in entry.findall("atom:author", ns)],
            })
        return entries
    except (requests.RequestException, ET.ParseError) as e:
        print(f"  [arxiv] failed for {name}: {e}", file=sys.stderr)
        return []


# ---------------------------------------------------------------------------
# SOURCE 3: OpenAlex  (open, unrestricted — good affiliation cross-check)
# ---------------------------------------------------------------------------

def fetch_openalex(name: str) -> dict | None:
    try:
        r = requests.get(
            f"{OPENALEX_API}/authors",
            params={"search": name, "per-page": 1},
            timeout=15,
        )
        r.raise_for_status()
        results = r.json().get("results", [])
        if not results:
            return None
        a = results[0]
        return {
            "openalexId": a.get("id"),
            "worksCount": a.get("works_count"),
            "citedByCount": a.get("cited_by_count"),
            "lastKnownInstitution": (a.get("last_known_institutions") or [{}])[0].get("display_name"),
        }
    except requests.RequestException as e:
        print(f"  [openalex] failed for {name}: {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# GOOGLE SCHOLAR  (enrichment only — see module docstring for caveats)
# ---------------------------------------------------------------------------

def fetch_google_scholar(name: str, use_proxy: bool = False) -> dict | None:
    try:
        from scholarly import scholarly, ProxyGenerator
    except ImportError:
        print("  [scholar] `scholarly` not installed, skipping. pip install scholarly", file=sys.stderr)
        return None

    if use_proxy:
        pg = ProxyGenerator()
        pg.FreeProxies()  # swap for pg.ScraperAPI(key) / pg.Luminati(...) for reliability
        scholarly.use_proxy(pg)

    try:
        search_query = scholarly.search_author(name)
        author = next(search_query, None)
        if author is None:
            return None
        filled = scholarly.fill(author, sections=["basics", "indices", "counts"])
        return {
            "scholarId": filled.get("scholar_id"),
            "citedby": filled.get("citedby"),
            "hindex": filled.get("hindex"),
            "interests": filled.get("interests", []),
            "profileUrl": f"https://scholar.google.com/citations?user={filled.get('scholar_id')}",
        }
    except StopIteration:
        return None
    except Exception as e:
        # scholarly raises assorted exceptions on CAPTCHA/block — fail soft
        print(f"  [scholar] failed for {name}: {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# MERGE LOGIC
# ---------------------------------------------------------------------------

def classify_field(papers: list[dict]) -> str:
    """Bucket a professor into sc / topo / mag / gen by keyword frequency in titles."""
    scores = {k: 0 for k in FIELD_KEYWORDS}
    for p in papers:
        title = (p.get("title") or "").lower()
        for field, kws in FIELD_KEYWORDS.items():
            if any(kw in title for kw in kws):
                scores[field] += 1
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "gen"


def fetch_one_person(pid: str, name: str, inst: str, known_id: str | None,
                      skip_scholar: bool, auto: bool = False) -> dict:
    """Run all sources for a single person, return the merged node dict."""
    print(f"Fetching: {name}{'  [auto-discovered]' if auto else ''}")

    ss = fetch_semantic_scholar(name, known_id=known_id)
    time.sleep(REQUEST_DELAY)
    oa = fetch_openalex(name)
    time.sleep(REQUEST_DELAY)
    arxiv_recent = fetch_arxiv_recent(name)
    time.sleep(REQUEST_DELAY)
    scholar = None if skip_scholar else fetch_google_scholar(name)
    time.sleep(REQUEST_DELAY)

    papers = (ss or {}).get("papers", [])
    return {
        "id": pid,
        "name": name,
        "inst": inst or (oa or {}).get("lastKnownInstitution", ""),
        "field": classify_field(papers),
        "h": (ss or {}).get("hIndex") or (scholar or {}).get("hindex") or 0,
        "tags": (scholar or {}).get("interests", [])[:3],
        "site": (scholar or {}).get("profileUrl", "—"),
        "_semanticScholarId": (ss or {}).get("semanticScholarId"),
        "_openalexId": (oa or {}).get("openalexId"),
        "_papers": papers,            # kept for edge-building, stripped before final write
        "_arxivRecent": arxiv_recent,
        "_autoDiscovered": auto,
    }


def build_nodes(seeds: list[dict], skip_scholar: bool) -> dict:
    """Fetch all three sources for each seed professor, merge into node records."""
    nodes = {}
    for person in seeds:
        nodes[person["id"]] = fetch_one_person(
            person["id"], person["name"], person.get("inst", ""),
            person.get("semanticScholarId"), skip_scholar,
        )
    return nodes


def discover_and_expand(nodes: dict, max_nodes: int, skip_scholar: bool) -> dict:
    """
    Look at every seed professor's actual paper list, count how often each
    non-seed coauthor appears, and fetch the most frequent ones as new nodes -
    real collaborators pulled from real data instead of a hand-typed list.
    """
    seed_names_lower = {n["name"].lower() for n in nodes.values()}
    counts = Counter()
    for n in nodes.values():
        for paper in n["_papers"]:
            for a in paper.get("authors", []):
                nm = (a.get("name") or "").strip()
                if nm and nm.lower() not in seed_names_lower and len(nm.split()) >= 2:
                    counts[nm] += 1

    slots = max(0, max_nodes - len(nodes))
    candidates = [name for name, _ in counts.most_common(slots * 2)]  # overfetch, some will fail

    added = 0
    for name in candidates:
        if added >= slots:
            break
        pid = slugify(name)
        if pid in nodes:
            continue
        node = fetch_one_person(pid, name, "", None, skip_scholar, auto=True)
        if node["_semanticScholarId"] is None:
            continue  # couldn't resolve this name to a real profile, skip rather than guess
        nodes[pid] = node
        seed_names_lower.add(name.lower())
        added += 1

    print(f"\nAuto-discovery added {added} collaborator nodes.")
    return nodes


def build_edges(nodes: dict) -> list[dict]:
    """
    Cross-reference each professor's Semantic Scholar paper list against every
    other node's name to find joint publications -> graph edges. Checks BOTH
    people's paper lists (not just one), since a paper might be in person A's
    fetched list but miss person B's list if B has >100 papers total (the API
    call caps at 100, oldest/newest ordering isn't guaranteed).
    """
    def find_shared(papers, other_surname):
        found = {}
        for paper in papers:
            author_names = [a.get("name", "").lower() for a in paper.get("authors", [])]
            if any(other_surname in an for an in author_names):
                found[paper.get("title")] = {
                    "title": paper.get("title"),
                    "venue": paper.get("venue") or "—",
                    "year": paper.get("year"),
                    "arxiv": (paper.get("externalIds") or {}).get("ArXiv", "—"),
                }
        return found

    edges = []
    ids = list(nodes.keys())
    for i, a_id in enumerate(ids):
        for b_id in ids[i + 1:]:
            a_surname = nodes[a_id]["name"].lower().split()[-1]
            b_surname = nodes[b_id]["name"].lower().split()[-1]
            shared = {}
            shared.update(find_shared(nodes[a_id]["_papers"], b_surname))
            shared.update(find_shared(nodes[b_id]["_papers"], a_surname))
            if shared:
                edges.append({"s": a_id, "t": b_id, "papers": list(shared.values())})
    return edges


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=str, help="path to a JSON file of seed professors")
    parser.add_argument("--no-scholar", action="store_true", help="skip Google Scholar lookups")
    parser.add_argument("--expand", action="store_true",
                         help="auto-discover additional collaborators from real coauthor data")
    parser.add_argument("--max-nodes", type=int, default=130,
                         help="total node cap when --expand is used (default 130)")
    parser.add_argument("--out", type=str, default="data/graph.json")
    args = parser.parse_args()

    seeds = DEFAULT_SEEDS
    if args.seed:
        seeds = json.loads(Path(args.seed).read_text())

    nodes = build_nodes(seeds, skip_scholar=args.no_scholar)

    if args.expand:
        nodes = discover_and_expand(nodes, args.max_nodes, skip_scholar=args.no_scholar)

    edges = build_edges(nodes)

    # strip internal fields before writing the public graph.json
    clean_nodes = []
    for n in nodes.values():
        n = dict(n)
        n.pop("_papers", None)
        n.pop("_arxivRecent", None)
        clean_nodes.append(n)

    out = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "nodes": clean_nodes,
        "edges": edges,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {len(clean_nodes)} nodes, {len(edges)} edges -> {out_path}")


if __name__ == "__main__":
    main()
