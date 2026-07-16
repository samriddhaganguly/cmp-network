#!/usr/bin/env python3
"""
build_graph.py
----------------
Pulls professor/publication data from three sources:
  1. Google Scholar   (via `scholarly` - unofficial scraper, fragile, see notes below)
  2. Semantic Scholar  (official Academic Graph API - primary source of truth)
  3. arXiv             (official API - catches new preprints fastest)

Cross-references co-authorship *within your seed list* to build graph edges,
and writes data/graph.json in the exact shape cmp_network.html expects.

USAGE
  python build_graph.py                 # full run, writes data/graph.json
  python build_graph.py --no-scholar     # skip Google Scholar (faster, no CAPTCHA risk)
  python build_graph.py --seed seeds.json

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
"""

import json
import time
import argparse
import sys
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
    "topo": ["topological", "berry curvature", "chern", "weyl", "dirac semimetal"],
    "mag":  ["magnon", "spintronic", "skyrmion", "spin caloritronic", "antiferromagnet"],
}

REQUEST_DELAY = 1.2  # seconds between API calls, be polite to free tiers

# ---------------------------------------------------------------------------
# SEED LIST — the professors you want tracked.
# `name` is used for lookups; give Semantic Scholar / OpenAlex IDs directly
# once you know them (avoids ambiguous-name mismatches on common names).
# ---------------------------------------------------------------------------

DEFAULT_SEEDS = [
    {"id": "nganguli",  "name": "Nirmal Ganguli",   "inst": "IISER Bhopal"},
    {"id": "jmeyer",    "name": "Julia S. Meyer",   "inst": "CEA Grenoble / Neel Institute"},
    {"id": "jrobinson", "name": "Jason W. A. Robinson", "inst": "University of Cambridge"},
    {"id": "nbirge",    "name": "Norman O. Birge",  "inst": "Michigan State University"},
    {"id": "agolubov",  "name": "Alexander A. Golubov", "inst": "University of Twente"},
    {"id": "meschrig",  "name": "Matthias Eschrig", "inst": "Royal Holloway, University of London"},
    {"id": "jlinder",   "name": "Jacob Linder",     "inst": "NTNU Trondheim"},
]

# ---------------------------------------------------------------------------
# SOURCE 1: Semantic Scholar  (primary backbone)
# ---------------------------------------------------------------------------

def fetch_semantic_scholar(name: str) -> dict | None:
    """Search for an author by name, return their profile + paper list."""
    try:
        r = requests.get(
            f"{SEMANTIC_SCHOLAR_API}/author/search",
            params={"query": name, "fields": "name,affiliations,hIndex,paperCount,citationCount"},
            timeout=15,
        )
        r.raise_for_status()
        results = r.json().get("data", [])
        if not results:
            return None
        author = results[0]  # best match; refine with an explicit ID for common names
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


def build_nodes(seeds: list[dict], skip_scholar: bool) -> dict:
    """Fetch all three sources for each seed professor, merge into node records."""
    nodes = {}
    for person in seeds:
        pid, name = person["id"], person["name"]
        print(f"Fetching: {name}")

        ss = fetch_semantic_scholar(name)
        time.sleep(REQUEST_DELAY)
        oa = fetch_openalex(name)
        time.sleep(REQUEST_DELAY)
        arxiv_recent = fetch_arxiv_recent(name)
        time.sleep(REQUEST_DELAY)
        scholar = None if skip_scholar else fetch_google_scholar(name)
        time.sleep(REQUEST_DELAY)

        papers = (ss or {}).get("papers", [])
        nodes[pid] = {
            "id": pid,
            "name": name,
            "inst": person.get("inst", (oa or {}).get("lastKnownInstitution", "")),
            "field": classify_field(papers),
            "h": (ss or {}).get("hIndex") or (scholar or {}).get("hindex") or 0,
            "tags": (scholar or {}).get("interests", [])[:3],
            "site": (scholar or {}).get("profileUrl", "—"),
            "_semanticScholarId": (ss or {}).get("semanticScholarId"),
            "_openalexId": (oa or {}).get("openalexId"),
            "_papers": papers,            # kept for edge-building, stripped before final write
            "_arxivRecent": arxiv_recent,
        }
    return nodes


def build_edges(nodes: dict) -> list[dict]:
    """
    Cross-reference each professor's Semantic Scholar paper list against every
    other seed professor's name to find joint publications -> graph edges.
    """
    edges = []
    ids = list(nodes.keys())
    for i, a_id in enumerate(ids):
        for b_id in ids[i + 1:]:
            a_papers = nodes[a_id]["_papers"]
            b_name_lower = nodes[b_id]["name"].lower()
            shared = []
            for paper in a_papers:
                author_names = [a.get("name", "").lower() for a in paper.get("authors", [])]
                if any(b_name_lower.split()[-1] in an for an in author_names):  # match on surname
                    shared.append({
                        "title": paper.get("title"),
                        "venue": paper.get("venue") or "—",
                        "year": paper.get("year"),
                        "arxiv": (paper.get("externalIds") or {}).get("ArXiv", "—"),
                    })
            if shared:
                edges.append({"s": a_id, "t": b_id, "papers": shared})
    return edges


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=str, help="path to a JSON file of seed professors")
    parser.add_argument("--no-scholar", action="store_true", help="skip Google Scholar lookups")
    parser.add_argument("--out", type=str, default="data/graph.json")
    args = parser.parse_args()

    seeds = DEFAULT_SEEDS
    if args.seed:
        seeds = json.loads(Path(args.seed).read_text())

    nodes = build_nodes(seeds, skip_scholar=args.no_scholar)
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
