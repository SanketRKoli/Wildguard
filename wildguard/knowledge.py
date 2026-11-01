"""Free, key-less public APIs: Wikipedia (description, photo), GBIF + Wikidata (IUCN Red List status).
Every call is time-limited and guarded - if the network is down the app simply shows 'not available'."""
import os
from functools import lru_cache
from urllib.parse import quote

import requests

UA = {"User-Agent": "WildGuard/1.0 (open-source educational project)"}
TIMEOUT = 8
OFFLINE = os.environ.get("WG_OFFLINE") == "1"

STATUS_NAMES = {
    "CR": "Critically Endangered", "EN": "Endangered", "VU": "Vulnerable", "NT": "Near Threatened",
    "LC": "Least Concern", "DD": "Data Deficient", "EW": "Extinct in the Wild", "EX": "Extinct", "NE": "Not Evaluated",
}
STATUS_COLOURS = {
    "CR": "#b71c1c", "EN": "#e65100", "VU": "#f9a825", "NT": "#9e9d24", "LC": "#2e7d32",
    "DD": "#607d8b", "EW": "#4a148c", "EX": "#000000", "NE": "#9e9e9e",
}
_LONG = {"CRITICALLY": "CR", "ENDANGERED": "EN", "VULNERABLE": "VU", "NEAR": "NT", "LEAST": "LC",
         "DEFICIENT": "DD", "WILD": "EW", "EXTINCT": "EX"}


def normalise_status(text):
    """'LEAST_CONCERN', 'least concern', 'endangered species', 'EN' ... -> 'LC' / 'EN' ..."""
    if not text:
        return None
    t = str(text).strip().upper().replace("_", " ")
    if t in STATUS_NAMES:
        return t
    if "EXTINCT" in t and "WILD" in t:
        return "EW"
    if "CRITICALLY" in t:
        return "CR"
    if "NEAR" in t and "THREAT" in t:
        return "NT"
    for key, code in _LONG.items():
        if key in t:
            return code
    return None


def _get(url, **kw):
    if OFFLINE:
        return None
    try:
        r = requests.get(url, headers=UA, timeout=TIMEOUT, **kw)
        return r if r.status_code == 200 else None
    except Exception:
        return None


@lru_cache(maxsize=512)
def wikipedia(title):
    """{'extract', 'image', 'url'} or {}"""
    if not title:
        return {}
    r = _get("https://en.wikipedia.org/api/rest_v1/page/summary/" + quote(title.replace(" ", "_"), safe=""))
    if not r:
        return {}
    try:
        j = r.json()
        if j.get("type") == "disambiguation":
            return {}
        return {"extract": j.get("extract", ""), "image": (j.get("thumbnail") or {}).get("source", ""),
                "url": ((j.get("content_urls") or {}).get("desktop") or {}).get("page", "")}
    except Exception:
        return {}


@lru_cache(maxsize=512)
def iucn_status(scientific):
    """Returns {'code': 'EN', 'label': 'Endangered', 'source': 'GBIF'} or {}"""
    if not scientific:
        return {}
    # 1) GBIF
    m = _get("https://api.gbif.org/v1/species/match", params={"name": scientific, "kingdom": "Animalia"})
    if m:
        try:
            key = m.json().get("usageKey")
            if key:
                r = _get(f"https://api.gbif.org/v1/species/{key}/iucnRedListCategory")
                if r:
                    j = r.json()
                    code = normalise_status(j.get("category") or j.get("code"))
                    if code:
                        return {"code": code, "label": STATUS_NAMES[code], "source": "IUCN Red List via GBIF"}
        except Exception:
            pass
    # 2) Wikidata
    q = ('SELECT ?l WHERE { ?t wdt:P225 "%s" . ?t wdt:P141 ?s . ?s rdfs:label ?l . FILTER(LANG(?l)="en") } LIMIT 5'
         % scientific.replace('"', ""))
    r = _get("https://query.wikidata.org/sparql", params={"query": q, "format": "json"})
    if r:
        try:
            for b in r.json()["results"]["bindings"]:
                code = normalise_status(b["l"]["value"])
                if code:
                    return {"code": code, "label": STATUS_NAMES[code], "source": "IUCN Red List via Wikidata"}
        except Exception:
            pass
    return {}


@lru_cache(maxsize=256)
def resolve_scientific(common_name):
    """Common name -> scientific name through GBIF (used only by the CLIP fallback)."""
    r = _get("https://api.gbif.org/v1/species/search",
             params={"q": common_name, "rank": "SPECIES", "highertaxonKey": 1, "limit": 1})
    try:
        res = r.json().get("results", []) if r else []
        return res[0].get("canonicalName", "") if res else ""
    except Exception:
        return ""


def species_info(scientific, common):
    sci = scientific or resolve_scientific(common)
    wiki = wikipedia(common) or wikipedia(sci)
    return {"scientific": sci, "common": common, "wiki": wiki, "status": iucn_status(sci)}
