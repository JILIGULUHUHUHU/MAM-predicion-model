import time
import pandas as pd
import requests


BASE_URL = "https://rest.uniprot.org/uniprotkb/search"


def query_uniprot_go(go_term, organism="9606", reviewed=True):
    """Query UniProt for proteins annotated with a specific GO term."""
    query = f"(go:{go_term}) AND (organism_id:{organism})"
    if reviewed:
        query += " AND (reviewed:true)"

    return _query_uniprot(query)


def query_uniprot_keyword(keyword, organism="9606", reviewed=True):
    """Query UniProt for proteins annotated with a specific keyword."""
    query = f'(keyword:"{keyword}") AND (organism_id:{organism})'
    if reviewed:
        query += " AND (reviewed:true)"

    return _query_uniprot(query)


def _query_uniprot(query):
    """Execute a UniProt REST API query with pagination."""
    params = {"query": query, "format": "json", "size": 500}
    results = []
    url = BASE_URL

    while url:
        try:
            resp = requests.get(
                url, params=params if url == BASE_URL else None, timeout=60
            )
            resp.raise_for_status()
        except requests.exceptions.HTTPError as e:
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 10))
                time.sleep(wait)
                continue
            raise e

        data = resp.json()
        results.extend(data.get("results", []))

        link_header = resp.headers.get("Link", "")
        url = None
        for part in link_header.split(","):
            if 'rel="next"' in part:
                url = part.split(";")[0].strip("<>")
                time.sleep(0.3)
                break

    return _parse_uniprot_results(results)


def _parse_uniprot_results(results):
    """Parse UniProt API results into a DataFrame."""
    rows = []
    for entry in results:
        seq_info = entry.get("sequence", {})
        seq = seq_info.get("value", "") if isinstance(seq_info, dict) else ""
        if not seq:
            continue

        genes = entry.get("genes", [])
        gene_name = genes[0].get("geneName", {}).get("value", "") if genes else ""

        rows.append({
            "uniprot_id": entry.get("primaryAccession", ""),
            "entry_name": entry.get("uniProtkbId", ""),
            "gene": gene_name,
            "sequence": seq,
            "length": len(seq),
            "organism": entry.get("organism", {}).get("scientificName", ""),
        })

    return pd.DataFrame(rows)


def fetch_sequences_by_accessions(accessions):
    """Fetch full sequences for a list of UniProt accession IDs."""
    results = []
    for i in range(0, len(accessions), 100):
        batch = accessions[i : i + 100]
        query = " OR ".join(f"(accession:{acc})" for acc in batch)
        df = _query_uniprot(query)
        results.append(df)
        time.sleep(0.5)
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()
