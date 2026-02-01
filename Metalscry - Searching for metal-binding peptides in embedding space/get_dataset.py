"""
Helper functions for creating metalscry dataset.
"""
from typing import Any, Dict, List, Set, Tuple, Optional, Sequence, Iterable
import gzip
import io
from collections import defaultdict
import itertools
import pandas as pd
import ast
import re
import json
import os
import numpy as np
from pathlib import Path
import time
import requests


METAL_SYMBOLS = {
    '3CO', '3NI', '4TI', '6MO', 'AG', 'AL', 'AU', 'AU3', 'BA', 'CA', 'CD', 'CE', 'CO',
      'CR', 'CS', 'CU', 'CU1', 'DY', 'ER3', 'EU', 'EU3', 'FE', 'FE2', 'HG', 'HO3', 'IR',
        'IR3', 'K', 'LA', 'LI', 'LU', 'MG', 'MN', 'MN3', 'NA', 'NI', 'OS', 'OS4', 'PB',
          'PD', 'PR', 'PT', 'RB', 'RH', 'RH3', 'RU', 'SM', 'SR', 'TB', 'V', 'YB', 'ZN'
}

def is_metal_ligand(code: str) -> bool:
    return code.upper() in METAL_SYMBOLS

def _split_row(line: str):
    # BioLiP says tab-delimited; fall back to any whitespace if no tab found
    return line.rstrip("\n").split("\t") if "\t" in line else line.split()

def count_metals(path: str, *, metal_only: bool = True) -> int:
    """Count the number of metal-containing ligands in a BioLiP2 file."""
    count = 0
    for raw in _open_biolip2(path):
        if not raw or raw.startswith('#'):
            continue
        cols = raw.split('\t')
        parsed = _parse_line_to_partial(cols)
        if parsed is None:
            continue
        pdb_id, chain_id, ligand_id, residues, biolip_seq = parsed
        if metal_only and not is_metal_ligand(ligand_id):
            continue
        count += 1
    return count

def get_rand_non_metal(path: str):
    
    for raw in _open_biolip2(path):
        if not raw or raw.startswith('#'):
            continue
        cols = raw.split('\t')
        parsed = _parse_line_to_partial(cols)
        if parsed is None:
            continue
        pdb_id, chain_id, ligand_id, residues, biolip_seq = parsed
        if not is_metal_ligand(ligand_id):
            return {
                "pdb_ids": pdb_id,
                "ligand": ligand_id,
                "sequence": biolip_seq
			}

def count_rows_in_db(path: str) -> int:
    """Count the number of rows in a BioLiP2 database file."""
    count = 0
    for raw in _open_biolip2(path):
        if not raw or raw.startswith('#'):
            continue
        count += 1
    return count

def _parse_line_to_partial(cols: Sequence[str]) -> Optional[Tuple[str, str, str, List[str], str]]:
    """Extract (pdb, chain, ligand_id, residues[], biolip_seq) from columns.

    Returns None if the line is malformed.
    """
    if len(cols) < 21:
        return None
    pdb_id   = cols[0].strip()
    chain_id = cols[1].strip()
    ligand   = cols[4].strip()
    residues_field = cols[7].strip()  # binding-site residues (PDB numbering)
    biolip_seq     = cols[20].strip()
    residues = residues_field.split() if residues_field else []
    return pdb_id, chain_id, ligand, residues, biolip_seq

def parse_raw_biolip2(inpath: str, outpath: str)  -> "pd.DataFrame":
        # open input (gz or plain)
    opener = gzip.open if inpath.endswith(".gz") else open
    kept = 0
    with opener(inpath, "rt", encoding="utf-8", newline="") as fin, \
         gzip.open(outpath, "wt", encoding="utf-8", newline="") as fout:
        for line in fin:
            if not line.strip():
                continue
            parts = _split_row(line)
            parsed = _parse_line_to_partial(parts)
            # col 05 = ligand CCD ID (0-based idx 4)
            if len(parts) > 4 and is_metal_ligand(parts[4]):
                pdb_id, chain_id, ligand_id, residues, biolip_seq = parsed
                fout.write(f"{pdb_id}\t{chain_id}\t{ligand_id}\t{residues}\t{biolip_seq}\n")
                kept += 1
    print("wrote rows:", kept)

def save_raw_metal_ds(inpath: str, outpath: str) -> None:
    # open input (gz or plain)
    opener = gzip.open if inpath.endswith(".gz") else open
    kept = 0
    with opener(inpath, "rt", encoding="utf-8", newline="") as fin, \
         gzip.open(outpath, "wt", encoding="utf-8", newline="") as fout:
        for line in fin:
            if not line.strip():
                continue
            parts = _split_row(line)
            # col 05 = ligand CCD ID (0-based idx 4)
            if len(parts) > 4 and is_metal_ligand(parts[4]):
                # ensure newline is present
                if not line.endswith("\n"):
                    line += "\n"
                fout.write(line)
                kept += 1
    print("wrote rows:", kept)

def get_unique_metals(path: str) -> Set[str]:
    """Get a set of unique metal ligand IDs from a BioLiP2 file."""
    unique_metals = set()
    for raw in _open_biolip2(path):
        if not raw or raw.startswith('#'):
            continue
        cols = _split_row(raw)
        parsed = _parse_line_to_partial(cols)
        if parsed is None:
            continue
        pdb_id, chain_id, ligand_id, residues, biolip_seq = parsed
        if is_metal_ligand(ligand_id):
            unique_metals.add(ligand_id)
    return unique_metals

"""
Filtering
"""

def count_rows_no_sequence(path: str) -> int:
    """Count the number of rows in a BioLiP2 file with no receptor sequence."""
    count = 0
    for raw in _open_biolip2(path):
        if not raw or raw.startswith('#'):
            continue
        cols = raw.split('\t')
        parsed = _parse_line_to_partial(cols)
        if parsed is None:
            continue
        pdb_id, chain_id, ligand_id, residues, biolip_seq = parsed
        if biolip_seq.strip():
            count += 1
            print(count)
    return count

def _open_biolip2(path: str) -> Iterable[str]:
    """Yield lines from a BioLiP2 annotation file (.txt or .gz)."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No such file: {path}")
    if path.lower().endswith('.gz'):
        f: io.TextIOBase = gzip.open(path, 'rt', encoding='utf-8', errors='replace')
    else:
        f = open(path, 'r', encoding='utf-8', errors='replace')
    try:
        for line in f:
            yield line.rstrip('\n')
    finally:
        f.close()

def get_row(idx: int, path: str) -> Dict[str, Any]:
    row = next(itertools.islice(_open_biolip2(path), idx, idx+1)).split('\t')
    out = {}
    out['pdb_ids'] = sorted(list(row['pdb_ids'].strip().split(',')))
    out['chains'] = sorted(list(row[1].strip().split(',')))
    out['sequence'] = row[2].strip()
    residuedict =  row["metal_residues"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    out['metal_residues'] = residuedict
    return out

def _pairwise_identity(seq1: str, seq2: str) -> float:
    """
    Global sequence identity in [0,1] using Bio.Align.PairwiseAligner if available.
    Falls back to difflib.SequenceMatcher().ratio() if Biopython is not installed.

    Identity = (# identical residues) / (aligned length without gaps)
    """
    a = (seq1 or "").upper().replace("*", "")
    b = (seq2 or "").upper().replace("*", "")
    if not a or not b:
        return 0.0

    try:
        from Bio import Align
        aligner = Align.PairwiseAligner()
        aligner.mode = "global"
        aligner.match_score = 2.0
        aligner.mismatch_score = -1.0
        aligner.open_gap_score = -2.0
        aligner.extend_gap_score = -0.5

        alignment = aligner.align(a, b)[0]

        matches = 0
        aligned_len = 0
        for (a_start, a_end), (b_start, b_end) in zip(*alignment.aligned):
            block_len = min(a_end - a_start, b_end - b_start)
            if block_len <= 0:
                continue
            sub_a = a[a_start:a_start + block_len]
            sub_b = b[b_start:b_start + block_len]
            aligned_len += block_len
            matches += sum(1 for x, y in zip(sub_a, sub_b) if x == y)

        return (matches / aligned_len) if aligned_len else 0.0

    except Exception:
        import difflib
        return difflib.SequenceMatcher(a=a, b=b, autojunk=False).ratio()

def merge_by_sequence_similarity(df: pd.DataFrame, identity_threshold: float = 0.95) -> pd.DataFrame:
    """
    Cluster sequences at ≥ identity_threshold and merge metadata across clusters.

    Expected columns (collapsed form):
        ['pdb','sequence','chains','metal_residues']

    Returns columns:
        ['pdb_ids','chains','sequence','metal_residues']
    """
    required = {"pdb_ids", "sequence", "chains", "metal_residues"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"merge_by_sequence_similarity: missing columns {sorted(missing)}")

    df = df.reset_index(drop=True)
    seqs = df["sequence"].astype(str).tolist()
    n = len(seqs)

    # Union-Find
    parent = list(range(n))
    rank = [0] * n

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    # Build similarity graph with a quick length-coverage screen
    for i in range(n):
        for j in range(i + 1, n):
            la, lb = len(seqs[i]), len(seqs[j])
            if max(la, lb) == 0:
                continue
            coverage = min(la, lb) / max(la, lb)
            if coverage < 0.90:  # avoid merging vastly different lengths
                continue
            ident = _pairwise_identity(seqs[i], seqs[j])
            if ident >= identity_threshold:
                union(i, j)

    # Collect clusters
    clusters: Dict[int, List[int]] = {}
    for idx in range(n):
        r = find(idx)
        clusters.setdefault(r, []).append(idx)

    # Reduce clusters
    out = []
    for _, idxs in clusters.items():
        # representative = longest sequence
        rep_idx = max(idxs, key=lambda k: len(seqs[k]))
        rep_seq = seqs[rep_idx]

        pdb_ids: Set[str] = set()
        chain_set: Set[str] = set()

        # We now want: metal -> List[List[str]] (preserve original lists)
        metal_lists: Dict[str, List[List[str]]] = {}
        metal_seen: Dict[str, Set[tuple]] = {}

        for k in idxs:
            row = df.iloc[k]
            pdb_ids.add(str(row["pdb_ids"]))

            # chains can be list or comma-joined string
            chains_val = row.get("chains", [])
            if isinstance(chains_val, str):
                for c in (c.strip() for c in chains_val.split(",") if chains_val):
                    if c:
                        chain_set.add(c)
            else:
                for c in chains_val or []:
                    chain_set.add(str(c))

            # merge metal_residues, but keep each original residue list intact
            mm = df["metal_residues"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
            for m, reslist in mm.items():
                key = str(m).upper()
                metal_lists.setdefault(key, [])
                metal_seen.setdefault(key, set())

                # Normalize to a list of lists
                if isinstance(reslist, list) and reslist and isinstance(reslist[0], list):
                    lists_to_add = reslist
                elif isinstance(reslist, list):
                    lists_to_add = [reslist]
                else:
                    lists_to_add = [[reslist]]

                # Clean, dedupe by tuple identity, and append
                for lst in lists_to_add:
                    cleaned = [str(r).strip() for r in lst if str(r).strip()]
                    tup = tuple(cleaned)
                    if cleaned and tup not in metal_seen[key]:
                        metal_lists[key].append(cleaned)
                        metal_seen[key].add(tup)

        out.append({
            "pdb_ids": sorted(list(pdb_ids)),
            "chains": sorted(list(chain_set)),
            "sequence": rep_seq,
            "metal_residues": metal_lists,   # dict[str, list[list[str]]]
        })

    return pd.DataFrame.from_records(out, columns=["pdb_ids", "chains", "sequence", "metal_residues"])
    
def filter_by_sequence_length(df: pd.DataFrame, min_len: int = 60, max_len: int = 300) -> pd.DataFrame:
    """Keep rows whose sequence length is in [min_len, max_len]."""
    m = df["sequence"].str.len().between(min_len, max_len, inclusive="both")
    return df.loc[m].reset_index(drop=True)

def collapse_pdb_and_sequence(path: str) -> pd.DataFrame:
    """Groups rows with the same (pdb_id, sequence) and builds:
       - metal_residues: {metal_ion -> [residue list, ...]}
       - also aggregates chains

    Output dataframe columns:
        pdb_id (str),
        chains (str; comma-joined unique chain IDs),
        metal_residues (dict[str, list[list[str]]]),
        sequence (str)
    """
    grouped: Dict[Tuple[str, str], dict] = {}
    parsed_rows = 0
    # Primary path: parse with the same pattern as the original code
    try:
        for raw in _open_biolip2(path):
            if not raw or raw.startswith('#'):
                continue
            cols = raw.split('\t')
            #parsed = _parse_line_to_partial(cols)
            if cols is None:
                continue
            pdb_id, chain_id, ligand_id, residues, biolip_seq = cols
            parsed_rows += 1

            pdb_id = str(pdb_id).upper()
            chain_id = str(chain_id)
            ligand_id = str(ligand_id).upper()
            biolip_seq = str(biolip_seq)

            key = (pdb_id, biolip_seq)
            rec = grouped.get(key)
            if rec is None:
                rec = {
                    "pdb_ids": pdb_id,
                    "chains": set(),
                    "metal_residues": defaultdict(list),  # metal -> list of residue lists
                    "sequence": biolip_seq,
                    "_seen": defaultdict(set),            # metal -> set of tuples for dedupe
                }
                grouped[key] = rec

            rec["chains"].add(chain_id)
            res_list = _to_residue_list(residues)
            if res_list:
                tup = tuple(res_list)
                if tup not in rec["_seen"][ligand_id]:
                    rec["metal_residues"][ligand_id].append(res_list)
                    rec["_seen"][ligand_id].add(tup)
    except Exception as e:
        # If the custom line parser can't handle this file, try a tidy TSV fallback.
        print("error", e)
        pass
    # finalize rows
    rows = []
    for rec in grouped.values():
        rows.append({
            "pdb_ids": rec["pdb_ids"],
            "chains": ",".join(sorted(rec["chains"])),
            "metal_residues": dict(rec["metal_residues"]),
            "sequence": rec["sequence"],
        })

    return pd.DataFrame(rows, columns=["pdb_ids", "chains", "metal_residues", "sequence"])    

def _to_residue_list(x: Any) -> List[str]:
    """Normalize residues into List[str] (handles lists or strings)."""
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        vals = list(x)
    elif isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
            try:
                v = ast.literal_eval(s)
                if isinstance(v, (list, tuple)):
                    vals = list(v)
                else:
                    vals = [str(v)]
            except Exception:
                vals = re.split(r"[,\s]+", s.strip("[]() "))
        else:
            vals = re.split(r"[,\s]+", s)
    else:
        vals = [str(x)]

    out, seen = [], set()
    for v in vals:
        vv = str(v).strip()
        if vv and vv not in seen:
            seen.add(vv)
            out.append(vv)
    return out

def write_df_to_tsv_gz(df: pd.DataFrame, outpath: str) -> None:
    """
    Write a DataFrame to a tab-delimited .txt.gz file.

    Each row in the DataFrame becomes a row in the output file.
    """
    df["pdb_ids"] = df["pdb_ids"].apply(lambda x: ",".join(x) if isinstance(x, list) else str(x))
    df["chains"] = df["chains"].apply(lambda x: ",".join(x) if isinstance(x, list) else str(x))
    df["metal_residues"] = df["metal_residues"].apply(json.dumps)  # safe JSON string
    if "embeddings" in df.columns:
        df["embeddings"] = df["embeddings"].apply(
            lambda t: t.detach().cpu().numpy().tolist() if hasattr(t, "detach") else str(t)
        )
    with gzip.open(outpath, "wt", encoding="utf-8", newline="") as fout:
        df.to_csv(fout, sep="\t", index=False)
        
"""
Mapping UniProt index to pdb
"""

def fetch_sifts_per_residue(
    pdb_id: str, *, cache_dir: str = ".sifts_cache", force: bool = False, sleep: float = 0.0
) -> Dict[str, Any]:
    """
    Fetch PDBe SIFTS UniProt mappings for a PDB entry using the single endpoint:
      https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pid}
    Returns the OUTER payload shape: {pid_lower: {"UniProt": {...}}}
    """
    pid = pdb_id.strip().lower()
    os.makedirs(cache_dir, exist_ok=True)
    cache = Path(cache_dir) / f"{pid}.sifts.uniprot.json"

    if cache.exists() and not force:
        return json.loads(cache.read_text())

    headers = {"Accept": "application/json", "User-Agent": "metalloprotein_search/0.1"}
    url = f"https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pid}"
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()

    payload = r.json() or {}
    if pid not in payload or not payload[pid]:
        raise RuntimeError(f"No UniProt mapping returned for {pdb_id!r} from {url}")

    cache.write_text(json.dumps(payload))
    if sleep:
        time.sleep(sleep)
    return payload

def _build_chain_maps_from_sifts(
    data: Dict[str, Any],
    pdb_id: str,
    chain_id: str,
    *,
    lenient_index_fallback: bool = True,  # <= enable index-only fallback
) -> Optional[Dict[str, Any]]:
    pid = pdb_id.strip().lower()
    entry = data[pid] if pid in data and isinstance(data[pid], dict) else data
    uni = entry.get("UniProt", {})
    if not uni:
        return None

    def _coalesce_resnum(side: Dict[str, Any]) -> Optional[int]:
        v = side.get("author_residue_number")
        if isinstance(v, int):
            return v
        v = side.get("residue_number")
        if isinstance(v, int):
            return v
        try:
            return int(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    # Collect segments for this chain
    segs: List[Tuple[Optional[int], Optional[int], int, int, str]] = []
    for acc, acc_obj in uni.items():
        for seg in acc_obj.get("mappings", []):
            ch = seg.get("chain_id") or seg.get("auth_asym_id") or seg.get("struct_asym_id")
            if ch != chain_id:
                continue
            a0 = _coalesce_resnum(seg.get("start", {}))
            a1 = _coalesce_resnum(seg.get("end",   {}))
            u0 = seg.get("unp_start")
            u1 = seg.get("unp_end")
            if not (isinstance(u0, int) and isinstance(u1, int)):
                continue
            segs.append((a0, a1, u0, u1, acc))

    if not segs:
        return None

    # Sort segments: ones with known PDB numbers first, then by UniProt start
    segs.sort(key=lambda t: (t[0] is None, t[0] if t[0] is not None else 10**9, t[2]))

    # Expand pairs
    pairs: List[Tuple[Optional[int], int, str]] = []  # (pdb_resnum_or_None, unp_res, acc)
    for a0, a1, u0, u1, acc in segs:
        n = (u1 - u0 + 1)
        if isinstance(a0, int) and isinstance(a1, int):
            n = min(n, (a1 - a0 + 1))
            if n <= 0: 
                continue
            for k in range(n):
                pairs.append((a0 + k, u0 + k, acc))
        else:
            # Missing PDB residue numbers
            if not lenient_index_fallback:
                continue
            if n <= 0:
                continue
            for k in range(n):
                pairs.append((None, u0 + k, acc))  # index-only fallback

    if not pairs:
        return None

    # Choose accession with largest coverage
    cov: Dict[str, int] = {}
    for _, __, acc in pairs:
        cov[acc] = cov.get(acc, 0) + 1
    primary_acc = max(cov, key=cov.get)

    # Keep only primary accession and define indices
    prim = [(p, u) for (p, u, a) in pairs if a == primary_acc]

    # Indexing order: PDB residue numbers when available, else keep original order (by u)
    prim.sort(key=lambda t: (t[0] is None, t[0] if t[0] is not None else 10**9, t[1]))

    pdb_idx_to_unp: Dict[int, int] = {}
    unp_to_pdb_idx: Dict[int, int] = {}
    pdb_resnum_to_idx: Dict[int, int] = {}

    for idx, (pdb_resnum, unp_res) in enumerate(prim):
        pdb_idx_to_unp[idx] = unp_res
        unp_to_pdb_idx.setdefault(unp_res, idx)
        if isinstance(pdb_resnum, int):
            pdb_resnum_to_idx.setdefault(pdb_resnum, idx)

    return {
        "unp_acc": primary_acc,
        "unp_to_pdb_idx": unp_to_pdb_idx,
        "pdb_idx_to_unp": pdb_idx_to_unp,
        "pdb_resnum_to_idx": pdb_resnum_to_idx,  # may be partial when fallback used
    }

def add_uniprot_pdb_mapping(df,
                            *,
                            pdb_col: str = "pdb_ids",
                            chain_col: str = "chains",
                            cache_dir: str = ".sifts_cache",
                            embeddings_have_bos_eos: bool = True):
    """
    Supports rows where pdb/chain are scalars OR lists:
      - If lists (e.g., pdb_ids=['1CC3','5SYD'], chains=['A','B']), we fetch for each pair.
      - Choose the pair with the largest residue coverage as the primary mapping.
    Returns a NEW DataFrame with the same added columns as before:
      unp_acc, unp_to_pdb_idx, pdb_idx_to_unp, pdb_resnum_to_idx, bos_offset
    Plus a convenience column:
      pdb_chain_maps: dict keyed by "PDB:CHAIN" -> same map dict as above for each pair.
    """
    rows = []
    for _, row in df.iterrows():
        # Read values that might be scalar or list; normalize to lists
        pdb_val = row.get(pdb_col, row.get("pdb_ids"))
        chain_val = row.get(chain_col, row.get("chains"))

        pdb_list = pdb_val if isinstance(pdb_val, (list, tuple)) else [pdb_val]
        chain_list = chain_val if isinstance(chain_val, (list, tuple)) else [chain_val]

        # If lengths mismatch, zip the min length and move on
        n = min(len(pdb_list), len(chain_list))
        pairs = [(str(pdb_list[i]).lower(), str(chain_list[i])) for i in range(n)]

        per_pair = {}  # "pdb:chain" -> maps
        best_key = None
        best_coverage = -1

        for pdb_id, chain in pairs:
            try:
                sifts = fetch_sifts_per_residue(pdb_id, cache_dir=cache_dir)
                print("sifts: ",sifts)
                maps = _build_chain_maps_from_sifts(sifts, pdb_id, chain)
                print("maps: ", maps)
            except Exception as e:
                #print(sifts)
                print("maps error: ", e)
                maps = None

            key = f"{pdb_id.upper()}:{chain}"
            if maps:
                per_pair[key] = maps
                coverage = len(maps.get("pdb_idx_to_unp", {}))  # how many residues mapped
                if coverage > best_coverage:
                    best_coverage = coverage
                    best_key = key
            else:
                per_pair[key] = None

        # Build output row
        newrow = row.to_dict()
        newrow["unp_acc"] = None
        newrow["unp_to_pdb_idx"] = {}
        newrow["pdb_idx_to_unp"] = {}
        newrow["pdb_resnum_to_idx"] = {}
        newrow["bos_offset"] = 1 if embeddings_have_bos_eos else 0
        newrow["pdb_chain_maps"] = per_pair  # keep all pair maps for optional use

        if best_key and per_pair[best_key]:
            print("unp_acc: ", maps["unp_acc"], ", pdb_id: ", pdb_val)
            maps = per_pair[best_key]
            newrow["unp_acc"] = maps["unp_acc"]
            newrow["unp_to_pdb_idx"] = maps["unp_to_pdb_idx"]
            newrow["pdb_idx_to_unp"] = maps["pdb_idx_to_unp"]
            newrow["pdb_resnum_to_idx"] = maps["pdb_resnum_to_idx"]

        rows.append(newrow)

    return pd.DataFrame(rows, columns=list(rows[0].keys()))

def map_res_list_to_token_cols(row, res_list, numbering: str = "uniprot", pair_key: str | None = None):
    """
    Convert a binding `res_list` (e.g., ['D176','N208',...]) to token columns:
      numbering='uniprot' -> use UniProt positions
      numbering='pdb'     -> use PDB author residue numbers
    If the row has multiple PDB:chain pairs, you can force one by passing pair_key='1CC3:A'.
    Otherwise, it uses the primary mapping chosen in add_uniprot_pdb_mapping.
    """
    def _digits(s):
        ds = "".join(ch for ch in str(s) if ch.isdigit())
        return int(ds) if ds else None

    bos = int(row.get("bos_offset", 0))

    # pick maps
    if pair_key:
        per_pair = row.get("pdb_chain_maps", {})
        maps = per_pair.get(pair_key)
        if not maps:
            return []
        unp_to_pdb_idx = maps.get("unp_to_pdb_idx", {})
        pdb_resnum_to_idx = maps.get("pdb_resnum_to_idx", {})
    else:
        unp_to_pdb_idx = row.get("unp_to_pdb_idx", {})
        pdb_resnum_to_idx = row.get("pdb_resnum_to_idx", {})

    cols = []
    if numbering == "uniprot":
        for r in res_list:
            num = _digits(r)
            if num is None:
                continue
            i = unp_to_pdb_idx.get(num)
            if i is not None:
                cols.append(i + bos)
    else:  # numbering == "pdb"
        for r in res_list:
            num = _digits(r)
            if num is None:
                continue
            i = pdb_resnum_to_idx.get(num)
            if i is not None:
                cols.append(i + bos)
    return cols