"""

"""
from esm.models.esmc import ESMC
from concurrent.futures import ThreadPoolExecutor
from typing import Sequence, Iterable, Dict, Any
from esm.sdk.api import (
    ESMProtein,
    LogitsConfig,
    LogitsOutput,
    ProteinType,
    ESMProteinError
)
import os
import gzip
import pandas as pd
import ast
import torch
import numpy as np
import io
import json
import itertools
from sklearn.metrics import (
    precision_recall_curve, average_precision_score,
    roc_curve, roc_auc_score
)
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import math
from functools import lru_cache
from typing import Dict, List, Tuple, Optional


# Global caches
_MODEL = None          # singleton model
_TRAIN_IDX = None      # normalized [N, D] tensor for train seq embeddings
_TRAIN_IDX_N = None    # number of rows cached, to detect table changes

def get_row(idx: int, df: pd.DataFrame) -> Dict[str, Any]:
    #row = next(itertools.islice(_open_biolip2(path), idx, idx+1)).split('\t')
    row = df.iloc[idx].to_dict()
    return row

EMBEDDING_CONFIG = LogitsConfig(
    sequence=True, return_embeddings=True, return_hidden_states=True
)

def get_ESMC(model_name: str = "esmc_600m", device: str = "cuda"):
    """Singleton model loader to avoid re-initialization overhead."""
    global _MODEL
    if _MODEL is None:
        _MODEL = ESMC.from_pretrained(model_name).to(device).eval()
    return _MODEL

@lru_cache(maxsize=4096)
def _embed_cached(seq: str, model_name: str = "esmc_600m"):
    client = get_ESMC(model_name=model_name)
    outs = embed_sequence(client, seq)
    res = outs.embeddings.squeeze(0)       # [L, D]
    seq_vec = get_sequence_embedding(outs) # [D]
    return res, seq_vec

def _maybe_build_train_index(train_df, device: str = "cuda"):
    """CUDA+bfloat16 index for fast cosine; does NOT mutate train_df."""
    global _TRAIN_IDX, _TRAIN_IDX_N
    if torch is None or not torch.cuda.is_available():
        return  # no GPU: the fast path won't be used

    n = len(train_df)
    # Reuse if same size; if you change train_df rows, clear _TRAIN_IDX/_TRAIN_IDX_N
    if _TRAIN_IDX is not None and _TRAIN_IDX_N == n:
        return

    dev = torch.device(device)
    embs = []
    for e in train_df["seq_embeddings"]:
        t = e if torch.is_tensor(e) else torch.as_tensor(np.asarray(e))
        # BEFORE:
        # t32 = t.to(device=dev, dtype=torch.float32)
        # t32 = torch.nn.functional.normalize(t32, dim=-1)
        # t16 = t32.to(dtype=torch.bfloat16)
        # AFTER (no L2 norm, keep raw magnitudes):
        #t16 = t.to(device=dev, dtype=torch.float32).to(dtype=torch.bfloat16)
        t32 = t.to(device=dev, dtype=torch.float32)

        embs.append(t32)
    _TRAIN_IDX = torch.stack(embs, dim=0)   # [N, D] bf16
    _TRAIN_IDX_N = n

# def _fast_cosine_scores(query_vec, train_df):
#     if torch is None or not torch.cuda.is_available():
#         return None

#     # Build/ensure CUDA bf16 index
#     _maybe_build_train_index(train_df, device="cuda")
#     if _TRAIN_IDX is None:
#         return None

#     # Make a working query copy on same device/dtype as index (bf16, cuda)
#     if torch.is_tensor(query_vec):
#         q32 = query_vec.to(device=_TRAIN_IDX.device, dtype=torch.float32)
#     else:
#         q32 = torch.as_tensor(query_vec, device=_TRAIN_IDX.device, dtype=torch.float32)
#     # BEFORE: q32 = torch.nn.functional.normalize(q32, dim=-1)
#     # q16 = q32.to(dtype=torch.bfloat16)
#     # sc = (_TRAIN_IDX @ q16).to(dtype=torch.float32).cpu().numpy()  # raw dot
#     #q16 = q32.to(dtype=torch.bfloat16)
#     sc = (_TRAIN_IDX @ q32).float().cpu().numpy()  # raw dot
#     return sc

def _fast_cosine_scores(query_vec: torch.Tensor, table) -> list[float] | None:
    """
    Fast cosine between query_vec [D] and each row['seq_embeddings'] [D] or [T,D].
    Everything runs on CUDA; returns a Python list of floats.
    """
    if not isinstance(query_vec, torch.Tensor):
        return None  # fall back to slower path if you want

    device = torch.device("cuda")
    q = query_vec.to(device=device, dtype=torch.float32)
    if q.ndim == 2:             # tolerate [T,D]
        q = q.mean(0)
    q = q / (q.norm() + 1e-12)  # [D]

    Emats = []
    for e in table["seq_embeddings"]:
        t = e if isinstance(e, torch.Tensor) else torch.as_tensor(e)
        t = t.to(device=device, dtype=torch.float32)
        if t.ndim == 2:         # tolerate [T,D] stored
            t = t.mean(0)
        Emats.append(t)         # each [D]

    if not Emats:
        return None

    X = torch.stack(Emats, dim=0)                # [N, D]
    X = X / (X.norm(dim=1, keepdim=True) + 1e-12)
    sims = torch.mv(X, q)                        # [N]

    return sims.detach().cpu().tolist()

def embed_sequence(client: ESMC, protein_sequence: str) -> LogitsOutput:
    protein = ESMProtein(sequence=protein_sequence)
    protein_tensor = client.encode(protein)
    output = client.logits(protein_tensor, EMBEDDING_CONFIG)
    return output

def batch_embed(
    model: ESMC, inputs: Sequence[ProteinType]
) -> Sequence[LogitsOutput]:
    """Forge supports auto-batching. So batch_embed() is as simple as running a collection
    of embed calls in parallel using asyncio.
    """
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(embed_sequence, model, protein) for protein in inputs
        ]
        results = []
        for future in futures:
            try:
                results.append(future.result())
            except Exception as e:
                results.append(ESMProteinError(500, str(e)))
    return results

def unwrap_embedding(out):
    if hasattr(out, "embeddings"):       # good case
        return out.embeddings.detach().cpu()
    else:                                # error case
        return str(out)

def _parse_site_groups(metal_residues):
    """
    Keep the grouping: return dict[metal] -> list[list[(resnum:int, aa:str)]].
    """
    groups = {}
    for metal, site_lists in (metal_residues or {}).items():
        g = []
        for site in site_lists:
            one = []
            for tag in site:
                aa = ''.join(ch for ch in tag if ch.isalpha()).upper()
                ns = ''.join(ch for ch in tag if ch.isdigit())
                if ns:
                    one.append((int(ns), aa))
            if one:
                g.append(one)
        groups[metal] = g
    return groups


def _compute_token_positions_for_labels(sequence, labels, delta, bos_offset):
    """
    Build positions for ONE site (labels = list[(resnum, aa)]) using a single delta.
    No AA-match requirement for inclusion. Also return diagnostics.
    """
    L = len(sequence)
    bos = int(bos_offset)
    all_pos, validated = [], []
    inbounds = matches = 0
    for resnum, aa in labels:
        j = resnum + int(delta)       # 1-based token index
        si = j - bos                  # 0-based sequence index
        if 0 <= si < L:
            inbounds += 1
            all_pos.append(int(j))
            if sequence[si].upper() == aa.upper():
                matches += 1
                validated.append(int(j))
    diag = {"n_labels": len(labels), "n_inbounds": inbounds, "n_matches": matches}
    return all_pos, validated, diag


def _delta_from_maps_row(row):
    """
    Row-level deltas from the mapping dicts (kept as scalars).
    """
    bos = int(row.get("bos_offset", 0))
    pdb_resnum_to_idx = row.get("pdb_resnum_to_idx")
    pdb_idx_to_unp    = row.get("pdb_idx_to_unp")

    pdb_delta = None
    uniprot_delta = None
    pdb_cov = unp_cov = 0.0
    pdb_npairs = unp_npairs = 0

    # PDB numbering tends to be linear: idx == resnum-1
    if isinstance(pdb_resnum_to_idx, dict) and pdb_resnum_to_idx:
        ok = all(pdb_resnum_to_idx.get(r) == (r - 1) for r in pdb_resnum_to_idx.keys())
        if ok:
            pdb_delta = bos - 1
        else:
            r, i = next(iter(pdb_resnum_to_idx.items()))
            pdb_delta = (i + bos) - r
        pdb_cov = 1.0
        pdb_npairs = len(pdb_resnum_to_idx)

    # UniProt map often linear: unp == idx + u0  ⇒ Δ = bos - u0
    if isinstance(pdb_idx_to_unp, dict) and pdb_idx_to_unp:
        items = sorted(pdb_idx_to_unp.items())
        i0, u0 = items[0]
        linear = all(u == (i + u0) for i, u in items)
        if linear:
            uniprot_delta = bos - u0
        else:
            i, u = items[0]
            uniprot_delta = (i + bos) - u
        unp_cov = 1.0
        unp_npairs = len(items)

    return pdb_delta, uniprot_delta, pdb_cov, unp_cov, pdb_npairs, unp_npairs

def _letter_vote_candidates(sequence, labels, bos_offset, top_k=5):
    """
    Return a list of (delta, 'letter_vote') sorted by vote count (desc),
    then by |delta| (asc). We consider matches to generate candidates,
    but we'll later score candidates by (inbounds, matches).
    """
    from collections import Counter
    L = len(sequence)
    bos = int(bos_offset)
    votes = Counter()

    for resnum, aa in labels:
        tgt = aa.upper()
        for si in range(L):                  # si is 0-based sequence index
            if sequence[si].upper() == tgt:
                j = si + bos                # token index j (1-based if bos==1)
                d = j - resnum             # candidate delta aligning this letter
                votes[d] += 1

    if not votes:
        return []

    # sort by (votes desc, |delta| asc)
    sorted_ds = sorted(votes.items(), key=lambda kv: (kv[1], -abs(kv[0])), reverse=True)
    deltas = [int(d) for d, _ in sorted_ds[:top_k]]
    return [(d, "letter_vote") for d in deltas]


def _span_fit_candidates(labels, sequence_len, bos_offset):
    """
    Deltas that make ALL labels in-bounds (regardless of AA match).
    Using the feasible range Δ ∈ [Δmin, Δmax], return Δmin, Δmax, and midpoint.
    """
    L = int(sequence_len)
    bos = int(bos_offset)
    resnums = [r for r, _ in labels]
    if not resnums:
        return []

    rmin, rmax = min(resnums), max(resnums)
    # in-bounds condition: bos <= (resnum + Δ) <= bos + L - 1
    dmin = bos - rmin
    dmax = (bos + L - 1) - rmax

    if dmin > dmax:
        # no single delta can put all labels in-bounds
        return []

    # consider extremes + midpoint
    mids = (dmin + dmax) // 2
    candidates = sorted({int(dmin), int(dmax), int(mids)}, key=lambda d: (abs(d), d))
    return [(d, "span_fit") for d in candidates]

def _choose_delta_for_site(sequence, bos, site_labels, trial_deltas, *, lv_top_k=5):
    """
    Pick the best delta for ONE site.

    Candidate set (in this order for provenance):
      1) dict-derived deltas: [('pdb', Δpdb), ('uniprot', Δu)]
      2) letter-vote top-K:   [('letter_vote', Δ1), ...]
      3) span-fit deltas:     [('span_fit', Δmin), ('span_fit', Δmid), ('span_fit', Δmax)]

    Scoring: maximize (n_inbounds, n_matches), then tie-break by source priority
             (pdb > uniprot > letter_vote > span_fit), then by smaller |Δ|.
    Returns (delta_used, source, diag, positions_all, positions_valid).
    """
    src_rank = {"pdb": 3, "uniprot": 2, "letter_vote": 1, "span_fit": 0}

    cand = []
    seen = {}
    for src, dlt in trial_deltas:
        if dlt in seen:
            if src_rank.get(src, -1) > src_rank.get(seen[dlt], -1):
                for i, (s, d) in enumerate(cand):
                    if d == dlt:
                        cand[i] = (src, dlt)
                        break
                seen[dlt] = src
        else:
            cand.append((src, int(dlt)))
            seen[int(dlt)] = src

    for d, s in _letter_vote_candidates(sequence, site_labels, bos, top_k=lv_top_k):
        if d not in seen:
            cand.append((s, d))
            seen[d] = s

    for d, s in _span_fit_candidates(site_labels, len(sequence), bos):
        if d not in seen:
            cand.append((s, d))
            seen[d] = s

    best_key = None
    best_pack = None
    for src, dlt in cand:
        all_pos, val_pos, diag = _compute_token_positions_for_labels(sequence, site_labels, dlt, bos)
        key = (diag["n_inbounds"], diag["n_matches"], src_rank.get(src, -1), -abs(int(dlt)))
        if (best_key is None) or (key > best_key):
            best_key = key
            best_pack = (int(dlt), src, diag, all_pos, val_pos)

        if diag["n_inbounds"] == diag["n_labels"] and diag["n_matches"] == diag["n_labels"]:
            break

    if best_pack is None:
        return 0, "default", {"n_labels": len(site_labels), "n_inbounds": 0, "n_matches": 0}, [], []

    delta, src, diag, pos_all, pos_val = best_pack

    # ---------- Single-rescue off-by-one (only if exactly one mismatch & all in-bounds) ----------
    if diag.get("n_inbounds", 0) == diag.get("n_labels", 0) and (diag.get("n_inbounds", 0) - diag.get("n_matches", 0) == 1):
        rescued_j, rescued_label = _single_rescue_off_by_one(sequence, bos, site_labels, delta)
        if rescued_j is not None:
            # add rescued position to validated set (keep originals intact for "unvalidated" downstream use)
            pos_val = sorted({*pos_val, int(rescued_j)})
            # mark diag for transparency
            diag = dict(diag)  # copy
            diag["rescue"] = True
            diag["rescue_label"] = rescued_label
            diag["rescue_shift"] = int(rescued_j - (rescued_j - 1)) if False else None  # kept for clarity
            # post-rescue, validated count equals labels (since all in-bounds and single mismatch fixed)
            diag["n_matches"] = diag["n_labels"]

    return int(delta), src, diag, pos_all, pos_val

def _single_rescue_off_by_one(sequence, bos_offset, labels, delta):
    """
    If all labels are in-bounds and exactly one is an AA-mismatch, try to rescue
    by shifting that one position by ±1. Return (rescued_j:int|None, rescued_label:str|None).
    """
    expl = _explain_site_misses(sequence, bos_offset, labels, delta)
    mism = [e for e in expl if e["in_bounds"] and not e["aa_match"]]
    if len(mism) != 1:
        return None, None

    e = mism[0]
    seq = sequence or ""
    L = len(seq)
    si = e["seq_idx"]
    aa = e["aa"].upper()

    for shift in (-1, 1):
        si2 = si + shift
        if 0 <= si2 < L and seq[si2].upper() == aa:
            return e["j"] + shift, e["label"]  # rescued token position, label string (e.g. "C101")

    return None, None

def _collect_site_cols_for_row(row, allow_single_mismatch=True):
    """
    Build the set of template *column indices* (0-based) to use for ranking.

    Source rules:
      - Prefer validated positions per site.
      - If a site has all labels in-bounds and exactly one mismatch:
            - if rescue filled it, validated already equals full set (use validated)
            - else (no rescue), use the unvalidated positions (tolerant mode)

    IMPORTANT: Stored token positions are 1-based "token" indices (j).
               Convert to 0-based embedding columns by: j0 = j - bos_offset.
    """
    cols = set()
    by_val = row.get("token_positions_validated_by_site") or {}
    by_all = row.get("token_positions_by_site") or {}
    diags  = row.get("site_diag_by_site") or {}

    # For safe clamping:
    emb = row.get("res_embeddings")
    Lt = int(emb.shape[0]) if emb is not None else 0
    bos = int(row.get("bos_offset", 0))

    metals = set(by_all.keys()) | set(by_val.keys())
    for metal in metals:
        sites_all = by_all.get(metal, [])
        sites_val = by_val.get(metal, [])
        sites_dg  = diags.get(metal, [])
        for sidx, all_pos in enumerate(sites_all):
            val_pos = sites_val[sidx] if sidx < len(sites_val) else []
            d       = sites_dg[sidx]  if sidx < len(sites_dg)  else {}
            nlab = int(d.get("n_labels", 0))
            ninb = int(d.get("n_inbounds", 0))
            nmat = int(d.get("n_matches", 0))

            if val_pos:
                use = val_pos
            elif allow_single_mismatch and nlab > 0 and ninb == nlab and (ninb - nmat == 1):
                # tolerant use of the full, unvalidated set
                use = all_pos
            else:
                use = val_pos  # stays empty when we don't allow unvalidated

            for j in use:
                j0 = int(j) - bos               # <-- convert 1-based token -> 0-based column
                if 0 <= j0 < Lt:
                    cols.add(j0)

    return sorted(cols)

def get_table(df: pd.DataFrame, embed: bool = True, model_name: str = "esmc_600m") -> pd.DataFrame:
    """
    Build a results table with per-site delta selection.
      1) Try dict-derived deltas (PDB, then UniProt) for each site.
      2) If a site's labels are OOB or letter-mismatch under those deltas, fall back to site-local letter-vote.
      3) Always write token positions from the site's chosen delta.

    Requires helpers:
      - _parse_site_groups(metal_residues)
      - _delta_from_maps_row(row)
      - _choose_delta_for_site(sequence, bos, site_labels, trial_deltas)
      - _compute_token_positions_for_labels(sequence, labels, delta, bos_offset)
      - (your existing embed_sequence, get_ESMC, get_sequence_embedding)
    """
    import ast
    from collections import Counter

    rows = []
    client = get_ESMC(model_name=model_name) if embed else None

    for _, row in df.iterrows():
        newrow = row.to_dict()

        # ---------------- Embeddings ----------------
        if embed:
            outs = embed_sequence(client, newrow['sequence'])
            res_emb = outs.embeddings.detach().cpu().squeeze(0)   # (T, D)
            seq_emb = get_sequence_embedding(outs).detach().cpu() # (D,)
            newrow['res_embeddings'] = res_emb
            newrow['seq_embeddings'] = seq_emb
            L = int(res_emb.shape[0])
        else:
            newrow['res_embeddings'] = None
            newrow['seq_embeddings'] = None
            L = len(newrow.get("sequence", ""))

        seq = newrow.get("sequence", "")
        bos = int(newrow.get("bos_offset", 0))

        # ---------------- metal_residues normalization ----------------
        mr = newrow.get("metal_residues")
        if isinstance(mr, str):
            try:
                mr = ast.literal_eval(mr)
            except Exception:
                mr = {}
        mr = mr or {}
        newrow["metal_residues"] = mr

        # Keep per-site grouping: dict[metal] -> list[list[(resnum:int, aa:str)]]
        site_groups = _parse_site_groups(mr)

        # Also keep your old "res_positions" (flattened ints by metal) for convenience
        newrow["res_positions"] = {}
        total_labeled = 0
        for metal, sites in site_groups.items():
            ints = []
            for site in sites:
                for (resnum, _aa) in site:
                    ints.append(int(resnum))
                    total_labeled += 1
            newrow["res_positions"][metal] = ints
        newrow["n_res_sites"] = total_labeled  # total number of labeled residues (not #sites)

        # ---------------- dict-derived deltas (row-level) ----------------
        # Produces compact scalars for provenance: pdb_delta, uniprot_delta, and simple coverage/npairs.
        pdb_delta, unp_delta, pdb_cov, unp_cov, pdb_n, unp_n = _delta_from_maps_row(newrow)
        trial_deltas = []
        if pdb_delta is not None:
            trial_deltas.append(("pdb", int(pdb_delta)))
        if unp_delta is not None:
            trial_deltas.append(("uniprot", int(unp_delta)))

        # Record row-level deltas & stats
        newrow["pdb_delta"] = None if pdb_delta is None else int(pdb_delta)
        newrow["uniprot_delta"] = None if unp_delta is None else int(unp_delta)
        newrow["pdb_delta_coverage"] = float(pdb_cov)
        newrow["uniprot_delta_coverage"] = float(unp_cov)
        newrow["pdb_delta_npairs"] = int(pdb_n)
        newrow["uniprot_delta_npairs"] = int(unp_n)

        # ---------------- per-site delta selection ----------------
        token_positions_by_site = {}            # dict[metal] -> list[list[int]]
        token_positions_valid_by_site = {}      # dict[metal] -> list[list[int]]
        deltas_by_site = {}                     # dict[metal] -> list[int]
        delta_sources_by_site = {}              # dict[metal] -> list[str]
        site_diag_by_site = {}                  # dict[metal] -> list[dict]

        total_inbounds = 0
        total_matches = 0

        # Also collect for a row-level "mode" delta if most sites agree
        all_site_deltas = []

        for metal, sites in site_groups.items():
            metal_pos_sites = []
            metal_pos_valid_sites = []
            metal_deltas = []
            metal_delta_srcs = []
            metal_site_diags = []

            for labels in sites:
                d_used, d_src, diag, pos_all, pos_val = _choose_delta_for_site(
                    seq, bos, labels, trial_deltas
                )
                metal_pos_sites.append(pos_all)
                metal_pos_valid_sites.append(pos_val)
                metal_deltas.append(d_used)
                metal_delta_srcs.append(d_src)
                metal_site_diags.append(diag)

                total_inbounds += diag.get("n_inbounds", 0)
                total_matches += diag.get("n_matches", 0)
                all_site_deltas.append((d_used, d_src, diag.get("n_matches", 0)))

            token_positions_by_site[metal] = metal_pos_sites
            token_positions_valid_by_site[metal] = metal_pos_valid_sites
            deltas_by_site[metal] = metal_deltas
            delta_sources_by_site[metal] = metal_delta_srcs
            site_diag_by_site[metal] = metal_site_diags

        # ---------------- flatten positions for backward compatibility ----------------
        token_positions_flat = {
            metal: [p for site in sites for p in site]
            for metal, sites in token_positions_by_site.items()
        }
        newrow['token_positions_by_site'] = token_positions_by_site
        newrow['token_positions_validated_by_site'] = token_positions_valid_by_site
        newrow['token_positions'] = token_positions_flat
        newrow['n_token_sites'] = sum(len(v) for v in token_positions_flat.values())
        newrow['deltas_by_site'] = deltas_by_site
        newrow['delta_sources_by_site'] = delta_sources_by_site
        newrow['site_diag_by_site'] = site_diag_by_site

        # Row-level "delta_used": majority vote by delta across sites (tie-break by total matches)
        delta_counter = Counter(d for (d, _s, _m) in all_site_deltas)
        if delta_counter:
            mode_delta, mode_count = max(delta_counter.items(), key=lambda kv: (kv[1], -abs(kv[0])))
            # Was there more than one distinct delta?
            if len(delta_counter) == 1:
                mode_source = next((s for (d, s, _m) in all_site_deltas if d == mode_delta), "mixed")
                newrow['delta_used'] = int(mode_delta)
                newrow['delta_used_source'] = mode_source
            else:
                # Mixed deltas across sites; keep the most common delta for reference, mark source as 'mixed'
                newrow['delta_used'] = int(mode_delta)
                newrow['delta_used_source'] = 'mixed'
        else:
            newrow['delta_used'] = None
            newrow['delta_used_source'] = 'none'

        # Diagnostics roll-up
        newrow['mapping_hits_inbounds'] = int(total_inbounds)
        newrow['mapping_aa_matches'] = int(total_matches)

        # ---------------- drop heavy maps to keep the table lean ----------------
        # (You asked to store only the numeric 'change' i.e., delta; these dicts are no longer needed.)
        for k in ("pdb_resnum_to_idx", "unp_to_pdb_idx", "pdb_idx_to_unp", "pdb_chain_maps"):
            if k in newrow:
                newrow.pop(k, None)

        rows.append(newrow)

    # Build DF using keys from first row
    table = pd.DataFrame.from_records(rows, columns=list(rows[0].keys()) if rows else None)
    return table


# def get_table(df: pd.DataFrame, embed= True, model_name="esmc_600m") -> pd.DataFrame:
#     rows = []
#     client = get_ESMC(model_name=model_name)
#     for _, row in df.iterrows():
#         newrow = row.to_dict()

#         outs = embed_sequence(client, newrow['sequence'])               # LogitsOutput
#         res_emb = outs.embeddings.detach().cpu().squeeze(0)          # (T, D) remove batch
#         seq_emb = get_sequence_embedding(outs).detach().cpu()        # (D,) or whatever you return
#         newrow["metal_residues"] = ast.literal_eval(newrow["metal_residues"])
#         newrow['res_embeddings'] = res_emb                   # (T, D)
#         newrow['seq_embeddings'] = seq_emb
        
#         mr = newrow.get("metal_residues") or {}
#         if isinstance(mr, str):
#             try:
#                 mr = ast.literal_eval(mr)
#             except Exception:
#                 mr = {}

#         res_pos_dict = {}
#         for metal, sites in mr.items():
#             # sites is list of lists like [['D176','N208',...], [...]]
#             ints = [
#                 int(''.join(ch for ch in r if ch.isdigit()))
#                 for site in sites for r in site
#                 if any(ch.isdigit() for ch in r)
#             ]
#             res_pos_dict[metal] = ints  # <- plain lists of ints

#         newrow["res_positions"] = res_pos_dict
        
#         m   = newrow.get("pdb_resnum_to_idx") or {}
#         bos = int(newrow.get("bos_offset", 0))
#         L   = int(res_emb.shape[0])  # res_emb defined above when embed=True

#         tok_pos = {}
#         for metal, nums in res_pos_dict.items():
#             idxs = []
#             for n in nums:
#                 i = m.get(n)
#                 if i is not None:
#                     j = i + bos
#                     if 0 <= j < L:
#                         idxs.append(int(j))
#             tok_pos[metal] = idxs

#         newrow["token_positions"] = tok_pos

#         rows.append(newrow)
#     table = pd.DataFrame.from_records(rows, columns=list(rows[0].keys()))
#     return table

# def get_sequence_embedding(res_embeddings) -> torch.Tensor:
#     """Convert ESM-C LogitsOutput to a single fixed-size tensor per sequence.
#     Uses the mean of the last hidden layer embeddings over the sequence length.
#     """
#     return torch.mean(res_embeddings.hidden_states, dim=-2).squeeze(1).mean(0)

# def get_sequence_embedding(res_embeddings) -> torch.Tensor:
#     """Final-layer token-mean (no cross-layer averaging)."""
#     h_last = res_embeddings.hidden_states[-1]   # [B, T, D] or [T, D]
#     return h_last.mean(dim=-2).squeeze(0)       # avg over tokens → [D]

def get_sequence_embedding(outs):
    E = outs.embeddings.squeeze(0).to(torch.float32)  # same source as res_embeddings
    return E.mean(dim=0)

# def cosine_mean(A: torch.Tensor, B: torch.Tensor) -> float:
#     # BEFORE: cosine of the means
#     # return torch.nn.functional.cosine_similarity(A.mean(0), B.mean(0), dim=0).item()
#     # AFTER: raw dot of the means
#     a = A.mean(0)
#     b = B.mean(0)
#     return float(torch.dot(a, b))

# def cosine_mean(a, b):
#     import numpy as np
#     a = np.asarray(a, dtype=np.float32).ravel()
#     b = np.asarray(b, dtype=np.float32).ravel()
#     na = np.linalg.norm(a)
#     nb = np.linalg.norm(b)
#     if na == 0.0 or nb == 0.0:
#         return 0.0
#     return float(np.dot(a, b) / (na * nb))

def cosine_mean(A: torch.Tensor, B: torch.Tensor) -> float:
    # Accept 1D [D] or 2D [T,D]; make both live on the same CUDA device
    if not isinstance(A, torch.Tensor): A = torch.as_tensor(A)
    if not isinstance(B, torch.Tensor): B = torch.as_tensor(B)
    # Move both to CUDA
    device = torch.device("cuda")
    A = A.to(device)
    B = B.to(device)

    # Mean-pool only if needed
    a = A.mean(0) if A.ndim == 2 else A  # [D]
    b = B.mean(0) if B.ndim == 2 else B  # [D]

    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()	

def cosine_max(A: torch.Tensor, B: torch.Tensor) -> float:
    # BEFORE: cosine of the per-dim maxima
    # return torch.nn.functional.cosine_similarity(A.max(0).values, B.max(0).values, dim=0).item()
    # AFTER: raw dot of the maxima
    a = A.max(0).values
    b = B.max(0).values
    return float(torch.dot(a, b))

def get_per_res_similarity_matrix(A: torch.Tensor, B: torch.Tensor, metric: str = "dot") -> torch.Tensor:
    """
    Returns a Torch tensor [Lq, Lt] on CUDA.
    A: query_res_emb [Lq, D]
    B: template_res_emb [Lt, D]
    """
    device = torch.device("cuda")
    A = (A if isinstance(A, torch.Tensor) else torch.as_tensor(A)).to(device=device, dtype=torch.float32)
    B = (B if isinstance(B, torch.Tensor) else torch.as_tensor(B)).to(device=device, dtype=torch.float32)

    if metric == "dot" or metric == "site_dot_sum":
        return A @ B.t()  # [Lq, Lt]
    elif metric == "cosine":
        An = A / (A.norm(dim=1, keepdim=True) + 1e-12)
        Bn = B / (B.norm(dim=1, keepdim=True) + 1e-12)
        return An @ Bn.t()
    else:
        # default to raw dot
        return A @ B.t()

def _softmax_temperature(x, temperature: float):
    # x: 1D array-like of global similarities G_t
    z = np.asarray(x, dtype=float) / float(temperature)
    z = z - np.max(z)                       # numerical stability
    e = np.exp(z)
    return e / e.sum()

def _softmax_tau(x, tau=0.2):
    z = np.asarray(x, float) / max(tau, 1e-8)
    z -= z.max()
    e = np.exp(z)
    return e / (e.sum() + 1e-12)

def _union_site_cols(token_positions) -> list[int]:
    """Flatten/unique metal-site token indices from a row's token_positions dict."""
    cols = set()
    if token_positions:
        for idxs in token_positions.values():
            if idxs:
                cols.update(int(i) for i in idxs)
    return sorted(cols)

def _site_dot_sum_vector(query_res_emb: torch.Tensor,
                         template_res_emb: torch.Tensor,
                         site_cols: list[int]):
    """
    Per-query-residue site score vector v where
    v[i] = sum_j S[i, j] over j in site_cols.

    Returns (v, order), both torch tensors:
      - v: [Lq] evidence vector
      - order: [Lq] indices sorted by v (desc)
    """
    if not site_cols:
        return None, None

    # S is torch [Lq, Lt]; implement dot or cosine inside your function
    S = get_per_res_similarity_matrix(query_res_emb, template_res_emb)  # torch
    if S is None or S.ndim != 2 or S.shape[1] == 0:
        return None, None

    cols = torch.as_tensor(site_cols, dtype=torch.long, device=S.device)
    # Guard against OOB indices
    cols = cols[(cols >= 0) & (cols < S.shape[1])]
    if cols.numel() == 0:
        return None, None

    sub = S.index_select(1, cols)       # [Lq, |J|]
    v = sub.sum(dim=1)                  # [Lq]
    order = torch.argsort(v, descending=True)
    return v, order


def _site_dot_mean_vector(query_res_emb: torch.Tensor,
                          template_res_emb: torch.Tensor,
                          site_cols: list[int]):
    """
    Per-query-residue site score vector v where
    v[i] = mean_j S[i, j] over j in site_cols.

    Returns (v, order), both torch tensors.
    """
    if not site_cols:
        return None, None

    S = get_per_res_similarity_matrix(query_res_emb, template_res_emb)  # torch
    if S is None or S.ndim != 2 or S.shape[1] == 0:
        return None, None

    cols = torch.as_tensor(site_cols, dtype=torch.long, device=S.device)
    cols = cols[(cols >= 0) & (cols < S.shape[1])]
    if cols.numel() == 0:
        return None, None

    sub = S.index_select(1, cols)       # [Lq, |J|]
    v = sub.mean(dim=1)                 # [Lq]  <-- only difference vs sum
    order = torch.argsort(v, descending=True)
    return v, order

def search_top_k_seq_embeddings(
    query_seq,
    table,
    k: int = 5,
    metric: str = "site_dot_sum",   # now also accepts "site_dot_mean"
    dev=None,
    model_name: str = "esmc_600m",
    top_m: int = 6
):
    # Embed once (cached)
    query_res_emb, query_emb = _embed_cached(query_seq, model_name=model_name)

    if metric in ("cosine_mean", "cosine_max"):
        # existing behavior (unchanged) ...
        if metric == "cosine_mean" and torch is not None:
            _scores = _fast_cosine_scores(query_emb, table)
            scores = _scores if _scores is not None else [
                cosine_mean(query_emb, emb) for emb in table["seq_embeddings"]
            ]
        elif metric == "cosine_max":
            scores = [cosine_max(query_emb, emb) for emb in table["seq_embeddings"]]
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        top_k_idx = np.argsort(scores)[-k:][::-1]
        out = []
        for rank, idx in enumerate(top_k_idx, start=1):
            row = table.iloc[idx]
            emb2 = row["res_embeddings"]
            S = get_per_res_similarity_matrix(query_res_emb, emb2, metric=metric)
            out.append({
                "rank": int(rank),
                "idx": int(idx),
                "global_score": float(scores[idx]),
                "similarity_matrix": S,
                "template_row": row,
                "pdb_ids": row.get("pdb_ids"),
                "metal_residues": row.get("metal_residues"),
                "token_positions": row.get("token_positions", {}),
                "res_positions": row.get("res_positions"),
            })
        if dev:
            print(f"Top-{k} templates chosen (metric={metric}):")
            for r in out:
                print(f"  rank={r['rank']} idx={r['idx']} score={r['global_score']:.4f} pdb_ids={r.get('pdb_ids')}")
        return out

    elif metric in ("site_dot_sum", "site_dot_mean"):
        # Site-aware ranking via sum or mean collapse over site columns
        scores = []
        cache = {}
        use_mean = (metric == "site_dot_mean")

        for idx, row in table.iterrows():
            site_cols = _collect_site_cols_for_row(row, allow_single_mismatch=True)  # 0-based
            if not site_cols:
                scores.append(float("-inf"))
                cache[idx] = None
                continue

            # Choose reducer
            if use_mean:
                v, order = _site_dot_mean_vector(query_res_emb, row["res_embeddings"], site_cols)
            else:
                v, order = _site_dot_sum_vector(query_res_emb, row["res_embeddings"], site_cols)

            if v is None or order is None:
                scores.append(float("-inf"))
                cache[idx] = None
                continue

            m = max(1, top_m)
            top_vals = v.index_select(0, order[:m])     # torch [m]
            glob = float(top_vals.sum().item())         # keep total evidence over top-m
            scores.append(glob)
            cache[idx] = (v, order, site_cols, glob)

        # keep only finite scores and take top-k
        valid = [(i, s) for i, s in enumerate(scores) if np.isfinite(s)]
        valid = sorted(valid, key=lambda t: t[1], reverse=True)[:k]
        top_k_idx = [i for i, _ in valid]

        out = []
        for rank, idx in enumerate(top_k_idx, start=1):
            row = table.iloc[idx]
            v, order, site_cols, glob = cache[idx]

            m = max(1, top_m)
            sorted_v = torch.sort(v, descending=True).values
            if v.numel() > m:
                top_vals = sorted_v[:m]
                rest = sorted_v[m:]
                gap_ratio = float((top_vals.mean() / (rest.median() + 1e-9)).item())
            else:
                gap_ratio = float("inf")

            out.append({
                "rank": int(rank),
                "idx": int(idx),
                "global_score": glob,          # sum over top-m of the (sum-or-mean) per-res v
                "site_cols": site_cols,        # used template columns (0-based)
                "per_query_site_scores": v,    # torch [Lq]
                "top_query_indices": order[:m].tolist(),
                "top_query_scores": v.index_select(0, order[:m]).tolist(),
                "gap_ratio": gap_ratio,
                "template_row": row,
                "pdb_ids": row.get("pdb_ids"),
                "metal_residues": row.get("metal_residues"),
                "token_positions": row.get("token_positions", {}),
                "res_positions": row.get("res_positions"),
            })
        if dev:
            print(f"Top-{k} templates chosen (metric={metric}, top_m={top_m}):")
            for r in out:
                print(f"  rank={r['rank']} idx={r['idx']} score={r['global_score']:.4f} gap={r['gap_ratio']:.2f} pdb_ids={r.get('pdb_ids')}")
        return out

    else:
        raise ValueError(f"Unsupported metric: {metric}")


def _flatten_validated_site_cols(row) -> list[int]:
    """
    Optional helper: union of validated-by-site indices.
    Falls back to plain token_positions if validated is missing/empty.
    """
    # prefer validated-by-site if you want stricter mapping
    vbs = row.get("token_positions_validated_by_site")
    if isinstance(vbs, dict) and any(vbs.values()):
        cols = set()
        for sites in vbs.values():
            for site in sites or []:
                for j in site or []:
                    cols.add(int(j))
        if cols:
            return sorted(cols)
    # else use the unvalidated union (default for downstream usage)
    return _union_site_cols(row.get("token_positions"))


def _site_evidence_from_union_cols(
    query_res_emb: torch.Tensor,
    template_res_emb: torch.Tensor,
    site_cols: list[int],
    top_m: int
):
    """
    CUDA-safe: all Torch ops, no NumPy.
    Returns:
      site_score (float), v (torch.Tensor[Lq]), order (torch.LongTensor[Lq]), gap_ratio (float)
    """
    if not site_cols:
        return float("-inf"), None, None, float("inf")

    S = get_per_res_similarity_matrix(query_res_emb, template_res_emb)  # [Lq, Lt] on CUDA
    idx = torch.as_tensor(site_cols, device=S.device, dtype=torch.long)
    v = S.index_select(dim=1, index=idx).sum(dim=1)                     # [Lq] CUDA

    m = max(1, int(top_m))
    # top-m sum for the score
    top_vals, top_idx = torch.topk(v, k=m, largest=True)
    site_score = float(top_vals.sum().item())

    # gap ratio = mean(top-m) / median(rest)
    if v.numel() > m:
        sorted_v = torch.sort(v, descending=True).values
        rest = sorted_v[m:]
        gap_ratio = float((top_vals.mean() / (rest.median() + 1e-9)).item())
    else:
        gap_ratio = float("inf")

    order = torch.argsort(v, dim=0, descending=True)
    return site_score, v, order, gap_ratio

def _site_vector_for_metric(query_res_emb, templ_res_emb, site_cols, site_metric: str):
    if site_metric == "site_dot_mean":
        return _site_dot_mean_vector(query_res_emb, templ_res_emb, site_cols)
    elif site_metric == "site_dot_sum":
        return _site_dot_sum_vector(query_res_emb, templ_res_emb, site_cols)
    else:
        raise ValueError(f"Unsupported site_metric: {site_metric}")

def search_top_k_hybrid(
    query_seq: str,
    table: pd.DataFrame,
    k: int = 10,
    *,
    prefilter_N: int = None,            # keep after stage-1; default = max(5*k, 50)
    global_metric: str = "cosine_mean", # "cosine_mean" or "cosine_max"
    site_metric: str = "site_dot_sum",  # <-- NEW: "site_dot_sum" or "site_dot_mean"
    site_top_m: int = 6,                # sum of top-m entries of v
    use_validated: bool = False,        # False -> use token_positions (unvalidated)
    model_name: str = "esmc_600m",
    dev: bool = False,
):
    """
    Two-stage search:
      Stage-1: rank all templates by global sequence-embedding similarity.
      Stage-2: for the top-N from stage-1, compute per-residue metal-site evidence vector v
               over the union of site token columns using `site_metric`:
                 - site_dot_sum:  v[i] = sum_j S[i, j]
                 - site_dot_mean: v[i] = mean_j S[i, j]
               Then the site_score is SUM of the top-m entries of v (total evidence).
               (Global order is preserved; we do not re-rank by site evidence.)
    """
    # --- embed query once
    query_res_emb, query_emb = _embed_cached(query_seq, model_name=model_name)

    # --- Stage-1: global sequence score (cosine over seq_embeddings)
    if global_metric == "cosine_mean" and torch is not None:
        
        global_scores = _fast_cosine_scores(query_emb, table)
        
        if global_scores is None:
            global_scores = [cosine_mean(query_emb, emb) for emb in table["seq_embeddings"]]
    elif global_metric == "cosine_mean":
        global_scores = [cosine_mean(query_emb, emb) for emb in table["seq_embeddings"]]
    elif global_metric == "cosine_max":
        global_scores = [cosine_max(query_emb, emb) for emb in table["seq_embeddings"]]
    else:
        raise ValueError(f"Unsupported global_metric: {global_metric}")

    # choose how many to keep for Stage-2
    n = len(table)
    if prefilter_N is None:
        prefilter_N = max(5 * k, 50)
    prefilter_N = min(max(k, prefilter_N), n)

    # indices of top-N globally similar templates
    topN_idx = np.argsort(global_scores)[-prefilter_N:][::-1]

    # --- Stage-2: site evidence only for the top-N (preserve global order)
    hits = []
    m = max(1, site_top_m)

    for rank, idx in enumerate(topN_idx[:k], start=1):
        row = table.iloc[idx]

        # which columns to use (0-based)
        site_cols = (
            _flatten_validated_site_cols(row) if use_validated
            else _collect_site_cols_for_row(row, allow_single_mismatch=True)
        )

        # compute per-residue evidence vector v with the chosen reducer
        if site_cols:
            v, order = _site_vector_for_metric(query_res_emb, row["res_embeddings"], site_cols, site_metric)
        else:
            v, order = None, None

        # aggregate site score (sum of top-m entries of v)
        if v is not None and order is not None and v.numel() > 0:
            top_vals = v.index_select(0, order[:m]) if v.numel() >= m else v
            site_score = float(top_vals.sum().item())

            # gap ratio (mean of top-m vs median of the rest)
            sorted_v = torch.sort(v, descending=True).values
            if v.numel() > m:
                rest = sorted_v[m:]
                gap_ratio = float((sorted_v[:m].mean() / (rest.median() + 1e-9)).item())
            else:
                gap_ratio = float("inf")
        else:
            site_score = float("-inf")
            gap_ratio = float("nan")
            order = torch.empty(0, dtype=torch.long, device=query_res_emb.device) if torch.is_tensor(query_res_emb) else None

        hits.append({
            "rank": int(rank),                      # rank by GLOBAL score (not re-ranked)
            "idx": int(idx),
            "global_score": float(global_scores[idx]),
            "site_score": float(site_score),        # sum of top-m entries of v
            "gap_ratio": float(gap_ratio),          # “pop-out” diagnostic
            "site_cols": site_cols,                 # union of template metal-site columns used (0-based)
            "per_query_site_scores": v,             # torch [Lq] or None
            "top_query_indices": (order[:m].tolist() if order is not None and order.numel() else []),
            "top_query_scores": (v.index_select(0, order[:m]).tolist() if (v is not None and order is not None and order.numel()) else []),
            "template_row": row,
            "pdb_ids": row.get("pdb_ids"),
            "metal_residues": row.get("metal_residues"),
            "token_positions": row.get("token_positions", {}),
            "res_positions": row.get("res_positions"),
        })

    if dev:
        print(f"Top-{k} templates chosen (Stage-1 metric={global_metric}, Stage-2={site_metric}, top_m={site_top_m}):")
        for h in hits:
            print(f"  rank={h['rank']} idx={h['idx']} "
                  f"global={h['global_score']:.4f} site={h['site_score']:.4f} "
                  f"gap={h['gap_ratio']:.2f} pdb_ids={h.get('pdb_ids')}")

    return hits
    
def _explain_site_misses(sequence, bos_offset, labels, delta):
    """
    For a single site (labels=[(resnum, aa), ...]) and a chosen delta, explain
    which labels fell OOB and which failed the AA check.
    Returns a list of dicts: [{'label':'C101', 'resnum':101, 'aa':'C', 'j':6, 'in_bounds':True, 'aa_match':True, 'seq_aa':'C'}, ...]
    """
    seq = sequence or ""
    L = len(seq)
    bos = int(bos_offset)
    out = []
    for resnum, aa in labels:
        j = int(resnum) + int(delta)            # 1-based position
        si = j - bos                             # 0-based index into seq
        inb = (0 <= si < L)
        seqaa = seq[si].upper() if inb else None
        aamat = (seqaa == aa.upper()) if inb else False
        out.append({
            "label": f"{aa}{resnum}",
            "resnum": int(resnum),
            "aa": aa.upper(),
            "j": int(j),
            "seq_idx": int(si),
            "in_bounds": bool(inb),
            "seq_aa": seqaa,
            "aa_match": bool(aamat),
        })
    return out


def why_empty(table, prefer_validated=True, only_problem_rows=True, max_rows=None):
    """
    Print human-friendly diagnostics for rows where not all labeled residues
    are represented in token positions.

    Uses columns produced by your new get_table:
      - metal_residues (original labels per site)
      - token_positions_by_site
      - token_positions_validated_by_site
      - deltas_by_site
      - delta_sources_by_site
      - site_diag_by_site

    prefer_validated=True  -> compare against token_positions_validated_by_site
                           -> if missing, fall back to token_positions_by_site
    """
    import ast

    def parse_groups(metal_residues):
        # same logic you used in _parse_site_groups
        groups = {}
        mr = metal_residues
        if isinstance(mr, str):
            try:
                mr = ast.literal_eval(mr)
            except Exception:
                mr = {}
        mr = mr or {}
        for metal, site_lists in mr.items():
            g = []
            for site in site_lists:
                one = []
                for tag in site:
                    aa = ''.join(ch for ch in tag if ch.isalpha()).upper()
                    ns = ''.join(ch for ch in tag if ch.isdigit())
                    if ns:
                        one.append((int(ns), aa))
                if one:
                    g.append(one)
            groups[metal] = g
        return groups

    printed = 0
    for ridx, row in table.iterrows():
        seq = row.get("sequence", "")
        bos = int(row.get("bos_offset", 0))
        groups = parse_groups(row.get("metal_residues", {}))

        by_site = None
        if prefer_validated:
            by_site = row.get("token_positions_validated_by_site") or None
        if by_site is None:
            by_site = row.get("token_positions_by_site") or {}

        deltas_by_site = row.get("deltas_by_site") or {}
        sources_by_site = row.get("delta_sources_by_site") or {}
        diags_by_site = row.get("site_diag_by_site") or {}

        row_had_problem = False
        lines = []

        for metal, sites in groups.items():
            sites_toks = (by_site.get(metal) or [])
            sites_dlts = (deltas_by_site.get(metal) or [])
            sites_srcs = (sources_by_site.get(metal) or [])
            sites_diags = (diags_by_site.get(metal) or [])

            for sidx, labels in enumerate(sites):
                nlab = len(labels)
                toks = sites_toks[sidx] if sidx < len(sites_toks) else []
                dlt  = sites_dlts[sidx] if sidx < len(sites_dlts) else None
                src  = sites_srcs[sidx] if sidx < len(sites_srcs) else "?"
                diag = sites_diags[sidx] if sidx < len(sites_diags) else {}

                inb  = int(diag.get("n_inbounds", 0))
                mat  = int(diag.get("n_matches", 0))
                got  = len(toks)

                problem = (got < nlab)
                if only_problem_rows and not problem:
                    continue

                if problem:
                    row_had_problem = True

                # explain label-by-label (which are OOB vs AA mismatch)
                detail = []
                if dlt is not None:
                    expl = _explain_site_misses(seq, bos, labels, dlt)
                    oob = [e["label"] for e in expl if not e["in_bounds"]]
                    mis = [e["label"] for e in expl if e["in_bounds"] and not e["aa_match"]]
                    if oob:
                        detail.append(f"OOB: {oob}")
                    if mis:
                        detail.append(f"AA-mismatch: {mis}")

                lines.append(
                    f"  - {metal}[site {sidx}] labels={nlab} got={got} inbounds={inb} "
                    f"matches={mat} Δ={dlt} ({src})" + (f"  {'; '.join(detail)}" if detail else "")
                )

        if lines and (row_had_problem or not only_problem_rows):
            print(f"row {ridx}:")
            for ln in lines:
                print(ln)
            printed += 1
            if max_rows is not None and printed >= max_rows:
                break