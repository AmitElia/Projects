
# helpers_min.py
# Minimal utilities extracted/simplified from helpersv1.py
# - Embedding with ESMC
# - Build a lean table with per-row embeddings and metal-site columns
# - Two-stage top-k search (global dot + residue-site evidence)
#
# Assumptions (by design, only minimal normalization and bounds filtering):
#   * Input dataframe has columns: ["sequence", "metal_residues"]
#   * metal_residues may be a dict or a JSON string of:
#         { metal: [ [ "C101", "H205", ... ], [...], ... ] }
#     If it's a bare list of sites [[...], ...], it's wrapped under a dummy key.
#   * Residue numbers are 1-based positions in the provided sequence.
#   * Similarities are raw dot products (no normalization).
#   * If any site indices fall outside the template length, they are ignored.
#
# Usage:
#   table = get_table_simple(df, embed=True, model_name="esmc_600m", device="cuda")
#   hits  = search_topk_hybrid_simple(query_seq, table, k=10, site_metric="sum", site_top_m=6)

from functools import lru_cache
from typing import Dict, List
import torch
import pandas as pd
import numpy as np
import json

from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig, LogitsOutput

# ---- Embedding config ----
EMBEDDING_CONFIG = LogitsConfig(sequence=True, return_embeddings=True)

# ---- Model init ----
def get_ESMC(model_name: str = "esmc_600m", device: str = "cuda"):
    dev = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
    model = ESMC.from_pretrained(model_name).to(dev).eval()
    return model

def embed_sequence(model: ESMC, sequence: str) -> LogitsOutput:
    protein = ESMProtein(sequence=sequence)
    tensor  = model.encode(protein)
    return model.logits(tensor, EMBEDDING_CONFIG)

def get_sequence_embedding(outs: LogitsOutput) -> torch.Tensor:
    # Mean over tokens of final embeddings
    E = outs.embeddings.squeeze(0).to(torch.float32)   # [L, D]
    return E.mean(dim=0)                               # [D]

@lru_cache(maxsize=2048)
def _embed_cached(seq: str, model_name: str = "esmc_600m", device: str = "cuda"):
    model = get_ESMC(model_name, device)
    outs  = embed_sequence(model, seq)
    res   = outs.embeddings.squeeze(0).to(torch.float32).cpu()  # [L, D] on CPU
    vec   = res.mean(dim=0)                                     # [D]  on CPU
    return res, vec

# ---- Metal-site parsing ----
def _metal_sites_to_cols(metal_residues: Dict[str, List[List[str]]] | list | str) -> List[int]:
    """
    Convert a dict (or JSON string) of metal sites to 0-based column indices J in template res_embeddings.
    Each tag like 'C101' contributes residue index 100 (since residues are 1-based).
    Union across all metals/sites; return sorted unique indices.
    Accepts:
        metal_residues: dict[str, list[list[str]]] OR list[list[str]] OR JSON of either.
    """
    if isinstance(metal_residues, str):
        metal_residues = json.loads(metal_residues)
    if isinstance(metal_residues, list):
        metal_residues = {"_": metal_residues}

    cols = set()
    for sites in metal_residues.values():
        for site in sites:
            for tag in site:
                num = int(''.join(ch for ch in tag if ch.isdigit()))  # e.g., "C101" -> 101
                cols.add(num - 1)  # 0-based
    return sorted(cols)

# ---- Table builder (lean) ----
def get_table_simple(df: pd.DataFrame, *, embed: bool = True,
                     model_name: str = "esmc_600m", device: str = "cuda") -> pd.DataFrame:
    rows = []
    model = get_ESMC(model_name, device) if embed else None

    for _, r in df.iterrows():
        row = r.to_dict()
        if embed:
            outs = embed_sequence(model, row["sequence"])
            row["res_embeddings"] = outs.embeddings.squeeze(0).to(torch.float32).cpu()  # [L, D]
            row["seq_embeddings"] = row["res_embeddings"].mean(dim=0)                   # [D]
        row["site_cols"] = _metal_sites_to_cols(row["metal_residues"])                 # list[int]
        rows.append(row)

    return pd.DataFrame.from_records(rows, columns=list(rows[0].keys()) if rows else None)

# ---- Similarities ----
def _dot(a: torch.Tensor, b: torch.Tensor) -> float:
    a = torch.as_tensor(a, dtype=torch.float32).view(-1)
    b = torch.as_tensor(b, dtype=torch.float32).view(-1)
    return float(torch.dot(a, b).item())

def _sim_matrix(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    # A: [Lq, D], B: [Lt, D]  ->  S = A @ B^T  [Lq, Lt]
    A = torch.as_tensor(A, dtype=torch.float32)
    B = torch.as_tensor(B, dtype=torch.float32)
    return A @ B.t()

def _site_vector(query_res_emb: torch.Tensor,
                 templ_res_emb: torch.Tensor,
                 site_cols: List[int],
                 reducer: str = "sum") -> torch.Tensor:
    """
    v[i] = sum_{j in J} S[i,j]       if reducer == 'sum'
    v[i] = (1/|J|) * sum_{j in J} S[i,j] if reducer == 'mean'
    If J after bounds-filtering is empty, return zeros (no evidence).
    """
    S = _sim_matrix(query_res_emb, templ_res_emb)        # [Lq, Lt]
    Lq, Lt = S.shape

    # bounds-filter J to valid template positions
    J = [j for j in site_cols if 0 <= j < Lt]
    if len(J) == 0:
        return torch.zeros(Lq, dtype=torch.float32)

    J = torch.as_tensor(J, dtype=torch.long)
    sub = S.index_select(1, J)                           # [Lq, |J|]
    if reducer == "mean":
        return sub.mean(dim=1)                           # [Lq]
    return sub.sum(dim=1)                                # [Lq]

# ---- Two-stage Top-K search ----
def search_topk_hybrid_simple(query_seq: str,
                              table: pd.DataFrame,
                              k: int = 10,
                              site_metric: str = "sum",     # 'sum' or 'mean'
                              site_top_m: int = 6,
                              model_name: str = "esmc_600m",
                              device: str = "cuda"):
    # Embed query once
    q_res, q_vec = _embed_cached(query_seq, model_name, device)

    # Stage-1: global score by raw dot product
    global_scores = [ _dot(q_vec, row["seq_embeddings"]) for _, row in table.iterrows() ]

    # Grab top-k indices by global score
    idx_sorted = np.argsort(global_scores)[-k:][::-1].tolist()

    # Stage-2: for those, compute residue-site evidence
    hits = []
    for rank, idx in enumerate(idx_sorted, start=1):
        row = table.iloc[idx]
        v = _site_vector(q_res, row["res_embeddings"], row["site_cols"], reducer=site_metric)
        # site_score = sum of top-m entries of v
        m = max(1, int(site_top_m))
        tt = min(m, v.numel())
        if tt == 0:
            site_score = 0.0
            top_vals = torch.empty(0, dtype=torch.float32)
            top_idx  = torch.empty(0, dtype=torch.long)
        else:
            top_vals, top_idx = torch.topk(v, k=tt, largest=True)
            site_score = float(top_vals.sum().item())

        hits.append({
            "rank": int(rank),
            "idx": int(idx),
            "global_score": float(global_scores[idx]),
            "site_score": site_score,
            "site_cols": row["site_cols"],
            "top_query_indices": top_idx.tolist(),
            "top_query_scores": top_vals.tolist(),
            "template_row": row,
        })
    hits = pd.DataFrame.from_records(hits)
    return hits
