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

def get_ESMC(model_name: str = "esmc_300m", device: str = "cuda"):
    """Singleton model loader to avoid re-initialization overhead."""
    global _MODEL
    if _MODEL is None:
        _MODEL = ESMC.from_pretrained(model_name).to(device).eval()
    return _MODEL

@lru_cache(maxsize=4096)
def _embed_cached(seq: str):
    client = get_ESMC()
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
        # normalize in float32 for numerical stability, then cast to bf16 for GEMM
        t32 = t.to(device=dev, dtype=torch.float32)
        t32 = torch.nn.functional.normalize(t32, dim=-1)
        t16 = t32.to(dtype=torch.bfloat16)
        embs.append(t16)

    _TRAIN_IDX = torch.stack(embs, dim=0)          # [N, D], cuda, bfloat16
    _TRAIN_IDX_N = n

def _fast_cosine_scores(query_vec, train_df):
    if torch is None or not torch.cuda.is_available():
        return None

    # Build/ensure CUDA bf16 index
    _maybe_build_train_index(train_df, device="cuda")
    if _TRAIN_IDX is None:
        return None

    # Make a working query copy on same device/dtype as index (bf16, cuda)
    if torch.is_tensor(query_vec):
        q32 = query_vec.to(device=_TRAIN_IDX.device, dtype=torch.float32)
    else:
        q32 = torch.as_tensor(query_vec, device=_TRAIN_IDX.device, dtype=torch.float32)

    # normalize in f32 (stable), then cast to bf16 to match index
    q32 = torch.nn.functional.normalize(q32, dim=-1)
    q16 = q32.to(dtype=torch.bfloat16)

    # GPU bf16 matmul; then convert result to f32 before .cpu().numpy()
    sc = (_TRAIN_IDX @ q16).to(dtype=torch.float32).cpu().numpy()   # [N]
    return sc

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

def unwrap_embedding(out):
    if hasattr(out, "embeddings"):       # good case
        return out.embeddings.detach().cpu()
    else:                                # error case
        return str(out)

def get_table(df: pd.DataFrame, embed= True) -> pd.DataFrame:
    rows = []
    client = get_ESMC()
    for _, row in df.iterrows():
        newrow = row.to_dict()

        outs = embed_sequence(client, newrow['sequence'])               # LogitsOutput
        res_emb = outs.embeddings.detach().cpu().squeeze(0)          # (T, D) remove batch
        seq_emb = get_sequence_embedding(outs).detach().cpu()        # (D,) or whatever you return
        newrow["metal_residues"] = ast.literal_eval(newrow["metal_residues"])
        newrow['res_embeddings'] = res_emb                   # (T, D)
        newrow['seq_embeddings'] = seq_emb
        
        mr = newrow.get("metal_residues") or {}
        if isinstance(mr, str):
            try:
                mr = ast.literal_eval(mr)
            except Exception:
                mr = {}

        res_pos_dict = {}
        for metal, sites in mr.items():
            # sites is list of lists like [['D176','N208',...], [...]]
            ints = [
                int(''.join(ch for ch in r if ch.isdigit()))
                for site in sites for r in site
                if any(ch.isdigit() for ch in r)
            ]
            res_pos_dict[metal] = ints  # <- plain lists of ints

        newrow["res_positions"] = res_pos_dict
        
        m   = newrow.get("pdb_resnum_to_idx") or {}
        bos = int(newrow.get("bos_offset", 0))
        L   = int(res_emb.shape[0])  # res_emb defined above when embed=True

        tok_pos = {}
        for metal, nums in res_pos_dict.items():
            idxs = []
            for n in nums:
                i = m.get(n)
                if i is not None:
                    j = i + bos
                    if 0 <= j < L:
                        idxs.append(int(j))
            tok_pos[metal] = idxs

        newrow["token_positions"] = tok_pos

        rows.append(newrow)
    table = pd.DataFrame.from_records(rows, columns=list(rows[0].keys()))
    return table

def get_sequence_embedding(res_embeddings) -> torch.Tensor:
    """Convert ESM-C LogitsOutput to a single fixed-size tensor per sequence.
    Uses the mean of the last hidden layer embeddings over the sequence length.
    """
    return torch.mean(res_embeddings.hidden_states, dim=-2).squeeze(1).mean(0)

def cosine_mean(A: torch.Tensor, B: torch.Tensor) -> float:
    a = A.mean(0)   # [D]
    b = B.mean(0)   # [D]
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()



def cosine_max(A, B):
    a = A.max(0).values
    b = B.max(0).values
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()

def cosine_vec(A: torch.Tensor, B: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(A, B, dim=0).item()

def get_per_res_similarity_matrix(emb1, emb2, metric="cosine_mean", alpha=0.5, beta=0.5):
    if torch is None or not torch.cuda.is_available():
        # CPU fallback: your original slow path or vectorized f32
        return np.array([[cosine_mean(e1, e2) for e2 in emb2] for e1 in emb1], dtype=float)

    # Move inputs to CUDA; normalize in f32; cast to bf16 for GEMM
    A = emb1 if torch.is_tensor(emb1) else torch.as_tensor(emb1)
    B = emb2 if torch.is_tensor(emb2) else torch.as_tensor(emb2)

    A32 = A.to(device="cuda", dtype=torch.float32)
    B32 = B.to(device="cuda", dtype=torch.float32)
    A32 = torch.nn.functional.normalize(A32, dim=-1)
    B32 = torch.nn.functional.normalize(B32, dim=-1)

    A16 = A32.to(dtype=torch.bfloat16)
    B16 = B32.to(dtype=torch.bfloat16)

    S16 = A16 @ B16.T                  # GEMM in bf16 on CUDA (fast)
    return S16.to(dtype=torch.float32).cpu().numpy()   # safe for numpy

def _softmax_temperature(x, temperature: float):
    # x: 1D array-like of global similarities G_t
    z = np.asarray(x, dtype=float) / float(temperature)
    z = z - np.max(z)                       # numerical stability
    e = np.exp(z)
    return e / e.sum()

def _pick_best_row(sites_df: pd.DataFrame, metal_of_interest: str | None = None):
    """
    sites_df: DataFrame with columns ['metal','votes','mass', ...]
      - 'votes' is a 1D np.ndarray (length L), normalized or zeros
      - 'mass' is the pre-normalization sum (float)

    Returns: integer *positional* index into sites_df for the best site,
             or None if no row matches the requested metal.
    """
    df = sites_df
    if metal_of_interest is not None:
        df = df[df['metal'] == metal_of_interest]
        if df.empty:
            return None
    #print("votes: ", df['votes'])
    # tiebreaker: how concentrated the (normalized) votes are
    df = df.assign(
        peak=df['votes'].apply(lambda v: float(np.max(np.asarray(v))) if np.size(v) else 0.0)
    )

    # choose row with max mass, then max peak
    best_label = df.sort_values(['mass', 'peak'], ascending=False).index[0]

    # return *positional* index into the original sites_df (so caller can use .iloc)
    best_iloc = int(np.where(sites_df.index == best_label)[0][0])
    #print("best_iloc: ", best_iloc)
    #print("row: ", df.iloc[best_iloc])
    return best_iloc, df.iloc[best_iloc]['metal']

def reweight_profiles(preds_df, temperature, threshold=None):
    """Recompute 'profile' (and optionally 'hits') for a new temperature without re-searching."""
    for idx in preds_df.index:
        topk   = preds_df.at[idx, "topk"] or []
        vecs   = preds_df.at[idx, "chosen_vectors"] or []
        if not topk or not vecs:
            continue
        g = [float(r.get("global_score", 0.0)) for r in topk]
        w = _softmax_temperature(g, temperature=float(temperature))
        prof = np.zeros_like(vecs[0], dtype=float)
        for wi, v in zip(w, vecs):
            prof += wi * np.asarray(v, float)
        preds_df.at[idx, "profile"] = prof
        if threshold is not None:
            L = len(prof)
            preds_df.at[idx, "hits"] = [i for i in range(1, L - 1) if prof[i] >= float(threshold)]
    return preds_df

def find_most_similar_query_residue(S_t, j):
    i_star = int(np.argmax(S_t[:, j]))   # argmax over query residues
    return i_star

def map_binding_residues(L, t, prior_q=None, c_max=1) -> pd.DataFrame:
    """
    Map template token positions to query residues using t['token_positions'].
    Returns one row per metal (no per-site split).
    """
    S  = t["similarity_matrix"]              # (L_query, L_template)
    Lq, Lt = S.shape
    if L is None or L != Lq:
        L = Lq

    rows = []
    token_positions = t.get("token_positions", {}) or {}

    for metal, token_cols in token_positions.items():
        # If your token_positions are already clean ints, this could just be: token_cols = list(token_cols)
        token_cols = [int(c) for c in (token_cols or []) if c is not None and 0 <= int(c) < Lt]

        votes = np.zeros(L, dtype=float)
        used  = np.zeros(L, dtype=int)

        for pos in token_cols:
            i_star = find_most_similar_query_residue(S, pos)
            vote   = float(S[i_star, pos]) + (float(prior_q.get(i_star, 0)) if prior_q else 0.0)
            if used[i_star] < c_max:
                votes[i_star] += vote
                used[i_star]  += 1

        mass = float(votes.sum())
        votes_norm = (votes / mass) if mass > 0.0 else votes

        rows.append({
            "metal":         metal,
            "token_cols":    token_cols,                      # template columns used
            "mass":          mass,
            "votes":         np.asarray(votes_norm, float),   # length = L (query)
            # optional passthrough if present on t:
            "res_positions": (t.get("res_positions") or {}).get(metal),
        })

    return pd.DataFrame(rows)

def search_top_k_seq_embeddings(query_seq, table, k=5, metric="cosine_mean", dev=None):
    # Embed once (cached)
    query_res_emb, query_emb = _embed_cached(query_seq)

    # Compute scores
    if metric == "cosine_mean" and torch is not None:
        _scores = _fast_cosine_scores(query_emb, table)
        if _scores is not None:
            scores = _scores
        else:
            scores = [cosine_mean(query_emb, emb) for emb in table["seq_embeddings"]]
    elif metric == "cosine_max":
        scores = [cosine_max(query_emb, emb) for emb in table["seq_embeddings"]]
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    # Top-k indices (descending)
    top_k_idx = np.argsort(scores)[-k:][::-1]

    # Build outputs — include fields used by dev prints
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
            "pdb_ids": row["pdb_ids"] if "pdb_ids" in row else None,
            "metal_residues": row["metal_residues"] if "metal_residues" in row else None,
            "token_positions": row["token_positions"] if "token_positions" in row else {},
        	"res_positions":   row["res_positions"]   if "res_positions"   in row else None,
        })
    if dev:
        print(f"Top-{k} templates chosen (metric={metric}):")
        for r in out:
            print(f"  rank={r['rank']} idx={r['idx']} score={r['global_score']:.4f} pdb_ids={r.get('pdb_ids')}")
    return out

def predict_res_weighted_sum(query_seq, topk, prior_q=None, threshold=0.5, c_max=1,
                             temperature=0.1, metal_of_interest=None, dev=None, use_raw_votes=True):
    """
    Returns:
        hits, metals, profile, chosen_vectors, chosen_masses
    """
    if dev:
        print("predicting residues by weighted sum...")

    if not topk:
        return [], [], np.array([]), [], []

    # Use similarity matrix to infer query length (no re-embedding)
    S0 = topk[0].get("similarity_matrix")
    L = int(S0.shape[0]) if S0 is not None else len(query_seq)

    chosen_vectors, chosen_masses, global_scores, metals = [], [], [], []

    # 1) pick ONE site per template
    for t in topk:
        table = map_binding_residues(L, t, prior_q=prior_q, c_max=c_max)
        best = _pick_best_row(table, metal_of_interest=metal_of_interest)
        if best is None:
            continue
        best_iloc, metal = best
        P_norm = np.asarray(table["votes"][best_iloc], float)   # normalized vector
        mass   = float(table["mass"][best_iloc])                # pre-normalization strength
        P_use  = (P_norm * mass) if use_raw_votes else P_norm
        chosen_vectors.append(P_use)
        chosen_masses.append(mass)
        global_scores.append(float(t.get("global_score", 0.0)))
        metals.append(metal)

    if not chosen_vectors:
        return [], [], np.zeros(L, float), [], []

    # reweight chosen site-vectors across templates
    w = _softmax_temperature(global_scores, temperature=temperature)
    profile = np.zeros(L, float)
    for w_t, P_t in zip(w, chosen_vectors):
        profile += w_t * np.asarray(P_t, float)

    hits = [i for i in range(1, L - 1) if profile[i] >= threshold]
    if dev:
        print("w: ", w)
        print(f"predicted {len(hits)} binding residues at threshold {threshold}")
    return hits, metals, profile, chosen_vectors, chosen_masses

def predict_binding(query_seq, table, k=5, metric="cosine_mean", prior_q=None,
                    threshold=0.5, c_max=1, temperature=0.1,
                    metal_of_interest=None, dev=None, use_raw_votes=True):
    """
    Picks the best site per template and combines via temperature-weighted sum.
    """
    topk = search_top_k_seq_embeddings(query_seq, table, k=k, metric=metric, dev=dev)
    if dev:
        for r in topk:
            print("rank:", r.get("rank"), ", pdb_ids:", r.get("pdb_ids"), ", residues:", r.get("metal_residues"))
    hits, metals, profile, chosen_vectors, chosen_masses = predict_res_weighted_sum(
        query_seq, topk, prior_q=prior_q, threshold=threshold, c_max=c_max,
        temperature=temperature, metal_of_interest=metal_of_interest, dev=dev,
        use_raw_votes=use_raw_votes
    )
    return hits, metals, profile, topk, chosen_vectors, chosen_masses

def test_run(df, train_df, k=5, prior_q=None, threshold=0.01, temperature=0.1,
             use_raw_votes=True):  # <—
    out = pd.DataFrame(index=df.index, columns=["hits","metals","profile","topk",
                                                "chosen_vectors","chosen_masses"], dtype=object)
    for idx, row in df.iterrows():
        hits, metals, profile, topk, chosen_vectors, chosen_masses = predict_binding(
            row["sequence"], train_df, k=k, metric="cosine_mean",
            prior_q=prior_q, threshold=threshold, c_max=1,
            temperature=temperature, metal_of_interest=None, dev=False,
            use_raw_votes=use_raw_votes
        )
        out.at[idx,"hits"]           = list(map(int, hits or []))
        out.at[idx,"metals"]         = list(metals or [])
        out.at[idx,"profile"]        = np.asarray(profile) if profile is not None else None
        out.at[idx,"topk"]           = topk
        out.at[idx,"chosen_vectors"] = [np.asarray(v) for v in (chosen_vectors or [])]
        out.at[idx,"chosen_masses"]  = [float(m) for m in (chosen_masses or [])]
    return out

def labels_and_scores_from_profile(pred_df, truth_series, seq_series, base0=True):
    y_true_all, y_score_all = [], []
    for idx in pred_df.index:
        prof = pred_df.at[idx, "profile"]
        if prof is None: continue
        prof = np.asarray(prof, dtype=float)
        if prof.ndim == 2:  # safety
            prof = prof.sum(axis=1)
        L = min(len(seq_series.loc[idx]), len(prof))
        prof = prof[:L]
        y = np.zeros(L, dtype=int)
        d = truth_series.loc[idx]
        if isinstance(d, dict):
            for lst in d.values():
                for p in (lst or []):
                    i = int(p) if base0 else int(p) - 1
                    if 0 <= i < L: y[i] = 1
        y_true_all.append(y); y_score_all.append(prof)
    if not y_true_all: return np.array([]), np.array([])
    return np.concatenate(y_true_all), np.concatenate(y_score_all)

def curves_location(preds, truth, seqs):
    y_true, y_score = labels_and_scores_from_profile(preds, truth, seqs, base0=True)
    if y_true.size == 0 or y_true.sum() == 0:
        raise ValueError("No positives found for location evaluation.")
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    return (prec, rec, ap), (fpr, tpr, auc)

def _norm01(v):
    v = np.asarray(v, dtype=float)
    s = v.sum()
    return v / s if s > 0 else v

def _entropy01(p):
    """Shannon entropy normalized to [0,1] by log(L); returns 0 for delta peak, 1 for uniform."""
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1.0)
    p = p / p.sum()
    H = -np.sum(p * np.log(p))
    return H / math.log(len(p)) if len(p) > 1 else 0.0

def metals_in_truth(d):
    if not isinstance(d, dict): return set()
    return {m for m, lst in d.items() if lst}

def metal_scores_from_vectors(topk, metals, chosen_vectors, chosen_masses=None,
                              temperature=0.1, prior_q=None,
                              alpha=0.5, use_mass=True):
    """
    Per-metal score = sum_i  w_i(global) * conf_i(shape) * mhat_i(mass)
    where conf_i(shape) = alpha*peak + (1-alpha)*(1-entropy) computed on the *normalized* vector,
    and mhat_i is the template's mass normalized across the chosen sites for this query.
    """
    if not topk or not chosen_vectors or not metals:
        return {}

    # global weights
    g = [float(r.get("global_score", 0.0)) for r in topk]
    w = _softmax_temperature(g, temperature=temperature)

    # masses per chosen site
    if chosen_masses is None or len(chosen_masses) != len(chosen_vectors):
        masses = np.array([np.asarray(v, float).sum() for v in chosen_vectors], dtype=float)
    else:
        masses = np.asarray(chosen_masses, dtype=float)

    mhat = masses / masses.sum() if (use_mass and masses.sum() > 0) else np.ones_like(masses)

    scores = defaultdict(float)
    for wi, mi, m, vec in zip(w, mhat, metals, chosen_vectors):
        v = np.asarray(vec, dtype=float)
        # for shape we always normalize the vector itself
        if v.sum() > 0:
            v_norm = v / v.sum()
            peak   = float(v_norm.max())
            sharp  = 1.0 - _entropy01(v_norm)      # 1 = sharp, 0 = flat
        else:
            peak = sharp = 0.0
        conf = alpha*peak + (1.0 - alpha)*sharp    # shape confidence
        scores[m] += wi * conf * mi                # include mass as evidence
    if prior_q:
        for m, prior in prior_q.items():
            if m in scores:
                scores[m] *= float(prior)
    return dict(scores)

def identity_labels_scores(pred_df, truth_series, temperature=0.1, prior_q=None,
                           alpha=0.5, normalize_vectors=True, use_mass=True):
    """
    Micro-avg labels & scores across metals using chosen_vectors + topk weights.
    """
    # universe of metals seen anywhere (truth or predictions)
    all_metals = set()
    for idx in pred_df.index:
        all_metals |= metals_in_truth(truth_series.loc[idx])
        tk = pred_df.at[idx, "topk"]
        if tk:
            for r in tk:
                all_metals |= set((r.get("metal_residues") or {}).keys())
        ms = pred_df.at[idx, "metals"] or []
        all_metals |= set(ms)
    all_metals = sorted(all_metals)

    y_true, y_score = [], []
    for idx in pred_df.index:
        truth_m = metals_in_truth(truth_series.loc[idx])
        topk     = pred_df.at[idx, "topk"]
        metals   = pred_df.at[idx, "metals"] or []
        vectors  = pred_df.at[idx, "chosen_vectors"] or []
        masses   = pred_df.at[idx, "chosen_masses"]  or []
        scores_d = metal_scores_from_vectors(
            topk, metals, vectors, chosen_masses=masses,
            temperature=temperature, prior_q=prior_q,
            alpha=alpha, use_mass=use_mass
        )
        for m in all_metals:
            y_true.append(1 if m in truth_m else 0)
            y_score.append(float(scores_d.get(m, 0.0)))
    return np.array(y_true), np.array(y_score), all_metals

def curves_identity(preds, truth, temperature=0.1, prior_q=None, alpha=0.5):
    y_true, y_score, metals = identity_labels_scores(
        preds, truth, temperature=temperature, prior_q=prior_q, alpha=alpha
    )
    if y_true.size == 0 or y_true.sum() == 0:
        raise ValueError("No positives found for identity evaluation.")
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    return (prec, rec, ap), (fpr, tpr, auc)

def _label_for_key_and_metric(key, metric_value, metric_name):
    k, th, T = key
    return f"k={k}, T={T} ({metric_name}={metric_value:.3f})"

def plot_grid_pr_ax(curves_dict, ax, title_prefix, select=None, equal_square=True):
    for key, v in curves_dict.items():
        if select and not select(key):
            continue
        prec, rec, ap = v["pr"]
        ax.plot(rec, prec, lw=1.6,
                label=_label_for_key_and_metric(key, ap, "AP"))
    if equal_square:
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"{title_prefix} — Precision–Recall", pad=8)
    handles, labels = ax.get_legend_handles_labels()
    return ax, handles, labels

def plot_grid_roc_ax(curves_dict, ax, title_prefix, select=None,
                     equal_square=True, max_fpr=None, log_x=False, pauc_at=None):
    """
    Adds three options:
      max_fpr: float in (0,1], zoom x-axis to [0, max_fpr]
      log_x:   bool, semilog x (spreads small FPRs)
      pauc_at: float in (0,1], append normalized partial AUC up to this FPR to the label
    """
    for key, v in curves_dict.items():
        if select and not select(key):
            continue
        fpr, tpr, auc = v["roc"]
        fpr = np.asarray(fpr, float)
        tpr = np.asarray(tpr, float)

        label = _label_for_key_and_metric(key, auc, "AUC")

        # partial AUC (trapezoid) up to pauc_at, normalized by pauc_at so in [0,1]
        if pauc_at is not None:
            cap = float(pauc_at)
            # ensure coverage at the cap by linear interpolation
            if fpr[-1] < cap:
                # append the cap point at tpr=1 (ROC always reaches (1,1))
                fpr_cap = np.append(fpr, cap)
                tpr_cap = np.append(tpr, 1.0)
            else:
                tpr_at_cap = np.interp(cap, fpr, tpr)
                mask = fpr <= cap
                fpr_cap = np.concatenate([fpr[mask], [cap]])
                tpr_cap = np.concatenate([tpr[mask], [tpr_at_cap]])
            pauc = np.trapz(tpr_cap, fpr_cap) / cap
            label += f", pAUC@{cap:.2f}={pauc:.3f}"

        if log_x:
            # avoid log(0) by clipping zeros to a tiny epsilon
            eps = 1e-8
            fpr_plot = np.clip(fpr, eps, 1.0)
            ax.semilogx(fpr_plot, tpr, lw=1.6, label=label)
        else:
            ax.plot(fpr, tpr, lw=1.6, label=label)

    # baseline (chance)
    if log_x:
        # draw chance line sampled in log space for looks
        x = np.logspace(-8, 0, 200)
        ax.semilogx(x, x, "--", lw=1, color="tab:orange")
    else:
        ax.plot([0, 1], [0, 1], "--", lw=1, color="tab:orange")

    # axes formatting
    if max_fpr is not None:
        ax.set_xlim((1e-8 if log_x else 0.0), max_fpr)
        if not equal_square:
            ax.set_aspect('auto')
    else:
        ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    if equal_square and max_fpr is None:
        ax.set_aspect('equal', adjustable='box')

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{title_prefix} — ROC", pad=8)

    handles, labels = ax.get_legend_handles_labels()
    return ax, handles, labels

def operating_point(y_true, y_score, tau):
    y_true = np.asarray(y_true).astype(int)
    y_hat  = (np.asarray(y_score) >= float(tau)).astype(int)
    TP = int(((y_hat==1)&(y_true==1)).sum())
    FP = int(((y_hat==1)&(y_true==0)).sum())
    FN = int(((y_hat==0)&(y_true==1)).sum())
    TN = int(((y_hat==0)&(y_true==0)).sum())
    prec = TP/(TP+FP) if (TP+FP) else 0.0
    rec  = TP/(TP+FN) if (TP+FN) else 0.0
    fpr  = FP/(FP+TN) if (FP+TN) else 0.0
    tpr  = rec
    return prec, rec, fpr, tpr

def taus_from_quantiles(y_score, quantiles, positive_only=True):
    """Choose τ from score quantiles; avoids ‘all zeros’ by default."""
    s = np.asarray(y_score, float)
    if positive_only:
        s = s[s > 0]
    if s.size == 0:
        return []
    qs = [q for q in (quantiles or []) if 0.0 < q < 1.0]
    return list(np.unique(np.quantile(s, qs)))

def tau_for_target_fpr(y_true, y_score, target_fpr):
    """
    Approximate τ that achieves ≈ target FPR by using the (1 - target_fpr)
    quantile of the NEGATIVE score distribution.
    """
    s = np.asarray(y_score, float)
    y = np.asarray(y_true).astype(int)
    neg = s[y == 0]
    if neg.size == 0:
        return None
    # choose the smallest τ such that P(score >= τ | negative) <= target_fpr
    # 'higher' makes τ conservative (FPR <= target)
    try:
        tau = np.quantile(neg, 1.0 - float(target_fpr), method="higher")
    except TypeError:
        # numpy <1.22 fallback
        tau = np.quantile(neg, 1.0 - float(target_fpr), interpolation="higher")
    return float(tau)

def evaluate_grid(
    test_x, test_y, train_df,
    k_list, temp_list,
    *,
    threshold=0.01,
    marker_taus_loc=None, marker_taus_id=None,
    marker_quantiles_loc=None, marker_quantiles_id=None,
    marker_fprs_loc=None, marker_fprs_id=None,
    prior_q=None, alpha=0.5, use_raw_votes=True, use_mass=True
):
    curves_loc, curves_id, rows = {}, {}, {}
    markers_loc, markers_id = {}, {}

    marker_taus_loc        = list(marker_taus_loc or [])
    marker_taus_id         = list(marker_taus_id  or [])
    marker_quantiles_loc   = list(marker_quantiles_loc or [])
    marker_quantiles_id    = list(marker_quantiles_id  or [])
    marker_fprs_loc        = list(marker_fprs_loc or [])
    marker_fprs_id         = list(marker_fprs_id  or [])

    for k in k_list:
        # Compute predictions ONCE for this k (temperature-independent pieces)
        base_T = (temp_list[0] if temp_list else 0.1)
        preds = test_run(test_x, train_df, k=k, threshold=threshold,
                         temperature=base_T, use_raw_votes=use_raw_votes)

        for T in temp_list:
            # Reweight profiles for this temperature (no new search/embedding)
            reweight_profiles(preds, temperature=T, threshold=None)

            # ---- LOCATION ----
            y_true_loc, y_score_loc = labels_and_scores_from_profile(
                preds, test_y, test_x["sequence"], base0=True
            )
            if y_true_loc.size == 0 or y_true_loc.sum() == 0:
                raise ValueError("No positives found for location evaluation.")
            precL, recL, _ = precision_recall_curve(y_true_loc, y_score_loc)
            apL  = average_precision_score(y_true_loc, y_score_loc)
            fprL, tprL, _ = roc_curve(y_true_loc, y_score_loc)
            aucL = roc_auc_score(y_true_loc, y_score_loc)
            curves_loc[(k, threshold, T)] = {"pr": (precL, recL, apL), "roc": (fprL, tprL, aucL)}

            # ---- IDENTITY ----
            y_true_id, y_score_id, _ = identity_labels_scores(
                preds, test_y, temperature=T, prior_q=prior_q,
                alpha=alpha, use_mass=use_mass
            )
            if y_true_id.size == 0 or y_true_id.sum() == 0:
                raise ValueError("No positives found for identity evaluation.")
            precI, recI, _ = precision_recall_curve(y_true_id, y_score_id)
            apI  = average_precision_score(y_true_id, y_score_id)
            fprI, tprI, _ = roc_curve(y_true_id, y_score_id)
            aucI = roc_auc_score(y_true_id, y_score_id)
            curves_id[(k, threshold, T)] = {"pr": (precI, recI, apI), "roc": (fprI, tprI, aucI)}

            # ---- markers (unchanged) ----
            pr_pts_loc, roc_pts_loc = [], []
            for tau in marker_taus_loc:
                P, R, F, TPR = operating_point(y_true_loc, y_score_loc, tau)
                pr_pts_loc.append((R, P, float(tau), f"τ={tau:g}"))
                roc_pts_loc.append((F, TPR, float(tau), f"τ={tau:g}"))
            for tau in taus_from_quantiles(y_score_loc, marker_quantiles_loc, positive_only=True):
                P, R, F, TPR = operating_point(y_true_loc, y_score_loc, tau)
                pr_pts_loc.append((R, P, float(tau), f"Q={tau:.3g}"))
                roc_pts_loc.append((F, TPR, float(tau), f"Q={tau:.3g}"))
            for tgt in marker_fprs_loc:
                tau = tau_for_target_fpr(y_true_loc, y_score_loc, tgt)
                if tau is not None:
                    P, R, F, TPR = operating_point(y_true_loc, y_score_loc, tau)
                    pr_pts_loc.append((R, P, float(tau), f"FPR@{tgt:g}"))
                    roc_pts_loc.append((F, TPR, float(tau), f"FPR@{tgt:g}"))
            markers_loc[(k, T)] = {"pr": pr_pts_loc, "roc": roc_pts_loc}

            pr_pts_id, roc_pts_id = [], []
            for tau in marker_taus_id:
                P, R, F, TPR = operating_point(y_true_id, y_score_id, tau)
                pr_pts_id.append((R, P, float(tau), f"τ={tau:g}"))
                roc_pts_id.append((F, TPR, float(tau), f"τ={tau:g}"))
            for tau in taus_from_quantiles(y_score_id, marker_quantiles_id, positive_only=True):
                P, R, F, TPR = operating_point(y_true_id, y_score_id, tau)
                pr_pts_id.append((R, P, float(tau), f"Q={tau:.3g}"))
                roc_pts_id.append((F, TPR, float(tau), f"Q={tau:.3g}"))
            for tgt in marker_fprs_id:
                tau = tau_for_target_fpr(y_true_id, y_score_id, tgt)
                if tau is not None:
                    P, R, F, TPR = operating_point(y_true_id, y_score_id, tau)
                    pr_pts_id.append((R, P, float(tau), f"FPR@{tgt:g}"))
                    roc_pts_id.append((F, TPR, float(tau), f"FPR@{tgt:g}"))
            markers_id[(k, T)] = {"pr": pr_pts_id, "roc": roc_pts_id}

            rows[(k, T)] = {
                "k": k, "temperature": T, "threshold": threshold,
                "AP_location": apL, "AUC_location": aucL,
                "AP_identity": apI, "AUC_identity": aucI,
            }

    summary_df = pd.DataFrame(list(rows.values())).sort_values(["k","temperature"]).reset_index(drop=True)
    markers = {"location": markers_loc, "identity": markers_id}
    return summary_df, curves_loc, curves_id, markers

