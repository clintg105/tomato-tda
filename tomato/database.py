from __future__ import annotations

import re
import json
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from itertools import product
from time import perf_counter
from pprint import pformat

from ripser import ripser

from tomato.utils import load_critic_review_df, tomato_data_path, load_movie_df
from tomato.encoding import bert_encode_reviews, bow_encode_reviews, tfidf_encode_reviews
from tomato.metrics import pdist2, wasserstein_distances_sinkhorn_parallel, wasserstein_distances_parallel, geodesic_isomap

class Mode(Enum):
    """What should happen if the artefact is missing on disk?"""
    AUTO  = "auto"   # load if present, otherwise compute + persist
    DISK  = "disk"   # only load – raise if the file is absent
    FORCE = "force"  # recompute even if a file exists, then overwrite


class TDAManager:
    """
    Lazy loading / file system memoization / artifact store.

    Root
    └─ {dataset}/
       ├─ encoding.parquet
       └─ {transform}/
            ├─ metric/{metric}.npy
            └─ tda/
                ├─ full/{metric}-{downsample}.npz
                └─ split/{split}/{metric}-{downsample}.npz
    
    Example:
    from tomato.database import TDAManager

    tm = TDAManager()                                       # init
    df_critic = tm.get("df_critic")                         # get full baseline dataset
    df_critic_small = tm.get("bert_unif1k")                 # get computed downsample df_small
    df_enc = tm.get("bert_unif1k","encoding")               # get *cached* encoding df_enc
    X = tm.get("bert_unif1k","pooled")                      # get computed pooled mat of vectors X
    mat1 = tm.get("bert_unif1k","cls","cos")                # get computed cos distance mat on CLS tokens
    mat1 = tm.get("bert_unif1k","pooled","cos")             # get computed cos distance mat on pooled vectors
    mat2 = tm.get("bert_unif1k","pooled","cos_ws01")        # get *cached* Sinkhorn r=0.01 distance mat with cos as base
    """

    # construction
    def __init__(self):
        self.root   = tomato_data_path()
        self._memo: Dict[Tuple[str, ...], Any] = {}      # full path‑tuple → object

    # public API
    def get(self, *keys: str, mode: Mode = Mode.AUTO) -> Any:
        """
        Grab (and cache) an artefact. Examples:
        >>> tm.get("bert_unif5k", "df_critic")
        >>> tm.get("bert_unif5k", "encoding")
        >>> tm.get("bert_unif5k", "metric", "cos_wass_5nn")
        """
        if not keys:
            raise ValueError("At least one key required")

        # immutable lookup key for the memo‑table
        K = tuple(keys)

        # FORCE -> recompute no matter what
        if mode is Mode.FORCE:
            value = self._compute(K)
            self._memo[K] = value
            return value

        # already cached in memory?
        if K in self._memo:
            return self._memo[K]

        # try disk or compute, depending on mode
        try:
            value = self._load_from_disk(K)
        except FileNotFoundError as err:
            if mode is Mode.DISK:
                raise
            value = self._compute(K, err.filename)

        self._memo[K] = value
        return value

    # -- helpers --
    def _load_from_disk(self, K: Tuple[str, ...]) -> Any:
        """
        Attempt to hydrate *K* directly from disk; raise FileNotFoundError
        if the expected file is not present.
        """
        enc_dir = self.root / K[0]

        if len(K) == 2 and K[1] == "encoding":
            f = enc_dir / "encoding.parquet"
            return pd.read_parquet(f) # raises if missing

        if len(K) == 3: # .npy
            f = enc_dir / K[1] / "metric" / f"{K[2]}.npy"
            return np.load(f) # raises if missing
        
        if len(K) == 4: # (ds, red, metric, split)
            ds, red, metric, split = K
            tdadir = (self.root / ds / red / "tda" /
                      ("full" if split == "full" else Path("split") / split))
            f = tdadir / f"{metric}.npz"
            return np.load(f, allow_pickle=True)["out"] # raises if missing

        raise FileNotFoundError  # unknown pattern – treat as “not on disk”

    def _compute(self, K: Tuple[str, ...], path: str = None) -> Any:
        """
        Dispatch to a concrete builder.
        TODO: Make branches single‑line calls to “_make_*” helpers.
        """
        # "global" data
        if len(K) == 1:
            if K[0] == "df_critic":
                return load_critic_review_df()
            if K[0] == "df_movie":
                return load_movie_df()
            if K[0] == "df_full":
                df_critic = self.get('df_critic')
                df_movie = self.get('df_movie')

                # merge critic-level data and movie-level data 
                df_full = df_critic.merge(
                    df_movie,
                    on='rotten_tomatoes_link',
                    how='inner',
                    suffixes=('', '_y')
                ).dropna(
                    subset=['review_content']
                )
                # identify "pure" drama vs "pure" comedy 
                df_full['drama_or_comedy'] = np.where(
                    (
                        df_full['genres'].str.contains('Drama', na=False) & 
                        ~df_full['genres'].str.contains('Comedy', na=False)
                    ),
                    'Drama',
                    np.where(
                        (
                            df_full['genres'].str.contains('Comedy', na=False) & 
                            ~df_full['genres'].str.contains('Drama', na=False)
                        ),
                        'Comedy',
                        None
                    )
                )
                return df_full
            
            else:
                return self.get(K[0], "df_critic")
        
        # downsampled/encoded data
        if len(K) == 2:
            enc, downsample = K[0].split("_")
            if K[1] == "df_critic":
                df_full = self.get('df_critic')
                df_full = df_full[~df_full.review_content.isna()]
                # parse downsample. Ex: unif5k
                m = re.fullmatch(r"([a-zA-Z]+)(\d+)k?", downsample)
                key = m.group(1).lower()
                num = int(m.group(2)) * (1000 if downsample.endswith("k") else 1)
                if key == "unif":
                    return df_full.sample(num,random_state=42)
                if key == "strat":
                    df_full = self.get('df_full')
                    # get two most prolific critics 
                    top_two_critics = df_full.critic_name.value_counts()[:2].index.tolist()
                    # stratified random sample, 800 samples spready across 16 buckets 9
                    cols = ['critic_name', 'content_rating', 'drama_or_comedy', 'review_type']
                    df_strat = df_full[
                        df_full.critic_name.isin(top_two_critics) &
                        df_full.content_rating.isin(['R', 'PG']) &
                        df_full.drama_or_comedy.notna()
                    ].groupby(
                        cols
                    ).sample(
                        50, random_state=12
                    ).reset_index(
                        drop=True
                    )
                    unique_vals = {col: df_strat[col].unique().tolist() for col in cols}
                    print(f"Stratified categories: {unique_vals}")
                    return df_strat
                else:
                    raise ValueError(f"Unknown sample spec '{key}'")
            if K[1] == "df_full":
                df_small = self.get(K[0],'df_critic')
                df_full = df_small.merge(
                    self.get('df_movie'),
                    on='rotten_tomatoes_link',
                    how='inner',
                    suffixes=('', '_y')
                )
                return df_full
            elif K[1] == "encoding":
                df_small = self.get(K[0],'df_critic')
                texts = df_small.review_content.astype(str).tolist()
                ids = df_small.index.tolist()
                if enc == "bert":
                    df_enc = bert_encode_reviews(texts, ids, "bert-base-uncased")
                elif enc == "bow":
                    df_enc = bow_encode_reviews(texts, ids)
                elif enc == "tfidf":
                    df_enc = tfidf_encode_reviews(texts, ids)
                else:
                    raise ValueError(f"Unknown encoding spec '{enc}'")
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                df_enc.to_parquet(path, compression='snappy')
                return df_enc
            # Everything below requires encoding and is used in base metrics 
            # (this should not impact wasserstein metrics, but it may later 
            # if we subtract PCs)
            elif K[1] == "pooled":
                df_enc = self.get(K[0], "encoding")
                dim_cols = [c for c in df_enc.columns if c.startswith('dim_')]
                mean_vectors = np.vstack(
                    df_enc\
                        .groupby('review_id')\
                        .apply(lambda g: np.average(
                            g[dim_cols].values, weights=g.attention_mask, axis=0))\
                        .values)
                return mean_vectors
            elif K[1] == "cls":
                df_enc = self.get(K[0], "encoding")
                dim_cols = [c for c in df_enc.columns if c.startswith('dim_')]
                cls_vectors = df_enc[df_enc['token_id'] == 0][dim_cols].values
                return cls_vectors
            else:
                ValueError(f"Unrecognised reduction method: {K[1]}")

        
        # metric data
        if len(K) == 3:
            spec = K[2]
            wass_check = contains_re(spec, ['wass', r'ws\d+'])
            geod_check = contains_re(spec, [r'\d+nn'])
            if geod_check[0]:
                spec_base = geod_check[3]
                X = self.get(K[0], K[1], spec_base)

                n = int(re.fullmatch(r"(\d+)nn", geod_check[2]).group(1))
                D = geodesic_isomap(X, n)
            elif wass_check[0]:
                # TODO: not impacted by K[1] (recheck other subdirs?)
                df_enc = self.get(K[0], "encoding")
                spec_base = wass_check[3]
                args = get_pdist2_args_from_base(spec_base)

                wass_spec = wass_check[2]
                if wass_spec == "ws01":
                    D = wasserstein_distances_sinkhorn_parallel(df_enc, *args, 0.01)
                elif wass_spec == "wass":
                    D = wasserstein_distances_parallel(df_enc, *args)
                else:
                    raise ValueError(f"unknown encoding spec '{spec}'")
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                np.save(path, D)
            else:
                # base metric
                X = self.get(K[0], K[1])
                args = get_pdist2_args_from_base(spec)
                D = pdist2(X,X,*args)
            return D

        # TDA data
        if len(K) == 4:                                      # (ds, red, metric, split)
            ds, red, metric, split = K
            if path is None:
                base = self.root / ds / red / "tda"
                path = (base / "full" / f"{metric}.npz" if split == "full"
                        else base / "split" / split / f"{metric}.npz")
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)

            D   = self.get(ds, red, metric)
            idx = (np.arange(D.shape[0]) if split == "full"
                   else self._split_idx(self.get(ds, "df_critic"), split))
            if idx.size == 0:
                raise ValueError(f"Empty split '{split}'")
            D = D[np.ix_(idx, idx)]

            out = ripser(D, distance_matrix=True, maxdim=2)
            if out["dgms"][0].size > 0:  # Check if there are any points in the 0-dimensional persistence diagram
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                np.savez(path, out=out, allow_pickle=True)
            else:
                print(f"Warning: Persistence diagram is empty. No data saved to {path}.")
            return out
        
        raise KeyError(f"Unrecognised path: {K!r}")
    
    def _split_idx(self, df: pd.DataFrame, spec: str) -> np.ndarray:
        """
        Return row-indices matching an underscore-delimited *spec* drawn from
        {critic_name, content_rating, drama_or_comedy, review_type}.
        Each token must match values in *exactly one* of those columns.
        Ambiguous or unknown tokens raise.
        """
        cols = ['critic_name', 'content_rating', 'drama_or_comedy', 'review_type']
        mask = np.ones(len(df), bool)
        for tok in spec.split('_'):                       # usually one token
            hit = [c for c in cols if tok in df[c].values]
            if len(hit) != 1:
                raise ValueError(f"Ambiguous / unknown split-token '{tok}'")
            mask &= (df[hit[0]] == tok)
        return np.flatnonzero(mask)
    
    def generate_all(tm: "TDAManager") -> None:
        """
        Exhaustively enumerates ( dataset, reduction, metric, split ) keys
        and drives `tm.get(...)` so *every* artefact is computed and persisted.
        """
        class_dict = {'critic_name': ['Dennis Schwartz', 'Roger Ebert'], 'content_rating': ['PG', 'R'], 'drama_or_comedy': ['Comedy', 'Drama'], 'review_type': ['Fresh', 'Rotten']}
        flattened = [item for sublist in class_dict.values() for item in sublist]
        CONFIG = {
            "DATASETS": ["bert_strat800", "bow_strat800", "tfidf_strat800"],
            "REDUCTIONS": ["pooled"],
            "BASE_METRICS": ["cos", "mp1"],
            "WASS_SPECS": ["ws01"],
            "KNN_K": [5, 10],
            "SPLITS": ["full"] + flattened,
        }
        
        def _all_metrics():
            for b in CONFIG["BASE_METRICS"]:
                yield b
                for w in CONFIG["WASS_SPECS"]:
                    bw = f"{b}_{w}"
                    yield bw
                    yield from (f"{bw}_{k}nn" for k in CONFIG["KNN_K"])
                yield from (f"{b}_{k}nn" for k in CONFIG["KNN_K"])

        # pretty banner
        banner = {
            "datasets": CONFIG["DATASETS"],
            "reductions": CONFIG["REDUCTIONS"],
            "metrics": list(_all_metrics()),
            "splits": CONFIG["SPLITS"],
        }
        print("▁▁ TDAManager GENERATE-ALL ▁▁\n" + pformat(banner, sort_dicts=False) + "\n")

        # guarantee global frames exist early (warms cache & avoids racey IO)
        for k in ("df_critic", "df_movie", "df_full"):
            _tic = perf_counter()
            tm.get(k)
            print(f"{k:<60} ➜ {perf_counter() - _tic:6.1f}s")

        # main cartesian product
        for ds, red in product(CONFIG["DATASETS"], CONFIG["REDUCTIONS"]):
            hdr = f"\n── {ds}/{red} ─────────────────────────────────────────"
            print(hdr)

            # make sure encoding & vectors exist
            for key in ("encoding", red):
                _tic = perf_counter()
                tm.get(ds, key)
                print(f"{'get('+ds+','+key+')':<60} ➜ {perf_counter() - _tic:6.1f}s")

            # all metrics for this (ds, red)
            for metric in _all_metrics():
                _tic = perf_counter()
                tm.get(ds, red, metric)
                print(f"{ds},{red},{metric:<20} (metric)        ➜ {perf_counter() - _tic:6.1f}s")

                # generate TDA diagrams for every requested split
                for split in CONFIG["SPLITS"]:
                    _tic = perf_counter()
                    tm.get(ds, red, metric, split)
                    print(f"{ds},{red},{metric},{split:<15} (tda) ➜ {perf_counter() - _tic:6.1f}s")

def get_pdist2_args_from_base(spec):
    if spec == "cos":
        return ["cosine"]
    elif spec == "mp1":
        return ["minkowski",1]
    elif spec == "mp2":
        return ["minkowski",2]
    elif spec == "mpinf":
        return ["minkowski",np.inf]
    else:
        raise ValueError(f"unknown encoding spec '{spec}'")
    


def contains_re(s, pats):
    """
    Return (True, pat, match, without_match) if any _-split part fully matches a pattern.
    Otherwise return (False, None, s, s).
    """
    parts = s.split('_')
    for i, x in enumerate(parts):
        for p in pats:
            if re.fullmatch(p, x):
                match = parts[i]
                without_match = '_'.join(parts[:i] + parts[i+1:])
                return True, p, match, without_match
    return False, None, s, s
