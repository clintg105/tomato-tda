from __future__ import annotations

import re
import json
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from tomato.utils import load_critic_review_df, tomato_data_path, load_movie_df
from tomato.encoding import bert_encode_reviews
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
            tdadir = enc_dir / K[1] / "tda" / (
                "full" if K[3] == "full" else Path("split") / K[3]
            )
            f = tdadir / f"{K[2]}.npz"
            return np.load(f, allow_pickle=True)["dgms"] # raises if missing

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
                    how='inner'
                ).dropna(
                    subset=['review_content']
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
                    # get two most prolific critics 
                    top_two_critics = df_full.critic_name.value_counts()[:2].index.tolist()
                    # stratified random sample, 800 samples spready across 16 buckets 9
                    df_strat = df_full[
                        df_full.critic_name.isin(top_two_critics) &
                        df_full.content_rating.isin(['R', 'PG']) &
                        df_full.drama_or_comedy.notna()
                    ].groupby(
                        ['critic_name', 'content_rating', 'drama_or_comedy', 'review_type']
                    ).sample(
                        50, random_state=12
                    ).reset_index(
                        drop=True
                    )
                    return df_strat
                else:
                    raise ValueError(f"Unknown sample spec '{key}'")
            elif K[1] == "encoding":
                df_small = self.get(K[0],'df_critic')
                if enc == "bert":
                    texts = df_small.review_content.astype(str).tolist()
                    ids = df_small.index.tolist()
                    df_enc = bert_encode_reviews(texts, ids, "bert-base-uncased")
                    Path(path).parent.mkdir(parents=True, exist_ok=True)
                    df_enc.to_parquet(path, compression='snappy')
                    return df_enc
                else:
                    raise ValueError(f"Unknown encoding spec '{enc}'")
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
                if wass_spec == "wass":
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
        if len(K) == 4: # (ds, red, metric, split)
            ds, red, metric, split = K
            if path is None:
                base = self.root / ds / red / "tda"
                tdir = base / ("full" if split == "full"
                               else Path("split") / split)
                path = tdir / f"{metric}.npz"
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)

            # D sub-mat
            D = self.get(ds, red, metric)
            if split == "full":
                idx = np.arange(D.shape[0])
            else:
                # *Minimal* canned splits; extend as needed.
                df = self.get(ds, "df_critic")
                if split == "fresh":
                    idx = np.where(df.is_fresh.values)[0]
                elif split == "rotten":
                    idx = np.where(~df.is_fresh.values)[0]
                else: # treat as critic name, example
                    idx = np.where(df.critic_name == split)[0]
                if idx.size == 0:
                    raise ValueError(f"Unknown/empty split '{split}'")
            D = D[np.ix_(idx, idx)]

            # run Rip & cache
            from ripser import ripser
            dgms = ripser(D, distance_matrix=True, maxdim=1)["dgms"]
            np.savez(path, dgms=dgms)
            return dgms
        
        raise KeyError(f"Unrecognised path: {K!r}")

def get_pdist2_args_from_base(spec):
    if spec == "cos":
        return ["cosine"]
    elif spec == "mp1":
        return ["minkowski",1]
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