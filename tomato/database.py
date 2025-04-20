from __future__ import annotations

import re
import json
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from tomato.utils import load_critic_review_df, tomato_data_path
from tomato.encoding import bert_encode_reviews

class Mode(Enum):
    """What should happen if the artefact is missing on disk?"""
    AUTO  = "auto"   # load if present, otherwise compute + persist
    DISK  = "disk"   # only load – raise if the file is absent
    FORCE = "force"  # recompute even if a file exists, then overwrite


class TDAManager:
    """
    Interface to on‑disk cache.

    Root
    └─ Encodings/{dataset}/
       ├─ encoding.parquet
       ├─ distances/{metric}.npy
       └─ tda/
          ├─ full/{metric}-{downsample}.npz
          └─ split/{split}/{metric}-{downsample}.npz
    """
    _re_metric   = re.compile(r"^[a-z0-9_]+$")
    _re_down     = re.compile(r"^[a-zA-Z0-9]+$")
    _re_split    = re.compile(r"^[a-zA-Z0-9_\-]+$")

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
        if len(K) == 1:
            if K[0] == "df_critic":
                return load_critic_review_df()
        else:
            enc_dir = self.root / K[0]

            if len(K) == 2 and K[1] == "encoding":
                f = enc_dir / "encoding.parquet"
                return pd.read_parquet(f) # raises if missing

            if len(K) == 3 and K[1] == "distances": # .npy
                f = self._enc_dir(K[0]) / "distances" / f"{K[2]}.npy"
                return np.load(f) # raises if missing

        raise FileNotFoundError  # unknown pattern – treat as “not on disk”

    def _compute(self, K: Tuple[str, ...], path: str = None) -> Any:
        """
        Dispatch to a concrete builder.
        TODO: Make branches single‑line calls to “_make_*” helpers.
        """
        
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
                else:
                    raise ValueError(f"unknown sample spec '{key}'")
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
                    raise ValueError(f"unknown encoding spec '{enc}'")
            elif K[1] == "pooled_vectors":
                df_enc = self.get(K[0], "encoding")
                dim_cols = [c for c in df_enc.columns if c.startswith('dim_')]
                mean_vectors = np.vstack(
                    df_enc\
                        .groupby('review_id')\
                        .apply(lambda g: np.average(
                            g[dim_cols].values, weights=g.attention_mask, axis=0))\
                        .values)
                return mean_vectors

        
        if len(K) == 3:
            if K[1] == "metric":
                mean_vectors = self.get(K[0], "pooled_vectors")
            
        raise KeyError(f"Unrecognised path: {K!r}")


