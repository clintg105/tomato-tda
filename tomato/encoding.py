from __future__ import annotations

import torch, pandas as pd
from transformers import BertTokenizer, BertModel
from joblib import Parallel, delayed
from tqdm import tqdm

import numpy as np
from sklearn.decomposition import PCA

import re, numpy as np, pandas as pd
import nltk
from typing import List
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# Globals used by worker processes.
T, M = None, None
def bert_encode_reviews(
    texts: list[str], 
    ids: list[int],
    model_name: str = "bert-base-uncased", 
    max_len: int = 512, 
    nCores: int = -1
) -> pd.DataFrame:
    """Parallel BERT encoding across reviews; returns a DataFrame with a row per token."""

    def enc(rid, txt, max_len=512, model_name="bert-base-uncased"):
        global T, M

        if not isinstance(txt, str):
            raise ValueError(f"Invalid input type for text (ID {rid}): {type(txt)} — {txt!r}")
        
        # Lazy initialization: if not already set in this worker, load the model and tokenizer.
        if T is None or M is None:
            T = BertTokenizer.from_pretrained(model_name)
            M = BertModel.from_pretrained(model_name).eval()
        
        rows = []
        inp = T(txt, return_tensors="pt", truncation=True, max_length=max_len)
        with torch.no_grad():
            out = M(**inp).last_hidden_state.squeeze(0)
        att = inp['attention_mask'].squeeze(0).tolist()
        
        for i, vec in enumerate(out):
            rows.append((rid, i, att[i], *vec.tolist()))
        return rows
    
    data = list(zip(ids, texts))
    res = Parallel(n_jobs=nCores)(
        delayed(enc)(rid, txt, max_len, model_name) 
        for rid, txt in tqdm(data, desc="BERT Encoding", total=len(data))
    )
    rows = [row for sub in res for row in sub]
    if not rows: 
        return pd.DataFrame()
    dims = [f"dim_{i}" for i in range(len(rows[0]) - 3)]
    return pd.DataFrame(rows, columns=["review_id", "token_id", "attention_mask"] + dims)

def subtract_pcs(X, min_pc=None, max_pc=None):
    """
    Subtracts selected principal components from X.

    Args:
      X: ndarray of shape (n, d)
      min_pc: int, float, or None. If int, start index. If float in [0,1), min variance. None means 0.
      max_pc: int, float, or None. If int < 0, drop top -max_pc PCs. If float in (0,1], max variance. None means d.

    Returns:
      X with PCs in range [min_pc, max_pc) removed.

    Examples:
      subtract_pcs(X, 0, -2)        # drop top 2 PCs
      subtract_pcs(X, 2, 5)         # drop PCs 2 to 4
      subtract_pcs(X, 0.9, 1.0)     # drop top 10 percent variance
      subtract_pcs(X, 5, None)      # drop all PCs after 5
    """
    X -= X.mean(0)
    pca = PCA().fit(X)
    C, V, d = pca.components_, np.cumsum(pca.explained_variance_ratio_), X.shape[1]

    def idx(v, is_upper): return (
        d + v if is_upper and isinstance(v, int) and v < 0 else
        np.searchsorted(V, v, 'right' if is_upper else 'left') if isinstance(v, float) else
        (d if is_upper else 0) if v is None else 
        v
    )

    i, j = idx(min_pc, 0), idx(max_pc, 1)
    return X - X @ C[i:j].T @ C[i:j]

def download_if_missing(path, resource):
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(resource, quiet=True)
download_if_missing('corpora/stopwords', 'stopwords')
download_if_missing('corpora/wordnet', 'wordnet')
download_if_missing('taggers/averaged_perceptron_tagger_eng', 'averaged_perceptron_tagger_eng')

_STOP = set(stopwords.words("english"))
_LEM  = WordNetLemmatizer()

def _wn_pos(tag: str):
    return (wordnet.ADJ if tag.startswith("J") else
            wordnet.VERB if tag.startswith("V") else
            wordnet.NOUN if tag.startswith("N") else
            wordnet.ADV if tag.startswith("R") else
            wordnet.NOUN)

def _clean(txt: str) -> str:
    s = re.sub(r"[^a-z]", " ", txt.lower())
    s = re.sub(r"\s+", " ", s)
    words = [w for w in s.split() if w and w not in _STOP]
    return " ".join(_LEM.lemmatize(w, _wn_pos(t)) for w, t in pos_tag(words))

def _frame_from_matrix(mat: np.ndarray, ids: List[int]) -> pd.DataFrame:
    dims = [f"dim_{i}" for i in range(mat.shape[1])]
    rows = [(rid, 0, 1, *vec) for rid, vec in zip(ids, mat)]
    return pd.DataFrame(rows,
                        columns=["review_id", "token_id",
                                 "attention_mask"] + dims)

def bow_encode_reviews(texts: List[str],
                       ids:   List[int],
                       *,
                       binary=True,
                       min_df=2) -> pd.DataFrame:
    """Binary / count BoW embedding with BERT-style output layout."""
    if len(texts) != len(ids):
        raise ValueError("texts and ids must have same length")
    clean_texts = [_clean(t) for t in texts]
    vec = CountVectorizer(binary=binary, min_df=min_df)
    mat = vec.fit_transform(clean_texts).toarray()          # (n, d)
    return _frame_from_matrix(mat, ids)

def tfidf_encode_reviews(texts: List[str],
                         ids:   List[int],
                         *,
                         min_df=2,
                         ngram_range=(1, 3)) -> pd.DataFrame:
    """TF-IDF embedding with BERT-style output layout."""
    if len(texts) != len(ids):
        raise ValueError("texts and ids must have same length")
    clean_texts = [_clean(t) for t in texts]
    vec = TfidfVectorizer(min_df=min_df, ngram_range=ngram_range)
    mat = vec.fit_transform(clean_texts).toarray()          # (n, d)
    return _frame_from_matrix(mat, ids)


if __name__ == "__main__":
    texts = ["This movie was great!", "Not so good.", "Average film."] * 5
    ids = list(range(len(texts)))
    df = bert_encode_reviews(texts, ids, "bert-base-uncased")
    print(df.head())