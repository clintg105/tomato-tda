import torch, pandas as pd
from transformers import BertTokenizer, BertModel
from joblib import Parallel, delayed
from tqdm import tqdm

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

if __name__ == "__main__":
    texts = ["This movie was great!", "Not so good.", "Average film."] * 5
    ids = list(range(len(texts)))
    df = bert_encode_reviews(texts, ids, "bert-base-uncased")
    print(df.head())