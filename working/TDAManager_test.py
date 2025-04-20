import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[1] if '__file__' in globals() else Path().resolve().parents[0]
sys.path.insert(0, str(project_root))

from tomato.database import TDAManager

tm = TDAManager()
# get full baseline dataset
df_critic = tm.get("df_critic")
# get computed downsample df_small
df_critic_small = tm.get("bert_unif1k")
# get *cached* encoding df_enc
df_enc = tm.get("bert_unif1k","encoding")
# get computed pooled mat of vectors X
X = tm.get("bert_unif1k","pooled")
# get computed cos distance mat on CLS tokens
mat1 = tm.get("bert_unif1k","cls","cos") 
# get computed cos distance mat on pooled vectors
mat1 = tm.get("bert_unif1k","pooled","cos") 
# get *cached* Sinkhorn r=0.01 distance mat with cos as base
mat2 = tm.get("bert_unif1k","pooled","cos_ws01")
# Note: "pooled" does not impact the above result

print(mat2)