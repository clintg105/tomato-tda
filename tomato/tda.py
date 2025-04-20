import numpy as np
from ripser import ripser
from persim import plot_diagrams
import gudhi

def maxmin_landmarks(D, k, seed=None):
    """
    Select landmarks by max–min on a distance matrix.

    Args:
      D: (n,n) ndarray of pairwise distances.
      k: number of landmarks.
      seed: RNG seed for reproducibility.

    Returns:
      (L, W): landmark and witness index lists.
    """
    if seed is not None:
        np.random.seed(seed)
    n = D.shape[0]
    L = [np.random.randint(n)]
    for _ in range(k - 1):
        L.append(int(D[:, L].min(axis=1).argmax()))
    W = [i for i in range(n) if i not in L]
    return L, W


def witness_dm(D, k=100, maxdim=1, alpha2=1.0, seed=None):
    """
    Compute witness-complex persistence on a distance matrix.

    Args:
      D: (n,n) ndarray of pairwise distances.
      k: number of landmarks.
      maxdim: max homology dimension.
      alpha2: max squared alpha threshold.
      seed: RNG seed for landmark selection.

    Returns:
      list of (dim, (birth, death)) pairs.
    """
    L, W = maxmin_landmarks(D, k, seed)
    table = [
        sorted([(l, D[w, l]**2) for l in L], key=lambda x: x[1])
        for w in W
    ]
    st = gudhi.WitnessComplex(nearest_landmark_table=table)
    st = st.create_simplex_tree(
        max_alpha_square=alpha2,
        limit_dimension=maxdim
    )
    return to_dgms(st.persistence(homology_coeff_field=2))


def to_dgms(persistence):
    """
    Convert GUDHI persistence list to Ripser‐style diagrams.

    Args:
      persistence: list of (dim, (birth, death)) pairs.

    Returns:
      dgms: list of (m×2) ndarrays, dgms[i] for H_i.
    """
    maxdim = max((d for d, _ in persistence), default=0)
    dgms = []
    for i in range(maxdim + 1):
        pts = [pt for (d, pt) in persistence if d == i]
        dgms.append(np.array(pts) if pts else np.zeros((0, 2)))
    return dgms


if __name__ == "__main__":
    import sys
    import time
    from pathlib import Path
    import matplotlib.pyplot as plt

    project_root = Path(__file__).resolve().parents[1] if '__file__' in globals() else Path().resolve().parents[0]
    sys.path.insert(0, str(project_root))

    from tomato.database import TDAManager
    tm = TDAManager()
    D = tm.get("bert_unif1k", "pooled", "cos_ws01")

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    print("Ripser H0/H1:")
    t0 = time.time()
    dgms_ripser = ripser(D, maxdim=1, thresh=0.5, distance_matrix=True)['dgms']
    t1 = time.time()
    plot_diagrams(dgms_ripser, ax=axs[0], show=False)
    axs[0].set_title(f"Ripser (t = {t1 - t0:.2f}s)")

    print("Witness H0/H1:")
    t0 = time.time()
    dgms_witness = witness_dm(D, k=250, maxdim=1, alpha2=0.2, seed=42)
    t1 = time.time()
    plot_diagrams(dgms_witness, ax=axs[1], show=False)
    axs[1].set_title(f"Witness (t = {t1 - t0:.2f}s)")

    plt.tight_layout()
    plt.show()