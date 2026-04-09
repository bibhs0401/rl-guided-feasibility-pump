"""
instance_generator_sparse.py
Matches new_instance_generator_MIP.ipynb exactly, saves as sparse .npz.
"""

import os
import random
import numpy as np
from scipy import sparse


def generate_instance(seed=None):
    if seed is not None:
        random.seed(seed)

    n = 4000
    m = n * 3
    p = random.choice(range(5, 6))  # p=5

    # --- A matrix ---
    weight_bin  = [0.016667] * 31;  weight_bin[0]  = 0.5
    weight_cont = [0.008333] * 61;  weight_cont[30] = 0.5  # index 30 = value 0

    rows, cols, vals = [], [], []
    for i in range(m):
        for j in range(n):
            v = random.choices(
                range(31) if j < int(0.8 * n) else range(-30, 31),
                weights=weight_bin if j < int(0.8 * n) else weight_cont,
                k=1
            )[0]
            if v != 0:
                rows.append(i); cols.append(j); vals.append(float(v))

    A = sparse.csr_matrix((vals, (rows, cols)), shape=(m, n), dtype=np.float32)

    # --- b vector ---
    sparsity = n - np.diff(A.indptr)  # zeros per row
    b = np.array([random.randint(int(sparsity[i]) + 1, int(10 * sparsity[i]) + 1)
                  for i in range(m)], dtype=np.float32)

    # --- c matrix ---
    rng = range(-10 * p, 10 * p + 1)
    wc  = [0.5 / len(rng)] * len(rng);  wc[10 * p] = 0.5
    c   = [random.choices(rng, weights=wc, k=1)[0] for _ in range(n * p)]

    # sign-balancing (from notebook)
    pos, neg = [0] * n, [0] * n
    for idx in range(len(c)):
        col = idx % n
        if c[idx] < 0:
            neg[col] += 1
            if idx + n < len(c) and c[idx + n] < 0 and neg[col] > pos[col]:
                c[idx + n] *= -1
        elif c[idx] > 0:
            pos[col] += 1
            if idx + n < len(c) and c[idx + n] > 0 and neg[col] < pos[col]:
                c[idx + n] *= -1
        elif c[idx] == 0:
            if idx + n < len(c):
                if c[idx + n] < 0 and neg[col] > pos[col]:
                    c[idx + n] *= -1
                if c[idx + n] > 0 and neg[col] < pos[col]:
                    c[idx + n] *= -1

    c = np.array(c, dtype=np.float32).reshape(p, n)

    # --- d vector ---
    d = np.array([
        random.randint(int(sum(abs(v) for v in c[k] if v < 0)) + 1,
                       int(sum(abs(v) for v in c[k] if v < 0)) + 10)
        for k in range(p)
    ], dtype=np.float32)

    return {"A": A, "b": b, "c": c, "d": d, "n": n, "m": m, "p": p}


def save_instance(inst, path):
    if not path.endswith(".npz"):
        path += ".npz"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    sparse.save_npz(path, inst["A"])
    archive = dict(np.load(path, allow_pickle=False))
    archive.update({"b": inst["b"], "c": inst["c"], "d": inst["d"],
                    "n": np.array([inst["n"]]), "m": np.array([inst["m"]]),
                    "p": np.array([inst["p"]])})
    np.savez_compressed(path, **archive)


def load_instance(path, dense=False):
    if not path.endswith(".npz"):
        path += ".npz"
    arc = np.load(path, allow_pickle=False)
    A   = sparse.csr_matrix((arc["data"], arc["indices"], arc["indptr"]),
                             shape=tuple(arc["shape"])).astype(np.float32)
    return {"A": A.toarray() if dense else A,
            "b": arc["b"], "c": arc["c"], "d": arc["d"],
            "n": int(arc["n"][0]), "m": int(arc["m"][0]), "p": int(arc["p"][0])}


if __name__ == "__main__":
    out_dir = "./instances"
    random.seed(10)
    num_instances = 100
    for i in range(num_instances):
        inst = generate_instance()
        path = os.path.join(out_dir, f"instance_{i+1}.npz")
        save_instance(inst, path)
        mb = os.path.getsize(path) / 1024**2
        print(f"instance_{i+1}: n={inst['n']}, m={inst['m']}, p={inst['p']}, {mb:.1f} MB")
