"""
instance_generator_sparse.py
saves the instances as sparse .npz, significantly compact than the dense version(csv).
Used for training the PPO model.
"""

import os
import random
import argparse
import time
import numpy as np
from scipy import sparse


def generate_instance(seed=None, verbose=False, progress_every_rows=1000, p=None):
    if seed is not None:
        random.seed(seed)

    n = 4000
    m = n * 3
    if p is None:
        p = 3
    else:
        p = int(p)
        if p < 1:
            raise ValueError("p must be >= 1")
    if verbose:
        print(f"  [generate] Start: n={n}, m={m}, p={p}")

    # --- A matrix ---
    weight_bin  = [0.016667] * 31;  weight_bin[0]  = 0.5
    weight_cont = [0.008333] * 61;  weight_cont[30] = 0.5  # index 30 = value 0

    rows, cols, vals = [], [], []
    for i in range(m):
        if verbose and (i % progress_every_rows == 0 or i == m - 1):
            print(f"  [generate] Building A: row {i+1}/{m}")
        for j in range(n):
            v = random.choices(
                range(31) if j < int(0.8 * n) else range(-30, 31),
                weights=weight_bin if j < int(0.8 * n) else weight_cont,
                k=1
            )[0]
            if v != 0:
                rows.append(i); cols.append(j); vals.append(float(v))

    A = sparse.csr_matrix((vals, (rows, cols)), shape=(m, n), dtype=np.float32)
    if verbose:
        print("  [generate] A matrix done")

    # --- b vector ---
    sparsity = n - np.diff(A.indptr)  # zeros per row
    b = np.array([random.randint(int(sparsity[i]) + 1, int(10 * sparsity[i]) + 1)
                  for i in range(m)], dtype=np.float32)
    if verbose:
        print("  [generate] b vector done")

    # --- c matrix ---
    rng = range(-10 * p, 10 * p + 1)
    wc  = [0.5 / len(rng)] * len(rng);  wc[10 * p] = 0.5
    c   = [random.choices(rng, weights=wc, k=1)[0] for _ in range(n * p)]

    # sign-balancing to ensure the instance is feasible
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
    if verbose:
        print("  [generate] c matrix done")

    # --- d vector ---
    d = np.array([
        random.randint(int(sum(abs(v) for v in c[k] if v < 0)) + 1,
                       int(sum(abs(v) for v in c[k] if v < 0)) + 10)
        for k in range(p)
    ], dtype=np.float32)
    if verbose:
        print("  [generate] d vector done")

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


def parse_args():
    parser = argparse.ArgumentParser(description="Generate sparse MIP instances.")
    parser.add_argument("--out-dir", default="./instances", help="Output directory for .npz files")
    parser.add_argument("--num-instances", type=int, default=100, help="Number of instances to generate")
    parser.add_argument("--seed", type=int, default=10, help="Global random seed")
    parser.add_argument(
        "--p",
        type=int,
        default=3,
        help="Number of y objectives (rows in c matrix)",
    )
    parser.add_argument(
        "--progress-every-rows",
        type=int,
        default=1000,
        help="Print A-matrix row progress every N rows during generation",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out_dir = args.out_dir
    random.seed(args.seed)
    print(
        f"Starting generation: instances={args.num_instances}, seed={args.seed}, p={args.p}, out_dir={out_dir}"
    )
    for i in range(args.num_instances):
        print(f"\n[{i+1}/{args.num_instances}] Generating instance_{i+1}.npz")
        t0 = time.time()
        inst = generate_instance(
            verbose=True,
            progress_every_rows=args.progress_every_rows,
            p=args.p,
        )
        path = os.path.join(out_dir, f"instance_{i+1}.npz")
        print(f"  [save] Writing {path}")
        save_instance(inst, path)
        elapsed = time.time() - t0
        mb = os.path.getsize(path) / 1024**2
        print(
            f"[{i+1}/{args.num_instances}] Done: n={inst['n']}, m={inst['m']}, p={inst['p']}, {mb:.1f} MB, {elapsed:.1f}s"
        )
