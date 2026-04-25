"""
instance_generator_sparse.py
Generates instances and saves them as CPLEX LP files (.lp).

The inherent data-generation logic (A, b, c, d) is unchanged from the
sparse .npz version; only the on-disk format has been switched from a
compressed sparse archive to a human-readable LP formulation so the
instances can be solved directly by CPLEX / Gurobi / HiGHS / etc.

The LP file models:
    maximize   sum_i y_i
    subject to A x <= b                 (m rows)
               c_k x + d_k = y_k        (p objective-image equations)
               x_j in {0,1}             for j = 0..floor(0.8*n)-1   (binary)
               x_j >= 0                 for j = floor(0.8*n)..n-1   (continuous)
               y_i free                 for i = 0..p-1
"""

import os
import random
import argparse
import time
import numpy as np
from scipy import sparse


# Must match INTEGER_VARIABLE_FRACTION in mmp_fp_core.py
INTEGER_VARIABLE_FRACTION = 0.8


def generate_instance(
    seed=None,
    verbose=False,
    progress_every_rows=1000,
    p=None,
    n=4000,
    a=3.0,
):
    if seed is not None:
        random.seed(seed)

    n = int(n)
    if n < 1:
        raise ValueError("n must be >= 1")

    m = int(round(n * float(a)))
    if m < 1:
        raise ValueError("m must be >= 1; increase --a or --n")
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
    # Original dense generator: for each row j, count entries where A[i,j]==0
    # ("sparsity_counter"), then
    #   b[j] ~ UniformInteger[ sparsity + 1, 10 * sparsity + 1 ]
    # (see legacy `data()` loop that incremented sparsity_counter when int(A[k]) == 0.)
    nnz_per_row = np.diff(A.indptr)
    zeros_per_row = n - nnz_per_row
    b = np.array(
        [
            random.randint(int(zeros_per_row[i]) + 1, int(10 * zeros_per_row[i]) + 1)
            for i in range(m)
        ],
        dtype=np.float32,
    )

    # --- b vector (nnz-based / "paper" version — keep for later) ---
    # Paper-style: b[i] ~ Uniform[ns_A, 10 * ns_A + 1] where ns_A is the number of
    # NON-ZERO elements per row (actual nnz from CSR).
    # nnz_per_row = np.diff(A.indptr)
    # b = np.array(
    #     [
    #         random.randint(int(nnz_per_row[i]), int(10 * nnz_per_row[i]) + 1)
    #         for i in range(m)
    #     ],
    #     dtype=np.float32,
    # )
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


# -----------------------------------------------------------------------------
# Legacy .npz saver / loader  (commented out in favour of LP export below)
# -----------------------------------------------------------------------------
# def save_instance(inst, path):
#     if not path.endswith(".npz"):
#         path += ".npz"
#     os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
#     sparse.save_npz(path, inst["A"])
#     archive = dict(np.load(path, allow_pickle=False))
#     archive.update({"b": inst["b"], "c": inst["c"], "d": inst["d"],
#                     "n": np.array([inst["n"]]), "m": np.array([inst["m"]]),
#                     "p": np.array([inst["p"]])})
#     np.savez_compressed(path, **archive)
#
#
# def load_instance(path, dense=False):
#     if not path.endswith(".npz"):
#         path += ".npz"
#     arc = np.load(path, allow_pickle=False)
#     A   = sparse.csr_matrix((arc["data"], arc["indices"], arc["indptr"]),
#                              shape=tuple(arc["shape"])).astype(np.float32)
#     return {"A": A.toarray() if dense else A,
#             "b": arc["b"], "c": arc["c"], "d": arc["d"],
#             "n": int(arc["n"][0]), "m": int(arc["m"][0]), "p": int(arc["p"][0])}


# -----------------------------------------------------------------------------
# LP (CPLEX LP format) saver
# -----------------------------------------------------------------------------
def _format_coef(coef: float, is_first_term: bool) -> str:
    """
    Format a single linear term's sign and coefficient (without the variable).

    LP format requires explicit '+' / '-' separators between terms. The first
    term in an expression may be written with a leading minus but must not
    have a leading '+'.
    """
    # All coefficients produced by generate_instance are integer-valued
    # (stored as float32); use :g to keep the representation compact without
    # losing precision for values up to a few thousand.
    if coef >= 0:
        if is_first_term:
            return f"{coef:.10g} "
        return f"+ {coef:.10g} "
    else:
        if is_first_term:
            return f"-{-coef:.10g} "
        return f"- {-coef:.10g} "


def _write_sparse_row(f, col_indices, values, var_prefix: str, terms_per_line: int = 10):
    """Write a sparse linear expression as ' coef var' tokens. Returns True if any term was written."""
    wrote_any = False
    count = 0
    for col, val in zip(col_indices, values):
        coef = float(val)
        if coef == 0.0:
            continue
        f.write(_format_coef(coef, is_first_term=not wrote_any))
        f.write(f"{var_prefix}{int(col) + 1}")
        wrote_any = True
        count += 1
        if count % terms_per_line == 0:
            f.write("\n     ")
        else:
            f.write(" ")
    return wrote_any


def save_instance_lp(inst, path):
    """
    Write a generated instance to a CPLEX LP formatted file.

    Formulation written to disk:
        maximize   y_1 + y_2 + ... + y_p
        subject to
            (A x)_i <= b_i                     for i = 1..m
            (c_k x) - y_k = -d_k               for k = 1..p
            x_j in {0,1}                       for j = 1..floor(0.8*n)
            x_j >= 0                           for j = floor(0.8*n)+1..n
            y_k free                           for k = 1..p
    """
    if not path.endswith(".lp"):
        path += ".lp"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    A = inst["A"].tocsr()
    b = np.asarray(inst["b"]).reshape(-1)
    c = np.asarray(inst["c"]).reshape(inst["p"], inst["n"])
    d = np.asarray(inst["d"]).reshape(-1)
    n = int(inst["n"])
    m = int(inst["m"])
    p = int(inst["p"])

    int_cutoff = int(INTEGER_VARIABLE_FRACTION * n)

    with open(path, "w") as f:
        f.write(f"\\Problem: {os.path.basename(path)}\n")
        f.write(f"\\ n={n}, m={m}, p={p}\n")
        f.write(f"\\ integer variables: x1..x{int_cutoff} (binary)\n")
        f.write(f"\\ continuous variables: x{int_cutoff + 1}..x{n} (x >= 0)\n\n")

        # --- Objective ---
        f.write("Maximize\n")
        f.write(" obj: " + " + ".join(f"y{k + 1}" for k in range(p)) + "\n\n")

        # --- Constraints ---
        f.write("Subject To\n")

        # A x <= b
        indptr = A.indptr
        indices = A.indices
        data = A.data
        for i in range(m):
            f.write(f" c{i + 1}: ")
            row_start = indptr[i]
            row_end = indptr[i + 1]
            wrote_any = _write_sparse_row(
                f,
                indices[row_start:row_end],
                data[row_start:row_end],
                var_prefix="x",
            )
            if not wrote_any:
                # Empty row – LP format still needs a valid expression on the LHS.
                f.write("0 x1 ")
            f.write(f"<= {float(b[i]):.10g}\n")

        # Objective-image constraints: c_k x - y_k = -d_k
        for k in range(p):
            f.write(f" obj_y{k + 1}: ")
            row = c[k]
            nz = np.nonzero(row)[0]
            wrote_any = _write_sparse_row(
                f,
                nz,
                row[nz],
                var_prefix="x",
            )
            # Append the -y_k term
            f.write(("- " if wrote_any else "-") + f"y{k + 1} ")
            f.write(f"= {-float(d[k]):.10g}\n")

        # --- Bounds ---
        # Continuous x_j (j > int_cutoff) default to [0, +inf] which matches
        # the generator's intent, so we do not need to write them explicitly.
        # y variables are free.
        f.write("\nBounds\n")
        for k in range(p):
            f.write(f" y{k + 1} free\n")

        # --- Binary variables ---
        if int_cutoff > 0:
            f.write("\nBinary\n")
            per_line = 10
            for start in range(0, int_cutoff, per_line):
                chunk = " ".join(
                    f"x{j + 1}" for j in range(start, min(start + per_line, int_cutoff))
                )
                f.write(f" {chunk}\n")

        f.write("\nEnd\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate sparse MIP instances (LP format).")
    parser.add_argument("--out-dir", default="./instances", help="Output directory for .lp files")
    parser.add_argument("--num-instances", type=int, default=100, help="Number of instances to generate")
    parser.add_argument("--seed", type=int, default=10, help="Global random seed")
    parser.add_argument(
        "--p",
        type=int,
        default=3,
        help="Number of y objectives (rows in c matrix)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help=(
            "Fixed number of decision variables (overrides --random-n). "
            "The number of constraints is set to round(a * n)."
        ),
    )
    parser.add_argument(
        "--a",
        type=float,
        default=3.0,
        help="Multiplier used to compute constraints count: m = round(a * n).",
    )
    parser.add_argument(
        "--random-n",
        action="store_true",
        default=True,
        help=(
            "Pick n independently for each instance from {300, 400, ..., 1000} "
            "(paper: 'randomly from the integer set ranging from 300 to 1000, by 100'). "
            "Ignored when --n is set explicitly."
        ),
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

    # Paper: n randomly selected from {300, 400, 500, 600, 700, 800, 900, 1000}.
    # --n overrides this; otherwise --random-n (default True) applies.
    N_CHOICES = list(range(300, 1001, 100))
    use_random_n = (args.n is None)

    print(
        f"Starting generation: instances={args.num_instances}, seed={args.seed}, p={args.p}, "
        f"n={'random from {300..1000}' if use_random_n else args.n}, out_dir={out_dir}"
    )
    for i in range(args.num_instances):
        n_i = random.choice(N_CHOICES) if use_random_n else args.n
        print(f"\n[{i+1}/{args.num_instances}] Generating instance_{i+1}.lp  (n={n_i})")
        t0 = time.time()
        inst = generate_instance(
            verbose=True,
            progress_every_rows=args.progress_every_rows,
            p=args.p,
            n=n_i,
            a=args.a,
        )
        path = os.path.join(out_dir, f"instance_{i+1}.lp")
        print(f"  [save] Writing {path}")
        save_instance_lp(inst, path)
        elapsed = time.time() - t0
        mb = os.path.getsize(path) / 1024**2
        print(
            f"[{i+1}/{args.num_instances}] Done: n={inst['n']}, m={inst['m']}, p={inst['p']}, {mb:.1f} MB, {elapsed:.1f}s"
        )

    # -------------------------------------------------------------------------
    # Legacy .npz save path (kept for reference):
    # -------------------------------------------------------------------------
    # path = os.path.join(out_dir, f"instance_{i+1}.npz")
    # save_instance(inst, path)
