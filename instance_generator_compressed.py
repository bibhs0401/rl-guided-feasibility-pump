import argparse
import os
import random
import numpy as np

random.seed(10)
np.random.seed(10)


def rowcolumn():
    m, n, p = [], [], []
    n = [1500]  # number of decision variables (x) (j)
    m = [3000]  # number of constraints (i)
    p.append(3)  # number of objectives
    return m, n, p


def data(m, n, p):
    b, c, d, BigM = [], [], [], []
    prob = np.random.uniform(0.01, 0.03)
    A = np.random.choice([0, 1], size=m[0] * n[0], p=[1 - prob, prob])

    A_mat = np.zeros((m[0], n[0])).astype(np.int32)  # decision variables coefficient matrix
    k = 0
    for i in range(m[0]):
        for j in range(n[0]):
            A_mat[i][j] = int(A[k])
            k += 1

    for i in range(m[0]):
        count = 0
        for items in A_mat[i]:
            if items == 0:
                count += 1
        if count == n[0]:
            a = np.random.randint(0, n[0])
            A_mat[i][a] = 1

    for _ in range(n[0]):
        c.append(1)
    weight = [1 / 100 for _ in range(1, 101)]
    for _ in range(p[0] - 1):
        for _ in range(n[0]):
            z = random.choices([i for i in range(1, 101)], weights=weight, k=1)
            c.append(z[0])

    c_mat = np.zeros((p[0], n[0])).astype(np.int32)
    k = 0
    for i in range(p[0]):
        for j in range(n[0]):
            c_mat[i][j] = int(c[k])
            k += 1

    for _ in range(m[0]):
        b.append(1)

    for _ in range(p[0]):
        d.append(0)

    sorted_c_mat = c_mat
    for i in range(p[0]):
        sorted_c_mat[i] = sorted(c_mat[i], reverse=True)
        M = 0
        for j in range(n[0]):
            M += sorted_c_mat[i][j]
        BigM.append(M)

    return A_mat, b, c_mat, d, BigM


def write_compressed(idx: int, out_dir: str):
    m, n, p = rowcolumn()
    A_mat, b, c_mat, d, BigM = data(m, n, p)
    out_path = os.path.join(out_dir, f"matrices{idx}.npz")
    np.savez_compressed(out_path, A=A_mat, b=b, c=c_mat, d=d, BigM=BigM, m=m, n=n, p=p)
    return A_mat, out_path


def write_set_packing_lp(A_mat: np.ndarray, out_path: str):
    """LP export layer only; generation logic remains unchanged."""
    m, n = A_mat.shape
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"\\ Set Packing from legacy generator: m={m}, n={n}\n")
        f.write("Maximize\n")
        f.write(" obj: " + " + ".join(f"x{j + 1}" for j in range(n)) + "\n\n")
        f.write("Subject To\n")
        for i in range(m):
            nz_cols = np.where(A_mat[i] == 1)[0]
            lhs = " + ".join(f"x{j + 1}" for j in nz_cols)
            f.write(f" c{i + 1}: {lhs} <= 1\n")
        f.write("\nBinary\n")
        per_line = 12
        for start in range(0, n, per_line):
            chunk = " ".join(f"x{j + 1}" for j in range(start, min(start + per_line, n)))
            f.write(f" {chunk}\n")
        f.write("\nEnd\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Legacy generator + optional set-packing LP export."
    )
    parser.add_argument("--num-instances", type=int, default=10, help="Number of instances")
    parser.add_argument("--out-dir", type=str, default=".", help="Output directory")
    parser.add_argument("--seed", type=int, default=10, help="Random seed")
    parser.add_argument("--write-npz", action="store_true", help="Also write legacy .npz files")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)

    for j in range(1, args.num_instances + 1):
        A_mat, npz_path = write_compressed(j, args.out_dir)
        lp_path = os.path.join(args.out_dir, f"set_packing_{j}.lp")
        write_set_packing_lp(A_mat, lp_path)
        if not args.write_npz and os.path.exists(npz_path):
            os.remove(npz_path)
        print(f"[{j}/{args.num_instances}] wrote {lp_path}")


if __name__ == "__main__":
    main()
