from __future__ import annotations

import argparse
import os
from itertools import combinations


def odd_cycle_graph(n: int):
    assert n >= 3 and n % 2 == 1, "n must be an odd integer >= 3"
    edges = [(i, (i + 1) % n) for i in range(n)]
    return list(range(n)), edges


def petersen_graph():
    vertices = list(range(10))
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
    ]
    return vertices, edges


def kneser_graph(k: int):
    assert k >= 2, "k must be >= 2"
    ground = list(range(2 * k + 1))
    vertices = list(combinations(ground, k))
    edges = []
    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            if not set(vertices[i]) & set(vertices[j]):
                edges.append((i, j))
    return list(range(len(vertices))), edges


def circulant_graph(n: int, offsets: list[int]):
    assert n >= 3, "n must be >= 3"
    vertices = list(range(n))
    edge_set = set()
    for i in range(n):
        for d in offsets:
            j = (i + d) % n
            if i != j:
                edge_set.add((min(i, j), max(i, j)))
    return vertices, sorted(edge_set)


def chained_cycles_graph(cycle_size: int, num_cycles: int):
    assert cycle_size >= 3 and cycle_size % 2 == 1
    assert num_cycles >= 1
    vertices = [0]
    edges = []
    node_id = 1
    shared = 0
    for _ in range(num_cycles):
        cycle_nodes = [shared]
        for _ in range(cycle_size - 1):
            vertices.append(node_id)
            cycle_nodes.append(node_id)
            node_id += 1
        for i in range(cycle_size):
            u = cycle_nodes[i]
            v = cycle_nodes[(i + 1) % cycle_size]
            edge = (min(u, v), max(u, v))
            if edge not in edges:
                edges.append(edge)
        shared = cycle_nodes[-1]
    return vertices, edges


def build_lp(vertices, edges, graph_name: str, comment: str = "") -> str:
    n = len(vertices)
    lines = [
        f"\\ Set Packing instance - {graph_name}",
        f"\\ Vertices (sets): {n}",
        f"\\ Conflict edges: {len(edges)}",
    ]
    if comment:
        for line in comment.splitlines():
            lines.append(f"\\ {line}")
    lines.extend(
        [
            "\\ Designed to create fractional LP points and FP stall/cycle behavior.",
            "",
            "Maximize",
            "  obj: " + " + ".join(f"x{v + 1}" for v in vertices),
            "",
            "Subject To",
        ]
    )
    for idx, (u, v) in enumerate(edges, start=1):
        lines.append(f"  c{idx}: x{u + 1} + x{v + 1} <= 1")
    lines.extend(["", "Bounds"])
    for v in vertices:
        lines.append(f"  0 <= x{v + 1} <= 1")
    lines.extend(["", "Binary", "  " + " ".join(f"x{v + 1}" for v in vertices), "", "End"])
    return "\n".join(lines)


def make_instance(graph_type: str, param: int = 7, offsets: list[int] | None = None, count: int = 3):
    if graph_type == "odd_cycle":
        n = param if param % 2 == 1 else param + 1
        vertices, edges = odd_cycle_graph(n)
        name = f"OddCycle_C{n}"
        comment = f"Odd cycle C_{n}; LP relaxation has symmetric x_i = 1/2 behavior."
    elif graph_type == "petersen":
        vertices, edges = petersen_graph()
        name = "Petersen"
        comment = "Petersen graph; vertex-transitive, 10 nodes, 15 conflict edges."
    elif graph_type == "kneser":
        k = max(2, param)
        vertices, edges = kneser_graph(k)
        name = f"Kneser_K{2 * k + 1}_{k}"
        comment = f"Kneser graph K({2 * k + 1},{k}); high-symmetry conflict graph."
    elif graph_type == "circulant":
        n = param if param % 2 == 1 else param + 1
        offs = offsets if offsets else [1, max(1, (n - 1) // 3)]
        vertices, edges = circulant_graph(n, offs)
        name = f"Circulant_C{n}_S{'_'.join(map(str, offs))}"
        comment = f"Circulant graph C_{n} with offsets {offs}."
    elif graph_type == "chained_cycles":
        cycle_size = param if param % 2 == 1 else param + 1
        vertices, edges = chained_cycles_graph(cycle_size, count)
        name = f"ChainedCycles_{count}x_C{cycle_size}"
        comment = f"{count} chained odd cycles of size {cycle_size}."
    else:
        raise ValueError(f"Unknown graph type: {graph_type!r}")
    return vertices, edges, name, comment


BATCH_CONFIGS = [
    ("odd_cycle", 7, None, 1, "odd_cycle_C7.lp"),
    ("odd_cycle", 11, None, 1, "odd_cycle_C11.lp"),
    ("odd_cycle", 19, None, 1, "odd_cycle_C19.lp"),
    ("petersen", 10, None, 1, "petersen.lp"),
    ("kneser", 2, None, 1, "kneser_K5_2.lp"),
    ("kneser", 3, None, 1, "kneser_K7_3.lp"),
    ("circulant", 9, [1, 3], 1, "circulant_C9_1_3.lp"),
    ("circulant", 13, [1, 4], 1, "circulant_C13_1_4.lp"),
    ("chained_cycles", 5, None, 3, "chained_3x_C5.lp"),
    ("chained_cycles", 7, None, 4, "chained_4x_C7.lp"),
]


def generate_batch(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    print(f"Generating {len(BATCH_CONFIGS)} instances into {out_dir}")
    print(f"{'File':<35} {'Vars':>6} {'Constrs':>8} Graph")
    print("-" * 65)
    for graph_type, param, offsets, count, filename in BATCH_CONFIGS:
        vertices, edges, name, comment = make_instance(graph_type, param, offsets, count)
        path = os.path.join(out_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(build_lp(vertices, edges, name, comment))
        print(f"{filename:<35} {len(vertices):>6} {len(edges):>8} {name}")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate set-packing LP instances for FP stall tests.")
    parser.add_argument(
        "--type",
        default="odd_cycle",
        choices=["odd_cycle", "petersen", "kneser", "circulant", "chained_cycles"],
    )
    parser.add_argument("--param", type=int, default=7)
    parser.add_argument("--offsets", type=int, nargs="+", default=None)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--out", default="instance.lp")
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--out-dir", default="results/stalling_instances")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch:
        generate_batch(args.out_dir)
        return
    vertices, edges, name, comment = make_instance(args.type, args.param, args.offsets, args.count)
    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(build_lp(vertices, edges, name, comment))
    print(f"Written: {out_path}")
    print(f"Graph: {name}")
    print(f"Vars: {len(vertices)}")
    print(f"Constraints: {len(edges)}")


if __name__ == "__main__":
    main()
