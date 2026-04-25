"""Generate set-packing LP instances designed to cause FP stall/cycle behaviour.

Why the repair heuristic alone is insufficient
----------------------------------------------
For SPP with b = 1 everywhere, x = 0 is always feasible.  The greedy repair
therefore *always* finds some feasible solution by removing conflicting variables.
The challenge for the RL agent is not to find *any* feasible solution but to find
a *high-quality* one (large independent set).  That requires instances where:

  1. The LP relaxation has many variables at exactly 0.5 (highly fractional), so
     rounding causes many constraint violations and FP oscillates between rounded
     solutions.
  2. The constraint graph is dense, so the maximum independent set is small and
     repair has to remove most selected variables → repair quality ≈ 10–20% of LP
     optimal, creating a large quality-gap that makes the agent's perturbation
     choices matter.
  3. The graph is symmetric enough that FP cycles without escaping unless
     perturbed intelligently.

Graph families used
-------------------
  odd_cycle        — C_n with n odd: LP gives all-0.5, stalls deterministically.
  petersen         — 10 nodes, vertex-transitive, well-studied FP stall case.
  kneser           — K(2k+1,k): high chromatic number, highly fractional LP.
  circulant        — C_n(S): tunable symmetry via offset set S.
  chained_cycles   — odd cycles sharing a vertex: multi-component stall.
  mycielski        — Mk: triangle-free, χ=k, LP always all-0.5 → maximum stall.
  random_dense     — G(n,p): p≈0.5 gives α≈2log₂(n) ≪ n/2, huge LP-integer gap.

Usage
-----
  # Generate the default hard batch
  python generate_stalling_instances.py --batch --out-dir results/stalling_instances

  # Generate a larger pool of verified-hard random instances
  python generate_stalling_instances.py --hard-batch --n-candidates 50 \
      --out-dir results/hard_instances --verify

  # Generate a single instance
  python generate_stalling_instances.py --type mycielski --param 6 --out my.lp
"""
from __future__ import annotations

import argparse
import os
import random
from itertools import combinations
from typing import Optional


# ---------------------------------------------------------------------------
# Structured graph generators
# ---------------------------------------------------------------------------

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
    edge_set: set[tuple[int, int]] = set()
    for i in range(n):
        for d in offsets:
            j = (i + d) % n
            if i != j:
                edge_set.add((min(i, j), max(i, j)))
    return list(range(n)), sorted(edge_set)


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


# ---------------------------------------------------------------------------
# New: Mycielski graph  (maximally hard for FP)
# ---------------------------------------------------------------------------

def mycielski_graph(steps: int):
    """Build the Mycielski graph M_{steps}.

    M2 = K2  (2 nodes, 1 edge)
    M3 = C5  (5 nodes, 5 edges)           χ = 3
    M4 = Grötzsch graph (11 nodes, 20 e)  χ = 4
    M5 = 23 nodes, ~71 edges              χ = 5
    M6 = 47 nodes, ~200 edges             χ = 6  ← good training size
    M7 = 95 nodes, ~600 edges             χ = 7  ← hard training size

    Key property: triangle-free at all steps, so the LP relaxation *always*
    assigns x_i = 0.5 to every variable.  This is the most fractional LP
    possible and causes FP to oscillate maximally regardless of rounding.
    """
    assert steps >= 2, "steps must be >= 2"

    # Start with K2
    adj: set[tuple[int, int]] = {(0, 1)}
    n = 2  # current number of vertices

    for _ in range(steps - 2):
        new_edges: set[tuple[int, int]] = set()
        # For each original edge (a, b):
        #   shadow u_a = n + a gets an edge to b
        #   shadow u_b = n + b gets an edge to a
        for a, b in adj:
            ua, ub = n + a, n + b
            new_edges.add((min(ua, b), max(ua, b)))
            new_edges.add((min(a, ub), max(a, ub)))
        # Universal root w = 2n connects to every shadow u_i
        w = 2 * n
        for i in range(n):
            ui = n + i
            new_edges.add((min(w, ui), max(w, ui)))
        adj = adj | new_edges
        n = 2 * n + 1

    return list(range(n)), sorted(adj)


# ---------------------------------------------------------------------------
# New: Random dense graph  (huge LP-integer gap)
# ---------------------------------------------------------------------------

def random_dense_graph(n: int, density: float, seed: int = 0):
    """Random graph G(n, p) with edge probability p = density.

    For p ≈ 0.5 and n = 100, the maximum independent set has size ≈ 2 log₂(n) ≈ 13
    while the LP optimal is n/2 = 50.  This LP-integer gap of ~74% causes FP to
    stall repeatedly: rounding the all-0.5 LP solution violates ≈ p*n*(n-1)/4
    constraints, and repair must remove most selected variables to reach feasibility.

    The agent must learn targeted perturbations to escape these high-violation
    configurations, since na\"ive strategies all converge to poor solutions.
    """
    assert n >= 4, "n must be >= 4"
    assert 0.0 < density < 1.0, "density must be in (0, 1)"
    rng = random.Random(seed)
    edges = [
        (i, j)
        for i in range(n)
        for j in range(i + 1, n)
        if rng.random() < density
    ]
    return list(range(n)), edges


# ---------------------------------------------------------------------------
# LP file builder  (supports non-uniform profits)
# ---------------------------------------------------------------------------

def build_lp(
    vertices,
    edges,
    graph_name: str,
    comment: str = "",
    profits: Optional[list[float]] = None,
) -> str:
    """Write an LP file for the maximum-weight independent set / set-packing problem.

    Args:
        profits: per-variable profit coefficients.  Defaults to all-ones (maximum
                 independent set).  Non-uniform profits break LP symmetry and are
                 useful for stress-testing the agent under asymmetric reward gradients.
    """
    n = len(vertices)
    if profits is None:
        profits = [1.0] * n
    assert len(profits) == n

    lines = [
        f"\\ Set Packing instance - {graph_name}",
        f"\\ Vertices (sets): {n}",
        f"\\ Conflict edges: {len(edges)}",
    ]
    if comment:
        for line in comment.splitlines():
            lines.append(f"\\ {line}")
    lines.extend([
        "\\ Designed to create fractional LP points and FP stall/cycle behavior.",
        "",
        "Maximize",
        "  obj: " + " + ".join(
            (f"{profits[v]} x{v + 1}" if profits[v] != 1.0 else f"x{v + 1}")
            for v in vertices
        ),
        "",
        "Subject To",
    ])
    for idx, (u, v) in enumerate(edges, start=1):
        lines.append(f"  c{idx}: x{u + 1} + x{v + 1} <= 1")
    lines.extend(["", "Bounds"])
    for v in vertices:
        lines.append(f"  0 <= x{v + 1} <= 1")
    lines.extend(["", "Binary", "  " + " ".join(f"x{v + 1}" for v in vertices), "", "End"])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Optional hardness verification (requires CPLEX via docplex)
# ---------------------------------------------------------------------------

def verify_instance_hardness(
    lp_path: str,
    max_iters: int = 60,
    time_limit: float = 30.0,
    stall_length: int = 3,
    baseline_action: int = 2,
    cplex_threads: int = 1,
    min_stalls: int = 8,
    max_quality_ratio: float = 0.75,
) -> Optional[dict]:
    """Run baseline FP on the instance and return hardness metrics.

    Returns None if CPLEX / the FP runner is unavailable.  Returns a dict with:
        lp_fractionality   — mean |x_lp - round(x_lp)| at LP optimum
        num_stalls         — baseline FP stalls before termination
        quality_ratio      — best_feasible_obj / lp_obj  (0 = worst, 1 = solved LP)
        is_hard            — True if num_stalls >= min_stalls AND
                             quality_ratio <= max_quality_ratio
        repair_obj         — objective of the solution repair found on iteration 1
        lp_obj             — LP relaxation objective
    """
    try:
        from set_packing.fp_baseline_spp import FPConfig, SPPFeasibilityPump, run_baseline_fp
        from set_packing.spp_model import (
            feasibility_metrics,
            load_spp_instance,
            objective_value,
            repair_set_packing_solution,
        )
        import numpy as np
    except ImportError:
        return None

    try:
        instance = load_spp_instance(lp_path)
    except Exception:
        return None

    cfg = FPConfig(
        max_iterations=max_iters,
        time_limit=time_limit,
        stall_length=stall_length,
        baseline_action=baseline_action,
        stop_on_repaired_incumbent=False,
        cplex_threads=cplex_threads,
        verbose=False,
    )

    runner = SPPFeasibilityPump(instance, cfg)
    runner.reset()

    if runner.done or runner.x_lp is None:
        return None   # LP solve failed → skip

    # LP metrics
    lp_obj = float(np.dot(instance.profits, runner.x_lp))
    frac = float(np.mean(np.abs(runner.x_lp - np.round(runner.x_lp))))

    # What does repair give on the very first rounded solution?
    repaired, _ = repair_set_packing_solution(instance, runner.x_binary, cfg.tolerance)
    repair_obj = float(objective_value(instance, repaired))

    # Run baseline FP to completion
    result = run_baseline_fp(instance, cfg, perturb_on_stall=True)

    quality_ratio = (result.final_objective / lp_obj) if lp_obj > 1e-9 else 0.0
    is_hard = (result.num_stalls >= min_stalls) and (quality_ratio <= max_quality_ratio)

    return {
        "lp_fractionality": frac,
        "lp_obj": lp_obj,
        "repair_obj": repair_obj,
        "repair_quality_ratio": repair_obj / lp_obj if lp_obj > 1e-9 else 0.0,
        "num_stalls": result.num_stalls,
        "quality_ratio": quality_ratio,
        "is_hard": is_hard,
        "n": instance.n,
        "m": instance.m,
    }


# ---------------------------------------------------------------------------
# Instance factory
# ---------------------------------------------------------------------------

def make_instance(
    graph_type: str,
    param: int = 7,
    offsets: Optional[list[int]] = None,
    count: int = 3,
    seed: int = 0,
    density: float = 0.50,
):
    """Build (vertices, edges, name, comment) for the requested graph type."""
    if graph_type == "odd_cycle":
        n = param if param % 2 == 1 else param + 1
        vertices, edges = odd_cycle_graph(n)
        name = f"OddCycle_C{n}"
        comment = f"Odd cycle C_{n}; LP relaxation has symmetric x_i = 1/2 behaviour."

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

    elif graph_type == "mycielski":
        steps = max(2, param)
        vertices, edges = mycielski_graph(steps)
        name = f"Mycielski_M{steps}"
        comment = (
            f"Mycielski graph M{steps}: triangle-free, chromatic number = {steps}. "
            f"LP relaxation assigns x_i = 0.5 to ALL variables → maximum FP fractionality. "
            f"Independence number ≈ n / {steps}."
        )

    elif graph_type == "random_dense":
        vertices, edges = random_dense_graph(param, density, seed)
        name = f"RandomDense_n{param}_p{int(density * 100)}_s{seed}"
        comment = (
            f"Random graph G({param}, {density:.2f}) seed={seed}. "
            f"Expected independence number ≈ {max(1, round(2 * (param ** 0.5)))} << {param // 2} (LP opt). "
            f"Large LP-integer gap forces many FP stalls."
        )

    else:
        raise ValueError(f"Unknown graph type: {graph_type!r}")

    return vertices, edges, name, comment


# ---------------------------------------------------------------------------
# Batch configurations
# ---------------------------------------------------------------------------

# Classic small instances (kept for backward compatibility)
BATCH_CONFIGS = [
    # (type,           param, offsets,  count, filename)
    ("odd_cycle",      7,     None,     1,     "odd_cycle_C7.lp"),
    ("odd_cycle",      11,    None,     1,     "odd_cycle_C11.lp"),
    ("odd_cycle",      19,    None,     1,     "odd_cycle_C19.lp"),
    ("petersen",       10,    None,     1,     "petersen.lp"),
    ("kneser",         2,     None,     1,     "kneser_K5_2.lp"),
    ("kneser",         3,     None,     1,     "kneser_K7_3.lp"),
    ("circulant",      9,     [1, 3],   1,     "circulant_C9_1_3.lp"),
    ("circulant",      13,    [1, 4],   1,     "circulant_C13_1_4.lp"),
    ("chained_cycles", 5,     None,     3,     "chained_3x_C5.lp"),
    ("chained_cycles", 7,     None,     4,     "chained_4x_C7.lp"),
]

# Hard batch: larger instances with genuine LP-integer gaps.
# Mycielski M6/M7 have LP always all-0.5 (maximally fractional).
# Random dense G(n, 0.5) have independence number ≈ 2log₂(n) << n/2.
# Large chained cycles need many perturbations to escape.
HARD_BATCH_CONFIGS = [
    # Mycielski — maximally fractional LP, triangle-free
    ("mycielski",      5,     None,     1,  0.5,  0, "mycielski_M5.lp"),
    ("mycielski",      6,     None,     1,  0.5,  0, "mycielski_M6.lp"),
    ("mycielski",      7,     None,     1,  0.5,  0, "mycielski_M7.lp"),
    # Random dense — huge LP-integer gap, unpredictable cycling
    ("random_dense",   60,    None,     1,  0.50, 0, "random_dense_n60_p50_s0.lp"),
    ("random_dense",   60,    None,     1,  0.50, 1, "random_dense_n60_p50_s1.lp"),
    ("random_dense",   80,    None,     1,  0.50, 0, "random_dense_n80_p50_s0.lp"),
    ("random_dense",   80,    None,     1,  0.50, 1, "random_dense_n80_p50_s1.lp"),
    ("random_dense",   80,    None,     1,  0.50, 2, "random_dense_n80_p50_s2.lp"),
    ("random_dense",   100,   None,     1,  0.45, 0, "random_dense_n100_p45_s0.lp"),
    ("random_dense",   100,   None,     1,  0.45, 1, "random_dense_n100_p45_s1.lp"),
    # Large chained cycles — multi-component stall
    ("chained_cycles", 7,     None,     8,  0.5,  0, "chained_8x_C7.lp"),
    ("chained_cycles", 9,     None,     6,  0.5,  0, "chained_6x_C9.lp"),
    ("chained_cycles", 11,    None,     5,  0.5,  0, "chained_5x_C11.lp"),
    # Large Kneser — high chromatic number, symmetric
    ("kneser",         4,     None,     1,  0.5,  0, "kneser_K9_4.lp"),
    # Large circulant — tunable density
    ("circulant",      31,    [1,5,11], 1,  0.5,  0, "circulant_C31.lp"),
    ("circulant",      41,    [1,6,14], 1,  0.5,  0, "circulant_C41.lp"),
]


# ---------------------------------------------------------------------------
# Batch generation
# ---------------------------------------------------------------------------

def generate_batch(out_dir: str, configs=None, verify: bool = False):
    """Generate the standard (small) batch."""
    if configs is None:
        configs = BATCH_CONFIGS
    os.makedirs(out_dir, exist_ok=True)
    print(f"Generating {len(configs)} instances → {out_dir}")
    print(f"{'File':<42} {'Vars':>6} {'Edges':>7}  Graph")
    print("-" * 72)

    for entry in configs:
        graph_type, param, offsets, count, filename = entry[:5]
        density = entry[5] if len(entry) > 5 else 0.5
        seed    = entry[6] if len(entry) > 6 else 0

        vertices, edges, name, comment = make_instance(
            graph_type, param, offsets, count, seed=seed, density=density
        )
        path = os.path.join(out_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(build_lp(vertices, edges, name, comment))

        hardness_str = ""
        if verify:
            stats = verify_instance_hardness(path)
            if stats:
                hardness_str = (
                    f"  stalls={stats['num_stalls']:>3}"
                    f"  frac={stats['lp_fractionality']:.2f}"
                    f"  qual={stats['quality_ratio']:.2f}"
                    f"  {'HARD' if stats['is_hard'] else 'easy'}"
                )
            else:
                hardness_str = "  (no CPLEX)"

        print(f"{filename:<42} {len(vertices):>6} {len(edges):>7}  {name}{hardness_str}")


def generate_hard_batch(
    out_dir: str,
    n_candidates: int = 30,
    sizes: tuple[int, ...] = (60, 80, 100),
    density: float = 0.50,
    verify: bool = True,
    min_stalls: int = 8,
    max_quality_ratio: float = 0.75,
    cplex_threads: int = 1,
):
    """Generate a pool of random dense instances, filter by hardness, save the hard ones.

    This is the recommended way to build a training pool for RL.  It generates
    `n_candidates` random instances and keeps those where baseline FP stalls at
    least `min_stalls` times and finds a solution with quality ≤ `max_quality_ratio`
    of the LP optimal.

    If CPLEX is unavailable the instances are saved without filtering (all retained).
    """
    os.makedirs(out_dir, exist_ok=True)
    print(f"Generating {n_candidates} random dense candidates across sizes {sizes}")
    if verify:
        print(f"  Hardness filter: stalls >= {min_stalls}, quality <= {max_quality_ratio}")
    print()

    saved = 0
    skipped = 0
    seed = 0

    for n in sizes:
        for _ in range(n_candidates // len(sizes) + 1):
            vertices, edges, name, comment = make_instance(
                "random_dense", n, density=density, seed=seed
            )
            filename = f"hard_n{n}_p{int(density * 100)}_s{seed}.lp"
            path = os.path.join(out_dir, filename)

            # Write temporarily to run verification
            with open(path, "w", encoding="utf-8") as f:
                f.write(build_lp(vertices, edges, name, comment))

            if verify:
                stats = verify_instance_hardness(
                    path,
                    min_stalls=min_stalls,
                    max_quality_ratio=max_quality_ratio,
                    cplex_threads=cplex_threads,
                )
                if stats is None:
                    # CPLEX unavailable — keep all
                    status = "kept (no CPLEX)"
                elif stats["is_hard"]:
                    status = (
                        f"HARD  stalls={stats['num_stalls']:>3}"
                        f"  frac={stats['lp_fractionality']:.2f}"
                        f"  repair_q={stats['repair_quality_ratio']:.2f}"
                        f"  final_q={stats['quality_ratio']:.2f}"
                    )
                    saved += 1
                else:
                    os.remove(path)   # discard easy instance
                    skipped += 1
                    status = (
                        f"easy  stalls={stats['num_stalls']:>3}"
                        f"  qual={stats['quality_ratio']:.2f}  (removed)"
                    )
            else:
                status = "kept (no verify)"
                saved += 1

            print(f"  n={n:>3}  s={seed:<4}  {filename:<38}  {status}")
            seed += 1

            if saved + skipped >= n_candidates:
                break

    print(f"\nDone: {saved} hard instances saved, {skipped} easy instances discarded.")
    print(f"Output: {os.path.abspath(out_dir)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate set-packing LP instances for FP stall tests.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--type",
        default="odd_cycle",
        choices=["odd_cycle", "petersen", "kneser", "circulant",
                 "chained_cycles", "mycielski", "random_dense"],
        help="Graph family for a single instance.",
    )
    parser.add_argument("--param",   type=int,   default=7,    help="Primary size parameter.")
    parser.add_argument("--offsets", type=int,   nargs="+",    help="Circulant offsets.")
    parser.add_argument("--count",   type=int,   default=3,    help="Number of chained cycles.")
    parser.add_argument("--density", type=float, default=0.50, help="Edge density for random_dense.")
    parser.add_argument("--seed",    type=int,   default=0,    help="RNG seed for random_dense.")
    parser.add_argument("--out",     default="instance.lp",    help="Output LP file (single mode).")

    # Batch modes
    parser.add_argument("--batch",      action="store_true",
                        help="Generate the standard small batch (BATCH_CONFIGS).")
    parser.add_argument("--hard-batch", action="store_true",
                        help="Generate and filter a pool of hard random dense instances.")
    parser.add_argument("--out-dir",    default="results/stalling_instances",
                        help="Output directory for batch modes.")

    # Hard batch options
    parser.add_argument("--n-candidates", type=int,   default=30,
                        help="Number of random candidates to generate (--hard-batch).")
    parser.add_argument("--sizes",        type=int,   nargs="+", default=[60, 80, 100],
                        help="Instance sizes for --hard-batch.")
    parser.add_argument("--verify",       action="store_true",
                        help="Run baseline FP and filter out easy instances (needs CPLEX).")
    parser.add_argument("--min-stalls",     type=int,   default=8,
                        help="Minimum FP stalls for an instance to be 'hard'.")
    parser.add_argument("--max-quality",    type=float, default=0.75,
                        help="Maximum quality ratio (feasible obj / LP obj) for 'hard'.")
    parser.add_argument("--cplex-threads",  type=int,   default=1)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.hard_batch:
        generate_hard_batch(
            out_dir=args.out_dir,
            n_candidates=args.n_candidates,
            sizes=tuple(args.sizes),
            density=args.density,
            verify=args.verify,
            min_stalls=args.min_stalls,
            max_quality_ratio=args.max_quality,
            cplex_threads=args.cplex_threads,
        )
        return

    if args.batch:
        # Generate both the classic batch and the hard batch configs
        all_configs = BATCH_CONFIGS + HARD_BATCH_CONFIGS
        generate_batch(args.out_dir, configs=all_configs, verify=args.verify)
        return

    # Single instance
    vertices, edges, name, comment = make_instance(
        args.type, args.param, args.offsets, args.count,
        seed=args.seed, density=args.density,
    )
    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(build_lp(vertices, edges, name, comment))
    print(f"Written:      {out_path}")
    print(f"Graph:        {name}")
    print(f"Vars (n):     {len(vertices)}")
    print(f"Edges (m):    {len(edges)}")
    print(f"Density:      {len(edges) / max(1, len(vertices) * (len(vertices) - 1) / 2):.3f}")

    if args.verify:
        stats = verify_instance_hardness(
            out_path,
            min_stalls=args.min_stalls,
            max_quality_ratio=args.max_quality,
            cplex_threads=args.cplex_threads,
        )
        if stats:
            print()
            print(f"LP fractionality:  {stats['lp_fractionality']:.3f}  (1 = all vars at 0.5)")
            print(f"LP objective:      {stats['lp_obj']:.2f}")
            print(f"Repair objective:  {stats['repair_obj']:.2f}  ({stats['repair_quality_ratio']:.1%} of LP)")
            print(f"FP stalls:         {stats['num_stalls']}")
            print(f"Final quality:     {stats['quality_ratio']:.1%} of LP optimal")
            print(f"Hard:              {'YES' if stats['is_hard'] else 'NO'}")
        else:
            print("\n(hardness verification skipped — CPLEX not available)")


if __name__ == "__main__":
    main()
