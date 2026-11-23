"""Recompute Tables 1-2, the Section 5.4 diagnostics and the Section 5.5 bounds from the stored designs.

Usage (from the repository root):  python python/examples/reproduce_tables.py
"""
from pathlib import Path

import numpy as np

from stageaware import FollowUpProblem, dsd, relaxation_bound, trace_bound

DATA = Path(__file__).resolve().parents[2] / "data" / "selected_designs"


def load(tag, which):
    return np.loadtxt(DATA / f"{tag}_{which}.csv", delimiter=",", skiprows=1)


print("Table 1  Baseline prediction (Psi)")
print(f"{'(m,k,n2)':<10}{'n1':>4}{'r1':>4}{'q':>3}{'Psi aware':>11}{'Psi naive':>11}{'red.%':>8}{'h aware':>9}{'h naive':>9}")
for m, k, n2 in [(6, 3, 4), (8, 4, 6), (10, 4, 4)]:
    P, tag = FollowUpProblem(dsd(m)[:, :k]), f"m{m}_k{k}_n{n2}"
    A, N = load(tag, "psi_aware"), load(tag, "Iu_naive")
    pa, pn = P.psi(A), P.psi(N)
    print(f"{str((m, k, n2)):<10}{P.n1:>4}{P.r1:>4}{P.q:>3}{pa:>11.4f}{pn:>11.4f}"
          f"{100 * (pn - pa) / pn:>8.2f}{P.confounding_index(A):>9.3f}{P.confounding_index(N):>9.3f}")

print("\nTable 2  Contrast variance C under the stage-adjusted analysis")
print(f"{'(m,k,n2)':<10}{'C,Psi-aw':>10}{'C,Iu-sel':>10}{'red.%':>7}{'C,C-aw':>9}{'C,Cu-sel':>10}{'red.%':>7}{'Psi,C-aw':>10}{'Psi,Cu':>9}")
for m, k, n2 in [(6, 3, 4), (8, 4, 6), (10, 4, 4)]:
    P, tag = FollowUpProblem(dsd(m)[:, :k]), f"m{m}_k{k}_n{n2}"
    a, b = P.contrast(load(tag, "psi_aware")), P.contrast(load(tag, "Iu_naive"))
    c, d = P.contrast(load(tag, "C_aware")), P.contrast(load(tag, "Cu_pooled"))
    print(f"{str((m, k, n2)):<10}{a:>10.4f}{b:>10.4f}{100 * (b - a) / b:>7.2f}{c:>9.4f}{d:>10.4f}"
          f"{100 * (d - c) / d:>7.2f}{P.psi(load(tag, 'C_aware')):>10.4f}{P.psi(load(tag, 'Cu_pooled')):>9.4f}")

print("\nSection 5.4  Rank-saturating case (10,4,4)")
P = FollowUpProblem(dsd(10)[:, :4])
for name in ("C_aware", "Cu_pooled"):
    s = P.summary(load("m10_k4_n4", name))
    print(f"  {name:<10} |Delta|={s['norm_Delta']:.3f}  min eig S={s['min_eig_S']:.3f}  "
          f"Delta'S^-1Delta={s['leverage']:.3f}  C-C_u={s['pooled_gap']:.3f}")

print("\nSection 5.5  Lower bounds for (6,3,4)")
P = FollowUpProblem(dsd(6)[:, :3])
r = relaxation_bound(P, 4)
print(f"  best found Psi = {P.psi(load('m6_k3_n4', 'psi_aware')):.4f}")
print(f"  Proposition 5  = {trace_bound(P, 4):.4f}")
print(f"  Theorem 2      : Psi(pi) = {r['relaxed_value']:.4f}, cube bound = {r['cube_lower_bound']:.4f}")
