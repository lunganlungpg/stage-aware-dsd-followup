"""Lower bounds on the optimal exact Psi at fixed n2 (Section 4.2)."""
from __future__ import annotations

import itertools
import math

import numpy as np
from scipy.optimize import minimize_scalar

from .criteria import FollowUpProblem, model_exponents, z_matrix


def trace_bound(problem: FollowUpProblem, n2: int) -> float:
    """Proposition 5: Psi >= {tr(Q^1/2)}^2 / {tr(S1) + n2 (t - 3k/4)}."""
    ev = np.clip(np.linalg.eigvalsh(problem.Q), 0, None)
    return float(np.sqrt(ev).sum() ** 2 / (np.trace(problem.S1) + n2 * (problem.t - 0.75 * problem.k)))


def _sensitivity_polynomial(mu_pi, W, k):
    """Coefficients of d_pi(x) = (z(x) - E_pi z)' W (z(x) - E_pi z) as {exponent: coeff}."""
    ex = model_exponents(k)
    zero = (0,) * k
    polys = [[(e, 1.0), (zero, -float(mu_pi[j]))] for j, e in enumerate(ex)]
    co: dict = {}
    for i in range(len(ex)):
        for j in range(len(ex)):
            for ei, ci in polys[i]:
                for ej, cj in polys[j]:
                    e = tuple(a + b for a, b in zip(ei, ej))
                    co[e] = co.get(e, 0.0) + ci * cj * W[i, j]
    return co


def bernstein_upper(poly: dict, k: int, boxes: int = 6, degree: int = 4) -> float:
    """Upper bound on a polynomial over [-1,1]^k from tensor Bernstein coefficients on subboxes.

    Floating-point only: this is a numerical bound, not an interval-certified one.
    """
    idx = list(itertools.product(range(degree + 1), repeat=k))
    mx = -np.inf
    for sel in itertools.product(range(boxes), repeat=k):
        lims = [(-1 + 2 * j / boxes, -1 + 2 * (j + 1) / boxes) for j in sel]
        trans: dict = {}
        for powers, coeff in poly.items():
            for expand in itertools.product(*[range(v + 1) for v in powers]):
                c = coeff
                for j, (m, e) in enumerate(zip(powers, expand)):
                    a, b = lims[j]
                    c *= math.comb(m, e) * a ** (m - e) * (b - a) ** e
                trans[expand] = trans.get(expand, 0.0) + c
        for ind in idx:
            val = 0.0
            for powers, c in trans.items():
                if all(j <= i for j, i in zip(powers, ind)):
                    fac = 1.0
                    for j, i in zip(powers, ind):
                        fac *= math.comb(i, j) / math.comb(degree, j)
                    val += fac * c
            mx = max(mx, val)
    return mx + 1e-9


def relaxation_bound(problem: FollowUpProblem, n2: int, grid_n: int = 11,
                     iterations: int = 1600, boxes: int = 6) -> dict:
    """Theorem 2: Frank-Wolfe on a grid relaxation, then bound sup d_pi over the cube.

    Returns the relaxed value Psi(pi), the grid-only bound and the continuous-cube
    bound Psi(pi) - n2 {sup d_pi - E_pi d_pi}, using a Bernstein envelope for the sup.
    """
    k = problem.k
    grid = np.array(list(itertools.product(np.linspace(-1, 1, grid_n), repeat=k)))
    Z = z_matrix(grid)
    S1, Q = problem.S1, problem.Q
    w = np.ones(len(Z)) / len(Z)

    def stats(weights):
        m = weights @ Z
        S = S1 + n2 * ((Z.T * weights) @ Z - np.outer(m, m))
        Si = np.linalg.inv(S)
        W = Si @ Q @ Si
        cen = Z - m
        return float(np.trace(Si @ Q)), m, W, np.einsum("ij,jk,ik->i", cen, W, cen)

    obj, m, W, sens = stats(w)
    A = Z.T @ (w[:, None] * Z)
    for it in range(iterations):
        j = int(np.argmax(sens))
        if n2 * (sens[j] - w @ sens) < 1e-7:
            break
        zj = Z[j]

        def line(g):
            mean = (1 - g) * m + g * zj
            Snew = S1 + n2 * ((1 - g) * A + g * np.outer(zj, zj) - np.outer(mean, mean))
            try:
                return float(np.trace(np.linalg.solve(Snew, Q)))
            except np.linalg.LinAlgError:
                return np.inf

        opt = minimize_scalar(line, bounds=(0, 1), method="bounded", options={"xatol": 1e-8})
        if opt.fun >= obj - 1e-13:
            break
        g = float(opt.x)
        w *= 1 - g
        w[j] += g
        A = Z.T @ (w[:, None] * Z)
        obj, m, W, sens = stats(w)

    sup_cube = bernstein_upper(_sensitivity_polynomial(m, W, k), k, boxes=boxes)
    return dict(
        relaxed_value=obj,
        grid_lower_bound=max(0.0, obj - n2 * (sens.max() - w @ sens)),
        cube_lower_bound=max(0.0, obj - n2 * (sup_cube - w @ sens)),
        sup_d_upper=sup_cube, iterations=it + 1,
        support=grid[w > 1e-10], weights=w[w > 1e-10],
    )
