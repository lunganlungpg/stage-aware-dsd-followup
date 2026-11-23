"""Coordinate exchange with the centred rank-two update (Section 4.1, Proposition 4)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

from .criteria import FollowUpProblem, _spd_inverse, z_matrix

def _update_factors(a, b, zbar2, n2):
    U = np.column_stack([a - zbar2, b - a])
    K = np.array([[0.0, 1.0], [1.0, 1.0 - 1.0 / n2]])
    return U, K


def rank_two_update(S, a, b, zbar2, n2):
    """Proposition 4: S_new = S + U K U' after a Stage 2 regressor row changes from a to b.

    u = a - zbar2 and v = b - a, with zbar2 the Stage 2 mean before the change.
    """
    U, K = _update_factors(a, b, zbar2, n2)
    return S + U @ K @ U.T, U, K


def updated_value(value, Sinv, W, U, K):
    """Eq. (12): Psi_new = Psi - tr{(K^-1 + U'S^-1U)^-1 U'WU}.

    Returns inf when the update would make S singular or indefinite. Because K
    has one positive and one negative eigenvalue, S_new is positive definite
    exactly when det(K^-1 + U'S^-1U) < 0.
    """
    B = np.linalg.inv(K) + U.T @ Sinv @ U
    if not np.all(np.isfinite(B)) or np.linalg.det(B) >= -1e-12:
        return np.inf
    try:
        out = value - float(np.trace(np.linalg.solve(B, U.T @ W @ U)))
    except np.linalg.LinAlgError:
        return np.inf
    return out if out > 0 else np.inf


@dataclass
class ExchangeResult:
    X2: np.ndarray            # best terminal design under the search criterion
    value: float              # its criterion value
    terminal: list            # all terminal designs, one per start
    values: np.ndarray        # criterion value of each terminal design
    sweeps: np.ndarray        # sweeps used by each start
    converged: np.ndarray     # True if the start met the tolerance before max_sweeps


def _criterion(problem: FollowUpProblem, name: str):
    return {"psi": problem.psi, "contrast": problem.contrast,
            "unblocked": problem.unblocked_I, "pooled_contrast": problem.pooled_contrast}[name]


def coordinate_exchange(problem: FollowUpProblem, n2: int, criterion: str = "psi",
                        n_starts: int = 20, seed: int | None = None, grid: int = 9,
                        brackets: int = 3, max_sweeps: int = 25, tol: float = 1e-9,
                        rng: np.random.Generator | None = None) -> ExchangeResult:
    """Minimise a criterion over n2 follow-up runs in [-1, 1]^k.

    criterion: "psi" (stage-aware baseline), "contrast" (stage-aware C),
    "unblocked" (I_u, stage-naive) or "pooled_contrast" (C_u).
    The two stage-aware criteria use the rank-two update of Proposition 4;
    the others are evaluated directly. Every accepted move is re-evaluated
    directly, as in the paper.
    """
    if n2 < 2:
        raise ValueError("n2 >= 2 is required: a single follow-up run adds no information")
    rng = rng if rng is not None else np.random.default_rng(seed)
    f = _criterion(problem, criterion)
    centred = criterion in ("psi", "contrast")
    target = problem.Q if criterion == "psi" else problem.V
    scale = 1.0 if criterion == "psi" else 2.0
    k, gridpts = problem.k, np.linspace(-1, 1, grid)

    terminal, values, sweeps_used, conv = [], [], [], []
    for _ in range(n_starts):
        for _try in range(1000):
            X2 = rng.uniform(-1, 1, (n2, k))
            val = f(X2)
            if np.isfinite(val):
                break
        else:
            raise RuntimeError("could not find an estimable random start")
        converged, sweep = False, 0
        for sweep in range(1, max_sweeps + 1):
            before = val
            for i in range(n2):
                for j in range(k):
                    if centred:
                        Z2 = z_matrix(X2)
                        S = problem.S1 + (Z2 - Z2.mean(0)).T @ (Z2 - Z2.mean(0))
                        Sinv = _spd_inverse(S)
                        if Sinv is None:
                            continue
                        W = Sinv @ target @ Sinv
                        a, zbar2 = Z2[i], Z2.mean(0)

                        def evaluate(c, i=i, j=j, a=a, zbar2=zbar2, Sinv=Sinv, W=W):
                            x = X2[i].copy(); x[j] = c
                            b = z_matrix(x)[0]
                            U, K = _update_factors(a, b, zbar2, n2)
                            return scale * updated_value(val / scale, Sinv, W, U, K)
                    else:
                        def evaluate(c, i=i, j=j):
                            Y = X2.copy(); Y[i, j] = c
                            return f(Y)
                    vals = np.array([evaluate(c) for c in gridpts])
                    cands = [(val, X2[i, j])]
                    cands += [(float(v), float(c)) for c, v in zip(gridpts, vals) if np.isfinite(v)]
                    order = np.argsort(np.minimum(vals[:-1], vals[1:]))
                    for b_ in order[:brackets]:
                        res = minimize_scalar(evaluate, bounds=(gridpts[b_], gridpts[b_ + 1]),
                                              method="bounded", options={"xatol": 1e-6})
                        if np.isfinite(res.fun):
                            cands.append((float(res.fun), float(res.x)))
                    best_v, best_c = min(cands)
                    if best_v < val - tol:
                        trial = X2.copy(); trial[i, j] = best_c
                        checked = f(trial)          # direct re-evaluation
                        if checked < val - tol:
                            X2, val = trial, checked
            if before - val < tol:
                converged = True
                break
        terminal.append(X2); values.append(val); sweeps_used.append(sweep); conv.append(converged)
    values = np.asarray(values)
    b = int(np.argmin(values))
    return ExchangeResult(terminal[b], float(values[b]), terminal, values,
                          np.asarray(sweeps_used), np.asarray(conv))
