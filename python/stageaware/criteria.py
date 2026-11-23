"""Stage-adjusted design criteria for follow-up runs (Section 2 of the paper).

Model (1): Y1 = F1 beta + e1,  Y2 = F2 beta + 1 delta + e2,  with an
unrestricted fixed stage shift delta. Prediction is averaged over the
uniform measure on [-1, 1]^k. Regressor order in z(x): k linear terms,
k pure quadratic terms, then k(k-1)/2 two-factor interactions.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

_SING_TOL = 1e-9


# --------------------------------------------------------------------------
# Model terms and region moments
# --------------------------------------------------------------------------
def model_exponents(k: int) -> list[tuple[int, ...]]:
    """Exponent vectors of z(x) (intercept excluded) in the paper's order."""
    lin = [tuple(int(h == j) for h in range(k)) for j in range(k)]
    sq = [tuple(2 * int(h == j) for h in range(k)) for j in range(k)]
    inter = [tuple(int(h in (i, j)) for h in range(k))
             for i, j in itertools.combinations(range(k), 2)]
    return lin + sq + inter


def term_names(k: int) -> list[str]:
    return ([f"x{j+1}" for j in range(k)] + [f"x{j+1}^2" for j in range(k)]
            + [f"x{i+1}:x{j+1}" for i, j in itertools.combinations(range(k), 2)])


def z_matrix(X) -> np.ndarray:
    """Non-intercept regressors z(x) for each row of X (n x k)."""
    X = np.atleast_2d(np.asarray(X, float))
    k = X.shape[1]
    cols = ([X[:, i] for i in range(k)] + [X[:, i] ** 2 for i in range(k)]
            + [X[:, i] * X[:, j] for i, j in itertools.combinations(range(k), 2)])
    return np.column_stack(cols)


def f_matrix(X) -> np.ndarray:
    """Full model matrix [1, z(x)]."""
    Z = z_matrix(X)
    return np.column_stack([np.ones(len(Z)), Z])


def _cube_moment(e) -> float:
    out = 1.0
    for a in e:
        if a % 2:
            return 0.0
        out /= a + 1
    return out


def cube_moments(k: int):
    """Exact (mu_z, V, Omega) for the uniform measure on [-1, 1]^k (eq. 2)."""
    ex = [(0,) * k] + model_exponents(k)
    P = len(ex)
    Om = np.empty((P, P))
    for a in range(P):
        for b in range(P):
            Om[a, b] = _cube_moment([x + y for x, y in zip(ex[a], ex[b])])
    mu = Om[0, 1:].copy()
    V = Om[1:, 1:] - np.outer(mu, mu)
    return mu, V, Om


def _centred_scatter(Z: np.ndarray) -> np.ndarray:
    if len(Z) == 0:
        return np.zeros((Z.shape[1], Z.shape[1]))
    D = Z - Z.mean(axis=0)
    return D.T @ D


def _spd_inverse(A: np.ndarray):
    """Inverse of a symmetric matrix if numerically positive definite, else None."""
    A = (A + A.T) / 2
    ev = np.linalg.eigvalsh(A)
    if ev[0] <= _SING_TOL * max(1.0, ev[-1]):
        return None
    return np.linalg.inv(A)


def matrix_rank(A, tol: float = 1e-8) -> int:
    return int(np.linalg.matrix_rank(np.asarray(A, float), tol=tol))


# --------------------------------------------------------------------------
# Problem object: Stage 1 fixed, Stage 2 varies
# --------------------------------------------------------------------------
@dataclass
class FollowUpProblem:
    """Stage 1 design X1 (n1 x k, coded) is fixed; criteria are functions of X2."""

    X1: np.ndarray
    k: int = field(init=False)
    n1: int = field(init=False)
    p: int = field(init=False)
    t: int = field(init=False)
    r1: int = field(init=False)
    q: int = field(init=False)

    def __post_init__(self):
        self.X1 = np.atleast_2d(np.asarray(self.X1, float))
        self.n1, self.k = self.X1.shape
        self.p = (self.k + 1) * (self.k + 2) // 2
        self.t = self.p - 1
        self.F1 = f_matrix(self.X1)
        self.Z1 = self.F1[:, 1:]
        self.zbar1 = self.Z1.mean(axis=0)
        self.S1 = _centred_scatter(self.Z1)
        self.mu, self.V, self.Omega = cube_moments(self.k)
        d = self.mu - self.zbar1
        self.Q = self.V + np.outer(d, d)
        self.r1 = matrix_rank(self.F1)
        self.q = self.p - self.r1

    # ---- building blocks ------------------------------------------------
    def _X2(self, X2) -> np.ndarray:
        if X2 is None:
            return np.zeros((0, self.k))
        X2 = np.atleast_2d(np.asarray(X2, float))
        if X2.size == 0:
            return np.zeros((0, self.k))
        if X2.shape[1] != self.k:
            raise ValueError(f"X2 must have {self.k} columns")
        return X2

    def S(self, X2) -> np.ndarray:
        """Pooled within-stage scatter S = S1 + S2 (eq. 3)."""
        X2 = self._X2(X2)
        if len(X2) == 0:
            return self.S1.copy()
        return self.S1 + _centred_scatter(z_matrix(X2))

    def M_adjusted(self, X2) -> np.ndarray:
        """M_a = F1'F1 + F2'H2F2: information for beta after profiling delta."""
        X2 = self._X2(X2)
        M = self.F1.T @ self.F1
        if len(X2):
            F2 = f_matrix(X2)
            F2c = F2 - F2.mean(axis=0)
            M = M + F2c.T @ F2c
        return M

    def M_unblocked(self, X2) -> np.ndarray:
        X2 = self._X2(X2)
        M = self.F1.T @ self.F1
        if len(X2):
            F2 = f_matrix(X2)
            M = M + F2.T @ F2
        return M

    def rank_adjusted(self, X2) -> int:
        return matrix_rank(self.M_adjusted(X2))

    def residual_df(self, X2) -> int:
        """Residual degrees of freedom of the stage-adjusted fit (Section 3)."""
        n2 = len(self._X2(X2))
        return self.n1 + n2 - self.rank_adjusted(X2) - int(n2 > 0)

    # ---- Proposition 1 ----------------------------------------------------
    def psi(self, X2) -> float:
        """Excess integrated variance Psi = tr(S^-1 Q); Phi_I = 1/n1 + Psi (eq. 5)."""
        Si = _spd_inverse(self.S(X2))
        return np.inf if Si is None else float(np.trace(Si @ self.Q))

    def phi_I(self, X2) -> float:
        return 1.0 / self.n1 + self.psi(X2)

    def contrast(self, X2) -> float:
        """Integrated pairwise-difference variance C = 2 tr(S^-1 V) (eq. 6)."""
        Si = _spd_inverse(self.S(X2))
        return np.inf if Si is None else float(2 * np.trace(Si @ self.V))

    def prediction_variance(self, X2, x) -> np.ndarray:
        """Var{mu1_hat(x)}/sigma^2 = 1/n1 + (z(x)-zbar1)' S^-1 (z(x)-zbar1) (eq. 4)."""
        Si = _spd_inverse(self.S(X2))
        D = z_matrix(x) - self.zbar1
        if Si is None:
            return np.full(len(D), np.inf)
        return 1.0 / self.n1 + np.einsum("ij,jk,ik->i", D, Si, D)

    # ---- Propositions 2 and 3 ------------------------------------------------
    def unblocked_I(self, X2) -> float:
        """I_u = tr(M_u^-1 Omega), the criterion that ignores the stage effect."""
        Mi = _spd_inverse(self.M_unblocked(X2))
        return np.inf if Mi is None else float(np.trace(Mi @ self.Omega))

    def confounding_index(self, X2) -> float:
        """Stage-confounding index h = n2 f2bar' M_u^-1 f2bar (eq. 7)."""
        X2 = self._X2(X2)
        if not len(X2):
            return 0.0
        Mi = _spd_inverse(self.M_unblocked(X2))
        if Mi is None:
            return np.nan
        fb = f_matrix(X2).mean(axis=0)
        return float(len(X2) * fb @ Mi @ fb)

    def stage_cost(self, X2) -> float:
        """Right-hand side of (8): tr(M_a^-1 Omega) - tr(M_u^-1 Omega)."""
        X2 = self._X2(X2)
        Mi = _spd_inverse(self.M_unblocked(X2))
        if Mi is None:
            return np.nan
        h = self.confounding_index(X2)
        if h >= 1 - 1e-12:
            return np.inf
        z = Mi @ f_matrix(X2).mean(axis=0)
        return float(len(X2) * z @ self.Omega @ z / (1 - h))

    def displacement(self, X2) -> np.ndarray:
        """Delta = zbar2 - zbar1."""
        return z_matrix(self._X2(X2)).mean(axis=0) - self.zbar1

    def S_pool(self, X2) -> np.ndarray:
        """Non-intercept information under a single common intercept (total centred scatter)."""
        X2 = self._X2(X2)
        return _centred_scatter(np.vstack([self.Z1, z_matrix(X2)]))

    def pooled_contrast(self, X2) -> float:
        """C_u = 2 tr(S_pool^-1 V)."""
        Si = _spd_inverse(self.S_pool(X2))
        return np.inf if Si is None else float(2 * np.trace(Si @ self.V))

    def leverage(self, X2) -> float:
        """Directional leverage Delta' S^-1 Delta."""
        Si = _spd_inverse(self.S(X2))
        d = self.displacement(X2)
        return np.inf if Si is None else float(d @ Si @ d)

    def pooled_gap(self, X2) -> float:
        """Right-hand side of identity (10): C - C_u."""
        X2 = self._X2(X2)
        Si = _spd_inverse(self.S(X2))
        if Si is None:
            return np.inf
        n2 = len(X2)
        a = self.n1 * n2 / (self.n1 + n2)
        d = self.displacement(X2)
        return float(2 * a * (d @ Si @ self.V @ Si @ d) / (1 + a * d @ Si @ d))

    # ---- convenience ------------------------------------------------------
    def summary(self, X2) -> dict:
        X2 = self._X2(X2)
        S = self.S(X2)
        return dict(
            n1=self.n1, n2=len(X2), k=self.k, p=self.p, r1=self.r1, q=self.q,
            rank_Ma=self.rank_adjusted(X2), residual_df=self.residual_df(X2),
            Psi=self.psi(X2), Phi_I=self.phi_I(X2), C=self.contrast(X2),
            C_u=self.pooled_contrast(X2), I_u=self.unblocked_I(X2),
            h=self.confounding_index(X2), norm_Delta=float(np.linalg.norm(self.displacement(X2))),
            leverage=self.leverage(X2), pooled_gap=self.pooled_gap(X2),
            min_eig_S=float(np.linalg.eigvalsh(S)[0]),
        )


def minimum_followup_runs(problem: FollowUpProblem) -> int:
    """Theorem 1: the smallest useful n2 that can restore full rank (0 when q = 0)."""
    return 0 if problem.q == 0 else problem.q + 1


def useful_run_counts(problem: FollowUpProblem, n2_max: int) -> list[int]:
    """Section 3: n2 = 0 (only if q = 0) and n2 in {max(2, q+1), ..., n2_max}."""
    out = [0] if problem.q == 0 else []
    return out + list(range(max(2, problem.q + 1), n2_max + 1))
