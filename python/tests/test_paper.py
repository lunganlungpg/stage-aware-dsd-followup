"""Reproduce the numbers reported in the paper from the stored designs and check every identity."""
import itertools
from pathlib import Path

import numpy as np
import pytest

from stageaware import (FollowUpProblem, coordinate_exchange, cube_moments, dsd, f_matrix,
                        rank_two_update, relaxation_bound, trace_bound, updated_value,
                        z_matrix)
from stageaware.criteria import _spd_inverse

DATA = Path(__file__).resolve().parents[2] / "data" / "selected_designs"
SCEN = [(6, 3, 4), (8, 4, 6), (10, 4, 4)]


def load(m, k, n2, which):
    return np.loadtxt(DATA / f"m{m}_k{k}_n{n2}_{which}.csv", delimiter=",", skiprows=1)


def prob(m, k):
    return FollowUpProblem(dsd(m)[:, :k])


# ---------------------------------------------------------------- Table 1
TABLE1 = {  # n1, r1, q, Psi aware, Psi naive, h aware, h naive
    (6, 3, 4): (13, 10, 0, 0.3566, 0.5518, 0.494, 0.686),
    (8, 4, 6): (17, 13, 2, 0.4538, 0.5326, 0.476, 0.672),
    (10, 4, 4): (21, 12, 3, 0.7049, 0.7246, 0.460, 0.638),
}


@pytest.mark.parametrize("scen", SCEN)
def test_table1(scen):
    m, k, n2 = scen
    P = prob(m, k)
    n1, r1, q, pa, pn, ha, hn = TABLE1[scen]
    assert (P.n1, P.r1, P.q) == (n1, r1, q)
    A, N = load(m, k, n2, "psi_aware"), load(m, k, n2, "Iu_naive")
    assert P.psi(A) == pytest.approx(pa, abs=5e-5)
    assert P.psi(N) == pytest.approx(pn, abs=5e-5)
    assert P.confounding_index(A) == pytest.approx(ha, abs=5e-4)
    assert P.confounding_index(N) == pytest.approx(hn, abs=5e-4)


# ---------------------------------------------------------------- Table 2
TABLE2 = {  # C psi-aware, C Iu-selected, C C-aware, C Cu-selected, Psi C-aware, Psi Cu-selected
    (6, 3, 4): (0.5161, 0.5589, 0.4501, 0.4518, 0.4193, 0.4220),
    (8, 4, 6): (0.5698, 0.5832, 0.5201, 0.5282, 0.5721, 0.5973),
    (10, 4, 4): (0.6984, 0.7378, 0.6984, 1.1452, 0.7049, 0.9282),
}


@pytest.mark.parametrize("scen", SCEN)
def test_table2(scen):
    m, k, n2 = scen
    P = prob(m, k)
    exp = TABLE2[scen]
    got = (P.contrast(load(m, k, n2, "psi_aware")), P.contrast(load(m, k, n2, "Iu_naive")),
           P.contrast(load(m, k, n2, "C_aware")), P.contrast(load(m, k, n2, "Cu_pooled")),
           P.psi(load(m, k, n2, "C_aware")), P.psi(load(m, k, n2, "Cu_pooled")))
    for g, e in zip(got, exp):
        assert g == pytest.approx(e, abs=5e-5)


def test_section_5_4_mechanism():
    P = prob(10, 4)
    A, U = load(10, 4, 4, "C_aware"), load(10, 4, 4, "Cu_pooled")
    sa, su = P.summary(A), P.summary(U)
    assert sa["norm_Delta"] == pytest.approx(1.078, abs=1e-3)
    assert su["norm_Delta"] == pytest.approx(1.045, abs=1e-3)
    assert sa["min_eig_S"] == pytest.approx(1.371, abs=1e-3)
    assert su["min_eig_S"] == pytest.approx(0.306, abs=1e-3)
    assert sa["leverage"] == pytest.approx(0.165, abs=1e-3)
    assert su["leverage"] == pytest.approx(1.709, abs=1e-3)
    assert sa["pooled_gap"] == pytest.approx(0.021, abs=1e-3)
    assert su["pooled_gap"] == pytest.approx(0.502, abs=1e-3)


# ---------------------------------------------------------------- identities
def test_moments_match_quadrature():
    for k in (2, 3, 4):
        mu, V, Om = cube_moments(k)
        g, w = np.polynomial.legendre.leggauss(3)
        pts = np.array(list(itertools.product(g, repeat=k)))
        ww = np.prod(np.array(list(itertools.product(w / 2, repeat=k))), 1)
        F = f_matrix(pts)
        assert np.allclose((F * ww[:, None]).T @ F, Om)


@pytest.mark.parametrize("scen", SCEN)
def test_propositions_1_to_3(scen):
    m, k, n2 = scen
    P = prob(m, k)
    for which in ("psi_aware", "Iu_naive", "C_aware", "Cu_pooled"):
        X2 = load(m, k, n2, which)
        Ma = P.M_adjusted(X2)
        # Proposition 1: tr(Ma^-1 Omega) = 1/n1 + Psi
        assert np.trace(np.linalg.solve(Ma, P.Omega)) == pytest.approx(P.phi_I(X2), rel=1e-9)
        # pointwise variance (4) equals f' Ma^-1 f
        x = np.random.default_rng(1).uniform(-1, 1, (5, k))
        F = f_matrix(x)
        assert np.allclose(np.einsum("ij,jk,ik->i", F, np.linalg.inv(Ma), F),
                           P.prediction_variance(X2, x))
        # Proposition 2: (8) and h as a squared cosine
        lhs = P.phi_I(X2) - P.unblocked_I(X2)
        assert lhs == pytest.approx(P.stage_cost(X2), rel=1e-8)
        Fall = f_matrix(np.vstack([P.X1, X2]))
        s = np.r_[np.zeros(P.n1), np.ones(n2)]
        proj = Fall @ np.linalg.lstsq(Fall, s, rcond=None)[0]
        assert P.confounding_index(X2) == pytest.approx(s @ proj / n2, rel=1e-8)
        # Proposition 3: S_pool = S + alpha DD' and identity (10)
        a = P.n1 * n2 / (P.n1 + n2)
        D = P.displacement(X2)
        assert np.allclose(P.S_pool(X2), P.S(X2) + a * np.outer(D, D))
        assert P.contrast(X2) - P.pooled_contrast(X2) == pytest.approx(P.pooled_gap(X2), rel=1e-8)


def test_psi_zero_and_tissue_culture_numbers():
    P = prob(6, 3)
    assert P.psi(None) == pytest.approx(0.6237, abs=1e-4)
    A = load(6, 3, 4, "psi_aware")
    sigma = 1.25
    assert sigma * np.sqrt(1 / 13 + P.psi(A)) == pytest.approx(0.82, abs=0.005)
    assert sigma * np.sqrt(1 / 13 + P.psi(load(6, 3, 4, "Iu_naive"))) == pytest.approx(0.99, abs=0.005)
    # exact-centre replacement changes Psi negligibly and leaves one pure-error df
    A2 = A.copy(); A2[1:3] = 0.0
    assert abs(P.psi(A2) - P.psi(A)) < 6e-11 * 10
    S2 = z_matrix(A2) - z_matrix(A2).mean(0)
    assert np.linalg.matrix_rank(S2.T @ S2, tol=1e-8) == 2


def test_theorem_1_rank():
    # (10,4): r1 = 12, q = 3, so n2 = 3 cannot be full rank but n2 = 4 can.
    P = prob(10, 4)
    rng = np.random.default_rng(0)
    for _ in range(20):
        assert P.rank_adjusted(rng.uniform(-1, 1, (3, 4))) < P.p
    X2 = np.vstack([P.X1[0], rng.uniform(-1, 1, (3, 4))])  # construction in the proof
    assert P.rank_adjusted(X2) == P.p
    # n2 = 1 adds nothing: M_a = F1'F1
    assert np.allclose(P.M_adjusted(rng.uniform(-1, 1, (1, 4))), P.F1.T @ P.F1)


def test_projection_rank_counts():
    # Section 5.5: full-rank four-factor projections for m = 11..14; none for m <= 10
    expected = {11: (110, 330), 12: (165, 495), 13: (520, 715), 14: (728, 1001)}
    for m in range(5, 15):
        D = dsd(m)
        full = sum(np.linalg.matrix_rank(f_matrix(D[:, list(s)]), tol=1e-8) == 15
                   for s in itertools.combinations(range(m), 4))
        tot = len(list(itertools.combinations(range(m), 4)))
        if m <= 10:
            assert full == 0
        else:
            assert (full, tot) == expected[m]
        assert all(np.linalg.matrix_rank(f_matrix(D[:, list(s)]), tol=1e-8) == 10
                   for s in itertools.combinations(range(m), 3))
    # Section 5.1 class counts
    for m, k, q, count in [(8, 4, 2, 42), (10, 4, 3, 30), (10, 4, 1, 180)]:
        D = dsd(m)
        n = sum(15 - np.linalg.matrix_rank(f_matrix(D[:, list(s)]), tol=1e-8) == q
                for s in itertools.combinations(range(m), k))
        assert n == count


def test_rank_two_update():
    P = prob(6, 3)
    rng = np.random.default_rng(3)
    X2 = rng.uniform(-1, 1, (4, 3))
    Z2 = z_matrix(X2)
    S = P.S(X2)
    new = X2.copy(); new[2] = rng.uniform(-1, 1, 3)
    Snew, U, K = rank_two_update(S, Z2[2], z_matrix(new[2])[0], Z2.mean(0), 4)
    assert np.allclose(Snew, P.S(new))
    Si = _spd_inverse(S)
    assert updated_value(P.psi(X2), Si, Si @ P.Q @ Si, U, K) == pytest.approx(P.psi(new), rel=1e-9)


def test_trace_bound():
    assert trace_bound(prob(6, 3), 4) == pytest.approx(0.1949, abs=1e-4)


def test_relaxation_bound():
    r = relaxation_bound(prob(6, 3), 4)
    assert r["relaxed_value"] == pytest.approx(0.3331, abs=1e-4)
    assert r["cube_lower_bound"] == pytest.approx(0.3327, abs=1e-4)
    # best-found exact design is at most 7.2% above the optimum
    assert 0.3566 / r["cube_lower_bound"] - 1 < 0.072


def test_exchange_finds_good_design():
    P = prob(6, 3)
    res = coordinate_exchange(P, 4, "psi", n_starts=6, seed=1)
    assert res.value < 0.40 and res.value >= 0.3327
    assert res.value == pytest.approx(P.psi(res.X2))
