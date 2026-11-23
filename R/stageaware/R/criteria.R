# Stage-adjusted criteria (Section 2). Regressor order in z(x): k linear,
# k pure quadratic, then the k(k-1)/2 two-factor interactions.

#' Non-intercept quadratic regressors z(x)
#' @param X n x k matrix of coded settings in [-1, 1].
#' @export
z_matrix <- function(X) {
  if (is.null(dim(X))) X <- matrix(X, nrow = 1L)
  X <- as.matrix(X); k <- ncol(X)
  Z <- cbind(X, X^2)
  nm <- c(paste0("x", 1:k), paste0("x", 1:k, "^2"))
  if (k >= 2L) {
    pr <- utils::combn(k, 2L)
    inter <- matrix(apply(pr, 2L, function(ij) X[, ij[1]] * X[, ij[2]]), nrow = nrow(X))
    Z <- cbind(Z, inter)
    nm <- c(nm, apply(pr, 2L, function(ij) sprintf("x%d:x%d", ij[1], ij[2])))
  }
  Z <- matrix(Z, nrow = nrow(X))
  colnames(Z) <- nm
  Z
}

#' Full model matrix [1, z(x)]
#' @export
f_matrix <- function(X) {
  Z <- z_matrix(X)
  cbind(`(Intercept)` = 1, Z)
}

#' Exact moments of [1, z(x)] under the uniform measure on [-1, 1]^k
#' @return list(mu, V, Omega) as in eq. (2).
#' @export
cube_moments <- function(k) {
  E <- rbind(rep(0L, k), diag(k), 2L * diag(k))
  if (k >= 2L) {
    pr <- utils::combn(k, 2L)
    E <- rbind(E, t(apply(pr, 2L, function(ij) { e <- integer(k); e[ij] <- 1L; e })))
  }
  Om <- matrix(1, nrow(E), nrow(E))
  for (j in seq_len(k)) {
    pw <- outer(E[, j], E[, j], `+`)
    Om <- Om * ifelse(pw %% 2L == 1L, 0, 1 / (pw + 1))
  }
  mu <- Om[1L, -1L]
  list(mu = mu, V = Om[-1L, -1L] - tcrossprod(mu), Omega = Om)
}

centred_scatter <- function(Z) {
  if (nrow(Z) == 0L) return(matrix(0, ncol(Z), ncol(Z)))
  crossprod(sweep(Z, 2L, colMeans(Z), "-"))
}

spd_inverse <- function(A, tol = 1e-9) {
  A <- (A + t(A)) / 2
  ev <- eigen(A, symmetric = TRUE, only.values = TRUE)$values
  if (min(ev) <= tol * max(1, max(ev))) return(NULL)
  solve(A)
}

matrix_rank <- function(A, tol = 1e-8) {
  s <- svd(A, nu = 0L, nv = 0L)$d
  sum(s > tol * max(1, s[1L]))
}

#' Set up a follow-up design problem with a fixed Stage 1 design
#'
#' @param X1 n1 x k matrix: the projected screening design in coded units.
#' @return an object of class "followup_problem".
#' @export
followup_problem <- function(X1) {
  X1 <- as.matrix(X1); k <- ncol(X1); p <- (k + 1L) * (k + 2L) / 2L
  F1 <- f_matrix(X1); Z1 <- F1[, -1L, drop = FALSE]
  mom <- cube_moments(k); zbar1 <- colMeans(Z1); d <- mom$mu - zbar1
  r1 <- matrix_rank(F1)
  structure(list(X1 = X1, k = k, n1 = nrow(X1), p = p, t = p - 1L, F1 = F1, Z1 = Z1,
                 zbar1 = zbar1, S1 = centred_scatter(Z1), mu = mom$mu, V = mom$V,
                 Omega = mom$Omega, Q = mom$V + tcrossprod(d), r1 = r1, q = p - r1),
            class = "followup_problem")
}

as_X2 <- function(P, X2) {
  if (is.null(X2) || length(X2) == 0L) return(matrix(numeric(0), 0L, P$k))
  if (is.null(dim(X2))) X2 <- matrix(X2, nrow = 1L)
  X2 <- as.matrix(X2)
  if (ncol(X2) != P$k) stop("X2 must have ", P$k, " columns")
  X2
}

#' Pooled within-stage scatter S = S1 + S2 (eq. 3)
#' @export
S_within <- function(P, X2) {
  X2 <- as_X2(P, X2)
  if (nrow(X2) == 0L) return(P$S1)
  P$S1 + centred_scatter(z_matrix(X2))
}

#' Information for beta after profiling the stage shift, M_a
#' @export
M_adjusted <- function(P, X2) {
  X2 <- as_X2(P, X2); M <- crossprod(P$F1)
  if (nrow(X2)) M <- M + centred_scatter(f_matrix(X2))
  M
}

#' Unblocked information M_u = F'F
#' @export
M_unblocked <- function(P, X2) {
  X2 <- as_X2(P, X2); M <- crossprod(P$F1)
  if (nrow(X2)) M <- M + crossprod(f_matrix(X2))
  M
}

#' Excess integrated variance Psi = tr(S^-1 Q) (Proposition 1)
#' @export
psi <- function(P, X2) {
  Si <- spd_inverse(S_within(P, X2))
  if (is.null(Si)) Inf else sum(Si * t(P$Q))
}

#' Integrated pairwise-difference variance C = 2 tr(S^-1 V) (eq. 6)
#' @export
contrast_C <- function(P, X2) {
  Si <- spd_inverse(S_within(P, X2))
  if (is.null(Si)) Inf else 2 * sum(Si * t(P$V))
}

#' Unblocked I-criterion I_u = tr(M_u^-1 Omega)
#' @export
unblocked_I <- function(P, X2) {
  Mi <- spd_inverse(M_unblocked(P, X2))
  if (is.null(Mi)) Inf else sum(Mi * t(P$Omega))
}

#' Stage-confounding index h (eq. 7)
#' @export
confounding_index <- function(P, X2) {
  X2 <- as_X2(P, X2)
  if (nrow(X2) == 0L) return(0)
  Mi <- spd_inverse(M_unblocked(P, X2))
  if (is.null(Mi)) return(NA_real_)
  fb <- colMeans(f_matrix(X2))
  nrow(X2) * drop(crossprod(fb, Mi %*% fb))
}

#' Pooled contrast criterion C_u = 2 tr(S_pool^-1 V)
#' @export
pooled_C <- function(P, X2) {
  X2 <- as_X2(P, X2)
  Si <- spd_inverse(centred_scatter(rbind(P$Z1, z_matrix(X2))))
  if (is.null(Si)) Inf else 2 * sum(Si * t(P$V))
}

#' Directional leverage and the gap C - C_u of identity (10)
#' @export
pooled_diagnostics <- function(P, X2) {
  X2 <- as_X2(P, X2); n2 <- nrow(X2)
  S <- S_within(P, X2); Si <- spd_inverse(S)
  D <- colMeans(z_matrix(X2)) - P$zbar1
  a <- P$n1 * n2 / (P$n1 + n2)
  lev <- if (is.null(Si)) Inf else drop(crossprod(D, Si %*% D))
  gap <- if (is.null(Si)) Inf else
    2 * a * drop(crossprod(D, Si %*% P$V %*% Si %*% D)) / (1 + a * lev)
  list(norm_Delta = sqrt(sum(D^2)), leverage = lev, gap = gap,
       min_eig_S = min(eigen(S, symmetric = TRUE, only.values = TRUE)$values))
}

#' All criteria and diagnostics for a follow-up design
#' @export
design_summary <- function(P, X2) {
  X2 <- as_X2(P, X2); n2 <- nrow(X2)
  rk <- matrix_rank(M_adjusted(P, X2))
  c(list(n1 = P$n1, n2 = n2, p = P$p, r1 = P$r1, q = P$q, rank_Ma = rk,
         residual_df = P$n1 + n2 - rk - as.integer(n2 > 0L),
         Psi = psi(P, X2), Phi_I = 1 / P$n1 + psi(P, X2), C = contrast_C(P, X2),
         C_u = pooled_C(P, X2), I_u = unblocked_I(P, X2), h = confounding_index(P, X2)),
    pooled_diagnostics(P, X2))
}

#' Minimum number of follow-up runs that can restore full rank (Theorem 1)
#' @export
minimum_followup_runs <- function(P) if (P$q == 0L) 0L else P$q + 1L

#' Analytic lower bound of Proposition 5
#' @export
trace_bound <- function(P, n2) {
  ev <- pmax(eigen(P$Q, symmetric = TRUE, only.values = TRUE)$values, 0)
  sum(sqrt(ev))^2 / (sum(diag(P$S1)) + n2 * (P$t - 3 * P$k / 4))
}
