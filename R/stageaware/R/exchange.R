# Coordinate exchange with the centred rank-two update (Section 4.1).

#' Rank-two update of the within-stage scatter (Proposition 4)
#'
#' @param S current pooled within-stage scatter.
#' @param a,b old and new regressor rows z(x) of the changed Stage 2 run.
#' @param zbar2 Stage 2 regressor mean before the change.
#' @param n2 number of Stage 2 runs.
#' @return list(S_new, U, K).
#' @export
rank_two_update <- function(S, a, b, zbar2, n2) {
  U <- cbind(a - zbar2, b - a)
  K <- matrix(c(0, 1, 1, 1 - 1 / n2), 2L, 2L)
  list(S_new = S + U %*% K %*% t(U), U = U, K = K)
}

# Eq. (12); returns Inf when S_new would not be positive definite.
updated_value <- function(value, Sinv, W, U, K) {
  B <- solve(K) + crossprod(U, Sinv %*% U)
  if (!all(is.finite(B)) || det(B) >= -1e-12) return(Inf)
  out <- value - sum(diag(solve(B, crossprod(U, W %*% U))))
  if (out > 0) out else Inf
}

#' Coordinate exchange for follow-up runs
#'
#' @param P a followup_problem.
#' @param n2 number of follow-up runs (>= 2).
#' @param criterion "psi", "contrast", "unblocked" or "pooled_contrast".
#' @param n_starts number of random starts.
#' @param grid,brackets,max_sweeps,tol search settings as in Section 4.1.
#' @return list(X2, value, values, sweeps, converged, terminal).
#' @export
coordinate_exchange <- function(P, n2, criterion = c("psi", "contrast", "unblocked", "pooled_contrast"),
                                n_starts = 20L, grid = 9L, brackets = 3L,
                                max_sweeps = 25L, tol = 1e-9) {
  criterion <- match.arg(criterion)
  if (n2 < 2L) stop("n2 >= 2 is required")
  f <- switch(criterion, psi = function(X) psi(P, X), contrast = function(X) contrast_C(P, X),
              unblocked = function(X) unblocked_I(P, X), pooled_contrast = function(X) pooled_C(P, X))
  centred <- criterion %in% c("psi", "contrast")
  target <- if (criterion == "psi") P$Q else P$V
  scale <- if (criterion == "psi") 1 else 2
  k <- P$k; gp <- seq(-1, 1, length.out = grid)
  terminal <- vector("list", n_starts); vals <- sw <- numeric(n_starts); conv <- logical(n_starts)
  for (s in seq_len(n_starts)) {
    repeat { X2 <- matrix(stats::runif(n2 * k, -1, 1), n2, k); val <- f(X2); if (is.finite(val)) break }
    converged <- FALSE
    for (sweep in seq_len(max_sweeps)) {
      before <- val
      for (i in seq_len(n2)) for (j in seq_len(k)) {
        if (centred) {
          Z2 <- z_matrix(X2); zb <- colMeans(Z2)
          Si <- spd_inverse(P$S1 + centred_scatter(Z2))
          if (is.null(Si)) next
          W <- Si %*% target %*% Si; a <- Z2[i, ]
          evaluate <- function(cc) {
            x <- X2[i, ]; x[j] <- cc
            uk <- rank_two_update(0, a, drop(z_matrix(x)), zb, n2)
            scale * updated_value(val / scale, Si, W, uk$U, uk$K)
          }
        } else {
          evaluate <- function(cc) { Y <- X2; Y[i, j] <- cc; f(Y) }
        }
        gv <- vapply(gp, evaluate, numeric(1))
        cand_v <- c(val, gv[is.finite(gv)]); cand_x <- c(X2[i, j], gp[is.finite(gv)])
        ord <- order(pmin(gv[-grid], gv[-1L]))
        for (b in ord[seq_len(brackets)]) {
          o <- stats::optimize(evaluate, c(gp[b], gp[b + 1L]), tol = 1e-6)
          if (is.finite(o$objective)) { cand_v <- c(cand_v, o$objective); cand_x <- c(cand_x, o$minimum) }
        }
        bst <- which.min(cand_v)
        if (cand_v[bst] < val - tol) {
          trial <- X2; trial[i, j] <- cand_x[bst]
          chk <- f(trial)
          if (chk < val - tol) { X2 <- trial; val <- chk }
        }
      }
      if (before - val < tol) { converged <- TRUE; break }
    }
    terminal[[s]] <- X2; vals[s] <- val; sw[s] <- sweep; conv[s] <- converged
  }
  b <- which.min(vals)
  list(X2 = terminal[[b]], value = vals[b], values = vals, sweeps = sw,
       converged = conv, terminal = terminal)
}
