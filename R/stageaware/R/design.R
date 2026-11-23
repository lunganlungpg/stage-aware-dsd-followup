# Conference-matrix definitive screening designs (Paley construction).
# GF(9) elements are a + b*t with t^2 = 2 over GF(3), stored as a + 3b.

field_sub <- function(a, b, q) {
  if (q != 9L) return((a - b) %% q)
  ((a %% 3L - b %% 3L) %% 3L) + 3L * (((a %/% 3L) - (b %/% 3L)) %% 3L)
}

field_mul <- function(a, b, q) {
  if (q != 9L) return((a * b) %% q)
  aa <- a %% 3L; ab <- a %/% 3L; ba <- b %% 3L; bb <- b %/% 3L
  ((aa * ba + 2L * ab * bb) %% 3L) + 3L * ((aa * bb + ab * ba) %% 3L)
}

field_pow <- function(a, e, q) {
  v <- 1L
  if (e > 0L) for (j in seq_len(e)) v <- field_mul(v, a, q)
  v
}

#' Paley conference matrix of order q + 1
#'
#' @param q prime power, one of 5, 7, 9, 11, 13.
#' @return (q+1) x (q+1) matrix C with C'C = qI.
#' @export
paley_conference <- function(q) {
  q <- as.integer(q)
  if (!(q %in% c(5L, 7L, 9L, 11L, 13L))) stop("q must be 5, 7, 9, 11 or 13")
  chi <- function(v) if (v == 0L) 0L else if (field_pow(v, (q - 1L) %/% 2L, q) == 1L) 1L else -1L
  el <- 0:(q - 1L)
  Q <- outer(el, el, Vectorize(function(a, b) chi(field_sub(a, b, q))))
  C <- matrix(1, q + 1L, q + 1L)
  C[1L, 1L] <- 0
  C[-1L, -1L] <- Q
  if (q %% 4L == 3L) C[-1L, 1L] <- -1
  if (!isTRUE(all.equal(crossprod(C), diag(q, q + 1L)))) stop("conference identity failed")
  C
}

#' Definitive screening design [C; -C; 0] for m = 5, ..., 14 factors
#' @param m number of factors.
#' @export
dsd <- function(m) {
  mm <- if (m %% 2L == 0L) m else m + 1L
  if (!(mm %in% c(6L, 8L, 10L, 12L, 14L))) stop("supported m = 5, ..., 14")
  C <- paley_conference(mm - 1L)
  D <- rbind(C, -C, rep(0, mm))[, seq_len(m), drop = FALSE]
  colnames(D) <- paste0("x", seq_len(m))
  D
}
