data_dir <- function() {
  d <- system.file("extdata", package = "stageaware")
  if (!nzchar(d)) d <- file.path("..", "..", "..", "..", "data", "selected_designs")
  d
}
load_design <- function(tag, which)
  as.matrix(utils::read.csv(file.path(data_dir(), sprintf("%s_%s.csv", tag, which))))

scen <- list(c(6, 3, 4), c(8, 4, 6), c(10, 4, 4))
table1 <- list(c(13, 10, 0, 0.3566, 0.5518, 0.494, 0.686),
               c(17, 13, 2, 0.4538, 0.5326, 0.476, 0.672),
               c(21, 12, 3, 0.7049, 0.7246, 0.460, 0.638))
table2 <- list(c(0.5161, 0.5589, 0.4501, 0.4518, 0.4193, 0.4220),
               c(0.5698, 0.5832, 0.5201, 0.5282, 0.5721, 0.5973),
               c(0.6984, 0.7378, 0.6984, 1.1452, 0.7049, 0.9282))

test_that("Tables 1 and 2 are reproduced from the stored designs", {
  for (s in seq_along(scen)) {
    m <- scen[[s]][1]; k <- scen[[s]][2]; n2 <- scen[[s]][3]
    tag <- sprintf("m%d_k%d_n%d", m, k, n2)
    P <- followup_problem(dsd(m)[, 1:k])
    e <- table1[[s]]
    expect_equal(c(P$n1, P$r1, P$q), e[1:3])
    A <- load_design(tag, "psi_aware"); N <- load_design(tag, "Iu_naive")
    expect_equal(psi(P, A), e[4], tolerance = 5e-5, scale = 1)
    expect_equal(psi(P, N), e[5], tolerance = 5e-5, scale = 1)
    expect_equal(confounding_index(P, A), e[6], tolerance = 5e-4, scale = 1)
    expect_equal(confounding_index(P, N), e[7], tolerance = 5e-4, scale = 1)
    got <- c(contrast_C(P, A), contrast_C(P, N),
             contrast_C(P, load_design(tag, "C_aware")), contrast_C(P, load_design(tag, "Cu_pooled")),
             psi(P, load_design(tag, "C_aware")), psi(P, load_design(tag, "Cu_pooled")))
    expect_equal(got, table2[[s]], tolerance = 5e-5, scale = 1)
  }
})

test_that("Propositions 1-3 hold exactly", {
  P <- followup_problem(dsd(10)[, 1:4])
  for (w in c("psi_aware", "Iu_naive", "C_aware", "Cu_pooled")) {
    X2 <- load_design("m10_k4_n4", w)
    Ma <- M_adjusted(P, X2)
    expect_equal(sum(diag(solve(Ma, P$Omega))), 1 / P$n1 + psi(P, X2), tolerance = 1e-9)
    d <- pooled_diagnostics(P, X2)
    expect_equal(contrast_C(P, X2) - pooled_C(P, X2), d$gap, tolerance = 1e-8)
  }
  d <- pooled_diagnostics(P, load_design("m10_k4_n4", "Cu_pooled"))
  expect_equal(c(d$norm_Delta, d$min_eig_S, d$leverage, d$gap),
               c(1.045, 0.306, 1.709, 0.502), tolerance = 1e-3, scale = 1)
})

test_that("Theorem 1, Psi(0), Proposition 4 and Proposition 5", {
  P <- followup_problem(dsd(10)[, 1:4])
  expect_equal(minimum_followup_runs(P), 4L)
  set.seed(1)
  expect_lt(matrix_rank(M_adjusted(P, matrix(runif(12, -1, 1), 3))), P$p)
  expect_equal(matrix_rank(M_adjusted(P, rbind(P$X1[1, ], matrix(runif(12, -1, 1), 3)))), P$p)
  P6 <- followup_problem(dsd(6)[, 1:3])
  expect_equal(psi(P6, NULL), 0.6237, tolerance = 1e-4, scale = 1)
  expect_equal(trace_bound(P6, 4), 0.1949, tolerance = 1e-4, scale = 1)
  X2 <- matrix(runif(12, -1, 1), 4); Y2 <- X2; Y2[2, ] <- runif(3, -1, 1)
  Z2 <- z_matrix(X2)
  up <- rank_two_update(S_within(P6, X2), Z2[2, ], drop(z_matrix(Y2[2, ])), colMeans(Z2), 4)
  expect_equal(unname(up$S_new), unname(S_within(P6, Y2)), tolerance = 1e-10)
})

test_that("coordinate exchange reaches a good design", {
  set.seed(2)
  P6 <- followup_problem(dsd(6)[, 1:3])
  r <- coordinate_exchange(P6, 4, "psi", n_starts = 4)
  expect_lt(r$value, 0.40)
  expect_gte(r$value, 0.3327)
})
