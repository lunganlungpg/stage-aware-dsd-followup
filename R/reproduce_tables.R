# Recompute Tables 1-2 of the paper from the stored designs.
# Run from the repository root:   Rscript R/reproduce_tables.R
# Works without installing the package (sources the R files directly).

for (f in list.files("R/stageaware/R", full.names = TRUE)) source(f)
rd <- function(tag, w) as.matrix(read.csv(sprintf("data/selected_designs/%s_%s.csv", tag, w)))

rows <- lapply(list(c(6, 3, 4), c(8, 4, 6), c(10, 4, 4)), function(s) {
  m <- s[1]; k <- s[2]; n2 <- s[3]; tag <- sprintf("m%d_k%d_n%d", m, k, n2)
  P <- followup_problem(dsd(m)[, 1:k])
  A <- rd(tag, "psi_aware"); N <- rd(tag, "Iu_naive")
  Ca <- rd(tag, "C_aware");  Cp <- rd(tag, "Cu_pooled")
  data.frame(scenario = sprintf("(%d,%d,%d)", m, k, n2), n1 = P$n1, r1 = P$r1, q = P$q,
             Psi_aware = psi(P, A), Psi_naive = psi(P, N),
             h_aware = confounding_index(P, A), h_naive = confounding_index(P, N),
             C_Psi_aware = contrast_C(P, A), C_Iu_selected = contrast_C(P, N),
             C_C_aware = contrast_C(P, Ca), C_Cu_selected = contrast_C(P, Cp))
})
tab <- do.call(rbind, rows)
cat("Table 1 / Table 2 (stage-adjusted evaluation)\n")
print(format(tab, digits = 4), row.names = FALSE)

P <- followup_problem(dsd(10)[, 1:4])
cat("\nSection 5.4 (10,4,4):\n")
print(rbind(C_aware = unlist(pooled_diagnostics(P, rd("m10_k4_n4", "C_aware"))),
            Cu_pooled = unlist(pooled_diagnostics(P, rd("m10_k4_n4", "Cu_pooled")))), digits = 4)

cat("\nExample: a fresh stage-aware search for (6,3,4), 10 starts\n")
set.seed(20261123)
P6 <- followup_problem(dsd(6)[, 1:3])
res <- coordinate_exchange(P6, 4, "psi", n_starts = 10)
print(round(res$X2, 3)); cat("Psi =", round(res$value, 4), "\n")
