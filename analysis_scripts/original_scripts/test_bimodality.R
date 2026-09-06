############################################################
# Is the per-subject legality difference-score distribution
# bimodal in any experiment?
#
# Three lines of evidence:
#   (1) Hartigan's dip test  - null = unimodal
#   (2) Gaussian mixture BIC - does 2+ components beat 1?
#   (3) Measurement-noise floor - how much apparent spread /
#       lumpiness is expected from binomial sampling alone,
#       even if every subject shared one true effect?
############################################################

suppressMessages({
  library(plyr); library(dplyr); library(diptest); library(mclust)
})
source("analysis_functions.R")
set.seed(1)

worker_legality_diff <- function(data) {
  gen <- filter(data, gen == 0.5)
  by_leg <- gen %>% group_by(worker_ID, legality) %>%
    dplyr::summarize(yes_rate = mean(yesResponse),
                     n_yes = sum(yesResponse), n = n(), .groups = "drop")
  legal   <- by_leg %>% filter(legality ==  0.5) %>%
    dplyr::select(worker_ID, legal_rate = yes_rate, n_legal_yes = n_yes, n_legal = n)
  illegal <- by_leg %>% filter(legality == -0.5) %>%
    dplyr::select(worker_ID, illegal_rate = yes_rate, n_illegal_yes = n_yes, n_illegal = n)
  inner_join(legal, illegal, by = "worker_ID") %>%
    mutate(diff = legal_rate - illegal_rate)
}

per1 <- open_file("data/per1_filteredSubjects.csv", experiment = 1)
per2 <- open_file("data/per2_filteredSubjects.csv", experiment = 2)

sets <- list(
  Per1A = worker_legality_diff(filter(per1, condition_num <  5)),
  Per1B = worker_legality_diff(filter(per1, condition_num >= 5)),
  Per2  = worker_legality_diff(per2)
)

for (nm in names(sets)) {
  d  <- sets[[nm]]
  x  <- d$diff
  n  <- length(x)

  cat("\n=====================", nm, "=====================\n")
  cat(sprintf("N = %d | mean = %.3f | sd = %.3f\n", n, mean(x), sd(x)))

  ## (1) Hartigan's dip test
  dt <- dip.test(x)
  cat(sprintf("Dip test: D = %.4f, p = %.3f  (small p => reject unimodal)\n",
              dt$statistic, dt$p.value))

  ## (2) Gaussian mixture model, 1..4 components, chosen by BIC
  mc <- Mclust(x, G = 1:4, verbose = FALSE)
  bic_by_G <- sapply(1:4, function(g) {
    m <- Mclust(x, G = g, modelNames = "V", verbose = FALSE)  # V = unequal variances
    if (is.null(m)) NA else m$bic
  })
  cat("GMM BIC by #components (higher = better in mclust):\n")
  print(round(setNames(bic_by_G, paste0("G=", 1:4)), 1))
  cat(sprintf("  best-BIC model: G = %d\n", mc$G))
  # BIC advantage of best 2-component over 1-component
  cat(sprintf("  BIC(G=2) - BIC(G=1) = %.1f  (>~2 favors 2 comps, >~10 strong)\n",
              bic_by_G[2] - bic_by_G[1]))

  ## (3) Measurement-noise null: if EVERY subject had the SAME true effect
  ## equal to the group mean, what would the observed distribution look like?
  ## Simulate each subject's two binomial rates at the group mean legal/illegal
  ## endorsement probabilities, recompute diff, and re-run the dip test.
  p_legal   <- mean(d$n_legal_yes)   / d$n_legal[1]
  p_illegal <- mean(d$n_illegal_yes) / d$n_illegal[1]
  nl <- d$n_legal[1]; ni <- d$n_illegal[1]
  sim_dips <- replicate(2000, {
    sl <- rbinom(n, nl, p_legal)  / nl
    si <- rbinom(n, ni, p_illegal) / ni
    dip.test(sl - si)$statistic
  })
  cat(sprintf("Noise-only null (all subjects share the mean effect):\n"))
  cat(sprintf("  observed dip D = %.4f;  null D 95%%ile = %.4f;  p_sim = %.3f\n",
              dt$statistic, quantile(sim_dips, 0.95),
              mean(sim_dips >= dt$statistic)))
  # spread expected from pure binomial noise vs observed spread
  sim_sd_val <- sqrt(p_legal*(1-p_legal)/nl + p_illegal*(1-p_illegal)/ni)
  cat(sprintf("  sd expected from binomial noise alone = %.3f;  observed sd = %.3f\n",
              sim_sd_val, sd(x)))
}
cat("\n")
