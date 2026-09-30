############################################################
# Discrete listener types vs. continuous variation, with the
# binomial measurement process IN the likelihood.
#
# This is the trial-level counterpart of test_bimodality_varying_mean.R.
# That script modelled the per-listener legality difference score as
# Gaussian, which is a poor fit to what the score actually is: a
# difference of two binomial proportions from 18 (Per1) or 16 (Per2)
# test items per legality. Consequences of the Gaussian version:
#   * 81-88% of the variance in the difference score is binomial
#     sampling noise, so the fitted mixture components were mostly
#     noise kernels, not subpopulations;
#   * the score is bounded and discrete (19 / 15 achievable values),
#     with a large atom at exactly 0 (9.6% / 13.6% of listeners) that
#     a Gaussian cannot produce and that can masquerade as a mode;
#   * noise shrinks near ceiling, which matters in Per2 (mean legal
#     yes-rate 0.77, 5.2% of listeners at ceiling).
#
# Here each listener contributes their counts (k_legal of n_legal,
# k_illegal of n_illegal) and the two models differ only in how the
# LATENT per-listener legality effect is distributed:
#
#   MODEL 1' "discrete types, distance-varying mixture weight"
#     Listener i belongs to one of two latent classes with legality
#     effects theta_1 < theta_2 (logit scale). Talker distance moves
#     only the class probability:
#       k_L,i ~ Bin(n_L, logit^-1(alpha_i + theta_c/2))
#       k_I,i ~ Bin(n_I, logit^-1(alpha_i - theta_c/2))
#       alpha_i ~ N(mu_a, sd_a)          (listener yes-bias)
#       logit P(class 2) = g0 + g1 * d_i
#     6 parameters. g1 > 0 means the higher-effect class gets more
#     common as talker distance grows -- the trial-level version of
#     "the higher mode gets taller".
#
#   MODEL 2' "continuous effect, distance-varying mean"
#     One population, the legality effect varies continuously:
#       theta_i ~ N(b0 + b1 * d_i, sd_theta)
#       alpha_i ~ N(mu_a, sd_a)
#     5 parameters. This is the listener-level counterpart of the
#     random-slope logistic GLMM already reported
#     (response ~ legality * avg_distance_c + (1 + legality | worker_ID)),
#     so the comparison lines up with the regression analysis instead
#     of sitting beside it. It is not numerically identical to it:
#     aggregating to counts drops the (1 + legality | syllable) term,
#     so b1 here should be close to, but a little smaller than, the
#     glmer interaction once rescaled (glmer's is per unit of centered
#     distance; b1 is per SD).
#
# alpha (and theta in Model 2') are integrated out by Gauss-Hermite
# quadrature; see section 2. Nothing here needs the SD floor that the
# Gaussian mixture needed, because the measurement noise now lives in
# the binomial likelihood instead of being absorbed into a component
# variance -- a component cannot collapse onto the discreteness grid.
#
# Reported per data set:
#   (a) fitted parameters, AIC, BIC;
#   (b) quadrature precision check (logLik at the fitted parameters
#       recomputed with twice the nodes -- integration error must be
#       far below the AIC/BIC differences being interpreted);
#   (c) posterior-predictive checks the Gaussian models failed:
#       P(diff = 0), P(diff < 0), SD(diff);
#   (d) parametric bootstrap under Model 2' calibrating the AIC
#       difference (the models are not nested);
#   (e) a RECOVERY simulation: data generated from the fitted Model 1'
#       and refit, to show how often a genuine two-class world is
#       detected at these sample sizes. With ~85% of the difference-
#       score variance being noise, absence of evidence for classes
#       may just be the power limit, so this is not optional.
#
# CAVEAT that no model here fixes: talker distance is confounded with
# talker identity (one distance cluster per speaker pair), especially
# in Per1. "Farther talkers" and "these particular talkers" are not
# separable in this design.
############################################################

suppressMessages({
  library(plyr); library(dplyr); library(ggplot2)
})
source("analysis_functions.R")
set.seed(1)

## ---- configuration ------------------------------------------------------
## Node counts were set from a measured convergence study at fitted parameter
## values (Per1). Absolute error in the log-likelihood vs. a 400-node (1-D) /
## 160-node-per-dimension (2-D) reference:
##   Model 1' (1-D): Q=20 -> 5e-2 | Q=32 -> 9e-3 | Q=48 -> 2e-4 | Q=64 -> 9e-7
##   Model 2' (2-D): Q=20 -> 4e-2 | Q=30 -> 1e-2 | Q=40 -> 9e-4 | Q=60 -> 3e-6
## The reported fits therefore use 64 / 60, which puts integration error ~6
## orders of magnitude below the AIC differences being interpreted. Optimizing
## at that many nodes is wasteful, so each model is located at the cheaper node
## count and then refined and scored at the high one (see fit_ml).
N_GH_M1_FIT <- 32;  N_GH_M1_HI <- 64   # Model 1': one integral (alpha)
N_GH_M2_FIT <- 30;  N_GH_M2_HI <- 60   # Model 2': per dimension (alpha, theta)
N_GH_M1_SIM <- 32                       # nodes inside the simulation loops
N_GH_M2_SIM <- 20                       # ~4e-2 logLik error, i.e. 8e-2 on AIC;
                                        # negligible against the several-unit
                                        # spread of the simulated dAIC
B_BOOT   <- 120   # parametric-bootstrap replicates under Model 2'
B_REC    <- 60    # recovery-simulation replicates under Model 1'
SIM_SETS <- c("Per1", "Per2")   # sets that get the (slow) simulations
## Runtime: ~4 min for the fits, ~45 min including the simulations (each
## replicate refits both models, ~7 s). Set B_BOOT / B_REC to 0 to skip them.


################### 1 - data: one row per listener ###################

# Test-phase counts per listener per legality.
listener_counts <- function(data) {
  gen <- filter(data, gen == 0.5)
  by_leg <- gen %>%
    group_by(worker_ID, legality) %>%
    dplyr::summarize(k = sum(yesResponse), n = n(), .groups = "drop")
  legal   <- by_leg %>% filter(legality ==  0.5) %>%
    dplyr::select(worker_ID, kL = k, nL = n)
  illegal <- by_leg %>% filter(legality == -0.5) %>%
    dplyr::select(worker_ID, kI = k, nI = n)
  inner_join(legal, illegal, by = "worker_ID") %>%
    mutate(diff = kL / nL - kI / nI)
}

# Per-worker embedding distance; same construction as prep_data() in
# embedding_summary_table.R.
worker_avg_distance <- function(raw, sim_path) {
  sim <- read.csv(sim_path)
  sim$speaker1 <- sub("_[^_]+$", "", sim$S1)
  sim$speaker2 <- sub("_[^_]+$", "", sim$S2)
  sim$syl      <- sub("^.*_", "", as.character(sim$S1))
  sim_lookup_sym <- bind_rows(
    sim %>% dplyr::select(speaker1, speaker2, syl, distance),
    sim %>% dplyr::select(speaker1 = speaker2, speaker2 = speaker1, syl, distance))

  worker_speakers <- raw %>%
    mutate(speaker = sub("^study-|^test-(legal|illegal)-", "", as.character(label))) %>%
    distinct(worker_ID, speaker) %>%
    group_by(worker_ID) %>%
    arrange(speaker, .by_group = TRUE) %>%
    dplyr::summarize(speaker1 = first(speaker), speaker2 = nth(speaker, 2),
                     .groups = "drop")
  worker_syllables <- raw %>%
    mutate(syl = sub("^.*_", "", as.character(syllable))) %>%
    distinct(worker_ID, syl)

  worker_speakers %>%
    inner_join(worker_syllables, by = "worker_ID") %>%
    inner_join(sim_lookup_sym, by = c("speaker1", "speaker2", "syl")) %>%
    group_by(worker_ID) %>%
    dplyr::summarize(avg_distance = mean(distance), .groups = "drop")
}

per1_raw <- open_file("data/per1_filteredSubjects.csv", experiment = 1)
per2_raw <- open_file("data/per2_filteredSubjects.csv", experiment = 2)
dist1 <- worker_avg_distance(per1_raw,
                             "../../per_similarity_results/per1_speaker_pair_per_sim_full.csv")
dist2 <- worker_avg_distance(per2_raw,
                             "../../per_similarity_results/per2_speaker_pair_per_sim_full.csv")

build_set <- function(raw, dists) {
  listener_counts(raw) %>%
    inner_join(dists, by = "worker_ID") %>%
    mutate(d = as.numeric(scale(avg_distance)))   # z-scored within set
}

sets <- list(
  Per1  = build_set(per1_raw, dist1),
  Per1A = build_set(filter(per1_raw, condition_num <  5), dist1),
  Per1B = build_set(filter(per1_raw, condition_num >= 5), dist1),
  Per2  = build_set(per2_raw, dist2)
)


################### 2 - Gauss-Hermite quadrature ###################

# Nodes/weights for weight function exp(-x^2) via Golub-Welsch: the
# symmetric Jacobi matrix for Hermite polynomials has zero diagonal and
# off-diagonal sqrt(k/2); nodes are its eigenvalues and weights are
# sqrt(pi) * (first eigenvector component)^2. Done here rather than with
# statmod::gauss.quad so the script adds no dependency.
gauss_hermite <- function(Q) {
  k <- seq_len(Q - 1)
  J <- matrix(0, Q, Q)
  J[cbind(k, k + 1)] <- sqrt(k / 2)
  J[cbind(k + 1, k)] <- sqrt(k / 2)
  e   <- eigen(J, symmetric = TRUE)
  ord <- order(e$values)
  list(x = e$values[ord], w = sqrt(pi) * e$vectors[1, ord]^2)
}

# With these nodes,
#   integral phi(a; m, s) f(a) da  ~=  sum_q (w_q / sqrt(pi)) f(m + sqrt(2) s x_q)
# so the weights below already carry the 1/sqrt(pi); with f == 1 they sum to 1.
gh_cache <- new.env(parent = emptyenv())
gh_get <- function(Q) {
  key <- paste0("Q", Q)
  if (is.null(gh_cache[[key]])) {
    g <- gauss_hermite(Q)
    gh_cache[[key]] <- list(x = g$x, w = g$w / sqrt(pi))
  }
  gh_cache[[key]]
}
stopifnot(abs(sum(gh_get(20)$w) - 1) < 1e-12)          # weights normalized
stopifnot(abs(sum(gh_get(20)$w * (sqrt(2) * gh_get(20)$x)^2) - 1) < 1e-10)  # E[Z^2] = 1

# The count matrices do not depend on parameters, so build them once.
make_ws <- function(dat, Q) {
  n <- nrow(dat)
  list(n = n, Q = Q, d = dat$d,
       KL = matrix(dat$kL, n, Q), NL = matrix(dat$nL, n, Q),
       KI = matrix(dat$kI, n, Q), NI = matrix(dat$nI, n, Q))
}


################### 3 - the two likelihoods ###################

# ---- Model 1': two latent classes, mixture weight logistic in distance ----
# par = (mu_a, log sd_a, theta1, log(theta2 - theta1), g0, g1)
unpack_m1 <- function(par) {
  c(mu_a = par[1], sd_a = exp(par[2]),
    theta1 = par[3], theta2 = par[3] + exp(par[4]),
    g0 = par[5], g1 = par[6])
}

negll_m1 <- function(par, ws) {
  p  <- unpack_m1(par)
  gh <- gh_get(ws$Q)
  a  <- p[["mu_a"]] + sqrt(2) * p[["sd_a"]] * gh$x         # Q alpha nodes
  # class-conditional marginal likelihood per listener
  class_lik <- function(th) {
    PL <- matrix(plogis(a + th / 2), ws$n, ws$Q, byrow = TRUE)
    PI <- matrix(plogis(a - th / 2), ws$n, ws$Q, byrow = TRUE)
    as.vector((dbinom(ws$KL, ws$NL, PL) * dbinom(ws$KI, ws$NI, PI)) %*% gh$w)
  }
  L1 <- class_lik(p[["theta1"]])
  L2 <- class_lik(p[["theta2"]])
  w2 <- plogis(p[["g0"]] + p[["g1"]] * ws$d)
  -sum(log(pmax((1 - w2) * L1 + w2 * L2, 1e-300)))
}

# ---- Model 2': one population, legality effect Normal with mean in distance ----
# par = (mu_a, log sd_a, b0, b1, log sd_theta)
unpack_m2 <- function(par) {
  c(mu_a = par[1], sd_a = exp(par[2]),
    b0 = par[3], b1 = par[4], sd_theta = exp(par[5]))
}

# Two integrals. The theta nodes are listener-specific (their mean depends
# on d_i), so loop over the theta nodes and vectorize over listeners x alpha.
negll_m2 <- function(par, ws) {
  p  <- unpack_m2(par)
  gh <- gh_get(ws$Q)
  a  <- p[["mu_a"]] + sqrt(2) * p[["sd_a"]] * gh$x
  mu_th <- p[["b0"]] + p[["b1"]] * ws$d                    # n
  acc <- numeric(ws$n)
  for (r in seq_len(ws$Q)) {
    th <- mu_th + sqrt(2) * p[["sd_theta"]] * gh$x[r]      # n
    PL <- plogis(outer(th / 2,  a, "+"))                   # n x Q
    PI <- plogis(outer(-th / 2, a, "+"))
    acc <- acc + gh$w[r] *
      as.vector((dbinom(ws$KL, ws$NL, PL) * dbinom(ws$KI, ws$NI, PI)) %*% gh$w)
  }
  -sum(log(pmax(acc, 1e-300)))
}


################### 4 - fitting ###################

# Crude method-of-moments starts from the pooled rates.
start_values <- function(dat) {
  pL <- (sum(dat$kL) + 0.5) / (sum(dat$nL) + 1)
  pI <- (sum(dat$kI) + 0.5) / (sum(dat$nI) + 1)
  eff <- qlogis(pL) - qlogis(pI)                 # pooled legality effect, logits
  bias <- (qlogis(pL) + qlogis(pI)) / 2
  list(m1 = c(bias, log(0.7), 0, log(max(eff, 0.2) * 2), 0, 0),
       m2 = c(bias, log(0.7), eff, 0, log(0.7)))
}

# Multi-start ML. Mixture likelihoods are multimodal, so try several starts and
# keep the best; `polish` adds a Nelder-Mead pass. When `ws_hi` is supplied the
# optimum is located on the cheap grid `ws` and then refined and scored on the
# accurate grid `ws_hi`, which costs a handful of extra evaluations instead of
# running the whole search at the high node count.
fit_ml <- function(nll, start, ws, k, n_start = 1, jitter_sd = 0.4,
                   polish = FALSE, maxit = 1000, ws_hi = NULL) {
  best <- NULL
  for (i in seq_len(n_start)) {
    st <- if (i == 1) start else start + rnorm(length(start), 0, jitter_sd)
    f <- st
    if (polish) {
      f0 <- try(optim(st, nll, ws = ws, method = "Nelder-Mead",
                      control = list(maxit = 4000, reltol = 1e-10)), silent = TRUE)
      if (!inherits(f0, "try-error") && is.finite(f0$value)) f <- f0$par
    }
    fit <- try(optim(f, nll, ws = ws, method = "BFGS",
                     control = list(maxit = maxit)), silent = TRUE)
    if (inherits(fit, "try-error") || !is.finite(fit$value)) next
    if (is.null(best) || fit$value < best$value) best <- fit
  }
  if (is.null(best)) return(NULL)
  if (!is.null(ws_hi)) {
    ref <- try(optim(best$par, nll, ws = ws_hi, method = "BFGS",
                     control = list(maxit = 200)), silent = TRUE)
    if (!inherits(ref, "try-error") && is.finite(ref$value)) best <- ref
  }
  list(par = best$par, logLik = -best$value, k = k, conv = best$convergence)
}

aic <- function(f) -2 * f$logLik + 2 * f$k
bic <- function(f, n) -2 * f$logLik + log(n) * f$k

se_of <- function(nll, par, ws, idx) {
  H <- try(optimHess(par, nll, ws = ws), silent = TRUE)
  if (inherits(H, "try-error")) return(NA_real_)
  V <- try(solve(H), silent = TRUE)
  if (inherits(V, "try-error") || !all(is.finite(diag(V))) || diag(V)[idx] <= 0)
    return(NA_real_)
  sqrt(diag(V)[idx])
}


################### 5 - predicted distribution of the difference score ###################

# Exact predicted pmf over the achievable grid, by summing the joint
# Binomial x Binomial pmf over the quadrature nodes. This is what lets us
# check features the Gaussian models could not represent (the atom at 0).
pred_pmf <- function(par, model, d, nL, nI, Q = 40) {
  gh <- gh_get(Q)
  M  <- matrix(0, nL + 1, nI + 1)
  if (model == "m1") {
    p  <- unpack_m1(par)
    a  <- p[["mu_a"]] + sqrt(2) * p[["sd_a"]] * gh$x
    w2 <- plogis(p[["g0"]] + p[["g1"]] * d)
    for (cc in 1:2) {
      th <- if (cc == 1) p[["theta1"]] else p[["theta2"]]
      wc <- if (cc == 1) 1 - w2 else w2
      for (q in seq_along(a)) {
        M <- M + wc * gh$w[q] * outer(dbinom(0:nL, nL, plogis(a[q] + th / 2)),
                                      dbinom(0:nI, nI, plogis(a[q] - th / 2)))
      }
    }
  } else {
    p  <- unpack_m2(par)
    a  <- p[["mu_a"]] + sqrt(2) * p[["sd_a"]] * gh$x
    mu <- p[["b0"]] + p[["b1"]] * d
    for (r in seq_along(gh$x)) {
      th <- mu + sqrt(2) * p[["sd_theta"]] * gh$x[r]
      for (q in seq_along(a)) {
        M <- M + gh$w[r] * gh$w[q] *
          outer(dbinom(0:nL, nL, plogis(a[q] + th / 2)),
                dbinom(0:nI, nI, plogis(a[q] - th / 2)))
      }
    }
  }
  dv <- outer(0:nL / nL, 0:nI / nI, "-")
  agg <- tapply(as.vector(M), round(as.vector(dv), 8), sum)
  data.frame(diff = as.numeric(names(agg)), p = as.numeric(agg))
}

# Average the predicted pmf over the observed distances (the marginal
# prediction for the whole sample).
pred_pmf_avg <- function(par, model, dat, Q = 40) {
  ds <- dat %>% dplyr::count(d)
  out <- NULL
  for (i in seq_len(nrow(ds))) {
    pm <- pred_pmf(par, model, ds$d[i], dat$nL[1], dat$nI[1], Q = Q)
    pm$p <- pm$p * ds$n[i] / nrow(dat)
    out <- if (is.null(out)) pm else
      data.frame(diff = pm$diff, p = out$p + pm$p)
  }
  out
}

pp_checks <- function(pm) {
  c(p_zero = sum(pm$p[abs(pm$diff) < 1e-8]),
    p_neg  = sum(pm$p[pm$diff < -1e-8]),
    sd     = sqrt(sum(pm$p * pm$diff^2) - sum(pm$p * pm$diff)^2))
}


################### 6 - simulation helpers ###################

# Generate counts from Model 2' (continuous effect) at the observed distances.
sim_m2 <- function(par, dat) {
  p <- unpack_m2(par)
  n <- nrow(dat)
  al <- rnorm(n, p[["mu_a"]], p[["sd_a"]])
  th <- rnorm(n, p[["b0"]] + p[["b1"]] * dat$d, p[["sd_theta"]])
  dat %>% mutate(kL = rbinom(n, nL, plogis(al + th / 2)),
                 kI = rbinom(n, nI, plogis(al - th / 2)),
                 diff = kL / nL - kI / nI)
}

# Generate counts from Model 1' (two latent classes) at the observed distances.
sim_m1 <- function(par, dat) {
  p <- unpack_m1(par)
  n <- nrow(dat)
  al <- rnorm(n, p[["mu_a"]], p[["sd_a"]])
  cls <- rbinom(n, 1, plogis(p[["g0"]] + p[["g1"]] * dat$d))
  th  <- ifelse(cls == 1, p[["theta2"]], p[["theta1"]])
  dat %>% mutate(kL = rbinom(n, nL, plogis(al + th / 2)),
                 kI = rbinom(n, nI, plogis(al - th / 2)),
                 diff = kL / nL - kI / nI)
}

# Refit both models to one simulated data set with cheap settings and return
# the AIC / BIC differences (M1' - M2'; negative favors M1').
refit_diffs <- function(sdat, st) {
  ws1 <- make_ws(sdat, N_GH_M1_SIM)
  ws2 <- make_ws(sdat, N_GH_M2_SIM)
  f1 <- fit_ml(negll_m1, st$m1, ws1, k = 6L, n_start = 3, jitter_sd = 0.3, maxit = 400)
  f2 <- fit_ml(negll_m2, st$m2, ws2, k = 5L, n_start = 1, maxit = 400)
  if (is.null(f1) || is.null(f2)) return(c(dAIC = NA, dBIC = NA))
  n <- nrow(sdat)
  c(dAIC = aic(f1) - aic(f2), dBIC = bic(f1, n) - bic(f2, n))
}


################### 7 - fit, check, compare ###################

rows <- list(); fits <- list()

for (nm in names(sets)) {
  dat <- sets[[nm]]
  n   <- nrow(dat)
  st  <- start_values(dat)

  ws1    <- make_ws(dat, N_GH_M1_FIT); ws1_hi <- make_ws(dat, N_GH_M1_HI)
  ws2    <- make_ws(dat, N_GH_M2_FIT); ws2_hi <- make_ws(dat, N_GH_M2_HI)

  f1 <- fit_ml(negll_m1, st$m1, ws1, k = 6L, n_start = 8, polish = TRUE, ws_hi = ws1_hi)
  f2 <- fit_ml(negll_m2, st$m2, ws2, k = 5L, n_start = 4, polish = TRUE, ws_hi = ws2_hi)
  p1 <- unpack_m1(f1$par); p2 <- unpack_m2(f2$par)
  se_g1 <- se_of(negll_m1, f1$par, ws1_hi, 6)
  se_b1 <- se_of(negll_m2, f2$par, ws2_hi, 4)

  ## (b) quadrature precision: same parameters, twice the nodes. Refitting is
  ## unnecessary -- what matters is that integration error is orders of
  ## magnitude below the AIC/BIC differences being interpreted.
  err1 <- abs(negll_m1(f1$par, make_ws(dat, 2 * N_GH_M1_HI)) + f1$logLik)
  err2 <- abs(negll_m2(f2$par, make_ws(dat, 2 * N_GH_M2_HI)) + f2$logLik)

  ## (c) posterior-predictive checks on the difference score
  pm1 <- pred_pmf_avg(f1$par, "m1", dat)
  pm2 <- pred_pmf_avg(f2$par, "m2", dat)
  obs <- c(p_zero = mean(abs(dat$diff) < 1e-8),
           p_neg  = mean(dat$diff < -1e-8),
           sd     = sd(dat$diff))
  c1 <- pp_checks(pm1); c2 <- pp_checks(pm2)

  obs_dAIC <- aic(f1) - aic(f2)
  obs_dBIC <- bic(f1, n) - bic(f2, n)

  cat("\n=====================", nm, "=====================\n")
  cat(sprintf("N listeners = %d | items per legality: %d legal / %d illegal\n",
              n, dat$nL[1], dat$nI[1]))

  cat("\nModel 1'  two latent classes, distance-varying weight  (6 par)\n")
  cat(sprintf("  listener bias: mu_a = %.3f, sd_a = %.3f\n", p1[["mu_a"]], p1[["sd_a"]]))
  cat(sprintf("  class legality effects (logit): theta1 = %.3f, theta2 = %.3f\n",
              p1[["theta1"]], p1[["theta2"]]))
  cat(sprintf("  logit weight of higher-effect class: g0 = %.3f, g1 = %.3f%s\n",
              p1[["g0"]], p1[["g1"]],
              if (is.finite(se_g1))
                sprintf(" (SE %.3f, z = %.2f, p = %.4f)", se_g1, p1[["g1"]] / se_g1,
                        2 * pnorm(-abs(p1[["g1"]] / se_g1))) else " (SE not estimable)"))
  wq <- plogis(p1[["g0"]] + p1[["g1"]] * quantile(dat$d, c(0.1, 0.9), names = FALSE))
  cat(sprintf("  higher-class share at 10th pct distance = %.3f -> 90th pct = %.3f\n",
              wq[1], wq[2]))
  # avg_distance forms one tight cluster per talker pair, so the logistic class
  # probability can degenerate into a step function that simply indicates which
  # pair a listener heard. g1 is then unidentified (its SE explodes) and
  # "the higher-effect class becomes more common with distance" is the wrong
  # description of the fit. Same failure mode as in the Gaussian version.
  step_like <- abs(p1[["g1"]]) > 5 || (wq[1] < 0.02 && wq[2] > 0.98)
  if (step_like) {
    cat("  WARNING: class probability has collapsed to a step function of distance;\n")
    cat("           g1 is unidentified (it is indexing talker pair, not a gradient).\n")
  }
  cat(sprintf("  logLik = %.3f, AIC = %.2f, BIC = %.2f  (quadrature error %.2e)\n",
              f1$logLik, aic(f1), bic(f1, n), err1))

  cat("\nModel 2'  continuous effect, distance-varying mean  (5 par)\n")
  cat(sprintf("  listener bias: mu_a = %.3f, sd_a = %.3f\n", p2[["mu_a"]], p2[["sd_a"]]))
  cat(sprintf("  legality effect: b0 = %.3f, b1 = %.3f per SD of distance%s; sd_theta = %.3f\n",
              p2[["b0"]], p2[["b1"]],
              if (is.finite(se_b1))
                sprintf(" (SE %.3f, z = %.2f, p = %.4f)", se_b1, p2[["b1"]] / se_b1,
                        2 * pnorm(-abs(p2[["b1"]] / se_b1))) else "",
              p2[["sd_theta"]]))
  cat(sprintf("  logLik = %.3f, AIC = %.2f, BIC = %.2f  (quadrature error %.2e)\n",
              f2$logLik, aic(f2), bic(f2, n), err2))

  cat("\nPosterior-predictive checks on the difference score (observed / M1' / M2'):\n")
  cat(sprintf("  P(diff = 0) : %.3f / %.3f / %.3f\n", obs[["p_zero"]], c1[["p_zero"]], c2[["p_zero"]]))
  cat(sprintf("  P(diff < 0) : %.3f / %.3f / %.3f\n", obs[["p_neg"]],  c1[["p_neg"]],  c2[["p_neg"]]))
  cat(sprintf("  SD(diff)    : %.3f / %.3f / %.3f\n", obs[["sd"]],     c1[["sd"]],     c2[["sd"]]))

  cat(sprintf("\nAIC(M1') - AIC(M2') = %+.2f  ->  %s preferred (negative favors M1')\n",
              obs_dAIC, if (obs_dAIC < 0) "Model 1' (classes)" else "Model 2' (continuous)"))
  cat(sprintf("BIC(M1') - BIC(M2') = %+.2f\n", obs_dBIC))

  p_boot <- NA_real_; n_boot <- 0L
  rec_rate <- NA_real_; rec_rate_bic <- NA_real_; n_rec <- 0L
  if (nm %in% SIM_SETS && B_BOOT > 0) {
    ## (d) is the observed AIC edge larger than the mixture's extra flexibility
    ## buys when the CONTINUOUS model is true?
    bd <- replicate(B_BOOT, refit_diffs(sim_m2(f2$par, dat), st))
    bd <- bd[, is.finite(bd["dAIC", ]), drop = FALSE]
    n_boot <- ncol(bd); p_boot <- mean(bd["dAIC", ] <= obs_dAIC)
    cat(sprintf("\nParametric bootstrap under Model 2' (%d usable reps):\n", n_boot))
    cat(sprintf("  null dAIC: median %+.2f, 5th pct %+.2f\n",
                median(bd["dAIC", ]), quantile(bd["dAIC", ], 0.05, names = FALSE)))
    cat(sprintf("  p = %.3f  (small p => classes fit better than the continuous model explains)\n",
                p_boot))
  }
  if (nm %in% SIM_SETS && B_REC > 0) {
    ## (e) if a two-class world WERE true at these estimates, how often would
    ## this design detect it? Without this, "no evidence for classes" is
    ## uninterpretable given ~85% noise in the difference score.
    rd <- replicate(B_REC, refit_diffs(sim_m1(f1$par, dat), st))
    rd <- rd[, is.finite(rd["dAIC", ]), drop = FALSE]
    n_rec <- ncol(rd)
    rec_rate     <- mean(rd["dAIC", ] < 0)
    rec_rate_bic <- mean(rd["dBIC", ] < 0)
    cat(sprintf("\nRecovery under the fitted Model 1' (%d usable reps):\n", n_rec))
    cat(sprintf("  AIC picks M1' in %.0f%% of reps; BIC picks M1' in %.0f%%\n",
                100 * rec_rate, 100 * rec_rate_bic))
    cat(sprintf("  median dAIC = %+.2f, median dBIC = %+.2f\n",
                median(rd["dAIC", ]), median(rd["dBIC", ])))
    if (rec_rate < 0.8)
      cat("  NOTE: low recovery -- this design cannot reliably distinguish these\n",
          "        two accounts, so preferring M2' is weak evidence against classes.\n", sep = "")
  }

  rows[[nm]] <- data.frame(
    set = nm, n = n, n_items = dat$nL[1],
    mu_a = p1[["mu_a"]], sd_a_m1 = p1[["sd_a"]],
    theta1 = p1[["theta1"]], theta2 = p1[["theta2"]],
    g0 = p1[["g0"]], g1 = p1[["g1"]], g1_se = se_g1, g1_step_like = step_like,
    w_low = wq[1], w_high = wq[2],
    b0 = p2[["b0"]], b1 = p2[["b1"]], b1_se = se_b1,
    sd_theta = p2[["sd_theta"]], sd_a_m2 = p2[["sd_a"]],
    logLik_m1 = f1$logLik, logLik_m2 = f2$logLik,
    AIC_m1 = aic(f1), AIC_m2 = aic(f2), dAIC = obs_dAIC,
    BIC_m1 = bic(f1, n), BIC_m2 = bic(f2, n), dBIC = obs_dBIC,
    quad_err_m1 = err1, quad_err_m2 = err2,
    obs_p_zero = obs[["p_zero"]], m1_p_zero = c1[["p_zero"]], m2_p_zero = c2[["p_zero"]],
    obs_sd = obs[["sd"]], m1_sd = c1[["sd"]], m2_sd = c2[["sd"]],
    p_boot = p_boot, n_boot = n_boot,
    recovery_aic = rec_rate, recovery_bic = rec_rate_bic, n_rec = n_rec,
    row.names = NULL)
  fits[[nm]] <- list(m1 = f1, m2 = f2, data = dat)
}

summary_tbl <- bind_rows(rows)

cat("\n\n################ summary ################\n")
summary_tbl %>%
  transmute(set, n,
            AIC_M1 = round(AIC_m1, 1), AIC_M2 = round(AIC_m2, 1),
            dAIC = round(dAIC, 1), dBIC = round(dBIC, 1),
            theta1 = round(theta1, 2), theta2 = round(theta2, 2),
            g1 = round(g1, 2), b1 = round(b1, 3), sd_theta = round(sd_theta, 2),
            p_boot = round(p_boot, 3), recov_AIC = round(recovery_aic, 2),
            preferred = ifelse(dAIC < 0, "M1' classes", "M2' continuous")) %>%
  as.data.frame() %>%
  print(row.names = FALSE)

write.csv(summary_tbl, "model_results/bimodality_binomial_comparison.csv",
          row.names = FALSE)
cat("\nWrote model_results/bimodality_binomial_comparison.csv\n")


################### 8 - observed vs predicted pmf by distance tertile ###################

band_frames <- function(nm) {
  f <- fits[[nm]]; dat <- f$data
  brk <- quantile(dat$d, c(0, 1/3, 2/3, 1), names = FALSE)
  dat$band <- cut(dat$d, unique(brk), include.lowest = TRUE,
                  labels = c("low", "mid", "high")[seq_len(length(unique(brk)) - 1)])
  # Snap to the achievable grid before counting: kL/nL - kI/nI is not bit-identical
  # across (kL, kI) pairs with the same mathematical value, which would otherwise
  # split one bar into two overlapping ones.
  obs <- dat %>% mutate(diff = round(diff, 8)) %>% group_by(band) %>%
    dplyr::count(diff) %>% mutate(p = n / sum(n)) %>% ungroup() %>%
    mutate(set = nm)
  prd <- dat %>% group_by(band) %>%
    dplyr::summarize(dbar = mean(d), .groups = "drop") %>%
    rowwise() %>%
    do({
      b <- .$band; db <- .$dbar
      a1 <- pred_pmf(f$m1$par, "m1", db, dat$nL[1], dat$nI[1])
      a2 <- pred_pmf(f$m2$par, "m2", db, dat$nL[1], dat$nI[1])
      rbind(data.frame(band = b, diff = a1$diff, p = a1$p,
                       model = "M1': two latent classes"),
            data.frame(band = b, diff = a2$diff, p = a2$p,
                       model = "M2': continuous effect"))
    }) %>% ungroup() %>% mutate(set = nm)
  list(obs = obs, prd = prd)
}

plot_sets <- c("Per1", "Per2")
bf  <- lapply(plot_sets, band_frames)
lbl <- function(df) df %>%
  mutate(set = factor(set, levels = plot_sets,
                      labels = c("Experiment 1", "Experiment 2")),
         band = factor(band, levels = c("low", "mid", "high"),
                       labels = c("low distance", "mid distance", "high distance")))
obs_all <- lbl(bind_rows(lapply(bf, `[[`, "obs")))
prd_all <- lbl(bind_rows(lapply(bf, `[[`, "prd")))

p <- ggplot() +
  geom_col(data = obs_all, aes(x = diff, y = p), width = 0.04,
           fill = "grey78", color = "white") +
  geom_line(data = prd_all, aes(x = diff, y = p, color = model), linewidth = 0.7) +
  geom_point(data = prd_all, aes(x = diff, y = p, color = model), size = 0.7) +
  geom_vline(xintercept = 0, linetype = "dotted", color = "grey40") +
  facet_grid(set ~ band) +
  scale_color_manual(values = c("M1': two latent classes"  = "#b2182b",
                                "M2': continuous effect"   = "#2166ac")) +
  # The achievable support runs to +/-1, but no listener is anywhere near it;
  # clip to the occupied region so the comparison is legible.
  coord_cartesian(xlim = range(obs_all$diff) + c(-0.06, 0.06)) +
  labs(x = "Legality difference score (legal − illegal 'yes' rate)",
       y = "Probability of each achievable value", color = NULL) +
  theme_minimal(base_size = 11) +
  theme(legend.position = "bottom")

ggsave("plots/bimodality_binomial_fits.png", p, width = 10, height = 6, dpi = 150)
cat("Saved plots/bimodality_binomial_fits.png\n")
