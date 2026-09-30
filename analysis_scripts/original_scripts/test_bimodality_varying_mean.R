############################################################
# Two competing accounts of how the per-listener legality
# difference-score distribution changes with talker distance.
#
# Each listener contributes one difference score (legal minus
# illegal "yes" rate on generalization trials) and one
# avg_distance (mean per-syllable HuBERT/DTW distance for the
# talker pair they heard, over the syllables they heard). The
# distance predictor is z-scored WITHIN experiment, since Per1
# and Per2 distances live on different scales.
#
#   MODEL 1 ("shifting mixture weight"): the population is a
#     mixture of two latent listener types with FIXED component
#     means mu1 < mu2. Talker distance changes only the mixing
#     proportion -- the relative height of the higher-mean mode:
#       y_i ~ (1 - w_i) N(mu1, s1) + w_i N(mu2, s2)
#       logit(w_i) = g0 + g1 * d_i
#     A positive g1 means the higher mode gets taller as talker
#     distance grows.
#
#   MODEL 2 ("shifting unimodal mean"): at every level of talker
#     distance the distribution is a single unimodal Gaussian
#     whose MEAN moves with distance:
#       y_i ~ N(b0 + b1 * d_i, s)
#     This is the distributional analogue of the
#     legality x avg_distance_c interaction in the glmer fits.
#
# The two are NOT nested in either direction (Model 1 with
# g1 = 0 is a static two-component mixture; Model 2 with b1 = 0
# is a single Gaussian), so we compare them by AIC/BIC and then
# calibrate the AIC difference with a parametric bootstrap under
# Model 2 -- how often does the extra mixture flexibility win by
# this much when the unimodal account is TRUE?
#
# CAVEATS
#   * Difference scores are discrete: multiples of 1/18 (Per1) or
#     1/16 (Per2), and mostly binomial sampling noise around a
#     much smaller true effect (see test_bimodality.R). Gaussian
#     likelihoods are therefore an approximation. Both models make
#     the same approximation, so the comparison is fair, but the
#     absolute fit should not be over-read.
#   * To keep the mixture from chasing that grid, component SDs are
#     floored at half a quantum (0.5/n_items); a component narrower
#     than that is fitting the discreteness, not a subpopulation.
#   * Distance is confounded with talker identity, especially in
#     Per1, where the six pairs are well separated. Neither model
#     can tell "farther talkers" from "these particular talkers".
############################################################

suppressMessages({
  library(plyr); library(dplyr); library(ggplot2)
})
source("analysis_functions.R")
set.seed(1)

N_BOOT <- 500   # parametric-bootstrap replicates for the AIC calibration


################### 1 - data: one row per listener ###################

# legal - illegal "yes" rate on generalization trials, one row per worker.
worker_legality_diff <- function(data) {
  gen <- filter(data, gen == 0.5)
  by_leg <- gen %>%
    group_by(worker_ID, legality) %>%
    dplyr::summarize(yes_rate = mean(yesResponse), .groups = "drop")
  legal   <- by_leg %>% filter(legality ==  0.5) %>%
    dplyr::select(worker_ID, legal_rate   = yes_rate)
  illegal <- by_leg %>% filter(legality == -0.5) %>%
    dplyr::select(worker_ID, illegal_rate = yes_rate)
  inner_join(legal, illegal, by = "worker_ID") %>%
    mutate(diff = legal_rate - illegal_rate)
}

# Per-worker embedding distance; same construction as prep_data() in
# embedding_summary_table.R (two talkers per worker, averaged over the
# syllables that worker actually heard).
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

# Per1 splits into 1A (condition_num 1-4) and 1B (5-7); both sub-experiments and
# the pooled Experiment 1 are fit, so the result can be read either way.
build_set <- function(raw, dists) {
  worker_legality_diff(raw) %>%
    inner_join(dists, by = "worker_ID") %>%
    mutate(d = as.numeric(scale(avg_distance)))   # z-scored within this set
}

sets <- list(
  Per1  = build_set(per1_raw, dist1),
  Per1A = build_set(filter(per1_raw, condition_num <  5), dist1),
  Per1B = build_set(filter(per1_raw, condition_num >= 5), dist1),
  Per2  = build_set(per2_raw, dist2)
)
# Test items per listener per legality -> the quantum of the difference score.
set_n_items <- c("Per1" = 18, "Per1A" = 18, "Per1B" = 18, "Per2" = 16)


################### 2 - the two likelihoods ###################

# ---- Model 2: single Gaussian, mean linear in distance. Closed-form MLE. ----
fit_m2 <- function(y, d) {
  X   <- cbind(1, d)
  b   <- qr.solve(X, y)
  res <- y - X %*% b
  s   <- sqrt(mean(res^2))              # ML (not REML) sigma
  list(par = c(b0 = b[1], b1 = b[2], sigma = s),
       logLik = sum(dnorm(y, X %*% b, s, log = TRUE)),
       k = 3L)
}

# ---- Model 1: two fixed components, mixing weight logistic in distance. ----
# Parameterization keeps the fit identified and away from degenerate spikes:
#   mu1 = par[1]
#   mu2 = mu1 + exp(par[2])            -> mu2 > mu1, so "component 2" is always
#                                          the higher-mean mode and g1's sign is
#                                          interpretable
#   s_j = sd_floor + exp(par[.])       -> SDs bounded below by half a quantum
#   logit(w) = g0 + g1 * d
negll_m1 <- function(par, y, d, sd_floor, common_sd) {
  mu1 <- par[1]
  mu2 <- mu1 + exp(par[2])
  if (common_sd) {
    s1 <- s2 <- sd_floor + exp(par[3])
    g0 <- par[4]; g1 <- par[5]
  } else {
    s1 <- sd_floor + exp(par[3])
    s2 <- sd_floor + exp(par[4])
    g0 <- par[5]; g1 <- par[6]
  }
  w    <- plogis(g0 + g1 * d)
  dens <- (1 - w) * dnorm(y, mu1, s1) + w * dnorm(y, mu2, s2)
  -sum(log(pmax(dens, 1e-300)))
}

unpack_m1 <- function(par, sd_floor, common_sd) {
  mu1 <- par[1]; mu2 <- mu1 + exp(par[2])
  if (common_sd) {
    s1 <- s2 <- sd_floor + exp(par[3]); g0 <- par[4]; g1 <- par[5]
  } else {
    s1 <- sd_floor + exp(par[3]); s2 <- sd_floor + exp(par[4]); g0 <- par[5]; g1 <- par[6]
  }
  c(mu1 = mu1, mu2 = mu2, sd1 = s1, sd2 = s2, g0 = g0, g1 = g1)
}

# Multi-start ML fit; mixture likelihoods are multimodal, so try several starts
# and keep the best.
fit_m1 <- function(y, d, sd_floor, common_sd = FALSE, n_start = 12, hessian = FALSE) {
  qs   <- quantile(y, c(0.2, 0.5, 0.8), names = FALSE)
  spread <- max(sd(y), 1e-3)
  base_sd <- max(spread / 2 - sd_floor, 1e-3)
  starts <- list()
  for (i in seq_len(n_start)) {
    jit <- if (i == 1) 0 else 1
    mu1 <- qs[1] + jit * rnorm(1, 0, spread / 4)
    gap <- max(qs[3] - qs[1], spread / 2) * (if (i == 1) 1 else exp(rnorm(1, 0, 0.4)))
    sdp <- log(base_sd) + jit * rnorm(1, 0, 0.4)
    core <- if (common_sd) c(sdp) else c(sdp, sdp + jit * rnorm(1, 0, 0.3))
    starts[[i]] <- c(mu1, log(max(gap, 1e-3)), core,
                     jit * rnorm(1, 0, 0.5), jit * rnorm(1, 0, 0.5))
  }
  best <- NULL
  for (st in starts) {
    fit <- try(optim(st, negll_m1, y = y, d = d, sd_floor = sd_floor,
                     common_sd = common_sd, method = "Nelder-Mead",
                     control = list(maxit = 5000, reltol = 1e-10)), silent = TRUE)
    if (inherits(fit, "try-error") || !is.finite(fit$value)) next
    fit <- try(optim(fit$par, negll_m1, y = y, d = d, sd_floor = sd_floor,
                     common_sd = common_sd, method = "BFGS",
                     control = list(maxit = 2000)), silent = TRUE)
    if (inherits(fit, "try-error") || !is.finite(fit$value)) next
    if (is.null(best) || fit$value < best$value) best <- fit
  }
  if (is.null(best)) return(NULL)
  k  <- if (common_sd) 5L else 6L
  se_g1 <- NA_real_
  if (hessian) {
    H <- try(optimHess(best$par, negll_m1, y = y, d = d, sd_floor = sd_floor,
                       common_sd = common_sd), silent = TRUE)
    if (!inherits(H, "try-error")) {
      V <- try(solve(H), silent = TRUE)
      if (!inherits(V, "try-error") && all(is.finite(diag(V))) && diag(V)[k] > 0) {
        se_g1 <- sqrt(diag(V)[k])   # g1 is the last parameter, untransformed
      }
    }
  }
  list(par = unpack_m1(best$par, sd_floor, common_sd),
       raw_par = best$par, logLik = -best$value, k = k, se_g1 = se_g1)
}

aic <- function(f) -2 * f$logLik + 2 * f$k
bic <- function(f, n) -2 * f$logLik + log(n) * f$k


################### 3 - fit both models in every data set ###################

rows  <- list()
fits  <- list()

for (nm in names(sets)) {
  dat <- sets[[nm]]
  y   <- dat$diff
  d   <- dat$d
  n   <- length(y)
  sd_floor <- 0.5 / set_n_items[[nm]]

  m2 <- fit_m2(y, d)
  m1 <- fit_m1(y, d, sd_floor, common_sd = FALSE, hessian = TRUE)
  m1c <- fit_m1(y, d, sd_floor, common_sd = TRUE)

  # Parametric bootstrap: generate data from the FITTED Model 2 (unimodal,
  # distance-shifted mean) keeping the observed distances, refit both, and see
  # how extreme the observed AIC difference is under that null.
  b0 <- m2$par[[1]]; b1 <- m2$par[[2]]; s_m2 <- m2$par[[3]]
  obs_daic <- aic(m1) - aic(m2)
  boot_daic <- replicate(N_BOOT, {
    ystar <- rnorm(n, b0 + b1 * d, s_m2)
    f2 <- fit_m2(ystar, d)
    f1 <- fit_m1(ystar, d, sd_floor, common_sd = FALSE, n_start = 4)
    if (is.null(f1)) NA_real_ else aic(f1) - aic(f2)
  })
  boot_daic <- boot_daic[is.finite(boot_daic)]
  p_boot <- mean(boot_daic <= obs_daic)

  # Weight of the higher-mean mode at low / high talker distance.
  w_at <- plogis(m1$par[["g0"]] + m1$par[["g1"]] * quantile(d, c(0.1, 0.9), names = FALSE))

  cat("\n=====================", nm, "=====================\n")
  cat(sprintf("N listeners = %d | difference-score quantum = 1/%d | SD floor = %.4f\n",
              n, set_n_items[[nm]], sd_floor))
  cat(sprintf("avg_distance: mean = %.3f, sd = %.3f (predictor z-scored within set)\n",
              mean(dat$avg_distance), sd(dat$avg_distance)))

  cat("\nModel 1  two components, distance-varying mixture weight\n")
  cat(sprintf("  mu1 = %.4f (sd %.4f) ; mu2 = %.4f (sd %.4f)\n",
              m1$par[["mu1"]], m1$par[["sd1"]], m1$par[["mu2"]], m1$par[["sd2"]]))
  cat(sprintf("  logit weight of higher mode: g0 = %.3f, g1 = %.3f%s\n",
              m1$par[["g0"]], m1$par[["g1"]],
              if (is.finite(m1$se_g1))
                sprintf(" (SE %.3f, z = %.2f, p = %.4f)",
                        m1$se_g1, m1$par[["g1"]] / m1$se_g1,
                        2 * pnorm(-abs(m1$par[["g1"]] / m1$se_g1)))
              else " (SE not estimable)"))
  cat(sprintf("  higher-mode weight at 10th pct distance = %.3f -> 90th pct = %.3f\n",
              w_at[1], w_at[2]))
  # avg_distance takes only a handful of well-separated values (one cluster per
  # talker pair), so the logistic weight can degenerate into a step function
  # that simply indicates which pair a listener heard. When that happens g1 is
  # not identified and its SE explodes; the fit is still a valid likelihood but
  # "weight increases with distance" is no longer the right description of it.
  if (abs(m1$par[["g1"]]) > 10 || (w_at[1] < 0.01 && w_at[2] > 0.99)) {
    cat("  WARNING: mixture weight has collapsed to a step function of distance;\n")
    cat("           g1 is unidentified (it is indexing talker pair, not a gradient).\n")
  }
  cat(sprintf("  logLik = %.2f, k = %d, AIC = %.2f, BIC = %.2f\n",
              m1$logLik, m1$k, aic(m1), bic(m1, n)))
  cat(sprintf("  [equal-variance variant: k = %d, AIC = %.2f, BIC = %.2f]\n",
              m1c$k, aic(m1c), bic(m1c, n)))

  cat("\nModel 2  single unimodal Gaussian, distance-varying mean\n")
  cat(sprintf("  b0 = %.4f, b1 = %.4f per SD of distance, sigma = %.4f\n",
              m2$par[[1]], m2$par[[2]], m2$par[[3]]))
  cat(sprintf("  logLik = %.2f, k = %d, AIC = %.2f, BIC = %.2f\n",
              m2$logLik, m2$k, aic(m2), bic(m2, n)))

  better <- if (aic(m1) < aic(m2)) "Model 1 (mixture)" else "Model 2 (unimodal)"
  cat(sprintf("\nAIC(M1) - AIC(M2) = %+.2f  ->  %s preferred (negative favors M1)\n",
              obs_daic, better))
  cat(sprintf("BIC(M1) - BIC(M2) = %+.2f\n", bic(m1, n) - bic(m2, n)))
  cat(sprintf("Parametric bootstrap under Model 2 (%d usable reps):\n", length(boot_daic)))
  cat(sprintf("  null AIC difference: median %+.2f, 5th pct %+.2f\n",
              median(boot_daic), quantile(boot_daic, 0.05, names = FALSE)))
  cat(sprintf("  p = %.3f  (small p => the mixture fits better than unimodal noise explains)\n",
              p_boot))

  rows[[nm]] <- data.frame(
    set = nm, n = n,
    m1_mu1 = m1$par[["mu1"]], m1_mu2 = m1$par[["mu2"]],
    m1_sd1 = m1$par[["sd1"]], m1_sd2 = m1$par[["sd2"]],
    m1_g0 = m1$par[["g0"]], m1_g1 = m1$par[["g1"]], m1_g1_se = m1$se_g1,
    w_low = w_at[1], w_high = w_at[2],
    m2_b0 = m2$par[[1]], m2_b1 = m2$par[[2]], m2_sigma = m2$par[[3]],
    logLik_m1 = m1$logLik, logLik_m2 = m2$logLik,
    AIC_m1 = aic(m1), AIC_m2 = aic(m2), dAIC = obs_daic,
    BIC_m1 = bic(m1, n), BIC_m2 = bic(m2, n), dBIC = bic(m1, n) - bic(m2, n),
    AIC_m1_equalvar = aic(m1c), p_boot = p_boot,
    row.names = NULL)
  fits[[nm]] <- list(m1 = m1, m2 = m2, data = dat)
}

summary_tbl <- bind_rows(rows)

cat("\n\n################ summary ################\n")
summary_tbl %>%
  transmute(set, n,
            AIC_M1 = round(AIC_m1, 1), AIC_M2 = round(AIC_m2, 1),
            dAIC = round(dAIC, 1), dBIC = round(dBIC, 1),
            g1 = round(m1_g1, 3), w_low = round(w_low, 2), w_high = round(w_high, 2),
            b1 = round(m2_b1, 3), p_boot = round(p_boot, 3),
            preferred = ifelse(dAIC < 0, "M1 mixture", "M2 unimodal")) %>%
  as.data.frame() %>%
  print(row.names = FALSE)

write.csv(summary_tbl, "model_results/bimodality_varying_mean_comparison.csv",
          row.names = FALSE)
cat("\nWrote model_results/bimodality_varying_mean_comparison.csv\n")


################### 4 - fitted densities by distance tertile ###################

# The diagnostic that matters: within a narrow band of talker distance, does the
# distribution look like one hump that has moved, or like two fixed humps with
# shifted relative height?
tertile_curves <- function(nm) {
  f   <- fits[[nm]]
  dat <- f$data
  n_items <- set_n_items[[nm]]
  brk <- quantile(dat$d, c(0, 1/3, 2/3, 1), names = FALSE)
  dat$band <- cut(dat$d, unique(brk), include.lowest = TRUE,
                  labels = c("low", "mid", "high")[seq_len(length(unique(brk)) - 1)])
  xs <- seq(min(dat$diff) - 0.1, max(dat$diff) + 0.1, length.out = 400)
  # Bars on each experiment's own achievable grid (1/18 vs 1/16), scaled to
  # density so they are on the same axis as the fitted curves.
  bars <- dat %>%
    mutate(grid = round(diff * n_items) / n_items) %>%
    dplyr::count(band, grid) %>%
    group_by(band) %>%
    mutate(dens = n / sum(n) * n_items, width = 1 / n_items) %>%
    ungroup()
  curves <- dat %>% group_by(band) %>%
    dplyr::summarize(dbar = mean(d), .groups = "drop") %>%
    rowwise() %>%
    do({
      b <- .$band; dbar <- .$dbar
      w <- plogis(f$m1$par[["g0"]] + f$m1$par[["g1"]] * dbar)
      d1 <- (1 - w) * dnorm(xs, f$m1$par[["mu1"]], f$m1$par[["sd1"]]) +
        w * dnorm(xs, f$m1$par[["mu2"]], f$m1$par[["sd2"]])
      d2 <- dnorm(xs, f$m2$par[[1]] + f$m2$par[[2]] * dbar, f$m2$par[[3]])
      data.frame(band = b, x = rep(xs, 2), dens = c(d1, d2),
                 model = rep(c("M1: two components, shifting weight",
                               "M2: unimodal, shifting mean"), each = length(xs)))
    }) %>% ungroup()
  list(bars = mutate(bars, set = nm), curves = mutate(curves, set = nm))
}

plot_sets <- c("Per1", "Per2")
tc <- lapply(plot_sets, tertile_curves)
bars   <- bind_rows(lapply(tc, `[[`, "bars")) %>%
  mutate(set = factor(set, levels = plot_sets, labels = c("Experiment 1", "Experiment 2")),
         band = factor(band, levels = c("low", "mid", "high"),
                       labels = c("low distance", "mid distance", "high distance")))
curves <- bind_rows(lapply(tc, `[[`, "curves")) %>%
  mutate(set = factor(set, levels = plot_sets, labels = c("Experiment 1", "Experiment 2")),
         band = factor(band, levels = c("low", "mid", "high"),
                       labels = c("low distance", "mid distance", "high distance")))

p <- ggplot() +
  geom_col(data = bars, aes(x = grid, y = dens, width = width),
           fill = "grey75", color = "white") +
  geom_line(data = curves, aes(x = x, y = dens, color = model), linewidth = 0.8) +
  geom_vline(xintercept = 0, linetype = "dotted", color = "grey40") +
  facet_grid(set ~ band) +
  scale_color_manual(values = c("M1: two components, shifting weight" = "#b2182b",
                                "M2: unimodal, shifting mean"         = "#2166ac")) +
  labs(x = "Legality difference score (legal − illegal 'yes' rate)",
       y = "Density", color = NULL) +
  theme_minimal(base_size = 11) +
  theme(legend.position = "bottom")

ggsave("plots/bimodality_varying_mean_fits.png", p, width = 10, height = 6, dpi = 150)
cat("Saved plots/bimodality_varying_mean_fits.png\n")
