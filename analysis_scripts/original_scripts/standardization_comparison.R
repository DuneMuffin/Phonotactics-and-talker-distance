############################################################
# Standardization sensitivity check.
#
# All reported results use PAIR ("mutual") standardization, the
# Chernyak et al. (2024) procedure (mut_normalize_sequences in
# github.com/bronichern/percept_sim): for each comparison the two
# frame sequences are concatenated, z-scored together per dimension,
# then split apart before DTW.
#
# GLOBAL standardization -- one set of per-dimension means/SDs over
# all frames from all talkers in an experiment -- is available via
# `process_audio_full.py --standardization global`. This script is
# the one place the two are compared, so the choice of the published
# procedure is documented rather than assumed.
#
# Writes a single table: model_results/standardization_comparison.csv
############################################################

library(plyr)
library(dplyr)
library(lme4)

if (requireNamespace("rstudioapi", quietly = TRUE) && rstudioapi::isAvailable()) {
  setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
}
source("analysis_functions.R")

sim_path <- function(exp, standardization) {
  sprintf("../../per_similarity_results/%s_speaker_pair_per_sim_full%s.csv",
          exp, if (standardization == "global") "_global" else "")
}


# ---- 1. Distance-level agreement between the two standardizations ----
# Merged on the (S1, S2) comparison, i.e. one row per speaker-pair x syllable.

distance_agreement <- function(exp) {
  a <- read.csv(sim_path(exp, "pair"))
  b <- read.csv(sim_path(exp, "global"))
  m <- merge(a[c("S1", "S2", "distance")],
             b[c("S1", "S2", "distance")],
             by = c("S1", "S2"), suffixes = c("_pair", "_global"))
  stopifnot(nrow(m) == nrow(a), nrow(a) == nrow(b))

  m$pair_id <- paste(sub("_[^_]+$", "", m$S1), sub("_[^_]+$", "", m$S2))
  by_pair <- m %>%
    group_by(pair_id) %>%
    summarize(mean_pair   = mean(distance_pair),
              mean_global = mean(distance_global), .groups = "drop")

  data.frame(
    n_rows       = nrow(m),
    n_pairs      = nrow(by_pair),
    dist_r       = round(cor(m$distance_pair, m$distance_global), 4),
    pair_mean_r  = round(cor(by_pair$mean_pair, by_pair$mean_global), 4),
    pair_rho     = round(cor(by_pair$mean_pair, by_pair$mean_global, method = "spearman"), 4),
    range_ratio  = round(diff(range(by_pair$mean_global)) /
                         diff(range(by_pair$mean_pair)), 3))
}


# ---- 2. Data prep (same construction as embedding_summary_table.R) ----

prep_data <- function(file_path, experiment) {
  raw <- open_file(file_path, experiment = experiment)
  gen <- filter(raw, gen == 0.5)

  if (experiment == 1) {
    for (col in c("label", "syllable")) {
      v <- as.character(gen[[col]])
      v <- sub('french', 'FR', v); v <- sub('_F', '_F1', v)
      v <- sub('_M', '_M1', v);    v <- sub('_F1I', '_F2', v)
      gen[[col]] <- sub('NS', 'EN', v)
    }
    cn <- data.frame(condition_num = c(1, 2, 3, 4, 5, 6, 7),
                     condition = c("weak-dif", "strong-dif", "native_shared",
                                   "NN_shared", "NN_shared2", "strong-dif2", "weak-dif2"))
    gen <- merge(gen, cn)
    gen$xp_num <- ifelse(gen$condition_num < 5, 1, 2)
    gen$syllable_no_speaker <- substring(gen$syllable, 7, 9)
  }

  gen$response  <- as.factor(gen$response)
  gen$worker_ID <- as.factor(gen$worker_ID)
  gen$syllable_no_speaker <- as.factor(gen$syllable_no_speaker)
  list(gen = gen, raw = raw)
}

# Per-worker avg_distance: their two speakers x every syllable they heard
# (fam + test), averaged. Identical under both standardizations; only the
# distance table swaps.
attach_distance <- function(prepped, path) {
  gen <- prepped$gen
  raw <- prepped$raw

  sim <- read.csv(path)
  sim$speaker1 <- sub("_[^_]+$", "", sim$S1)
  sim$speaker2 <- sub("_[^_]+$", "", sim$S2)
  sim$syl      <- sub("^.*_", "", as.character(sim$S1))
  sim_lookup_sym <- bind_rows(
    sim %>% select(speaker1, speaker2, syl, distance),
    sim %>% select(speaker1 = speaker2, speaker2 = speaker1, syl, distance))

  worker_speakers <- raw %>%
    mutate(speaker = sub("^study-|^test-(legal|illegal)-", "", as.character(label))) %>%
    distinct(worker_ID, speaker) %>%
    group_by(worker_ID) %>%
    arrange(speaker, .by_group = TRUE) %>%
    summarize(speaker1 = first(speaker), speaker2 = nth(speaker, 2), .groups = "drop") %>%
    mutate(worker_ID = as.factor(worker_ID))

  worker_syllables <- raw %>%
    mutate(syl = sub("^.*_", "", as.character(syllable))) %>%
    distinct(worker_ID, syl) %>%
    mutate(worker_ID = as.factor(worker_ID))

  worker_avg_dist <- worker_speakers %>%
    inner_join(worker_syllables, by = "worker_ID") %>%
    inner_join(sim_lookup_sym, by = c("speaker1", "speaker2", "syl")) %>%
    group_by(worker_ID, speaker1, speaker2) %>%
    summarize(avg_distance = mean(distance), .groups = "drop")

  gen <- gen %>% left_join(worker_avg_dist, by = "worker_ID")
  gen$avg_distance_c <- scale(gen$avg_distance, scale = FALSE)[, 1]
  gen
}


# ---- 3. Fit the embedding model, pull the legality x distance interaction ----

fit_interaction <- function(data) {
  m <- glmer(response ~ legality * avg_distance_c
             + (1 + legality | worker_ID)
             + (1 + legality | syllable_no_speaker),
             data = data, family = "binomial",
             control = glmerControl(optimizer = "bobyqa"))
  cf <- summary(m)$coefficients["legality:avg_distance_c", ]
  list(beta = cf["Estimate"], SE = cf["Std. Error"],
       z = cf["z value"], p = cf["Pr(>|z|)"], AIC = AIC(m), n_obs = nobs(m))
}


# ---- 4. Run ----

per1_prep <- prep_data("data/per1_filteredSubjects.csv", experiment = 1)
per2_prep <- prep_data("data/per2_filteredSubjects.csv", experiment = 2)

agreement <- list(per1 = distance_agreement("per1"),
                  per2 = distance_agreement("per2"))

datasets <- list()
for (std in c("pair", "global")) {
  p1 <- attach_distance(per1_prep, sim_path("per1", std))
  p2 <- attach_distance(per2_prep, sim_path("per2", std))
  datasets[[std]] <- list(`Per1 (combined 1A+1B)` = p1,
                          `Per1A` = filter(p1, xp_num == 1),
                          `Per1B` = filter(p1, xp_num == 2),
                          `Per2`  = p2)
}

parent_exp <- c(`Per1 (combined 1A+1B)` = "per1", `Per1A` = "per1",
                `Per1B` = "per1", `Per2` = "per2")

results <- bind_rows(lapply(names(parent_exp), function(label) {
  fp <- fit_interaction(datasets$pair[[label]])
  fg <- fit_interaction(datasets$global[[label]])
  ag <- agreement[[parent_exp[[label]]]]
  data.frame(
    experiment       = label,
    n_obs            = fp$n_obs,
    # agreement between the two distance tables (experiment level)
    dist_r           = ag$dist_r,
    pair_mean_r      = ag$pair_mean_r,
    pair_rho         = ag$pair_rho,
    range_ratio      = ag$range_ratio,
    # legality x avg_distance_c interaction under each standardization
    beta_pair        = round(fp$beta, 4),
    beta_global      = round(fg$beta, 4),
    beta_diff        = round(fg$beta - fp$beta, 4),
    SE_pair          = round(fp$SE, 4),
    SE_global        = round(fg$SE, 4),
    z_pair           = round(fp$z, 3),
    z_global         = round(fg$z, 3),
    p_pair           = signif(fp$p, 3),
    p_global         = signif(fg$p, 3),
    AIC_pair         = round(fp$AIC, 2),
    AIC_global       = round(fg$AIC, 2),
    AIC_diff         = round(fg$AIC - fp$AIC, 2))
}))

print(results, row.names = FALSE)
write.csv(results, "model_results/standardization_comparison.csv", row.names = FALSE)
cat("\nWritten to model_results/standardization_comparison.csv\n")
