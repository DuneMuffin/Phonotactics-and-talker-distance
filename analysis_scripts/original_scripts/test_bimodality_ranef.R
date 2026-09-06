############################################################
# Bimodality check on NOISE-SHRUNK per-subject legality effects.
#
# Instead of raw (legal - illegal) rates, which are dominated by
# 16-18-trial sampling noise, we fit the embedding mixed models and
# extract each subject's random legality slope (BLUP) from
# (1 + legality | worker_ID). These are the model's estimate of each
# subject's legality effect with sampling noise shrunk out.
#
# CAVEAT: BLUPs are conditional modes under a Gaussian random-effect
# prior and are shrunk toward a single mode, so they RESIST showing
# bimodality by construction. Bimodality surviving that shrinkage is
# strong evidence; its absence is partly assumed. We therefore also
# report how much reliable between-subject variance exists at all
# (the random-slope SD) and whether it exceeds a near-zero floor.
############################################################

suppressMessages({
  library(plyr); library(dplyr); library(lme4); library(diptest); library(mclust); library(ggplot2)
})
source("analysis_functions.R")
set.seed(1)

# ---- data prep (per-worker avg_distance), same as embedding_summary_table.R ----
prep_data <- function(file_path, sim_path, experiment) {
  raw <- open_file(file_path, experiment = experiment)
  gen <- filter(raw, gen == 0.5)
  if (experiment == 1) {
    for (col in c("label","syllable")) {
      gen[[col]] <- as.character(gen[[col]])
      gen[[col]] <- sub('french','FR', gen[[col]]); gen[[col]] <- sub('_F','_F1', gen[[col]])
      gen[[col]] <- sub('_M','_M1', gen[[col]]);    gen[[col]] <- sub('_F1I','_F2', gen[[col]])
      gen[[col]] <- sub('NS','EN', gen[[col]])
    }
    cn <- data.frame(condition_num = 1:7,
                     condition = c("weak-dif","strong-dif","native_shared","NN_shared",
                                   "NN_shared2","strong-dif2","weak-dif2"))
    gen <- merge(gen, cn)
    gen$xp_num <- ifelse(gen$condition_num < 5, 1, 2)
    gen$syllable_no_speaker <- substring(gen$syllable, 7, 9)
  }
  gen$response <- as.factor(gen$response)
  gen$worker_ID <- as.factor(gen$worker_ID)
  gen$syllable_no_speaker <- as.factor(gen$syllable_no_speaker)

  sim <- read.csv(sim_path)
  sim$speaker1 <- sub("_[^_]+$", "", sim$S1); sim$speaker2 <- sub("_[^_]+$", "", sim$S2)
  sim$syl <- sub("^.*_", "", as.character(sim$S1))
  sim_lookup_sym <- bind_rows(
    sim %>% select(speaker1, speaker2, syl, distance),
    sim %>% select(speaker1 = speaker2, speaker2 = speaker1, syl, distance))
  worker_speakers <- raw %>%
    mutate(speaker = sub("^study-|^test-(legal|illegal)-", "", as.character(label))) %>%
    distinct(worker_ID, speaker) %>% group_by(worker_ID) %>% arrange(speaker, .by_group = TRUE) %>%
    summarize(speaker1 = first(speaker), speaker2 = nth(speaker, 2), .groups = "drop") %>%
    mutate(worker_ID = as.factor(worker_ID))
  worker_syllables <- raw %>% mutate(syl = sub("^.*_", "", as.character(syllable))) %>%
    distinct(worker_ID, syl) %>% mutate(worker_ID = as.factor(worker_ID))
  worker_avg_dist <- worker_speakers %>%
    inner_join(worker_syllables, by = "worker_ID") %>%
    inner_join(sim_lookup_sym, by = c("speaker1", "speaker2", "syl")) %>%
    group_by(worker_ID, speaker1, speaker2) %>%
    summarize(avg_distance = mean(distance), .groups = "drop")
  gen %>% left_join(worker_avg_dist, by = "worker_ID")
}

per1 <- prep_data("data/per1_filteredSubjects.csv",
                  "../../per_similarity_results/per1_speaker_pair_per_sim_full.csv", 1)
per2 <- prep_data("data/per2_filteredSubjects.csv",
                  "../../per_similarity_results/per2_speaker_pair_per_sim_full.csv", 2)

datasets <- list(
  Per1A = filter(per1, xp_num == 1),
  Per1B = filter(per1, xp_num == 2),
  Per2  = per2
)

ctrl <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))

results <- list()
for (nm in names(datasets)) {
  d <- datasets[[nm]]
  d$avg_distance_c <- scale(d$avg_distance, scale = FALSE)[, 1]   # center within experiment

  m <- glmer(response ~ legality * avg_distance_c
             + (1 + legality | worker_ID)
             + (1 + legality | syllable_no_speaker),
             data = d, family = "binomial", control = ctrl)

  # Per-subject random legality slope (BLUP): deviation from the population
  # legality effect, on the logit scale, with sampling noise shrunk out.
  re    <- ranef(m)$worker_ID
  slope <- re[["legality"]]

  vc      <- as.data.frame(VarCorr(m))
  slope_sd <- vc$sdcor[vc$grp == "worker_ID" & vc$var1 == "legality" & is.na(vc$var2)]
  fixed_leg <- fixef(m)[["legality"]]

  dt  <- dip.test(slope)
  mc  <- Mclust(slope, G = 1:4, verbose = FALSE)
  bic <- sapply(1:4, function(g) { mm <- Mclust(slope, G = g, modelNames = "V", verbose = FALSE)
                                   if (is.null(mm)) NA else mm$bic })

  cat("\n=====================", nm, "=====================\n")
  cat(sprintf("N subjects = %d | fixed legality effect (logit) = %.3f\n", length(slope), fixed_leg))
  cat(sprintf("random legality-slope SD (reliable between-subject spread, logit) = %.3f\n", slope_sd))
  cat(sprintf("BLUP dip test: D = %.4f, p = %.3f  (continuous, no ties)\n", dt$statistic, dt$p.value))
  cat("BLUP GMM BIC by #components:\n"); print(round(setNames(bic, paste0("G=", 1:4)), 1))
  cat(sprintf("  best-BIC G = %d ; BIC(G=2)-BIC(G=1) = %.1f\n", mc$G, bic[2] - bic[1]))

  results[[nm]] <- data.frame(experiment = nm, worker_ID = rownames(re),
                              slope = slope, total_effect = fixed_leg + slope)
}

all_re <- bind_rows(results) %>%
  mutate(experiment = factor(experiment, levels = c("Per1A", "Per1B", "Per2")))

exp_colors <- c("Per1A" = "#1b9e77", "Per1B" = "#d95f02", "Per2" = "#7570b3")

means <- all_re %>% group_by(experiment) %>%
  dplyr::summarize(m = mean(total_effect), .groups = "drop")

p <- ggplot(all_re, aes(x = total_effect, fill = experiment, color = experiment)) +
  geom_histogram(aes(y = after_stat(density)), bins = 30, alpha = 0.55, color = "white") +
  geom_density(linewidth = 0.9) +
  geom_vline(xintercept = 0, linetype = "dotted", color = "grey40") +
  geom_vline(data = means, aes(xintercept = m, color = experiment),
             linetype = "dashed", linewidth = 0.8, show.legend = FALSE) +
  scale_fill_manual(values = exp_colors) + scale_color_manual(values = exp_colors) +
  facet_wrap(~ experiment, ncol = 1, scales = "free_y") +
  labs(x = "Per-subject legality effect (noise-shrunk BLUP, logit scale)",
       y = "Density",
       title = "Noise-shrunk per-subject legality effects (random slopes)",
       subtitle = "Population legality effect + subject random slope; BLUPs are Gaussian-shrunk") +
  theme_minimal(base_size = 12) + theme(legend.position = "none")

ggsave("plots/all_experiments_legality_ranef_dist.png", p, width = 8, height = 9, dpi = 150)
cat("\nSaved plots/all_experiments_legality_ranef_dist.png\n")
