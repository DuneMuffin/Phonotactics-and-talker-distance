############################################################
# 1:1 embedding vs. categorical comparison.
# Gradient model:    response ~ legality * avg_distance_c
# Categorical model: response ~ legality * diffVshared
# Per1 = combined 1A+1B; Per2 unchanged.
############################################################

library(plyr)
library(dplyr)
library(lme4)
library(stringr)

if (requireNamespace("rstudioapi", quietly = TRUE) && rstudioapi::isAvailable()) {
  setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
}
source("analysis_functions.R")


# ---- Helper: prep gen data with per-worker avg_distance ----
prep_data <- function(file_path, sim_path, experiment) {
  raw <- open_file(file_path, experiment = experiment)
  gen <- filter(raw, gen == 0.5)

  if (experiment == 1) {
    gen$label    <- as.character(gen$label)
    gen$label    <- sub('french','FR', gen$label)
    gen$label    <- sub('_F','_F1',  gen$label)
    gen$label    <- sub('_M','_M1',  gen$label)
    gen$label    <- sub('_F1I','_F2',gen$label)
    gen$label    <- sub('NS','EN',   gen$label)
    gen$syllable <- as.character(gen$syllable)
    gen$syllable <- sub('french','FR', gen$syllable)
    gen$syllable <- sub('_F','_F1',  gen$syllable)
    gen$syllable <- sub('_M','_M1',  gen$syllable)
    gen$syllable <- sub('_F1I','_F2',gen$syllable)
    gen$syllable <- sub('NS','EN',   gen$syllable)

    cn <- data.frame(c(1,2,3,4,5,6,7),
                     c("weak-dif","strong-dif","native_shared","NN_shared",
                       "NN_shared2","strong-dif2","weak-dif2"))
    colnames(cn) <- c("condition_num","condition")
    gen <- merge(gen, cn)
    gen$xp_num <- ifelse(gen$condition_num < 5, 1, 2)
    gen$syllable_no_speaker <- substring(gen$syllable, 7, 9)
  } else {
    cn <- data.frame(condition_num = c(1, 2, 3),
                     condition = c("mixed-different",
                                   "non-native-different",
                                   "non-native-shared"))
    gen <- merge(gen, cn)
  }

  gen$response   <- as.factor(gen$response)
  gen$worker_ID  <- as.factor(gen$worker_ID)
  gen$syllable_no_speaker <- as.factor(gen$syllable_no_speaker)

  sim <- read.csv(sim_path)
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
    summarize(speaker1 = first(speaker),
              speaker2 = nth(speaker, 2),
              .groups = "drop") %>%
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


# ---- Helper: fit 1:1 gradient vs categorical models ----
# Both formulas have exactly one interaction term with legality.
fit_and_extract <- function(data, label, diff_var) {
  rand_eff <- "(1 + legality | worker_ID) + (1 + legality | syllable_no_speaker)"

  emb <- glmer(as.formula(paste("response ~ legality * avg_distance_c +", rand_eff)),
               data = data, family = "binomial",
               control = glmerControl(optimizer = "bobyqa"))

  cond <- glmer(as.formula(paste("response ~ legality *", diff_var, "+", rand_eff)),
                data = data, family = "binomial",
                control = glmerControl(optimizer = "bobyqa"))

  cf_e <- summary(emb)$coefficients
  cf_c <- summary(cond)$coefficients
  row_e <- "legality:avg_distance_c"
  row_c <- grep(":", rownames(cf_c), value = TRUE)[1]

  data.frame(experiment    = label,
             n_obs         = nobs(emb),
             n_workers     = length(unique(data$worker_ID)),
             beta_emb      = round(cf_e[row_e, "Estimate"],   4),
             SE_emb        = round(cf_e[row_e, "Std. Error"], 4),
             z_emb         = round(cf_e[row_e, "z value"],    3),
             p_emb         = signif(cf_e[row_e, "Pr(>|z|)"],  3),
             beta_cond     = round(cf_c[row_c, "Estimate"],   4),
             SE_cond       = round(cf_c[row_c, "Std. Error"], 4),
             z_cond        = round(cf_c[row_c, "z value"],    3),
             p_cond        = signif(cf_c[row_c, "Pr(>|z|)"],  3),
             AIC_emb       = round(AIC(emb),  2),
             AIC_cond      = round(AIC(cond), 2),
             dAIC          = round(AIC(emb) - AIC(cond), 2))
}


# ---- Prep ----
per1 <- prep_data("data/per1_filteredSubjects.csv",
                  "../../per_similarity_results/per1_speaker_pair_per_sim_full.csv",
                  experiment = 1)
per2 <- prep_data("data/per2_filteredSubjects.csv",
                  "../../per_similarity_results/per2_speaker_pair_per_sim_full.csv",
                  experiment = 2)

# Combined-Per1 diffVshared: dif vs. shared, ignoring sub-experiment.
per1$diffVshared <- ifelse(grepl("dif", per1$condition), 0.25, -0.25)

# Per2 diffVshared from original per2_analysis.R.
per2$diffVshared <- ifelse(per2$condition == 'mixed-different' |
                           per2$condition == 'non-native-different', 0.25, -0.5)


# ---- Run ----
results <- bind_rows(
  fit_and_extract(per1, "Per1 (combined 1A+1B)", "diffVshared"),
  fit_and_extract(per2, "Per2",                  "diffVshared")
)

print(results, row.names = FALSE)
write.csv(results, "model_results/embedding_interaction_summary_1to1.csv",
          row.names = FALSE)
