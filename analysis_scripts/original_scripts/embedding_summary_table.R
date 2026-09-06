############################################################
# Embedding-distance regression summary across all experiments.
# Outputs one table with: legality x avg_distance_c interaction
# beta, p-value, and AIC for Exp 1 (combined), 1A, 1B, and 2.
############################################################

library(plyr)
library(dplyr)
library(lme4)
library(stringr)
library(ggplot2)

if (requireNamespace("rstudioapi", quietly = TRUE) && rstudioapi::isAvailable()) {
  setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
}
source("analysis_functions.R")


# ---- Helper: prep gen data with per-worker avg_distance ----
prep_data <- function(file_path, sim_path, experiment) {
  raw <- open_file(file_path, experiment = experiment)
  gen <- filter(raw, gen == 0.5)

  # per1 needs the standard label/syllable transformations + speaker columns
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
  }

  gen$response   <- as.factor(gen$response)
  gen$worker_ID  <- as.factor(gen$worker_ID)
  gen$syllable_no_speaker <- as.factor(gen$syllable_no_speaker)

  # per-syllable pair distance lookup
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


# ---- Helper: fit embedding-only and condition-only models and return key stats ----
fit_and_extract <- function(data, label, condition_formula) {
  emb <- glmer(response ~ legality * avg_distance_c
               + (1 + legality | worker_ID)
               + (1 + legality | syllable_no_speaker),
               data = data, family = "binomial",
               control = glmerControl(optimizer = "bobyqa"))

  cond <- glmer(condition_formula,
                data = data, family = "binomial",
                control = glmerControl(optimizer = "bobyqa"))

  cf  <- summary(emb)$coefficients
  row <- "legality:avg_distance_c"
  data.frame(experiment = label,
             n_obs      = nobs(emb),
             n_workers  = length(unique(data$worker_ID)),
             beta       = round(cf[row, "Estimate"],   4),
             SE         = round(cf[row, "Std. Error"], 4),
             z          = round(cf[row, "z value"],    3),
             p          = signif(cf[row, "Pr(>|z|)"],  3),
             AIC_emb    = round(AIC(emb),  2),
             AIC_cond   = round(AIC(cond), 2),
             dAIC       = round(AIC(emb) - AIC(cond), 2))
}


# ---- Run all four ----
per1 <- prep_data("data/per1_filteredSubjects.csv",
                  "../../per_similarity_results/per1_speaker_pair_per_sim_full.csv",
                  experiment = 1)
per2 <- prep_data("data/per2_filteredSubjects.csv",
                  "../../per_similarity_results/per2_speaker_pair_per_sim_full.csv",
                  experiment = 2)

# Per1 condition contrasts (combined uses unified codes; 1A/1B use original per-experiment codes)
per1$diffVshared_combined <- ifelse(grepl("dif",    per1$condition), 0.25, -0.25)
per1$weakVstrong_combined <- ifelse(grepl("strong", per1$condition), 0.5,
                             ifelse(grepl("weak",   per1$condition), -0.5, 0))
per1$accent_combined      <- ifelse(per1$condition == "NN_shared",      0.5,
                             ifelse(per1$condition == "native_shared", -0.5, 0))

per1A <- filter(per1, xp_num == 1)
per1A$diffVshared <- ifelse(per1A$condition == 'weak-dif' | per1A$condition == 'strong-dif', 0.25, -0.25)
per1A$weakVstrong <- ifelse(per1A$condition == 'native_shared' | per1A$condition == 'NN_shared', 0,
                     ifelse(per1A$condition == 'strong-dif', 0.5, -0.5))
per1A$accent      <- ifelse(per1A$condition == 'NN_shared', 0.5,
                     ifelse(per1A$condition == 'native_shared', -0.5, 0))

per1B <- filter(per1, xp_num == 2)
per1B$diffVshared <- ifelse(per1B$condition == 'strong-dif2' | per1B$condition == 'weak-dif2', 0.25, -0.5)
per1B$weakVstrong <- ifelse(per1B$condition == 'NN_shared2', 0,
                     ifelse(per1B$condition == 'strong-dif2', 0.5, -0.5))

# Per2 condition contrasts (from original per2_analysis.R)
per2$diffVshared <- ifelse(per2$condition == 'mixed-different' | per2$condition == 'non-native-different', 0.25, -0.5)
per2$native      <- ifelse(per2$condition == 'mixed-different', 0.5, -0.25)

rand_eff <- "(1 + legality | worker_ID) + (1 + legality | syllable_no_speaker)"

results <- bind_rows(
  fit_and_extract(per1,  "Per1 (combined 1A+1B)",
                  as.formula(paste("response ~ legality * (diffVshared_combined + weakVstrong_combined + accent_combined) +", rand_eff))),
  fit_and_extract(per1A, "Per1A",
                  as.formula(paste("response ~ legality * (diffVshared + weakVstrong + accent) +", rand_eff))),
  fit_and_extract(per1B, "Per1B",
                  as.formula(paste("response ~ legality * (diffVshared + weakVstrong) +", rand_eff))),
  fit_and_extract(per2,  "Per2",
                  as.formula(paste("response ~ legality * (diffVshared + native) +", rand_eff)))
)

print(results, row.names = FALSE)
write.csv(results, "model_results/embedding_interaction_summary.csv", row.names = FALSE)


# ---- Combined faceted plot: false alarm rate (logit) by avg distance, all experiments ----

worker_fa <- function(data, label) {
  data %>%
    group_by(worker_ID, legality, avg_distance) %>%
    summarize(n_yes = sum(yesResponse), n_total = n(), .groups = "drop") %>%
    mutate(fa_logit = log((n_yes + 0.5) / (n_total - n_yes + 0.5)),
           Legality = factor(ifelse(legality == 0.5, "Legal", "Illegal"),
                             levels = c("Legal", "Illegal")),
           experiment = label)
}

combined_fa <- bind_rows(
  worker_fa(filter(per1, xp_num == 1), "Per1A"),
  worker_fa(filter(per1, xp_num == 2), "Per1B"),
  worker_fa(per2,                       "Per2")
) %>%
  mutate(experiment = factor(experiment, levels = c("Per1A", "Per1B", "Per2")))

p_combined <- ggplot(combined_fa, aes(x = avg_distance, y = fa_logit, color = Legality)) +
  geom_jitter(alpha = 0.35, size = 1.2, width = 0.1, height = 0) +
  geom_smooth(method = "lm", se = TRUE) +
  scale_color_manual(values = c("Legal" = "#1f77b4", "Illegal" = "#d62728")) +
  facet_wrap(~ experiment, scales = "free_x") +
  labs(x = "Average embedding distance between speakers",
       y = "False alarm rate (empirical logit)",
       title = "False alarm rate by speaker-pair embedding distance, all experiments") +
  theme_minimal(base_size = 12)

ggsave("plots/all_experiments_fa_by_distance.png", p_combined,
       width = 12, height = 4.5, dpi = 150)


# ---- Single (non-faceted) plot with all experiments overlaid ----
# Distances are z-scored within experiment (Per2's raw distances live on a
# different scale than Per1's because of speaker / utterance differences),
# so the x-axis is "distance relative to other speaker pairs in the same
# experiment" rather than raw DTW units.

combined_fa_z <- combined_fa %>%
  group_by(experiment) %>%
  mutate(avg_distance_z = as.numeric(scale(avg_distance))) %>%
  ungroup()

p_overlay <- ggplot(combined_fa_z, aes(x = avg_distance_z, y = fa_logit, color = Legality)) +
  geom_jitter(aes(shape = experiment), alpha = 0.3, size = 1.2, width = 0.05, height = 0) +
  geom_smooth(method = "lm", se = TRUE, linewidth = 1) +
  scale_color_manual(values = c("Legal" = "#1f77b4", "Illegal" = "#d62728")) +
  scale_shape_manual(values = c("Per1A" = 16, "Per1B" = 17, "Per2" = 15)) +
  labs(x = "Average embedding distance (z-scored within experiment)",
       y = "False alarm rate (empirical logit)",
       title = "False alarm rate by speaker-pair embedding distance (all experiments combined)",
       shape = "Experiment") +
  theme_minimal(base_size = 12)

ggsave("plots/all_experiments_combined_fa_by_distance.png", p_overlay,
       width = 9, height = 5.5, dpi = 150)
