############################################################
# Thomas Denby
# Rational Inference in Phonotactic Learning
#
# Regression analysis for Experiments 1A and 1B using
# average HuBERT embedding distance between familiarization
# and test speakers as a continuous predictor, replacing the
# categorical condition contrasts (diffVshared, weakVstrong, accent).
#
# Sections:
#   (1) data prep
#   (2) embedding distance lookup
#   (3) Exp 1A model
#   (4) Exp 1B model
############################################################

library(plyr)
library(dplyr)
library(lme4)
library(boot)
library(stringr)
library(ggplot2)
library(rstudioapi)

if (requireNamespace("rstudioapi", quietly = TRUE) && rstudioapi::isAvailable()) {
  setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
}
source("analysis_functions.R")


################### 1 - data prep ###################

per1_file_path <- 'data/per1_filteredSubjects.csv'
per1_data <- open_file(per1_file_path, experiment = 1)

all_gen_syllables <- filter(per1_data, gen == 0.5)
all_fam_syllables <- filter(per1_data, gen == -0.5)

# Standard label/syllable transformations (same as per1_analysis.R)
all_gen_syllables$label <- as.character(all_gen_syllables$label)
all_gen_syllables$label <- sub('french','FR',all_gen_syllables$label)
all_gen_syllables$label <- sub('_F','_F1',all_gen_syllables$label)
all_gen_syllables$label <- sub('_M','_M1',all_gen_syllables$label)
all_gen_syllables$label <- sub('_F1I','_F2',all_gen_syllables$label)
all_gen_syllables$label <- sub('NS','EN',all_gen_syllables$label)

all_gen_syllables$syllable <- as.character(all_gen_syllables$syllable)
all_gen_syllables$syllable <- sub('french','FR',all_gen_syllables$syllable)
all_gen_syllables$syllable <- sub('_F','_F1',all_gen_syllables$syllable)
all_gen_syllables$syllable <- sub('_M','_M1',all_gen_syllables$syllable)
all_gen_syllables$syllable <- sub('_F1I','_F2',all_gen_syllables$syllable)
all_gen_syllables$syllable <- sub('NS','EN',all_gen_syllables$syllable)

condition_names <- data.frame(c(1,2,3,4,5,6,7), c("weak-dif","strong-dif","native_shared","NN_shared","NN_shared2","strong-dif2","weak-dif2"))
colnames(condition_names) <- c("condition_num","condition")
all_gen_syllables <- merge(all_gen_syllables, condition_names)
all_gen_syllables$xp_num <- ifelse(all_gen_syllables$condition_num < 5, 1, 2)

all_gen_syllables$speaker <- substring(all_gen_syllables$label, nchar(all_gen_syllables$label)-4, nchar(all_gen_syllables$label))
all_gen_syllables$gender <- substring(all_gen_syllables$label, nchar(all_gen_syllables$label)-1, nchar(all_gen_syllables$label)-1)
all_gen_syllables$vowel <- substring(all_gen_syllables$syllable, 8, 8)
all_gen_syllables$accent <- substring(all_gen_syllables$speaker, 0, 2)
all_gen_syllables$accent <- ifelse(all_gen_syllables$accent == "FR", "French", "English")
all_gen_syllables$syllable_no_speaker <- substring(all_gen_syllables$syllable, 7, 9)

all_gen_syllables$response <- as.factor(all_gen_syllables$response)
all_gen_syllables$worker_ID <- as.factor(all_gen_syllables$worker_ID)
all_gen_syllables$syllable <- as.factor(all_gen_syllables$syllable)
all_gen_syllables$syllable_no_speaker <- as.factor(all_gen_syllables$syllable_no_speaker)


################### 2 - embedding distance lookup ###################

# Load per1 HuBERT pairwise distances (one row per speaker-pair x syllable).
# The "_full" CSV includes [y]-vowel syllables for French-French pairs (which
# the English speakers don't have recordings for) and is built using pair-level
# word availability rather than a global intersection.
per1_sim <- read.csv("../../per_similarity_results/per1_speaker_pair_per_sim_full.csv")

# Parse: speaker name (everything before the final _segment) and syllable code (final segment)
per1_sim$speaker1 <- sub("_[^_]+$", "", per1_sim$S1)
per1_sim$speaker2 <- sub("_[^_]+$", "", per1_sim$S2)
per1_sim$syl      <- sub("^.*_", "", as.character(per1_sim$S1))

# Symmetric lookup: each (speaker1, speaker2, syl) row plus its (speaker2, speaker1, syl) twin
sim_lookup_sym <- bind_rows(
  per1_sim %>% select(speaker1, speaker2, syl, distance),
  per1_sim %>% select(speaker1 = speaker2, speaker2 = speaker1, syl, distance)
)

# Each participant hears exactly two speakers across both fam and test phases.
worker_speakers <- per1_data %>%
  mutate(speaker = sub("^study-|^test-(legal|illegal)-", "", as.character(label))) %>%
  distinct(worker_ID, speaker) %>%
  group_by(worker_ID) %>%
  arrange(speaker, .by_group = TRUE) %>%
  summarize(speaker1 = first(speaker),
            speaker2 = nth(speaker, 2),
            .groups = "drop") %>%
  mutate(worker_ID = as.factor(worker_ID))

# Syllables each worker actually heard, across BOTH familiarization and test
# trials. (Some conditions, e.g. 1A strong-dif, hear French [y] only during
# fam — excluding fam would erase the strong-cue stimulus.)
worker_syllables <- per1_data %>%
  mutate(syl = sub("^.*_", "", as.character(syllable))) %>%
  distinct(worker_ID, syl) %>%
  mutate(worker_ID = as.factor(worker_ID))

# Per-worker avg distance: mean over (their speaker pair, syllables they heard)
# tuples that exist in the similarity CSV. Conditions whose stimuli include
# vowels not in the CSV (e.g. y-syllables for French-English pairs, since
# English speakers never have y recordings) just average over the subset
# that is computable.
worker_avg_dist <- worker_speakers %>%
  inner_join(worker_syllables, by = "worker_ID") %>%
  inner_join(sim_lookup_sym, by = c("speaker1", "speaker2", "syl")) %>%
  group_by(worker_ID, speaker1, speaker2) %>%
  summarize(avg_distance = mean(distance),
            n_syllables  = n(),
            .groups = "drop")

all_gen_syllables <- all_gen_syllables %>%
  left_join(worker_avg_dist, by = "worker_ID")

# Center avg_distance (not scaled) so legality main effect is interpretable at mean distance
all_gen_syllables$avg_distance_c <- scale(all_gen_syllables$avg_distance, scale = FALSE)[, 1]


################### 3 - combined model (Exp 1A + 1B) ###################

per1.glm.emb <- glmer(response ~ legality * avg_distance_c
                      + (1 + legality | worker_ID)
                      + (1 + legality | syllable_no_speaker)
                      , data = all_gen_syllables, family = "binomial",
                      control = glmerControl(optimizer = "bobyqa"))

summary(per1.glm.emb)

chi_sq_legality    <- chiReport.func(anova(per1.glm.emb, update(per1.glm.emb, .~. - legality)))
chi_sq_distance    <- chiReport.func(anova(per1.glm.emb, update(per1.glm.emb, .~. - avg_distance_c)))
chi_sq_interaction <- chiReport.func(anova(per1.glm.emb, update(per1.glm.emb, .~. - legality:avg_distance_c)))

capture.output(summary(per1.glm.emb)) %>%
  writeLines(con = "model_results/per1_combined_glm_embedding.txt")

paste0("AIC: ", AIC(per1.glm.emb),
       "\n legality main effect: ",          capture.output(chi_sq_legality),
       "\n distance main effect: ",          capture.output(chi_sq_distance),
       "\n legality:distance interaction: ", capture.output(chi_sq_interaction)) %>%
  writeLines(con = "model_results/per1_combined_chi_sq_embedding.txt")


################### 3b - Exp 1A model ###################

xp_1A <- filter(all_gen_syllables, xp_num == 1)

xp_1a.glm.emb <- glmer(response ~ legality * avg_distance_c
                       + (1 + legality | worker_ID)
                       + (1 + legality | syllable_no_speaker)
                       , data = xp_1A, family = "binomial",
                       control = glmerControl(optimizer = "bobyqa"))

summary(xp_1a.glm.emb)

chi_sq_legality_1a    <- chiReport.func(anova(xp_1a.glm.emb, update(xp_1a.glm.emb, .~. - legality)))
chi_sq_distance_1a    <- chiReport.func(anova(xp_1a.glm.emb, update(xp_1a.glm.emb, .~. - avg_distance_c)))
chi_sq_interaction_1a <- chiReport.func(anova(xp_1a.glm.emb, update(xp_1a.glm.emb, .~. - legality:avg_distance_c)))

capture.output(summary(xp_1a.glm.emb)) %>%
  writeLines(con = "model_results/per1A_glm_embedding.txt")

paste0("AIC: ", AIC(xp_1a.glm.emb),
       "\n legality main effect: ",          capture.output(chi_sq_legality_1a),
       "\n distance main effect: ",          capture.output(chi_sq_distance_1a),
       "\n legality:distance interaction: ", capture.output(chi_sq_interaction_1a)) %>%
  writeLines(con = "model_results/per1A_chi_sq_embedding.txt")


################### 3c - Exp 1B model ###################

xp_1B <- filter(all_gen_syllables, xp_num == 2)

xp_1b.glm.emb <- glmer(response ~ legality * avg_distance_c
                       + (1 + legality | worker_ID)
                       + (1 + legality | syllable_no_speaker)
                       , data = xp_1B, family = "binomial",
                       control = glmerControl(optimizer = "bobyqa"))

summary(xp_1b.glm.emb)

chi_sq_legality_1b    <- chiReport.func(anova(xp_1b.glm.emb, update(xp_1b.glm.emb, .~. - legality)))
chi_sq_distance_1b    <- chiReport.func(anova(xp_1b.glm.emb, update(xp_1b.glm.emb, .~. - avg_distance_c)))
chi_sq_interaction_1b <- chiReport.func(anova(xp_1b.glm.emb, update(xp_1b.glm.emb, .~. - legality:avg_distance_c)))

capture.output(summary(xp_1b.glm.emb)) %>%
  writeLines(con = "model_results/per1B_glm_embedding.txt")

paste0("AIC: ", AIC(xp_1b.glm.emb),
       "\n legality main effect: ",          capture.output(chi_sq_legality_1b),
       "\n distance main effect: ",          capture.output(chi_sq_distance_1b),
       "\n legality:distance interaction: ", capture.output(chi_sq_interaction_1b)) %>%
  writeLines(con = "model_results/per1B_chi_sq_embedding.txt")


################### 4 - AIC comparison: embedding vs. original condition contrasts ###################

# Reconstruct original contrast codes for 1A and 1B on the combined dataset
all_gen_syllables$diffVshared <- ifelse(grepl("dif", all_gen_syllables$condition),  0.25, -0.25)
all_gen_syllables$weakVstrong <- ifelse(grepl("strong", all_gen_syllables$condition), 0.5,
                                 ifelse(grepl("weak",   all_gen_syllables$condition), -0.5, 0))
all_gen_syllables$accent      <- ifelse(all_gen_syllables$condition == "NN_shared",     0.5,
                                 ifelse(all_gen_syllables$condition == "native_shared", -0.5, 0))

per1.glm.condition <- glmer(response ~ legality * (diffVshared + weakVstrong + accent)
                            + (1 + legality | worker_ID)
                            + (1 + legality | syllable_no_speaker)
                            , data = all_gen_syllables, family = "binomial",
                            control = glmerControl(optimizer = "bobyqa"))

aic_comparison <- data.frame(
  model = c("condition contrasts (categorical)", "embedding distance (continuous)"),
  k     = c(attr(logLik(per1.glm.condition), "df"), attr(logLik(per1.glm.emb), "df")),
  AIC   = c(AIC(per1.glm.condition), AIC(per1.glm.emb))
)

print(aic_comparison)
write.csv(aic_comparison, "model_results/per1_AIC_comparison.csv", row.names = FALSE)


################### 5 - scatterplot: per-participant false alarm rate by avg distance ###################

worker_fa <- all_gen_syllables %>%
  group_by(worker_ID, legality, avg_distance) %>%
  summarize(n_yes   = sum(yesResponse),
            n_total = n(),
            fa_logit = log((n_yes + 0.5) / (n_total - n_yes + 0.5)),
            .groups = "drop") %>%
  mutate(Legality = factor(ifelse(legality == 0.5, "Legal", "Illegal"),
                           levels = c("Legal", "Illegal")))

make_plot <- function(df, title) {
  ggplot(df, aes(x = avg_distance, y = fa_logit, color = Legality)) +
    geom_jitter(alpha = 0.4, size = 1.4, width = 0.15, height = 0) +
    geom_smooth(method = "lm", se = TRUE) +
    scale_color_manual(values = c("Legal" = "#1f77b4", "Illegal" = "#d62728")) +
    labs(x = "Average embedding distance between speakers",
         y = "False alarm rate (empirical logit)",
         title = title) +
    theme_minimal(base_size = 12)
}

ggsave("plots/per1_combined_fa_by_distance.png",
       make_plot(worker_fa, "Per1 (combined 1A + 1B): false alarm rate by speaker-pair embedding distance"),
       width = 8, height = 5.5, dpi = 150)

ggsave("plots/per1A_fa_by_distance.png",
       make_plot(worker_fa %>% semi_join(distinct(filter(all_gen_syllables, xp_num == 1), worker_ID), by = "worker_ID"),
                 "Per1A: false alarm rate by speaker-pair embedding distance"),
       width = 8, height = 5.5, dpi = 150)

ggsave("plots/per1B_fa_by_distance.png",
       make_plot(worker_fa %>% semi_join(distinct(filter(all_gen_syllables, xp_num == 2), worker_ID), by = "worker_ID"),
                 "Per1B: false alarm rate by speaker-pair embedding distance"),
       width = 8, height = 5.5, dpi = 150)


################### 6b - per-pair scatter: legality effect by embedding distance (Per1A) ###################

pair_effect_1A <- all_gen_syllables %>%
  filter(xp_num == 1) %>%
  group_by(worker_ID, speaker1, speaker2, avg_distance, legality) %>%
  summarize(fa = mean(yesResponse), .groups = "drop") %>%
  group_by(worker_ID, speaker1, speaker2, avg_distance) %>%
  summarize(legality_effect = fa[legality == 0.5] - fa[legality == -0.5],
            .groups = "drop") %>%
  group_by(speaker1, speaker2) %>%
  summarize(mean_legality_effect = mean(legality_effect),
            mean_distance        = mean(avg_distance),
            n_workers            = n(),
            .groups = "drop") %>%
  mutate(pair = paste(speaker1, "vs", speaker2))

fit_1A    <- lm(mean_legality_effect ~ mean_distance, data = pair_effect_1A)
r2_1A     <- summary(fit_1A)$r.squared
r_1A      <- cor(pair_effect_1A$mean_distance, pair_effect_1A$mean_legality_effect)
n_pairs_1A <- nrow(pair_effect_1A)

p_pair_1A <- ggplot(pair_effect_1A, aes(x = mean_distance, y = mean_legality_effect)) +
  geom_smooth(method = "lm", se = TRUE, color = "#2b83ba", linetype = "dashed", linewidth = 0.8) +
  geom_point(size = 4, color = "#2b83ba", alpha = 0.85) +
  ggrepel::geom_text_repel(aes(label = pair), size = 3.5, max.overlaps = Inf, box.padding = 0.5) +
  annotate("text", x = -Inf, y = Inf,
           label = sprintf("R² = %.3f\nr = %.3f\nN = %d pairs", r2_1A, r_1A, n_pairs_1A),
           hjust = -0.1, vjust = 1.3, fontface = "bold", size = 4.5) +
  labs(x = "Average embedding distance",
       y = "Mean legality effect (legal − illegal FA rate)",
       title = "Per1A: embedding distance vs. legality effect, by speaker pair") +
  theme_minimal(base_size = 12)

ggsave("plots/per1A_pair_legality_effect.png", p_pair_1A, width = 8, height = 6, dpi = 150)


################### 6 - density of per-syllable distances by speaker pair ###################

per1_sim$pair <- paste(per1_sim$speaker1, per1_sim$speaker2, sep = " vs ")

p_density <- ggplot(per1_sim, aes(x = distance, color = pair, fill = pair)) +
  geom_density(alpha = 0.15, linewidth = 0.7) +
  labs(x = "DTW distance per syllable",
       y = "Density",
       title = "Per1: distribution of per-syllable HuBERT distances by speaker pair",
       color = "Speaker pair", fill = "Speaker pair") +
  theme_minimal(base_size = 12)

ggsave("plots/per1_distance_density_by_pair.png", p_density,
       width = 10, height = 6, dpi = 150)
