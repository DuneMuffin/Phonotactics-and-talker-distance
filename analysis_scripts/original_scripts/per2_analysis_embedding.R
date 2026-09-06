############################################################
# Thomas Denby
# Rational Inference in Phonotactic Learning
#
# Regression analysis for Experiment 2 using average HuBERT
# embedding distance between familiarization and test speakers
# as a continuous predictor, replacing the categorical condition
# contrasts (diffVshared, native).
#
# Sections:
#   (1) data prep
#   (2) embedding distance lookup
#   (3) model
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

per2_file_path <- 'data/per2_filteredSubjects.csv'
per2_data <- open_file(per2_file_path, experiment = 2)

all_gen_syllables <- filter(per2_data, gen == 0.5)
all_fam_syllables <- filter(per2_data, gen == -0.5)

all_gen_syllables$response <- as.factor(all_gen_syllables$response)
all_gen_syllables$worker_ID <- as.factor(all_gen_syllables$worker_ID)
all_gen_syllables$syllable <- as.factor(all_gen_syllables$syllable)
all_gen_syllables$syllable_no_speaker <- as.factor(all_gen_syllables$syllable_no_speaker)

condition_names <- data.frame(c(1,2,3), c("mixed-different","non-native-different","non-native-shared"))
colnames(condition_names) <- c("condition_num","condition")
all_gen_syllables <- merge(all_gen_syllables, condition_names)

all_gen_syllables$label <- as.character(all_gen_syllables$label)
all_gen_syllables$speaker <- substring(all_gen_syllables$label, nchar(all_gen_syllables$label)-3, nchar(all_gen_syllables$label))
all_gen_syllables$gender <- substring(all_gen_syllables$label, nchar(all_gen_syllables$label), nchar(all_gen_syllables$label))
all_gen_syllables$gender_contrast <- ifelse(all_gen_syllables$gender == 'F', 0.5, -0.5)
all_gen_syllables$accent <- substring(all_gen_syllables$label, nchar(as.character(all_gen_syllables$label))-3, nchar(as.character(all_gen_syllables$label))-2)


################### 2 - embedding distance lookup ###################

# Load per2 HuBERT pairwise distances (one row per speaker-pair x syllable pair),
# from the "_full" CSV (pair-level word availability).
per2_sim <- read.csv("../../per_similarity_results/per2_speaker_pair_per_sim_full.csv")

per2_sim$speaker1 <- sub("_[^_]+$", "", per2_sim$S1)
per2_sim$speaker2 <- sub("_[^_]+$", "", per2_sim$S2)
per2_sim$syl      <- sub("^.*_", "", as.character(per2_sim$S1))

sim_lookup_sym <- bind_rows(
  per2_sim %>% select(speaker1, speaker2, syl, distance),
  per2_sim %>% select(speaker1 = speaker2, speaker2 = speaker1, syl, distance)
)

worker_speakers <- per2_data %>%
  mutate(speaker = sub("^study-|^test-(legal|illegal)-", "", as.character(label))) %>%
  distinct(worker_ID, speaker) %>%
  group_by(worker_ID) %>%
  arrange(speaker, .by_group = TRUE) %>%
  summarize(speaker1 = first(speaker),
            speaker2 = nth(speaker, 2),
            .groups = "drop") %>%
  mutate(worker_ID = as.factor(worker_ID))

worker_syllables <- per2_data %>%
  mutate(syl = sub("^.*_", "", as.character(syllable))) %>%
  distinct(worker_ID, syl) %>%
  mutate(worker_ID = as.factor(worker_ID))

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


################### 3 - model ###################

exp2.glm.emb <- glmer(response ~ legality * avg_distance_c
                      + (1 + legality | worker_ID)
                      + (1 + legality | syllable_no_speaker)
                      , data = all_gen_syllables, family = "binomial",
                      control = glmerControl(optimizer = "bobyqa"))

summary(exp2.glm.emb)

chi_sq_legality    <- chiReport.func(anova(exp2.glm.emb, update(exp2.glm.emb, .~. - legality)))
chi_sq_distance    <- chiReport.func(anova(exp2.glm.emb, update(exp2.glm.emb, .~. - avg_distance_c)))
chi_sq_interaction <- chiReport.func(anova(exp2.glm.emb, update(exp2.glm.emb, .~. - legality:avg_distance_c)))

capture.output(summary(exp2.glm.emb)) %>%
  writeLines(con = "model_results/per2_glm_embedding.txt")

paste0("AIC: ", AIC(exp2.glm.emb),
       "\n legality main effect: ",          capture.output(chi_sq_legality),
       "\n distance main effect: ",          capture.output(chi_sq_distance),
       "\n legality:distance interaction: ", capture.output(chi_sq_interaction)) %>%
  writeLines(con = "model_results/per2_chi_sq_embedding.txt")


################### 4 - AIC comparison: embedding vs. original condition contrasts ###################

all_gen_syllables$diffVshared <- ifelse(all_gen_syllables$condition == 'mixed-different' |
                                        all_gen_syllables$condition == 'non-native-different', 0.25, -0.5)
all_gen_syllables$native      <- ifelse(all_gen_syllables$condition == 'mixed-different', 0.5, -0.25)

exp2.glm.condition <- glmer(response ~ legality * (diffVshared + native)
                            + (1 + legality | worker_ID)
                            + (1 + legality | syllable_no_speaker)
                            , data = all_gen_syllables, family = "binomial",
                            control = glmerControl(optimizer = "bobyqa"))

aic_comparison <- data.frame(
  model = c("condition contrasts (categorical)", "embedding distance (continuous)"),
  k     = c(attr(logLik(exp2.glm.condition), "df"), attr(logLik(exp2.glm.emb), "df")),
  AIC   = c(AIC(exp2.glm.condition), AIC(exp2.glm.emb))
)

print(aic_comparison)
write.csv(aic_comparison, "model_results/per2_AIC_comparison.csv", row.names = FALSE)


################### 5 - scatterplot: per-participant false alarm rate by avg distance ###################

worker_fa <- all_gen_syllables %>%
  group_by(worker_ID, legality, avg_distance) %>%
  summarize(n_yes   = sum(yesResponse),
            n_total = n(),
            fa_logit = log((n_yes + 0.5) / (n_total - n_yes + 0.5)),
            .groups = "drop") %>%
  mutate(Legality = factor(ifelse(legality == 0.5, "Legal", "Illegal"),
                           levels = c("Legal", "Illegal")))

p <- ggplot(worker_fa, aes(x = avg_distance, y = fa_logit, color = Legality)) +
  geom_jitter(alpha = 0.4, size = 1.4, width = 0.04, height = 0) +
  geom_smooth(method = "lm", se = TRUE) +
  scale_color_manual(values = c("Legal" = "#1f77b4", "Illegal" = "#d62728")) +
  labs(x = "Average embedding distance between speakers",
       y = "False alarm rate (empirical logit)",
       title = "Per2: false alarm rate by speaker-pair embedding distance") +
  theme_minimal(base_size = 12)

ggsave("plots/per2_fa_by_distance.png", p, width = 8, height = 5.5, dpi = 150)


################### 6 - density of per-syllable distances by speaker pair ###################

per2_sim$pair <- paste(per2_sim$speaker1, per2_sim$speaker2, sep = " vs ")

p_density <- ggplot(per2_sim, aes(x = distance, color = pair, fill = pair)) +
  geom_density(alpha = 0.12, linewidth = 0.7) +
  labs(x = "DTW distance per syllable",
       y = "Density",
       title = "Per2: distribution of per-syllable HuBERT distances by speaker pair",
       color = "Speaker pair", fill = "Speaker pair") +
  theme_minimal(base_size = 12)

ggsave("plots/per2_distance_density_by_pair.png", p_density,
       width = 10, height = 6, dpi = 150)
