############################################################
# Distribution of per-subject legality difference scores.
#
# For each participant, the "difference score" is their
# false-alarm ("yes") rate on LEGAL test items minus their
# "yes" rate on ILLEGAL test items, computed over generalization
# (test) trials only (gen == 0.5). A positive score means the
# participant endorsed legal items more than illegal ones.
#
# Produces a histogram + density for each experiment (Per1A,
# Per1B, Per2) separately, a faceted view of the same three, an
# overlay at the experiment level (Per1 collapsed across 1A and
# 1B, vs Per2), and the same facet view collapsed to two panels
# (Experiment 1, Experiment 2) with each panel split by a
# within-experiment median split on per-worker talker distance.
############################################################

library(plyr)
library(dplyr)
library(ggplot2)

if (requireNamespace("rstudioapi", quietly = TRUE) && rstudioapi::isAvailable()) {
  setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
}
source("analysis_functions.R")


################### 1 - per-worker legality difference scores ###################

# Returns one row per worker: legal "yes" rate - illegal "yes" rate on test trials.
worker_legality_diff <- function(data) {
  gen <- filter(data, gen == 0.5)
  by_leg <- gen %>%
    group_by(worker_ID, legality) %>%
    summarize(yes_rate = mean(yesResponse), .groups = "drop")
  # legality: 0.5 = legal, -0.5 = illegal
  legal   <- by_leg %>% filter(legality ==  0.5) %>% select(worker_ID, legal_rate   = yes_rate)
  illegal <- by_leg %>% filter(legality == -0.5) %>% select(worker_ID, illegal_rate = yes_rate)
  inner_join(legal, illegal, by = "worker_ID") %>%
    mutate(diff = legal_rate - illegal_rate)
}

per1_data <- open_file("data/per1_filteredSubjects.csv", experiment = 1)
per2_data <- open_file("data/per2_filteredSubjects.csv", experiment = 2)

# Per1 splits into 1A (condition_num 1-4) and 1B (condition_num 5-7).
per1a_data <- filter(per1_data, condition_num < 5)
per1b_data <- filter(per1_data, condition_num >= 5)

diff_1A  <- worker_legality_diff(per1a_data) %>% mutate(experiment = "Per1A")
diff_1B  <- worker_legality_diff(per1b_data) %>% mutate(experiment = "Per1B")
diff_2   <- worker_legality_diff(per2_data)  %>% mutate(experiment = "Per2")

# Per1 collapsed across sub-experiments, for the experiment-level overlay.
diff_1   <- worker_legality_diff(per1_data)  %>% mutate(experiment = "Per1")

all_diff <- bind_rows(diff_1A, diff_1B, diff_2) %>%
  mutate(experiment = factor(experiment, levels = c("Per1A", "Per1B", "Per2")))


################### 2 - per-experiment histogram + density ###################

exp_colors <- c("Per1A" = "#1b9e77", "Per1B" = "#d95f02",
                "Per1"  = "#1b9e77", "Per2"  = "#7570b3")

# Test items per subject per legality (drives the discreteness of the difference
# score). The difference can only be a multiple of 1/n_items, so bins are aligned
# to that grid (binwidth = 1/n_items, centered on 0) — one bin per achievable
# value. Using a bin width that doesn't divide 1/n_items leaves phantom empty
# bars where no achievable value can ever fall.
# Per1A and Per1B both use 18, so the collapsed Per1 keeps the same grid.
exp_n_items <- c("Per1A" = 18, "Per1B" = 18, "Per1" = 18, "Per2" = 16)

make_hist <- function(df, label) {
  m  <- mean(df$diff)
  n  <- nrow(df)
  bw <- 1 / exp_n_items[[label]]
  ggplot(df, aes(x = diff)) +
    geom_histogram(aes(y = after_stat(density)), binwidth = bw, center = 0,
                   fill = exp_colors[[label]], color = "white", alpha = 0.65) +
    geom_density(color = exp_colors[[label]], linewidth = 1) +
    geom_vline(xintercept = 0, linetype = "dotted", color = "grey40") +
    geom_vline(xintercept = m, linetype = "dashed", color = exp_colors[[label]], linewidth = 0.9) +
    annotate("text", x = m, y = Inf,
             label = sprintf("mean = %.3f", m),
             hjust = -0.1, vjust = 1.5, color = exp_colors[[label]], fontface = "bold", size = 4) +
    labs(x = "Legality difference score (legal − illegal 'yes' rate)",
         y = "Density",
         title = sprintf("%s: per-subject legality difference scores (N = %d, %d items/legality)",
                         label, n, exp_n_items[[label]])) +
    theme_minimal(base_size = 12)
}

ggsave("plots/per1A_legality_diff_dist.png", make_hist(diff_1A, "Per1A"),
       width = 8, height = 5, dpi = 150)
ggsave("plots/per1B_legality_diff_dist.png", make_hist(diff_1B, "Per1B"),
       width = 8, height = 5, dpi = 150)
ggsave("plots/per2_legality_diff_dist.png",  make_hist(diff_2,  "Per2"),
       width = 8, height = 5, dpi = 150)


################### 3 - experiment-level overlay (Per1 vs Per2) ###################

# Per1 is collapsed across 1A and 1B here; the sub-experiment split is kept in
# the per-experiment histograms above and the facet below.
combined_diff <- bind_rows(diff_1, diff_2) %>%
  mutate(experiment = factor(experiment, levels = c("Per1", "Per2")))

combined_means <- combined_diff %>%
  group_by(experiment) %>%
  summarize(mean_diff = mean(diff), n = n(), .groups = "drop")

exp_means <- all_diff %>%
  group_by(experiment) %>%
  summarize(mean_diff = mean(diff), n = n(), .groups = "drop")

p_combined <- ggplot(combined_diff, aes(x = diff, color = experiment, fill = experiment)) +
  geom_density(alpha = 0.15, linewidth = 1) +
  geom_vline(xintercept = 0, linetype = "dotted", color = "grey40") +
  geom_vline(data = combined_means, aes(xintercept = mean_diff, color = experiment),
             linetype = "dashed", linewidth = 0.8, show.legend = FALSE) +
  scale_color_manual(values = exp_colors) +
  scale_fill_manual(values = exp_colors) +
  labs(x = "Legality difference score (legal \u2212 illegal 'yes' rate)",
       y = "Density",
       title = "Per-subject legality difference scores, Experiment 1 and Experiment 2",
       color = "Experiment", fill = "Experiment") +
  theme_minimal(base_size = 12)

ggsave("plots/all_experiments_legality_diff_dist.png", p_combined,
       width = 9, height = 5.5, dpi = 150)


# Also a faceted view. Each experiment has a different quantum (1/18 vs 1/16),
# so a single binwidth across facets would reintroduce phantom gaps. Instead,
# snap each subject's score to its experiment's achievable grid and count, then
# draw with geom_col (proportion of subjects at each achievable value).
facet_counts <- all_diff %>%
  mutate(n_items = exp_n_items[as.character(experiment)],
         diff_grid = round(diff * n_items) / n_items) %>%
  group_by(experiment, diff_grid) %>%
  dplyr::summarize(n = n(), .groups = "drop") %>%
  group_by(experiment) %>%
  mutate(prop = n / sum(n)) %>%
  ungroup()

p_facet <- ggplot(facet_counts, aes(x = diff_grid, y = prop, fill = experiment)) +
  geom_col(width = 0.045, alpha = 0.75) +
  geom_vline(xintercept = 0, linetype = "dotted", color = "grey40") +
  geom_vline(data = exp_means, aes(xintercept = mean_diff, color = experiment),
             linetype = "dashed", linewidth = 0.8, show.legend = FALSE) +
  scale_fill_manual(values = exp_colors) +
  scale_color_manual(values = exp_colors) +
  facet_wrap(~ experiment, ncol = 1) +
  labs(x = "Legality difference score (legal − illegal 'yes' rate)",
       y = "Proportion of subjects",
       title = "Per-subject legality difference scores by experiment (bins on achievable grid)") +
  theme_minimal(base_size = 12) +
  theme(legend.position = "none")

ggsave("plots/all_experiments_legality_diff_facet.png", p_facet,
       width = 8, height = 9, dpi = 150)


# Same facet view collapsed to the experiment level (Experiment 1 = Per1A +
# Per1B, vs Experiment 2), no title, sized for a manuscript column, with the
# two talker-distance halves of each experiment drawn separately.

# Per-worker embedding distance. Same construction as prep_data() in
# embedding_summary_table.R: each worker hears exactly two talkers across fam +
# test, and avg_distance is the mean per-syllable HuBERT/DTW distance for that
# talker pair over the syllables that worker actually heard.
worker_avg_distance <- function(raw, sim_path) {
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

dist_1 <- worker_avg_distance(per1_data,
                              "../../per_similarity_results/per1_speaker_pair_per_sim_full.csv")
dist_2 <- worker_avg_distance(per2_data,
                              "../../per_similarity_results/per2_speaker_pair_per_sim_full.csv")

# Median split on avg_distance WITHIN experiment — Per1 and Per2 distances live
# on different scales, so a pooled split would just re-encode experiment.
combined_diff_dist <- bind_rows(
    inner_join(diff_1, dist_1, by = "worker_ID"),
    inner_join(diff_2, dist_2, by = "worker_ID")) %>%
  mutate(experiment = factor(experiment, levels = c("Per1", "Per2"))) %>%
  group_by(experiment) %>%
  mutate(dist_group = factor(ifelse(avg_distance >= median(avg_distance),
                                    "High", "Low"),
                             levels = c("Low", "High"))) %>%
  ungroup()

dist_colors <- c("Low" = "#2166ac", "High" = "#b2182b")

# Proportions are within experiment x distance group, so the two halves are
# each their own distribution rather than two pieces of one.
split_counts <- combined_diff_dist %>%
  mutate(n_items = exp_n_items[as.character(experiment)],
         diff_grid = round(diff * n_items) / n_items) %>%
  group_by(experiment, dist_group, diff_grid) %>%
  dplyr::summarize(n = n(), .groups = "drop") %>%
  group_by(experiment, dist_group) %>%
  mutate(prop = n / sum(n)) %>%
  ungroup()

# Dodging only lines up if both groups have a bar at every achievable value, so
# fill the empty cells with zero-height bars.
split_counts <- merge(distinct(split_counts, experiment, diff_grid),
                      data.frame(dist_group = factor(c("Low", "High"),
                                                     levels = c("Low", "High"))),
                      by = NULL) %>%
  left_join(split_counts, by = c("experiment", "diff_grid", "dist_group")) %>%
  mutate(n = ifelse(is.na(n), 0L, n), prop = ifelse(is.na(prop), 0, prop))

split_means <- combined_diff_dist %>%
  group_by(experiment, dist_group) %>%
  dplyr::summarize(mean_diff = mean(diff), n = n(), .groups = "drop")

exp_facet_labels <- c("Per1" = "Experiment 1", "Per2" = "Experiment 2")

p_facet_combined <- ggplot(split_counts,
                           aes(x = diff_grid, y = prop, fill = dist_group)) +
  geom_col(width = 0.045, alpha = 0.75,
           position = position_dodge(width = 0.045)) +
  geom_vline(xintercept = 0, linetype = "dotted", color = "grey40") +
  geom_vline(data = split_means, aes(xintercept = mean_diff, color = dist_group),
             linetype = "dashed", linewidth = 0.6, show.legend = FALSE) +
  scale_fill_manual(values = dist_colors) +
  scale_color_manual(values = dist_colors) +
  facet_wrap(~ experiment, ncol = 1, labeller = labeller(experiment = exp_facet_labels)) +
  labs(x = "Legality difference score (legal − illegal 'yes' rate)",
       y = "Proportion of listeners",
       fill = "Talker distance") +
  theme_minimal(base_size = 9) +
  theme(legend.position = "bottom",
        legend.key.size = unit(0.35, "cm"),
        legend.margin = margin(t = -4))

ggsave("plots/per1_per2_legality_diff_facet.png", p_facet_combined,
       width = 3.5, height = 3, dpi = 300)


################### 4 - console summary ###################

cat("\nPer-subject legality difference score summary (experiment level):\n")
combined_diff %>%
  group_by(experiment) %>%
  summarize(n = n(),
            mean   = round(mean(diff), 4),
            sd     = round(sd(diff), 4),
            median = round(median(diff), 4),
            pct_positive = round(mean(diff > 0), 3),
            .groups = "drop") %>%
  as.data.frame() %>%
  print(row.names = FALSE)

cat("\nPer-subject legality difference score summary (talker-distance median split):\n")
combined_diff_dist %>%
  group_by(experiment, dist_group) %>%
  summarize(n = n(),
            mean_distance = round(mean(avg_distance), 3),
            mean   = round(mean(diff), 4),
            sd     = round(sd(diff), 4),
            median = round(median(diff), 4),
            pct_positive = round(mean(diff > 0), 3),
            .groups = "drop") %>%
  as.data.frame() %>%
  print(row.names = FALSE)

cat("\nPer-subject legality difference score summary (sub-experiments):\n")
all_diff %>%
  group_by(experiment) %>%
  summarize(n = n(),
            mean   = round(mean(diff), 4),
            sd     = round(sd(diff), 4),
            median = round(median(diff), 4),
            pct_positive = round(mean(diff > 0), 3),
            .groups = "drop") %>%
  as.data.frame() %>%
  print(row.names = FALSE)
