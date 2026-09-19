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
# Per1B, Per2) separately, a faceted view of the same three, and
# an overlay at the experiment level (Per1 collapsed across 1A and
# 1B, vs Per2).
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
