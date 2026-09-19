############################################################
# Per2 legality difference scores on their achievable grid.
#
# The density plots for Per2 look faintly bimodal, but none of
# the formal tests (dip, Mclust) found anything. This plot asks
# whether the appearance is an artifact of how FEW values the
# difference score can take.
#
# Every Per2 participant contributes exactly 16 legal and 16
# illegal generalization trials, so their difference score
# (legal "yes" rate - illegal "yes" rate) can only be k/16 for
# integer k in -16..16: a 33-value grid with nothing in between.
# One bar per achievable value, drawn narrow so the gaps between
# achievable values are visible, and the empty slots flanking the
# occupied range are kept on the axis.
#
# Writes plots/per2_diff_score_grid.png
############################################################

library(plyr)
library(dplyr)
library(ggplot2)

if (requireNamespace("rstudioapi", quietly = TRUE) && rstudioapi::isAvailable()) {
  setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
}
source("analysis_functions.R")

N_ITEMS <- 16          # legal (and illegal) generalization trials per subject
PAD     <- 2           # empty achievable slots to show either side

per2 <- open_file("data/per2_filteredSubjects.csv", experiment = 2)
gen  <- filter(per2, gen == 0.5)

# Guard the whole premise: the k/16 grid only holds if every subject really
# does have 16 of each. If that ever stops being true the grid is finer and
# this figure would quietly mislead.
counts <- gen %>% count(worker_ID, legality)
stopifnot(all(counts$n == N_ITEMS))

by_leg <- gen %>%
  group_by(worker_ID, legality) %>%
  summarize(yes_rate = mean(yesResponse), .groups = "drop")
legal   <- by_leg %>% filter(legality ==  0.5) %>% select(worker_ID, legal_rate   = yes_rate)
illegal <- by_leg %>% filter(legality == -0.5) %>% select(worker_ID, illegal_rate = yes_rate)
diffs <- inner_join(legal, illegal, by = "worker_ID") %>%
  mutate(diff = legal_rate - illegal_rate,
         k    = round(diff * N_ITEMS))

stopifnot(all(abs(diffs$diff - diffs$k / N_ITEMS) < 1e-9))

# One row per achievable value across the observed range (+/- PAD), so slots
# with no subjects appear as visible zeros rather than vanishing.
k_range <- seq(min(diffs$k) - PAD, max(diffs$k) + PAD)
tab <- diffs %>%
  count(k, name = "n_subjects") %>%
  right_join(data.frame(k = k_range), by = "k") %>%
  mutate(n_subjects = ifelse(is.na(n_subjects), 0L, n_subjects),
         diff = k / N_ITEMS) %>%
  arrange(k)

occupied_in_range <- sum(tab$n_subjects > 0)
span <- max(diffs$k) - min(diffs$k) + 1

subtitle <- sprintf(
  "N = %d subjects, 16 legal / 16 illegal test items each -> score can only be k/16\n%d of the %d achievable values inside the observed range are occupied",
  nrow(diffs), occupied_in_range, span)

p <- ggplot(tab, aes(x = diff, y = n_subjects)) +
  geom_col(width = 0.55 / N_ITEMS, fill = "#7570b3", colour = NA) +
  geom_vline(xintercept = 0, linetype = "dotted", colour = "grey40") +
  geom_vline(xintercept = mean(diffs$diff), linetype = "dashed",
             colour = "#d95f02", linewidth = 0.6) +
  scale_x_continuous(
    breaks = tab$diff,
    labels = ifelse(tab$k %% 2 == 0, sprintf("%.3f", tab$diff), "")) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.08))) +
  labs(x = "Legality difference score (legal - illegal false alarm rate)",
       y = "Number of subjects",
       title = "Per2 difference scores, one bar per achievable value",
       subtitle = subtitle) +
  theme_bw(base_size = 11) +
  theme(panel.grid.minor = element_blank(),
        panel.grid.major.x = element_blank(),
        axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
        plot.subtitle = element_text(size = 9, colour = "grey30"))

ggsave("plots/per2_diff_score_grid.png", p, width = 8, height = 4.5, dpi = 200)

cat(sprintf("N = %d | k range %d..%d | %d of %d in-range values occupied\n",
            nrow(diffs), min(diffs$k), max(diffs$k), occupied_in_range, span))
print(tab %>% select(k, diff, n_subjects), n = nrow(tab))
cat("-> plots/per2_diff_score_grid.png\n")
