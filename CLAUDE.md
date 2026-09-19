# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a speech perception research project that computes pairwise perceptual similarity distances between speaker utterances using HuBERT self-supervised embeddings and Dynamic Time Warping (DTW). It correlates embedding distances with human perceptual legality judgments from two psycholinguistics experiments.

## Repository Structure

All working scripts live in `analysis_scripts/similarity_scripts/`. Key files:

- **`app.py`** — Gradio web app deployed on Hugging Face Spaces; accepts WAV files + CSV mapping, returns pairwise embedding distances
- **`process_audio.py`** / **`process_audio2.py`** — Batch processors: MP3→WAV conversion, 16 kHz resampling, HuBERT embedding extraction, pairwise distance computation; outputs `mapping.csv` and `output_results.csv`
- **`analyze_per1.py`** / **`analyze_per2.py`** — Parse human listener response CSVs from `experiment_data/`, compute per-speaker legality effect differences
- **`process_audio_full.py`** — current batch processor: pair-level word availability, cross-vowel rows, `--standardization` (see "Audio pipeline (full)")
- **`plot_scatter_distance_vs_false_alarm.py`** — the `scatter_distance_vs_false_alarm` figures (Exp 1, Exp 2, combined); replaces the removed `plot_scatter.py` / `plot_scatter_exp1.py` / `plot_scatter_exp2.py` / `plot_scatter_combined.py`, which read the removed `output_results{,2}.csv`
- **`generate_table_viz.py`** — 5×5 speaker pairwise distance heatmap
- **`extractSingleChannel.praat`** — Praat script for extracting mono from stereo WAV before processing

## Running Scripts

```bash
cd analysis_scripts/similarity_scripts

# Install dependencies
pip install -r requirements.txt

# Launch the Gradio web app (localhost:7860)
python app.py

# Batch process audio → per_similarity_results/per{1,2}_speaker_pair_per_sim_full.csv
python3 process_audio_full.py

# Analyze perceptual experiment data
python analyze_per1.py
python analyze_per2.py

# Generate plots (saved to plots/)
python3 plot_scatter_distance_vs_false_alarm.py
python generate_table_viz.py
```

## Provenance convention

The embedding pipeline started as a reimplementation of Chernyak et al. (2024)
(repo: <https://github.com/bronichern/percept_sim>) and has since diverged. **Every pipeline step
documented here is tagged with its provenance — `[Chernyak et al.]` for a step inherited from that
work, `[ours]` for a choice we made — and any new step added to this file must carry the same tag.**
When a step is `[ours]`, say briefly why.

## Core Algorithm

1. **Feature extraction** `[Chernyak et al.]`: HuBERT base model (`facebook/hubert-base-ls960`),
   hidden state at layer 12.
2. **Standardization** `[Chernyak et al.]`: pair ("mutual") z-scoring — see "Standardization
   modes" below. This is what every reported result uses. A `global` alternative `[ours]` exists
   behind a flag and feeds one sensitivity table only.
3. **Distance** `[Chernyak et al.]`: DTW via `librosa.sequence.dtw`, normalized by
   `distance / (len(seq1) + len(seq2))`.

All audio must be 16 kHz mono WAV. Use `extractSingleChannel.praat` to convert stereo files first.
(The per1 `french_FI` recordings are 44.1 kHz and are resampled at load time; `torchaudio`'s
resampler and `librosa`'s differ enough to move those distances by ~0.1%, so the published numbers
assume `torchaudio` is installed.)

### Standardization modes

`process_audio_full.py --standardization {pair,global}` (default `pair`).

**All published results, model fits and plots use `pair`** — the Chernyak et al. (2024) procedure,
so the numbers stay comparable to that precedent. `global` exists only so the choice can be shown to
be immaterial; its output feeds `standardization_comparison.R` and nothing else.

- **`pair`** `[Chernyak et al.]` — "mutual" z-scoring, their `mut_normalize_sequences`. For each
  (speaker pair, syllable), the two frame sequences are concatenated, each of the 768 dimensions is
  z-scored over that concatenated stack, and the sequences are split apart again before DTW. The
  reference frame therefore changes with every comparison.
- **`global`** `[ours, sensitivity check only]` — one set of per-dimension means/SDs computed over
  ALL frames from ALL talkers in the experiment (saved to
  `per_similarity_results/per{1,2}_global_stats.npz`), applied to every recording before the
  identical DTW + length normalization. Motivation: the pair procedure divides each dimension by the
  SD of the two talkers being compared, so it inflates exactly the dimensions on which those talkers
  differ most, which could compress the between-talker distance more for pairs that are farther
  apart. **The check says it doesn't matter** — see the table below.

Both modes share one feature-extraction pass. Layer-12 features are cached as one `.npy` per
recording under `feature_cache/per{1,2}/` (gitignored, ~97 MB, regenerable) `[ours]`, so the second
mode costs seconds rather than a full re-embedding. `--force-features` re-extracts.

### The one standardization comparison

`cd analysis_scripts/original_scripts && Rscript standardization_comparison.R` writes the single
table `model_results/standardization_comparison.csv`. It reports, per model fit: how well the two
distance tables agree, and the `legality × avg_distance_c` interaction under each.

| experiment | dist r | pair-mean r | range ratio | β pair | β global | p pair | p global | AIC pair | AIC global |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Per1 (combined) | 0.940 | 0.987 | 0.94 | 0.1858 | 0.1822 | 6.0e-11 | 2.8e-10 | 19129.70 | 19130.69 |
| Per1A | 0.940 | 0.987 | 0.94 | 0.1716 | 0.1834 | 5.9e-07 | 4.0e-07 | 11121.25 | 11120.64 |
| Per1B | 0.940 | 0.987 | 0.94 | 0.1337 | 0.1203 | 1.6e-02 | 1.6e-02 | 8088.23 | 8088.23 |
| Per2 | 0.996 | 0.998 | 1.05 | 0.2764 | 0.2693 | 2.8e-05 | 3.4e-05 | 6990.25 | 6990.63 |

`dist r` is the Pearson correlation between the two methods across all (speaker-pair × syllable)
rows; `pair-mean r` is the same at the speaker-pair level; `range ratio` is the spread of pair-level
means under global ÷ under pair. Every β moves by ≤ 0.013, no p-value crosses a threshold, and the
largest AIC difference is 0.99. That is the whole justification for reporting the pair numbers and
not carrying two sets of results.

## Key Constants (Hardcoded)

- HuBERT layer: `12`
- Sampling rate: `16000` Hz
- Speakers (Exp 1): `EN_F1`, `EN_M1`, `FR_F1`, `FR_M1`, `FR_F2`
- Audio folder names map to experiment speaker labels: `NS_F→EN_F1`, `NS_M→EN_M1`, `french_F→FR_F1`, `french_M→FR_M1`, `french_FI→FR_F2`

## Data Files

- `mapping.csv` / `mapping2.csv` — speaker pair combinations (auto-generated by `process_audio.py`)
- `output_results.csv` / `output_results2.csv` — computed pairwise distances (S1, S2, distance)
- `experiment_data/per1_filteredSubjects.csv` / `per2_filteredSubjects.csv` — quality-filtered human listener responses (~21 MB and ~7 MB)

- `per_similarity_results/per{1,2}_speaker_pair_per_sim_full.csv` — per-syllable pairwise distances, pair standardization; **what every R script and plot reads**
- `per_similarity_results/per{1,2}_speaker_pair_per_sim_full_global.csv` — the same rows under global standardization; read only by `standardization_comparison.R`
- `per_similarity_results/per{1,2}_global_stats.npz` — the 768-dim mean/SD reference used by global mode, plus `n_frames` and `layer`, saved so the reference is inspectable
- `model_results/standardization_comparison.csv` — the single pair-vs-global sensitivity table

Audio files (~50 GB total) are tracked via Git LFS (`audio/per1_audio/`, `audio/per2_audio/`).
`feature_cache/` holds cached layer-12 features (gitignored, regenerable).

## Tech Stack

- **ML/Audio**: `torch` 2.2.2, `transformers` 4.46.2, `torchaudio` 2.2.2, `librosa` 0.10.2, `scipy` 1.14.1
- **Data**: `pandas` 2.2.3, `numpy` 1.26.4
- **Web**: `gradio` 5.5.0 (Hugging Face Spaces deployment, GPU-enabled)
- **Viz**: `matplotlib`, `seaborn`, `plotly`

## Embedding-distance regression analysis

The mixed-effects regressions live in `analysis_scripts/original_scripts/` alongside the original (condition-contrast-coded) scripts they replace:

- **`per1_analysis_embedding.R`** / **`per2_analysis_embedding.R`** — `glmer(response ~ legality * avg_distance_c + …)` for each experiment, with sub-experiment fits (1A, 1B) and an AIC comparison vs. the original condition-coded model. Generates per-participant scatterplots in `plots/`.
- **`standardization_comparison.R`** — the single pair-vs-global sensitivity table (see "The one standardization comparison"); the only script that reads the `_global` distance CSVs.
- **`embedding_summary_table.R`** — refits all four embedding models (Per1 combined, 1A, 1B, Per2) plus their condition-coded counterparts and writes `model_results/embedding_interaction_summary.csv` (β, SE, z, p, AICs, ΔAIC for the `legality × avg_distance_c` interaction), one row per experiment, pair standardization. Also generates `plots/all_experiments_fa_by_distance.png` (3-panel facet by experiment) and `plots/all_experiments_combined_fa_by_distance.png` (single overlay plot, distances z-scored within experiment so the three experiments share an x-axis). Compares only embedding-only vs condition-only models — no combined / matched / fully-controlled variants, since `avg_distance` is too collinear with the condition contrasts (especially in Per2, where it's perfectly collinear) for those comparisons to be interpretable.
- **`analysis_functions.R`** — shared helpers from the original analysis (`open_file`, `chiReport.func`, etc.); the embedding scripts source it.

The R scripts assume working directory `analysis_scripts/original_scripts/`, with `data/` (symlink → `../../experiment_data`), `model_results/` (symlink → `../../model_results`), and `plots/` (symlink → `../../plots`) resolving from there. The `setwd(rstudioapi::…)` line is a no-op when run via `Rscript`, so `cd analysis_scripts/original_scripts && Rscript per1_analysis_embedding.R` works.

### Per-participant embedding distance

Each participant hears exactly two distinct speakers across BOTH the familiarization and test phases (the "shared" vs "different" condition labels distinguish accent-group similarity, not phase-specific speaker identity). For each participant:

1. Extract their two unique speakers by stripping `^study-|^test-(legal|illegal)-` from `label` across all rows.
2. Extract every unique syllable they heard across all rows (fam + test combined — some conditions, e.g. 1A `strong-dif`, only hear French [y] during fam).
3. Look up the per-syllable HuBERT/DTW distance for each (speaker_pair, syllable) tuple in the `_full` similarity CSV and average them. The result is a single `avg_distance` per worker.

This construction `[ours]` is unchanged by the standardization mode — only the distance table swaps, which is what makes the sensitivity check a clean one-variable comparison.

Granularity in practice: distances vary at the **(sublist × speaker-pair counterbalance)** level, not strictly per worker. Each condition splits across many sublists (encoded in `listname` like `A1i`, `A2ii`, `B1i`) that draw different syllable subsets and assign different specific speakers; participants in the same sublist share an identical `avg_distance`, but participants in different sublists of the same condition do not. So per-participant ≠ per-condition averaging.

### Audio pipeline (full)

`analysis_scripts/similarity_scripts/process_audio_full.py` is the upgraded version of `process_audio.py`. Differences:

- **Pair-level word availability** `[ours]` instead of global intersection — French-French pairs include [y] syllables (which English speakers don't have recordings for). Why: a global intersection throws away the [y] stimuli that carry the strong accent cue.
- **Cross-vowel rows** `[ours]` for cross-accent pairs: French [y] vs English [u] in the same consonant frame (e.g. `french_M_fyf` vs `NS_M_fuf`). Why: English perceives French [y] as [u], so for `strong-dif` participants who hear French [y] tokens, the perceptually relevant comparison is against the matching English [u].
- **Selectable standardization** `[ours]` — `--standardization {pair,global}`, see "Standardization modes" above.

Outputs (row set is identical across modes):

| file | mode | rows |
| --- | --- | --- |
| `per_similarity_results/per1_speaker_pair_per_sim_full.csv` | pair (default) | 1,044 |
| `per_similarity_results/per2_speaker_pair_per_sim_full.csv` | pair (default) | 960 |
| `per_similarity_results/per1_speaker_pair_per_sim_full_global.csv` | global | 1,044 |
| `per_similarity_results/per2_speaker_pair_per_sim_full_global.csv` | global | 960 |
| `per_similarity_results/per{1,2}_global_stats.npz` | global | 768 means + 768 SDs, `n_frames`, `layer` |

The `_global` CSVs carry an extra `standardization = "global"` column so they can never be mistaken for the reported table; the pair CSVs keep their original `S1,S2,distance` columns. The older `per{1,2}_speaker_pair_per_sim.csv` files (built with global word intersection) are stale.

```bash
# pair standardization — reproduces the existing _full CSVs (default, unchanged)
python3 analysis_scripts/similarity_scripts/process_audio_full.py
# global standardization — sensitivity check only; writes the _global CSVs + stats npz
python3 analysis_scripts/similarity_scripts/process_audio_full.py --standardization global
# other flags: --experiment {per1,per2,both}  --output-dir DIR  --force-features
```

~5–10 min on CPU for the first run (feature extraction dominates); subsequent runs in either mode take seconds off the `feature_cache/` npy cache.

### Distance vs. legality effect scatters

`python3 analysis_scripts/similarity_scripts/plot_scatter_distance_vs_false_alarm.py` draws the
downstream view: pair-level embedding distance vs. the mean (legal − illegal) difference from
`experiment_data/per{1,2}_speaker_legality_diffs.csv`. Writes
`plots/scatter_distance_vs_false_alarm_exp1.png`, `..._exp2.png` and `..._combined.png`.

Current values (pair standardization): Exp 1 r = 0.835 (N = 6 pairs), Exp 2 r = 0.966 (N = 8),
pooled r = 0.456 (N = 14). The pooled r sits far below both within-experiment values because Per1
and Per2 distances live on different scales — pooling across experiments isn't meaningful, which is
what the combined figure shows.

**Two fixes vs. the superseded figures** (which came from `output_results{,2}.csv` via
`process_audio.py`):

1. `process_audio.py` resamples to 16 kHz **only inside its MP3-conversion branch**. The per1
   `french_FI` recordings are 44.1 kHz WAVs with no MP3 source, so they reached HuBERT at 44.1 kHz.
   That inflated both french_FI pair distances to ~25–26 instead of ~16–18 and pulled Exp 1's r down
   to 0.710. `process_audio_full.py` resamples at load time regardless of source format `[ours]`.
2. `per1_speaker_legality_diffs.csv` contains a duplicated `EN_F1,FR_F1` row, which the old scripts
   plotted twice (N = 7 for 6 pairs). The new script deduplicates on the pair key.

Exp 2 reproduces the old r exactly (0.966) — no french_FI, no resampling involved — which is the
check that the rest of the reconstruction is faithful.

### Still pair-standardized elsewhere

`app.py` (the Hugging Face Space) uses `mut_normalize_sequences` and cites Chernyak et al. (2024)
in its description. It is deliberately left on the paired method and is not part of the analysis
pipeline.
