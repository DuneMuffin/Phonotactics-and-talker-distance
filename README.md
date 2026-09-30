# Phonetics and Talker Distance

Speech-perception research on whether a self-supervised speech model's
representations of talker differences predict human phonotactic learning.

The project computes pairwise **perceptual similarity distances** between speaker
utterances using **HuBERT** self-supervised embeddings and **Dynamic Time
Warping (DTW)**, then correlates those embedding distances with human perceptual
legality judgments from two psycholinguistics experiments (Per1, which splits
into sub-experiments 1A and 1B, and Per2).

## Core idea

For each participant, we measure a **legality effect** — how much more they
endorse legal than illegal test items — and ask whether the acoustic distance
between the talkers they heard predicts the strength of that effect. Per-subject
distance is built by looking up the HuBERT/DTW distance for every
(speaker-pair × syllable) a participant heard and averaging.

## Algorithm

1. **Feature extraction** — HuBERT base model (`facebook/hubert-base-ls960`),
   hidden state at layer 12.
2. **Normalization** — paired sequences are concatenated, z-score normalized per
   dimension, then split.
3. **Distance** — DTW via `librosa.sequence.dtw`, normalized by
   `distance / (len(seq1) + len(seq2))`.

All audio must be 16 kHz mono WAV. `extractSingleChannel.praat` converts stereo
files to mono first.

## Repository structure

```
analysis_scripts/
  similarity_scripts/     Python: audio processing + HuBERT embedding + distances
    app.py                  Gradio web app (also deployed on Hugging Face Spaces)
    process_audio*.py       Batch: MP3→WAV, 16 kHz resample, embeddings, distances
    process_audio_full.py   Upgraded pipeline (pair-level word availability,
                            cross-vowel rows for cross-accent pairs)
    analyze_per{1,2}.py     Parse listener responses, per-speaker legality diffs
    plot_scatter*.py        Regression scatter plots with Pearson correlations
    generate_table_viz.py   5×5 speaker pairwise-distance heatmap
    extractSingleChannel.praat
    requirements.txt
  original_scripts/        R: mixed-effects regression analyses
    per{1,2}_analysis_embedding.R   glmer(response ~ legality * avg_distance_c + …)
    embedding_summary_table.R       Refit all models, write summary table + plots
    plot_legality_diff_distributions.R  Per-subject legality difference-score dists
    test_bimodality.R / test_bimodality_ranef.R  Unimodality checks (dip test,
                            Gaussian-mixture BIC, noise-shrunk random slopes)
    test_bimodality_varying_mean.R  Distance-varying mixture weight vs.
                            distance-varying unimodal mean (AIC/BIC +
                            parametric bootstrap)
    analysis_functions.R    Shared helpers (open_file, chiReport.func, …)
experiment_data/          Human listener response CSVs (already public elsewhere)
per_similarity_results/   Computed per-syllable pairwise distances (_full CSVs)
model_results/            Fitted-model summaries and comparison tables
plots/                    Generated figures
```

> **Audio files are not tracked in this repository** (distributed separately).

## Running

### Python pipeline

```bash
cd analysis_scripts/similarity_scripts
pip install -r requirements.txt

python app.py                 # Gradio web app (localhost:7860)
python process_audio_full.py  # Batch process audio → similarity CSVs
python analyze_per1.py        # Analyze perceptual experiment data
python plot_scatter_combined.py
```

`process_audio_full.py` outputs
`per_similarity_results/per1_speaker_pair_per_sim_full.csv` (1,044 rows) and
`per2_speaker_pair_per_sim_full.csv` (960 rows); the `_full` CSVs are the ones
the R scripts read.

### R analyses

The R scripts expect the working directory `analysis_scripts/original_scripts/`,
where `data/`, `model_results/`, and `plots/` are symlinks resolving to the
repository-root directories.

```bash
cd analysis_scripts/original_scripts
Rscript per1_analysis_embedding.R
Rscript per2_analysis_embedding.R
Rscript embedding_summary_table.R
```

R dependencies: `lme4`, `dplyr`, `plyr`, `ggplot2`, `stringr`, `boot`, and
(for the bimodality checks) `diptest`, `mclust`.

## Key constants

- HuBERT layer: `12`
- Sampling rate: `16000` Hz
- Speakers (Exp 1): `EN_F1`, `EN_M1`, `FR_F1`, `FR_M1`, `FR_F2`

## Tech stack

- **ML / audio**: `torch`, `transformers`, `torchaudio`, `librosa`, `scipy`
- **Data**: `pandas`, `numpy`
- **Web**: `gradio` (Hugging Face Spaces, GPU-enabled)
- **Stats / viz**: R (`lme4`), `matplotlib`, `seaborn`, `plotly`
