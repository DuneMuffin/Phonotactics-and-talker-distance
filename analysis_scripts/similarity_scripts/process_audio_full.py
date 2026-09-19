"""
Compute pairwise HuBERT embedding distances for ALL (speaker_pair, word) tuples
where both speakers in the pair have the word recorded.

Differs from process_audio.py in that it uses pair-level word availability
(intersection per pair) rather than a global intersection, so words available
to only a subset of speakers (e.g. [y]-vowel syllables present for the French
speakers but not the native English speakers) are still included for the
pairs that do share them.

Two standardization modes are available (--standardization):

  pair   (default)  Chernyak et al. (2024) procedure: the two frame sequences
                    being compared are concatenated, each of the 768 dimensions
                    is z-scored over that concatenated stack, and the sequences
                    are split apart again before DTW. Reproduces
                    `mut_normalize_sequences` in github.com/bronichern/percept_sim.

  global (ours)     One set of per-dimension means/SDs is computed over ALL
                    frames from ALL talkers in the experiment, and every
                    recording is standardized with those. Pair standardization
                    inflates the SD of exactly the dimensions on which the two
                    talkers differ most, which shrinks the between-talker
                    distance, and shrinks it more for pairs that are farther
                    apart; global standardization measures every pair in one
                    common space.

Both modes share the same feature-extraction step and the same row set; only
the standardization applied before DTW differs.
"""

import argparse
import os
from itertools import combinations

import numpy as np
import pandas as pd
import torch
from scipy.stats import zscore
from librosa.sequence import dtw as lib_dtw
from transformers import HubertModel
from tqdm import tqdm


try:
    import torchaudio
except ImportError:  # torchaudio is optional; soundfile can read the 16 kHz WAVs
    torchaudio = None


HUBERT_LAYER = 12
EXPECTED_SR = 16000

EXPERIMENTS = {
    "per1": {
        "audio_subdir": os.path.join("audio", "per1_audio"),
        "speakers": ['french_M', 'french_F', 'french_FI', 'NS_M', 'NS_F'],
    },
    "per2": {
        "audio_subdir": os.path.join("audio", "per2_audio"),
        "speakers": ['EN_F', 'EN_M', 'HI_F', 'HI_M', 'HU_F', 'HU_M'],
    },
}


def mut_normalize_sequences(sq1, sq2):
    sq1 = np.copy(sq1)
    sq2 = np.copy(sq2)
    len_sq1 = sq1.shape[0]
    arr = np.concatenate((sq1, sq2), axis=0)
    for dim in range(sq1.shape[1]):
        arr[:, dim] = zscore(arr[:, dim])
    return arr[:len_sq1, :], arr[len_sq1:, :]


def dtw_distance(f1, f2):
    """Length-normalized DTW between two (already standardized) frame sequences."""
    distance = lib_dtw(f1.transpose(), f2.transpose())[0][-1, -1]
    return distance / (len(f1) + len(f2))


def calc_distance(feat1, feat2):
    """Pair-standardized distance (Chernyak et al. 2024)."""
    f1, f2 = mut_normalize_sequences(feat1, feat2)
    return dtw_distance(f1, f2)


def discover_speaker_words(audio_dir, speakers):
    speaker_words = {s: set() for s in speakers}
    for filename in os.listdir(audio_dir):
        if not filename.endswith('.wav'):
            continue
        base = filename[:-4]
        for s in speakers:
            if base.startswith(s + "_"):
                speaker_words[s].add(base[len(s) + 1:])
                break
    return speaker_words


def load_wav(wav_path):
    """Load a WAV as a (channels, samples) float32 tensor. Prefers torchaudio;
    falls back to soundfile when torchaudio is unavailable or lacks a usable
    backend (both return the same [-1, 1) normalized samples for PCM WAV)."""
    if torchaudio is not None:
        try:
            return torchaudio.load(wav_path)
        except Exception:
            pass
    import soundfile as sf
    data, sr = sf.read(wav_path, dtype="float32", always_2d=True)
    return torch.from_numpy(data.T.copy()), sr


_WARNED_RESAMPLE = False


def resample(wav, orig_sr, new_sr):
    """Resample to 16 kHz. Only the per1 french_FI recordings (44.1 kHz) need
    this; torchaudio's resampler and librosa's differ enough to move those
    distances by ~0.1%, so torchaudio is used whenever it is importable."""
    global _WARNED_RESAMPLE
    if torchaudio is not None:
        return torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=new_sr)(wav)
    if not _WARNED_RESAMPLE:
        print("  WARNING: torchaudio unavailable; resampling with librosa. "
              "Distances for non-16 kHz recordings will differ slightly from "
              "the published values.")
        _WARNED_RESAMPLE = True
    import librosa
    return torch.from_numpy(
        librosa.resample(wav.numpy(), orig_sr=orig_sr, target_sr=new_sr))


def extract_one(audio_dir, speaker, word, model, device):
    wav_path = os.path.join(audio_dir, f"{speaker}_{word}.wav")
    wav, sr = load_wav(wav_path)
    if sr != EXPECTED_SR:
        wav = resample(wav, sr, EXPECTED_SR)
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)
    wav = wav.to(device)
    with torch.no_grad():
        out = model(wav, return_dict=True, output_hidden_states=True)
    return out.hidden_states[HUBERT_LAYER].squeeze(0).cpu().numpy()


def load_features(audio_dir, speaker_words, cache_dir=None, force=False):
    """Layer-12 features for every recording, reading/writing an npy cache.

    The model is only loaded if at least one recording is missing from the
    cache, so a second standardization mode over the same audio is free.
    """
    wanted = [(s, w) for s in sorted(speaker_words) for w in sorted(speaker_words[s])]

    features = {}
    missing = []
    for s, w in wanted:
        cache_path = cache_dir and os.path.join(cache_dir, f"{s}_{w}.npy")
        if cache_path and not force and os.path.exists(cache_path):
            features[(s, w)] = np.load(cache_path)
        else:
            missing.append((s, w))

    if cache_dir:
        print(f"  feature cache: {len(features)} hit, {len(missing)} to extract")

    if missing:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  loading HuBERT model on {device}...")
        model = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(device)
        model.eval()
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        for s, w in tqdm(missing, desc="extracting features"):
            feat = extract_one(audio_dir, s, w, model, device)
            features[(s, w)] = feat
            if cache_dir:
                np.save(os.path.join(cache_dir, f"{s}_{w}.npy"), feat)

    return features


def compute_global_stats(features):
    """Per-dimension mean/SD over all frames of all recordings (population SD,
    ddof=0, matching scipy.stats.zscore's default used by the pair mode)."""
    n_dim = next(iter(features.values())).shape[1]
    total = np.zeros(n_dim, dtype=np.float64)
    total_sq = np.zeros(n_dim, dtype=np.float64)
    n_frames = 0
    for feat in features.values():
        arr = feat.astype(np.float64)
        total += arr.sum(axis=0)
        total_sq += (arr ** 2).sum(axis=0)
        n_frames += arr.shape[0]
    mean = total / n_frames
    var = total_sq / n_frames - mean ** 2
    std = np.sqrt(np.clip(var, 0.0, None))
    return mean, std, n_frames


def enumerate_comparisons(speakers, speaker_words, verbose=True):
    """Every (S1_key, S2_key) comparison, in the canonical output row order:
    for each speaker pair, the shared-word rows followed by the cross-vowel rows."""
    comparisons = []
    for s1, s2 in combinations(speakers, 2):
        # Same-word comparisons (intersection of available words)
        common = sorted(speaker_words[s1] & speaker_words[s2])
        if verbose:
            print(f"  {s1} vs {s2}: {len(common)} shared words")
        for word in common:
            comparisons.append(((s1, word), (s2, word)))

        # Cross-vowel comparisons: French [y] vs English [u] in same consonant frame.
        # English perceives French [y] as [u], so for cross-accent pairs we want
        # to compare these even though the underlying audio differs in vowel.
        if is_french(s1) and is_english(s2):
            for fw, ew in cross_vowel_pairs(speaker_words[s1], speaker_words[s2]):
                comparisons.append(((s1, fw), (s2, ew)))
        elif is_french(s2) and is_english(s1):
            for fw, ew in cross_vowel_pairs(speaker_words[s2], speaker_words[s1]):
                comparisons.append(((s2, fw), (s1, ew)))
    return comparisons


def is_french(speaker):
    return speaker.startswith("french")


def is_english(speaker):
    return speaker.startswith("NS_")


def cross_vowel_pairs(french_words, english_words):
    """Map each French [y]-vowel word to its English [u]-vowel counterpart
    with the same consonant frame (e.g. 'fyf' -> 'fuf'). English speakers
    don't produce [y]; perceptually English listeners hear French [y] as [u].
    """
    pairs = []
    for fw in sorted(french_words):  # sorted so row order is deterministic
        if len(fw) >= 2 and fw[1] == 'y':
            ew = fw[0] + 'u' + fw[2:]
            if ew in english_words:
                pairs.append((fw, ew))
    return pairs


def process_experiment(audio_dir, speakers, output_path, standardization="pair",
                       cache_dir=None, stats_path=None, force_features=False):
    speaker_words = discover_speaker_words(audio_dir, speakers)
    print(f"\nAudio dir: {audio_dir}")
    for s in speakers:
        print(f"  {s}: {len(speaker_words[s])} words")

    print(f"\nLoading layer-{HUBERT_LAYER} features...")
    features = load_features(audio_dir, speaker_words, cache_dir=cache_dir,
                             force=force_features)

    if standardization == "global":
        mean, std, n_frames = compute_global_stats(features)
        print(f"\nGlobal stats over {n_frames} frames from {len(features)} recordings: "
              f"{len(mean)} dims, SD range [{std.min():.4f}, {std.max():.4f}]")
        if stats_path:
            os.makedirs(os.path.dirname(stats_path), exist_ok=True)
            np.savez(stats_path, mean=mean, std=std,
                     n_frames=n_frames, layer=HUBERT_LAYER)
            print(f"  saved global stats to {stats_path}")
        if np.any(std == 0) or np.any(np.isnan(std)) or np.any(np.isnan(mean)):
            raise ValueError("global stats contain zero or NaN SDs")
        # Standardize once per recording, not once per comparison.
        features = {k: (v - mean) / std for k, v in features.items()}

    print(f"\nComputing pairwise distances ({standardization} standardization)...")
    comparisons = enumerate_comparisons(speakers, speaker_words)

    results = []
    for (s1, w1), (s2, w2) in tqdm(comparisons, desc="DTW"):
        if standardization == "pair":
            d = calc_distance(features[(s1, w1)], features[(s2, w2)])
        else:
            d = dtw_distance(features[(s1, w1)], features[(s2, w2)])
        row = {'S1': f"{s1}_{w1}", 'S2': f"{s2}_{w2}", 'distance': d}
        if standardization == "global":
            row['standardization'] = "global"
        results.append(row)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"\nSaved {len(results)} rows to {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--standardization", choices=["pair", "global"], default="pair",
                        help="pair = Chernyak et al. (2024) mutual z-scoring (default); "
                             "global = one set of per-dimension stats over all talkers")
    parser.add_argument("--experiment", choices=["per1", "per2", "both"], default="both")
    parser.add_argument("--repo-root", default=None,
                        help="repo root (default: inferred from this file's location)")
    parser.add_argument("--output-dir", default=None,
                        help="where to write result CSVs (default: <repo>/per_similarity_results)")
    parser.add_argument("--cache-dir", default=None,
                        help="feature cache root (default: <repo>/feature_cache); "
                             "pass '' to disable caching")
    parser.add_argument("--force-features", action="store_true",
                        help="re-extract embeddings even when cached")
    args = parser.parse_args()

    repo_root = args.repo_root or os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", ".."))
    output_dir = args.output_dir or os.path.join(repo_root, "per_similarity_results")
    cache_root = os.path.join(repo_root, "feature_cache") if args.cache_dir is None \
        else (args.cache_dir or None)

    names = ["per1", "per2"] if args.experiment == "both" else [args.experiment]
    suffix = "_global" if args.standardization == "global" else ""

    for name in names:
        cfg = EXPERIMENTS[name]
        process_experiment(
            audio_dir=os.path.join(repo_root, cfg["audio_subdir"]),
            speakers=cfg["speakers"],
            output_path=os.path.join(output_dir,
                                     f"{name}_speaker_pair_per_sim_full{suffix}.csv"),
            standardization=args.standardization,
            cache_dir=os.path.join(cache_root, name) if cache_root else None,
            stats_path=os.path.join(output_dir, f"{name}_global_stats.npz"),
            force_features=args.force_features,
        )


if __name__ == "__main__":
    main()
