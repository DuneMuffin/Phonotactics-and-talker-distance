"""
Compute pairwise HuBERT embedding distances for ALL (speaker_pair, word) tuples
where both speakers in the pair have the word recorded.

Differs from process_audio.py in that it uses pair-level word availability
(intersection per pair) rather than a global intersection, so words available
to only a subset of speakers (e.g. [y]-vowel syllables present for the French
speakers but not the native English speakers) are still included for the
pairs that do share them.
"""

import os
from itertools import combinations

import numpy as np
import pandas as pd
import torch
import torchaudio
from scipy.stats import zscore
from librosa.sequence import dtw as lib_dtw
from transformers import HubertModel
from tqdm import tqdm


HUBERT_LAYER = 12
EXPECTED_SR = 16000


def mut_normalize_sequences(sq1, sq2):
    sq1 = np.copy(sq1)
    sq2 = np.copy(sq2)
    len_sq1 = sq1.shape[0]
    arr = np.concatenate((sq1, sq2), axis=0)
    for dim in range(sq1.shape[1]):
        arr[:, dim] = zscore(arr[:, dim])
    return arr[:len_sq1, :], arr[len_sq1:, :]


def calc_distance(feat1, feat2):
    f1, f2 = mut_normalize_sequences(feat1, feat2)
    distance = lib_dtw(f1.transpose(), f2.transpose())[0][-1, -1]
    return distance / (len(f1) + len(f2))


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


def extract_features(audio_dir, speaker_words, model, device):
    features = {}
    for s in sorted(speaker_words):
        for word in tqdm(sorted(speaker_words[s]), desc=f"features {s}"):
            wav_path = os.path.join(audio_dir, f"{s}_{word}.wav")
            wav, sr = torchaudio.load(wav_path)
            if sr != EXPECTED_SR:
                wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=EXPECTED_SR)(wav)
            if wav.shape[0] > 1:
                wav = torch.mean(wav, dim=0, keepdim=True)
            wav = wav.to(device)
            with torch.no_grad():
                out = model(wav, return_dict=True, output_hidden_states=True)
            features[(s, word)] = out.hidden_states[HUBERT_LAYER].squeeze(0).cpu().numpy()
    return features


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
    for fw in french_words:
        if len(fw) >= 2 and fw[1] == 'y':
            ew = fw[0] + 'u' + fw[2:]
            if ew in english_words:
                pairs.append((fw, ew))
    return pairs


def process_experiment(audio_dir, speakers, output_path):
    speaker_words = discover_speaker_words(audio_dir, speakers)
    print(f"\nAudio dir: {audio_dir}")
    for s in speakers:
        print(f"  {s}: {len(speaker_words[s])} words")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nLoading HuBERT model on {device}...")
    model = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(device)
    model.eval()

    features = extract_features(audio_dir, speaker_words, model, device)

    print("\nComputing pairwise distances...")
    results = []
    for s1, s2 in combinations(speakers, 2):
        # Same-word comparisons (intersection of available words)
        common = sorted(speaker_words[s1] & speaker_words[s2])
        print(f"  {s1} vs {s2}: {len(common)} shared words")
        for word in common:
            d = calc_distance(features[(s1, word)], features[(s2, word)])
            results.append({'S1': f"{s1}_{word}",
                            'S2': f"{s2}_{word}",
                            'distance': d})

        # Cross-vowel comparisons: French [y] vs English [u] in same consonant frame.
        # English perceives French [y] as [u], so for cross-accent pairs we want
        # to compare these even though the underlying audio differs in vowel.
        if is_french(s1) and is_english(s2):
            for fw, ew in cross_vowel_pairs(speaker_words[s1], speaker_words[s2]):
                d = calc_distance(features[(s1, fw)], features[(s2, ew)])
                results.append({'S1': f"{s1}_{fw}",
                                'S2': f"{s2}_{ew}",
                                'distance': d})
        elif is_french(s2) and is_english(s1):
            for fw, ew in cross_vowel_pairs(speaker_words[s2], speaker_words[s1]):
                d = calc_distance(features[(s2, fw)], features[(s1, ew)])
                results.append({'S1': f"{s2}_{fw}",
                                'S2': f"{s1}_{ew}",
                                'distance': d})

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"\nSaved {len(results)} rows to {output_path}")


if __name__ == "__main__":
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    process_experiment(
        audio_dir=os.path.join(repo_root, "audio", "per1_audio"),
        speakers=['french_M', 'french_F', 'french_FI', 'NS_M', 'NS_F'],
        output_path=os.path.join(repo_root, "per_similarity_results",
                                 "per1_speaker_pair_per_sim_full.csv")
    )

    process_experiment(
        audio_dir=os.path.join(repo_root, "audio", "per2_audio"),
        speakers=['EN_F', 'EN_M', 'HI_F', 'HI_M', 'HU_F', 'HU_M'],
        output_path=os.path.join(repo_root, "per_similarity_results",
                                 "per2_speaker_pair_per_sim_full.csv")
    )
