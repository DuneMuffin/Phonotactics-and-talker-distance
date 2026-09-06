import os
import torch
import torchaudio
import numpy as np
import pandas as pd
from scipy.stats import zscore
from librosa.sequence import dtw as lib_dtw
from transformers import HubertModel
from itertools import combinations
from tqdm import tqdm

# --- Functions from app.py ---

def mut_normalize_sequences(sq1, sq2, normalize: bool):
    if normalize:
        sq1 = np.copy(sq1)
        sq2 = np.copy(sq2)
        len_sq1 = sq1.shape[0]
        arr = np.concatenate((sq1, sq2), axis=0)
        for dim in range(sq1.shape[1]):
            arr[:, dim] = zscore(arr[:, dim])
        sq1 = arr[:len_sq1, :]
        sq2 = arr[len_sq1:, :]
    return sq1, sq2

def librosa_dtw(sq1, sq2):
    return lib_dtw(sq1.transpose(), sq2.transpose())[0][-1, -1]

def time_txt(time, time_frame=5):
    if time % time_frame == 0:
        return f"{round(time * 0.02, 2)}"
    return ""

def create_df(feats, speaker_len, names, time_frame=5):
    cols = [f"val {i}" for i in range(feats.shape[1])]
    df = pd.DataFrame(feats, columns=cols)
    df['idx'] = df.index
    time_index = {i: speaker_len[i] for i in range(len(speaker_len))}
    com_time_index = {i: sum(speaker_len[:i]) for i in range(len(speaker_len))}
    df_speaker_count = pd.Series(time_index)
    df_speaker_count = df_speaker_count.reindex(df_speaker_count.index.repeat(df_speaker_count.to_numpy())).rename_axis(
        'speaker_id').reset_index()
    df['speaker_id'] = df_speaker_count['speaker_id']
    df['speaker_len'] = df['speaker_id'].apply(lambda row: speaker_len[row])
    df['com_sum'] = df['speaker_id'].apply(lambda i: com_time_index[i])
    df['speaker'] = df['speaker_id'].apply(lambda i: names[i])
    df['time'] = df['idx'] - df['com_sum']
    df['time_txt'] = df[['time', 'speaker_len']].apply(lambda row: time_txt(row['time'], time_frame), axis=1)
    df_subset = df.copy()
    data_subset = df_subset[cols].values
    return data_subset, df_subset, cols

def calc_distance(df_subset, speaker1, speaker2, cols):
    features_speaker1 = df_subset[df_subset['speaker'] == speaker1][cols].to_numpy()
    features_speaker2 = df_subset[df_subset['speaker'] == speaker2][cols].to_numpy()
    features_speaker1, features_speaker2 = mut_normalize_sequences(features_speaker1, features_speaker2, True)
    distance = librosa_dtw(features_speaker1, features_speaker2)
    distance = distance / (len(features_speaker1) + len(features_speaker2))
    return distance

# --- Main Logic ---

audio_dir = 'audio'
expected_sr = 16000
speakers = ['french_M', 'french_F', 'french_FI', 'NS_M', 'NS_F']

# 1. Convert MP3 to WAV and identify speaker words
print("Converting MP3 to WAV and identifying words...")
speaker_words = {s: set() for s in speakers}

for filename in os.listdir(audio_dir):
    if filename.endswith('.mp3'):
        input_path = os.path.join(audio_dir, filename)
        wav_filename = filename.replace('.mp3', '.wav')
        wav_path = os.path.join(audio_dir, wav_filename)
        
        # Load and Resample
        wav, sr = torchaudio.load(input_path)
        if sr != expected_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=expected_sr)
            wav = resampler(wav)
        
        # Save as WAV (mono)
        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
        torchaudio.save(wav_path, wav, expected_sr)
        
        filename_for_parts = wav_filename
    elif filename.endswith('.wav'):
        filename_for_parts = filename
    else:
        continue

    parts = filename_for_parts.replace('.wav', '').split('_')
    if len(parts) >= 3:
        speaker_key = f"{parts[0]}_{parts[1]}"
        word = parts[2]
        if speaker_key in speakers:
            speaker_words[speaker_key].add(word)

common_words = set.intersection(*speaker_words.values())
print(f"Found {len(common_words)} common words.")

# 2. Generate mapping.csv
print("Generating mapping.csv...")
pairs = []
speaker_pairs = list(combinations(speakers, 2))
for word in sorted(common_words):
    for s1, s2 in speaker_pairs:
        pairs.append({'S1': f"{s1}_{word}", 'S2': f"{s2}_{word}"})

map_df = pd.DataFrame(pairs)
map_df.to_csv('mapping.csv', index=False)
print(f"Created mapping.csv with {len(pairs)} pairs.")

# 3. Load Model and Generate Features
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading HuBERT model on {device}...")
model = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(device)

all_wav_paths = []
for word in common_words:
    for s in speakers:
        all_wav_paths.append(os.path.join(audio_dir, f"{s}_{word}.wav"))

print(f"Extracting features for {len(all_wav_paths)} files...")
features = None
speaker_len = []
names = []
layer = 12

for wav_path in tqdm(all_wav_paths):
    wav, sr = torchaudio.load(wav_path)
    wav = wav.to(device)
    with torch.no_grad():
        outputs = model(wav, return_dict=True, output_hidden_states=True)
        wav_features = outputs.hidden_states[layer].squeeze().cpu().numpy()
    
    features = wav_features if features is None else np.concatenate([features, wav_features], axis=0)
    speaker_len.append(wav_features.shape[0])
    names.append(os.path.basename(wav_path).replace('.wav', ''))

# 4. Calculate distances
print("Calculating distances...")
data_subset, df_subset, hubert_feature_columns = create_df(features, speaker_len, names)

results = []
for index, row in tqdm(map_df.iterrows(), total=len(map_df)):
    try:
        dist = calc_distance(df_subset, row['S1'], row['S2'], hubert_feature_columns)
        results.append([row['S1'], row['S2'], dist])
    except Exception as e:
        print(f"Error calculating distance for {row['S1']} and {row['S2']}: {e}")

output_df = pd.DataFrame(results, columns=['S1', 'S2', 'distance'])
output_df.to_csv('output_results.csv', index=False)
print("Saved results to output_results.csv")
