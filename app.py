import numpy as np
import pandas as pd
import torch
from transformers import HubertModel
import torchaudio
from scipy.stats import zscore
from librosa.sequence import dtw as lib_dtw
import gradio as gr
import spaces
from itertools import combinations
import os


def mut_normalize_sequences(sq1, sq2, normalize: bool):
    """
    Normalize the sequences together by z-scoring each dimension.
    sq1: numpy array of shape (t1, d)
    sq2: numpy array of shape (t2, d)
    normalize: if True, normalize the sequences together
    """
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
    """
    Compute the Dynamic Time Warping distance between two sequences.
    sq1: numpy array of shape (t1, d)
    sq2: numpy array of shape (t2, d)
    """
    return lib_dtw(sq1.transpose(), sq2.transpose())[0][-1, -1]


def time_txt(time, time_frame=5):
    if time % time_frame == 0:
        return f"{round(time * 0.02, 2)}"
    return ""


def create_df(feats, speaker_len, names):
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
    assert len(df.loc[df['speaker'] == -1]) == 0
    assert len(df_speaker_count) == len(df)
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
    

# Model's label rate is 0.02 seconds. To not overflow the plot, time is shown every 5 samples (0.1 seconds).
# To change that, change "time_frame" below.
time_frame = 5

# @spaces.GPU(duration=120)
def grMeasureDistance(wav_paths):
	if wav_paths is None:
		gr.Warning("Please upload some sound files!")
		return None
	seed = 31415
	# Load wav files
	expected_sr = 16000
	wavs = []
	for wav_path in wav_paths:
		wav, sr = torchaudio.load(wav_path)
		if sr != expected_sr:
			print(f"Sampling rate of {wav_path} is not {expected_sr} -> Resampling the file")
			resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=expected_sr)
			wav = resampler(wav)
			wav.squeeze()
		wavs.append(wav)

	# Generate Features
	device_name = "cuda" if torch.cuda.is_available() else "cpu"
	device = torch.device(device_name)
	print(f'Running on {device_name}')

	model = HubertModel.from_pretrained("facebook/hubert-base-ls960")
	features = None
	speaker_len = []
	layer = 12
	names = [f.rsplit(".", 1)[0] for f in wav_paths]
	# Not batched to know the actual seqence shape
	for wav in wavs:
		wav_features = model(wav, return_dict=True, output_hidden_states=True).hidden_states[
			layer].squeeze().detach().numpy()
		features = wav_features if features is None else np.concatenate([features, wav_features], axis=0)
		speaker_len.append(wav_features.shape[0])

	# Create & Fill a dataframe with the details
	data_subset, df_subset, hubert_feature_columns = create_df(features, speaker_len, names)

	# Evaluate Distance of all speaker pairs
	distances_list = []
	wav_pairs = list(combinations(names, 2))
	for wav_pair in wav_pairs:
		S1 = wav_pair[0]
		S2 = wav_pair[1]
		# FULL DIMENSIONALITY
		distance = calc_distance(df_subset, S1, S2, hubert_feature_columns)
		distances_list.append([os.path.basename(S1), os.path.basename(S2), distance])
	return distances_list

#csv export function
def export_csv(d):
	if(len(d.iloc[0,0])>0):
		d.to_csv("output.csv")
		return gr.File(value="output.csv", visible=True)

def clearInterface():
	return gr.File(interactive=False, visible=False), gr.Dataframe(value=None)

#main GradIO interface
with gr.Blocks() as demo:
	gr.Markdown(
	"""
	# PS3-PDM: Perceptual Similarity Space for Speech-Pairwise Distance Matrix
	 """)
 
	with gr.Accordion("Click to view details", open=False):
		gr.Markdown(
		"""
		## Project
		- Perceptual Similarity Space for Speech
		- Supported by the National Science Foundation (DRL 2219843) and Binational Science Foundation (2022618)
 
		## Project team
		- [Matt Goldrick](https://faculty.wcas.northwestern.edu/matt-goldrick/)
		- [Yossi Keshet](https://keshet.net.technion.ac.il/)
		- [Ann Bradlow](https://faculty.wcas.northwestern.edu/ann-bradlow/)
		- [Seung-Eun Kim](https://seungeun-kim.github.io/)
		- [Roni Chernyak](https://bronichern.github.io/)
		- [Chun Liang Chan](https://staff.wcas.northwestern.edu/clc500/)
 
		## Description
		Takes a set of utterance files (.wav format) and generates all pair-wise distances of the corresponding trajectories in HuBERT embedding spaces. Methods are based on Kim et al. (2025) and Chernyak et al. (2024). We report distances for embeddings in the original embedding space of transformer layer 12, without any form of dimensionality reduction.
 
		## Requirements
		- All speech files must be in .wav format. (Note: It is recommended to normalize the loudness of the files.)
		- All speech files must contain productions of the identical linguistic content (i.e., same words in same order). 
		- For example, the files may contain productions of a given sentence by different talkers, or by a single talker under different conditions. 
		- Note that while the utility will return distance values for files with different content the interpretation of these values is meaningless.
 
		## Usage
		- Upload wav files.
		- Click 'run' to get distances.
		- Output (download in .csv format) consists of a table with 3 columns (sentence1, sentence2, distance) and (n*(n-1))/2 rows where n = number of sentences (.wav soundfiles).
 
		## Capacity limits
		- Processing time is approximately 7 times the duration of the input audio files. For example, a minute of audio can take up to 7 minutes to process. If processing is taking longer than expected, please refresh the page and reupload your files.
		- Ocassionally the app may fail when uploading a large number of files in a single session. Consider running in smaller batches if possible.
		- Networks with slower upload speeds may experience reduced performance.
 
		## References
		- Kim, S-E, Chernyak, B. R., Keshet, J., Goldrick, M., & Bradlow, A. R. (2025).  Predicting relative intelligibility from inter-talker distances in a perceptual similarity space for speech.  Psychonomic Bulletin and Review. https://doi.org/10.3758/s13423-025-02652-2
		- Chernyak, B. R., Bradlow, A. R., Keshet, J., & Goldrick, M., & (2024).  A perceptual similarity space for speech based on self-supervised speech representations.  Journal of the Acoustical Society of America, 155(6), 3915-3929.  

		"""
		)
	with gr.Row():
		inputFiles = gr.File(file_count="multiple", file_types=[".wav"])
		with gr.Column():
			runbtn = gr.Button("Run")
			csv = gr.File(interactive=False, visible=False)
			dataframe = gr.Dataframe(headers=["S1", "S2", "distance"], visible=True, row_count=[1, 'dynamic'])


	runbtn.click(fn=grMeasureDistance, inputs=inputFiles, outputs=dataframe)
	dataframe.change(export_csv, inputs=dataframe, outputs=csv)
	
	inputFiles.change(fn=clearInterface, inputs=None, outputs=[csv, dataframe])

if __name__ == "__main__":
	demo.launch(ssr_mode=False)
