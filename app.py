import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

import torch

from transformers import HubertModel
import torchaudio

from scipy.stats import zscore
from librosa.sequence import dtw as lib_dtw
import plotly.express as px
import plotly.graph_objs as go
import plotly.offline as pyo

import gradio as gr
import spaces

tsne_1 = 'tsne-3d-one'
tsne_2 = 'tsne-3d-two'
tsne_3 = 'tsne-3d-thr'


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


def tsne(data_subset, init='pca', early_exaggeration=12.0, lr='auto', n_comp=3, perplexity=40, iters=1000,
         random_state=None):
    tsne = TSNE(n_components=n_comp, verbose=1, perplexity=perplexity, n_iter=iters, init=init,
                early_exaggeration=early_exaggeration,
                learning_rate=lr, random_state=random_state)
    tsne_results = tsne.fit_transform(data_subset)
    return tsne_results


def fill_tsne(df_subset, tsne_results):
    print(tsne_results[:, 0].shape)
    df_subset[tsne_1] = tsne_results[:, 0]
    df_subset[tsne_2] = tsne_results[:, 1]
    if tsne_results.shape[1] == 3:
        df_subset[tsne_3] = tsne_results[:, 2]
    return df_subset


def plot_tsne(df_subset):
    #pyo.init_notebook_mode()
    fig = px.scatter_3d(df_subset, x=tsne_1, y=tsne_2, z=tsne_3,
                        color='speaker')
    fig.update_traces(mode='lines+markers+text')
    #pyo.iplot(fig, filename='jupyter-styled_bar')
    fig.write_html(f"tsne_plot_all.html")


def calc_distance(df_subset, speaker1, speaker2, cols):
    features_speaker1 = df_subset[df_subset['speaker'] == speaker1][cols].to_numpy()
    features_speaker2 = df_subset[df_subset['speaker'] == speaker2][cols].to_numpy()
    features_speaker1, features_speaker2 = mut_normalize_sequences(features_speaker1, features_speaker2, True)
    distance = librosa_dtw(features_speaker1, features_speaker2)
    distance = distance / (len(features_speaker1) + len(features_speaker2))
    return distance


def plot_two_speakers(speaker1, speaker2, max_s1=None, max_s2=None, df_subset=None, speakerLabel="audio1"):
    def axes_style3d(bgcolor = "rgb(20, 20, 20)", gridcolor="rgb(255, 255, 255)"): 
        return dict(showbackground =True, backgroundcolor=bgcolor, gridcolor=gridcolor, zeroline=False)
    dcp = df_subset.loc[df_subset['speaker'].isin([speaker1, speaker2])].copy().rename(
        columns={tsne_1: "x", tsne_2: 'y', tsne_3: 'z'})
    dcp1 = dcp.loc[(dcp['speaker'] == speaker1)].copy()
    dcp2 = dcp.loc[(dcp['speaker'] == speaker2)].copy()
    dcp1['clr'] = np.linspace(0, 1, dcp.loc[(dcp['speaker'] == speaker1)].shape[0])
    dcp2['clr'] = np.linspace(1, 0, dcp.loc[(dcp['speaker'] == speaker2)].shape[0])

    if max_s1 is not None:
        dcp1 = dcp1[:max_s1]

    if max_s2 is not None:
        dcp2 = dcp2[:max_s2]
    # S1
    fig = px.scatter_3d(dcp1, x='x', y='y', z='z',
                        color='clr', symbol='speaker',
                        text='time_txt',
                        labels={'x': 't-SNE-dim1', 'y': 't-SNE-dim2', 'z': 't-SNE-dim3'})
    fig.update_traces({'name':speakerLabel}, hovertemplate="audio1", marker_symbol='diamond', marker_coloraxis=None, marker_colorscale='burg',
                      mode='lines+markers+text', line_color='lightgray')
    fig.for_each_trace(lambda t: t.update(textfont_color='darkred'))

    # S2
    fig2 = px.scatter_3d(dcp2, x='x', y='y', z='z',
                         color='clr', symbol='speaker',
                         text='time_txt',
                         labels={'x': 't-SNE-dim1', 'y': 't-SNE-dim2', 'z': 't-SNE-dim3'})
    fig2.update_traces({'name':'baseline'}, hovertemplate="EF08", marker_coloraxis=None, marker_colorscale='ice', mode='lines+markers+text', line_color='lightgray')
    fig2.for_each_trace(lambda t: t.update(textfont_color='blue'))

    axis_style = axes_style3d(bgcolor='rgb(245, 249, 252)',) #transparent background color
    fig3 = go.Figure(data=fig.data + fig2.data)
    fig3.update_layout(scene=dict(
        xaxis = axis_style,
        yaxis = axis_style,
        zaxis = axis_style,
        xaxis_title='dimension 1 (t-SNE)',
        yaxis_title='dimension 2 (t-SNE)',
        zaxis_title='dimension 3 (t-SNE)'),
        margin=dict(r=20, b=10, l=10, t=10),
        legend_title="Speaker", )

    # fig3.show()
    #fig3.write_html(f"tsne_plot.html")
    return fig3

# Model's label rate is 0.02 seconds. To not overflow the plot, time is shown every 5 samples (0.1 seconds).
# To change that, change "time_frame" below.

time_frame = 5

@spaces.GPU
def grPlot(wav_paths, speakerLabel):
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

	df_subset_orig = df_subset.copy()
	data_subset_orig = data_subset.copy()


	#iters set to 300
	tsne_results = tsne(data_subset, init='pca', early_exaggeration=2.0, lr=100.0, n_comp=3, perplexity=40, iters=300,
						random_state=seed)
	df_subset = fill_tsne(df_subset, tsne_results)

	# Evaluate Distance of Two Speakers which will be the first two speakers in the wav_paths array
	S1 = names[0]
	S2 = names[1]

	# FULL DIMENSIONALITY
	distance = calc_distance(df_subset, S1, S2, hubert_feature_columns)
	print(f"Full Dim. Distance: {distance}")

	# TSNE DIMENSIONALITY
	cols = [tsne_1, tsne_2, tsne_3]
	distance = calc_distance(df_subset, S1, S2, cols)
	print(f"TSNE Dim. Distance: {distance}")

	# TSNE plot of the 2 speakers with no min and no max parameters
	fig = plot_two_speakers(S1, S2, None, None, df_subset, speakerLabel)
	
	return [fig, distance]


def grAudioInputs(audio1, audio2):
	wav_paths = [
	audio1,
	"audio/KEI_EF08_EN038.wav",
	"audio/KEI_KF04_EN038.wav",
	"audio/KEI_EF04_EN038.wav",
	"audio/KEI_EF05_EN038.wav",
		"audio/KEI_EF07_EN038.wav",
	"audio/KEI_EF09_EN038.wav",
	"audio/KEI_EM01_EN038.wav",
	"audio/KEI_EM02_EN038.wav",
	"audio/KEI_EM03_EN038.wav",
	"audio/KEI_EM05_EN038.wav",
	"audio/KEI_EM06_EN038.wav",
	"audio/KEI_KF01_EN038.wav",
	"audio/KEI_KF03_EN038.wav",
	"audio/KEI_KF05_EN038.wav",
	"audio/KEI_KF06_EN038.wav",
	"audio/KEI_KM01_EN038.wav",
	"audio/KEI_KM03_EN038.wav",
	"audio/KEI_KM04_EN038.wav",
	"audio/KEI_KM05_EN038.wav",
	"audio/KEI_KM06_EN038.wav"]
	
	#user uploaded audio1 with EF_EN038 as baseline
	grFig1, distance1 = grPlot(wav_paths, "audio1")
	
	wav_paths = [
	audio2,
	"audio/KEI_EF08_EN038.wav",
	"audio/KEI_KF04_EN038.wav",
	"audio/KEI_EF04_EN038.wav",
	"audio/KEI_EF05_EN038.wav",
	"audio/KEI_EF07_EN038.wav",
	"audio/KEI_EF09_EN038.wav",
	"audio/KEI_EM01_EN038.wav",
	"audio/KEI_EM02_EN038.wav",
	"audio/KEI_EM03_EN038.wav",
	"audio/KEI_EM05_EN038.wav",
	"audio/KEI_EM06_EN038.wav",
	"audio/KEI_KF01_EN038.wav",
	"audio/KEI_KF03_EN038.wav",
	"audio/KEI_KF05_EN038.wav",
	"audio/KEI_KF06_EN038.wav",
	"audio/KEI_KM01_EN038.wav",
	"audio/KEI_KM03_EN038.wav",
	"audio/KEI_KM04_EN038.wav",
	"audio/KEI_KM05_EN038.wav",
	"audio/KEI_KM06_EN038.wav"]
	
	#user uploaded audio2 with EF_EN038 as baseline	
	grFig2, distance2 = grPlot(wav_paths, "audio2")
	
	mdText = "Note: Darkness indicates time in recording (light = start; dark = end)"
	
	return [grFig1, mdText, distance1, grFig2, mdText, distance2]


demo = gr.Interface(
	grAudioInputs,
	[gr.Audio(sources=["microphone", "upload"], type="filepath"), gr.Audio(sources=["microphone", "upload"], type="filepath")],
	[gr.Plot(label="Perceptual Similarity Space Audio 1", min_width=300, show_label=False, container=True), 
    gr.Markdown(),
	gr.Textbox(label="t-SNE Distance"),
	gr.Plot(label="Perceptual Similarity Space Audio 2", min_width=300, show_label=False, container=True),
    gr.Markdown(),
	gr.Textbox(label="t-SNE Distance")],
	title="Perceptual Similarity Space Demo",
	description="<p>Demo based on Chernyak, Bradlow, Keshet, & Goldrick (2024) \"A perceptual space for speech based on self-supervised speech representations\" <a href='https://doi.org/10.1121/10.0026358' target='_blank'>https://doi.org/10.1121/10.0026358</a></p> <p>This demo allows you to upload or record two productions of the sentence \"The lady wore a coat\" for comparison against a common baseline – an English L1 production of that sentence. The output will be two visualizations, each showing a 3-dimensional projection of one of your recording vs. the baseline, along with the perceptual similarity space distance of that recording from the baseline.</p> <p>For example, you could compare a relatively high intelligibility L2 English talker’s production of \"The lady wore a coat\" to the same sentence as produced by a relatively low intelligibility L2 English talker. Based on our results, you’d expect the high intelligibility talker to typically have a smaller distance to the baseline relative to the low intelligibility talker.</p> <p><b>Requirements:</b> <ul><li>The two recordings must be of the sentence \"The lady wore a coat\" (the distance comparisons are not interpretable if different sentences are used).</li><li>Each must be a mono .wav file or you can record directly in your browser by clicking on the “Record” buttons below.</li><li>Any leading/trailing silence should be trimmed.  If you record directly, click the scissors icon to automatically trim leading/trailing silences.</li></ul></p><p>Download example files here: <a href='https://huggingface.co/spaces/MLSpeech/perceptual-similarity/resolve/main/audio/KEI_KF03_EN038.wav' download>high_intelligibility_talker</a>, <a href='https://huggingface.co/spaces/MLSpeech/perceptual-similarity/resolve/main/audio/KEI_KF01_EN038.wav' download>low_intelligibility_talker</a>. For reference, the baseline speaker can be downloaded <a href='https://huggingface.co/spaces/MLSpeech/perceptual-similarity/resolve/main/audio/KEI_EF08_EN038.wav' download>here</a>.",
	article="This work was supported by NSF Grant No. 2219843 and BSF Grant No. 2022618.",
	flagging_mode="never",
)

demo.launch()