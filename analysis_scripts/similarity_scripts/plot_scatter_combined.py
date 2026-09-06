import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def get_speaker(s):
    # Handles both french_F_fif and EN_F_fifhih
    return '_'.join(s.split('_')[:2])

def normalize_pair(s1, s2):
    return '__'.join(sorted([s1, s2]))

# 1. Load Embedding Distances from both result files
df_dist1 = pd.read_csv('output_results.csv')
df_dist2 = pd.read_csv('output_results2.csv')
df_dist = pd.concat([df_dist1, df_dist2])

df_dist['Speaker1'] = df_dist['S1'].apply(get_speaker)
df_dist['Speaker2'] = df_dist['S2'].apply(get_speaker)

# Calculate average embedding distance for each pair
avg_dist_df = df_dist.groupby(['Speaker1', 'Speaker2'])['distance'].mean().reset_index()
avg_dist_df['Pair_Key'] = avg_dist_df.apply(lambda r: normalize_pair(r['Speaker1'], r['Speaker2']), axis=1)
avg_dist_df = avg_dist_df.drop_duplicates(subset=['Pair_Key'])

# 2. Load and Map Experimental Data
# Mapping dictionary for Experiment 1
speaker_map_per1 = {
    'EN_F1': 'NS_F',
    'EN_M1': 'NS_M',
    'FR_F1': 'french_F',
    'FR_M1': 'french_M',
    'FR_F2': 'french_FI'
}

# Experiment 1
df_per1 = pd.read_csv('experiment_data/per1_speaker_legality_diffs.csv')
df_per1 = df_per1[['Speaker_1', 'Speaker_2', 'avg_diff']].dropna()
df_per1['Speaker1_Audio'] = df_per1['Speaker_1'].map(speaker_map_per1)
df_per1['Speaker2_Audio'] = df_per1['Speaker_2'].map(speaker_map_per1)
df_per1 = df_per1.dropna(subset=['Speaker1_Audio', 'Speaker2_Audio'])
df_per1['Pair_Key'] = df_per1.apply(lambda r: normalize_pair(r['Speaker1_Audio'], r['Speaker2_Audio']), axis=1)
df_per1['Experiment'] = 'Exp 1'
df_per1 = df_per1.rename(columns={'avg_diff': 'diff'})

# Experiment 2
df_per2 = pd.read_csv('experiment_data/per2_speaker_legality_diffs.csv')
def split_combo(combo):
    s1, s2 = combo.split('/')
    return s1, s2

df_per2[['S1_exp', 'S2_exp']] = df_per2['Speaker Combination'].apply(lambda x: pd.Series(split_combo(x)))
df_per2['Pair_Key'] = df_per2.apply(lambda r: normalize_pair(r['S1_exp'], r['S2_exp']), axis=1)
df_per2['Experiment'] = 'Exp 2'
df_per2 = df_per2.rename(columns={'Mean (Legal - Illegal) Difference': 'diff'})

# Combine Experiments
df_exp = pd.concat([
    df_per1[['Pair_Key', 'diff', 'Experiment']],
    df_per2[['Pair_Key', 'diff', 'Experiment']]
])

# 3. Merge Datasets
merged_df = pd.merge(avg_dist_df, df_exp, on='Pair_Key', how='inner')

# 4. Create Scatterplot
plt.figure(figsize=(12, 8))

# Define colors for experiments
palette = {'Exp 1': '#2b83ba', 'Exp 2': '#d7191c'}

# Scatter points
sns.scatterplot(
    data=merged_df,
    x='distance',
    y='diff',
    hue='Experiment',
    style='Experiment',
    s=150,
    palette=palette,
    edgecolor='black',
    alpha=0.8,
    zorder=5
)

# Individual Regression Lines
for exp in merged_df['Experiment'].unique():
    sns.regplot(
        data=merged_df[merged_df['Experiment'] == exp],
        x='distance',
        y='diff',
        scatter=False,
        color=palette[exp],
        line_kws={'linewidth': 2, 'label': f'{exp} Trend'},
        ci=None  # Disable confidence intervals for clarity
    )

# Overall Regression Line
sns.regplot(
    data=merged_df,
    x='distance',
    y='diff',
    scatter=False,
    line_kws={'color': 'gray', 'linewidth': 2, 'linestyle': '--', 'label': 'Overall Trend', 'alpha': 0.6},
    ci=None
)

# Annotate each point with the speaker pair
for i, row in merged_df.iterrows():
    label = row['Pair_Key'].replace('__', ' vs ')
    plt.text(
        row['distance'] + 0.05, 
        row['diff'], 
        label, 
        fontsize=8, 
        ha='left', 
        va='center',
        bbox=dict(facecolor='white', alpha=0.3, edgecolor='none', pad=0.5)
    )

plt.title('Embedding Distance vs. Perceptual Legality Effect (Combined Experiments)', fontsize=15, pad=20)
plt.xlabel('Average Embedding Distance (HuBERT + DTW)', fontsize=12)
plt.ylabel('Mean (Legal - Illegal) Difference', fontsize=12)

plt.legend(title='Experiment', title_fontsize='12', fontsize='10')
plt.grid(True, linestyle='--', alpha=0.6)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# Calculate correlations
overall_r = merged_df['distance'].corr(merged_df['diff'])
exp1_r = merged_df[merged_df['Experiment'] == 'Exp 1']['distance'].corr(merged_df[merged_df['Experiment'] == 'Exp 1']['diff'])
exp2_r = merged_df[merged_df['Experiment'] == 'Exp 2']['distance'].corr(merged_df[merged_df['Experiment'] == 'Exp 2']['diff'])

stats_text = (
    f"Overall r = {overall_r:.3f}\n"
    f"Exp 1 r = {exp1_r:.3f}\n"
    f"Exp 2 r = {exp2_r:.3f}\n"
    f"N = {len(merged_df)}"
)

plt.text(
    0.05, 0.95, 
    stats_text, 
    transform=plt.gca().transAxes, 
    fontsize=12, 
    fontweight='bold',
    va='top',
    bbox=dict(boxstyle="round,pad=0.5", facecolor='white', edgecolor='gray', alpha=0.9)
)

plt.tight_layout()
plt.savefig('scatter_distance_vs_false_alarm_combined.png', dpi=300)
print(f"Combined scatterplot saved to scatter_distance_vs_false_alarm_combined.png with {len(merged_df)} data points.")
print(f"Overall Pearson Correlation Coefficient (r): {overall_r:.3f}")
print(f"Exp 1 Correlation (r): {exp1_r:.3f}")
print(f"Exp 2 Correlation (r): {exp2_r:.3f}")
