import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Embedding Distances
df_dist = pd.read_csv('output_results.csv')

def get_speaker(s):
    return '_'.join(s.split('_')[:2])

df_dist['Speaker1'] = df_dist['S1'].apply(get_speaker)
df_dist['Speaker2'] = df_dist['S2'].apply(get_speaker)

# Calculate average embedding distance for each pair
avg_dist_df = df_dist.groupby(['Speaker1', 'Speaker2'])['distance'].mean().reset_index()

# Sort speaker names in each pair alphabetically to ensure we can match regardless of order
def normalize_pair(row, col1, col2):
    s1, s2 = sorted([row[col1], row[col2]])
    return f"{s1}__{s2}"

avg_dist_df['Pair_Key'] = avg_dist_df.apply(lambda r: normalize_pair(r, 'Speaker1', 'Speaker2'), axis=1)
# Drop duplicates as distance(A,B) == distance(B,A)
avg_dist_df = avg_dist_df.drop_duplicates(subset=['Pair_Key'])

# 2. Load and Map Experimental Data
df_exp = pd.read_csv('experiment_data/speaker_false_alarm_diffs.csv')

# Drop empty unnamed columns if any
df_exp = df_exp[['Speaker_1', 'Speaker_2', 'avg_diff']].dropna()

# Mapping dictionary from experiment names to audio names
speaker_map = {
    'EN_F1': 'NS_F',
    'EN_M1': 'NS_M',
    'FR_F1': 'french_F',
    'FR_M1': 'french_M',
    'FR_F2': 'french_FI'
}

df_exp['Speaker1_Audio'] = df_exp['Speaker_1'].map(speaker_map)
df_exp['Speaker2_Audio'] = df_exp['Speaker_2'].map(speaker_map)

# Drop any rows where a speaker wasn't mapped (e.g. EN_F2 which wasn't in our audio dataset)
df_exp = df_exp.dropna(subset=['Speaker1_Audio', 'Speaker2_Audio'])

# Create normalized key for merging
df_exp['Pair_Key'] = df_exp.apply(lambda r: normalize_pair(r, 'Speaker1_Audio', 'Speaker2_Audio'), axis=1)

# 3. Merge Datasets
merged_df = pd.merge(avg_dist_df, df_exp, on='Pair_Key', how='inner')

# 4. Create Scatterplot
plt.figure(figsize=(10, 6))

sns.regplot(
    data=merged_df,
    x='distance',
    y='avg_diff',
    scatter_kws={'s': 100, 'color': '#2b83ba', 'edgecolor': 'white'},
    line_kws={'color': '#d7191c', 'linewidth': 2, 'linestyle': '--'}
)

# Annotate each point with the speaker pair
for i, row in merged_df.iterrows():
    pair_label = f"{row['Speaker_1']} vs {row['Speaker_2']}"
    plt.text(
        row['distance'] + 0.1, 
        row['avg_diff'], 
        pair_label, 
        fontsize=9, 
        ha='left', 
        va='center',
        bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1)
    )

plt.title('Perceptual Embedding Distance vs. False Alarm Difference Rate', fontsize=14, pad=15)
plt.xlabel('Average Embedding Distance (HuBERT + DTW)', fontsize=12)
plt.ylabel('Average False Alarm Difference', fontsize=12)

# Add grid and styling
plt.grid(True, linestyle='--', alpha=0.7)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# Calculate and annotate correlation
correlation = merged_df['distance'].corr(merged_df['avg_diff'])
plt.text(
    0.05, 0.95, 
    f"r = {correlation:.2f}", 
    transform=plt.gca().transAxes, 
    fontsize=12, 
    fontweight='bold',
    va='top',
    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='gray', alpha=0.8)
)

plt.tight_layout()
plt.savefig('scatter_distance_vs_false_alarm.png', dpi=300)
print(f"Scatterplot saved to scatter_distance_vs_false_alarm.png with {len(merged_df)} data points.")
print(f"Pearson Correlation Coefficient (r): {correlation:.3f}")
