import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def get_speaker(s):
    return '_'.join(s.split('_')[:2])

def normalize_pair(s1, s2):
    return '__'.join(sorted([s1, s2]))

# 1. Load Embedding Distances (only output_results2.csv is needed for Exp 2)
df_dist = pd.read_csv('output_results2.csv')
df_dist['Speaker1'] = df_dist['S1'].apply(get_speaker)
df_dist['Speaker2'] = df_dist['S2'].apply(get_speaker)

# Calculate average embedding distance for each pair
avg_dist_df = df_dist.groupby(['Speaker1', 'Speaker2'])['distance'].mean().reset_index()
avg_dist_df['Pair_Key'] = avg_dist_df.apply(lambda r: normalize_pair(r['Speaker1'], r['Speaker2']), axis=1)
avg_dist_df = avg_dist_df.drop_duplicates(subset=['Pair_Key'])

# 2. Load and Map Experimental Data (Experiment 2 only)
df_per2 = pd.read_csv('experiment_data/per2_speaker_legality_diffs.csv')
def split_combo(combo):
    s1, s2 = combo.split('/')
    return s1, s2

df_per2[['S1_exp', 'S2_exp']] = df_per2['Speaker Combination'].apply(lambda x: pd.Series(split_combo(x)))
df_per2['Pair_Key'] = df_per2.apply(lambda r: normalize_pair(r['S1_exp'], r['S2_exp']), axis=1)
df_per2 = df_per2.rename(columns={'Mean (Legal - Illegal) Difference': 'diff'})

# 3. Merge Datasets
merged_df = pd.merge(avg_dist_df, df_per2, on='Pair_Key', how='inner')

# 4. Create Scatterplot
plt.figure(figsize=(10, 7))

# Scatter points
sns.regplot(
    data=merged_df,
    x='distance',
    y='diff',
    scatter_kws={'s': 150, 'color': '#d7191c', 'edgecolor': 'black', 'alpha': 0.8},
    line_kws={'color': '#d7191c', 'linewidth': 2, 'linestyle': '--'},
    ci=None
)

# Annotate each point with the speaker pair
for i, row in merged_df.iterrows():
    label = row['Pair_Key'].replace('__', ' vs ')
    plt.text(
        row['distance'] + 0.05, 
        row['diff'], 
        label, 
        fontsize=9, 
        ha='left', 
        va='center',
        bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1)
    )

plt.title('Embedding Distance vs. Perceptual Legality Effect (Experiment 2)', fontsize=15, pad=20)
plt.xlabel('Average Embedding Distance (HuBERT + DTW)', fontsize=12)
plt.ylabel('Mean (Legal - Illegal) Difference', fontsize=12)

plt.grid(True, linestyle='--', alpha=0.6)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# Calculate correlation
r_val = merged_df['distance'].corr(merged_df['diff'])
plt.text(
    0.05, 0.95, 
    f"Exp 2 Correlation (r) = {r_val:.3f}\nN = {len(merged_df)}", 
    transform=plt.gca().transAxes, 
    fontsize=14, 
    fontweight='bold',
    va='top',
    bbox=dict(boxstyle="round,pad=0.5", facecolor='white', edgecolor='gray', alpha=0.9)
)

plt.tight_layout()
plt.savefig('scatter_distance_vs_false_alarm_exp2.png', dpi=300)
print(f"Experiment 2 scatterplot saved to scatter_distance_vs_false_alarm_exp2.png")
print(f"Exp 2 Pearson Correlation Coefficient (r): {r_val:.3f}")
