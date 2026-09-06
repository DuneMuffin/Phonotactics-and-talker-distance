import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load results
df = pd.read_csv('output_results.csv')

# Extract speaker pair and word
def get_speaker(s):
    # s is like 'french_M_fif'
    return '_'.join(s.split('_')[:2])

df['Speaker1'] = df['S1'].apply(get_speaker)
df['Speaker2'] = df['S2'].apply(get_speaker)
df['Pair'] = df['Speaker1'] + " vs " + df['Speaker2']

# Calculate averages
avg_distances = df.groupby('Pair')['distance'].agg(['mean', 'std', 'count']).reset_index()
avg_distances = avg_distances.sort_values('mean')

print("Average Distances by Speaker Pair:")
print(avg_distances)

# Visualization
plt.figure(figsize=(12, 10))

# 1. Bar Chart of Means
plt.subplot(2, 1, 1)
sns.barplot(data=df, x='Pair', y='distance', order=avg_distances['Pair'], palette='viridis')
plt.title('Average Perceptual Distance by Speaker Pair (Across 72 Words)')
plt.xticks(rotation=15)
plt.ylabel('Average Distance (HuBERT + DTW)')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 2. Box Plot to show distribution across words
plt.subplot(2, 1, 2)
sns.boxplot(data=df, x='Pair', y='distance', order=avg_distances['Pair'], palette='viridis')
plt.title('Distribution of Perceptual Distances across Words')
plt.xticks(rotation=15)
plt.ylabel('Distance')
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('distance_analysis.png')
print("\nVisualization saved to distance_analysis.png")

# Group by Language Comparison
def get_lang_comp(pair):
    if 'french' in pair and 'NS' in pair:
        return 'French vs Native'
    if 'french' in pair and 'NS' not in pair:
        return 'French vs French'
    return 'Native vs Native'

df['ComparisonType'] = df['Pair'].apply(get_lang_comp)
lang_comp = df.groupby('ComparisonType')['distance'].mean().sort_values()
print("\nAverage Distance by Comparison Type:")
print(lang_comp)
