import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load results
df = pd.read_csv('output_results2.csv')

# Extract speakers from the 'S1' and 'S2' columns
def get_speaker(s):
    return '_'.join(s.split('_')[:2])

df['Speaker1'] = df['S1'].apply(get_speaker)
df['Speaker2'] = df['S2'].apply(get_speaker)

# Get the average for each pair
avg_df = df.groupby(['Speaker1', 'Speaker2'])['distance'].mean().reset_index()

# Dynamically get speakers list
speakers = sorted(list(set(df['Speaker1']).union(set(df['Speaker2']))))

# Create an empty dataframe for the matrix
matrix_df = pd.DataFrame(index=speakers, columns=speakers, dtype=float)

# Fill the matrix with the computed averages
for _, row in avg_df.iterrows():
    s1, s2 = row['Speaker1'], row['Speaker2']
    dist = row['distance']
    matrix_df.loc[s1, s2] = dist
    matrix_df.loc[s2, s1] = dist # Ensure symmetry

# Get min and max directly from the computed averages for the scale
min_val = avg_df['distance'].min()
max_val = avg_df['distance'].max()

# Create a mask to hide ONLY the diagonal
mask = np.eye(len(speakers), dtype=bool)

# Create the visualization
plt.figure(figsize=(10, 8)) # Slightly larger figure for 6x6 matrix

# Set background color of the plot so masked cells appear grey
plt.gca().set_facecolor('lightgrey')

# Use a heatmap to display the table visually
sns.heatmap(
    matrix_df, 
    annot=True,          # Show the numbers in the cells
    fmt=".2f",           # Format to 2 decimal places
    cmap="RdYlBu",       # Colormap (warm/red for low, cool/blue for high)
    mask=mask,           # Hide only the diagonal
    vmin=min_val,        # Set the minimum value for the color scale
    vmax=max_val,        # Set the maximum value for the color scale
    cbar_kws={'label': 'Average Perceptual Distance'},
    linewidths=.5,
    square=True,
    annot_kws={"size": 11, "weight": "bold"}
)

plt.title('Pairwise Average Perceptual Distances - Audio 2\n(HuBERT Layer 12 + DTW)', fontsize=15, pad=15)
plt.xlabel('Speaker 2', fontsize=13)
plt.ylabel('Speaker 1', fontsize=13)
plt.xticks(rotation=45)

# Adjust layout and save
plt.tight_layout()
plt.savefig('pairwise_table_heatmap2.png', dpi=300)
print(f"Visualization saved to pairwise_table_heatmap2.png (Scale: {min_val:.2f} to {max_val:.2f})")
