import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('experiment_data/per1_filteredSubjects.csv')

# Filter for test-legal and test-illegal rows
test_df = df[df['label'].str.contains('test-legal|test-illegal', na=False)].copy()

def extract_info(label):
    parts = label.split('-')
    if len(parts) >= 3:
        # e.g., "test", "legal", "NS_F" or "french_F"
        condition = parts[1]
        speaker = parts[2]
        return condition, speaker
    return None, None

test_df[['legality', 'speaker']] = test_df['label'].apply(lambda x: pd.Series(extract_info(x)))

# Convert response to numeric (yes=1, no=0)
test_df['response_numeric'] = test_df['response'].map({'yes': 1, 'no': 0})

# Get speakers per participant
# A participant typically hears two speakers.
participant_speakers = test_df.groupby('worker_ID')['speaker'].unique().apply(lambda x: '/'.join(sorted(x)))
participant_speakers = participant_speakers.reset_index().rename(columns={'speaker': 'speaker_combination'})

# Calculate average yes response rate per participant and legality
means = test_df.groupby(['worker_ID', 'legality'])['response_numeric'].mean().unstack()

# Handle cases where one condition might be missing for a participant (unlikely but safe)
if 'legal' not in means.columns: means['legal'] = 0
if 'illegal' not in means.columns: means['illegal'] = 0
means = means.fillna(0)

# Calculate difference: mean legal - mean illegal
means['diff'] = means['legal'] - means['illegal']
means = means.reset_index()

# Merge with speaker combinations
final_df = pd.merge(means, participant_speakers, on='worker_ID')

# Average the differences over each speaker combination
result = final_df.groupby('speaker_combination')['diff'].mean().reset_index()
result.columns = ['Speaker Combination', 'Mean (Legal - Illegal) Difference']

# Sort for better display
result = result.sort_values(by='Speaker Combination')

print(result.to_string(index=False))

# Compare with the existing per1_speaker_legality_diffs.csv
print("\n--- Comparison with existing results ---")
try:
    existing = pd.read_csv('experiment_data/per1_speaker_legality_diffs.csv')
    print("Existing data (Speaker_1, Speaker_2, avg_diff):")
    print(existing[['Speaker_1', 'Speaker_2', 'avg_diff']].dropna().to_string(index=False))
except Exception as e:
    print(f"Could not read existing file: {e}")
