from collections import defaultdict
import pandas as pd
import random

# Read and load the training dataset CSV file
df = pd.read_csv(r"D:\DLC_Main_Project-Brant-2025-04-01\training-datasets\iteration-0\UnaugmentedDataSet_DLC_Main_ProjectApr1\CollectedData_Brant.csv")
df = df[df.columns[1]]
df = df[3:]

# List_frames stores the video name and the number of frames in that video
list_frames = []
count = 0
for i in range(len(df)-1):
    if df.iloc[i] == df.iloc[i+1]:
        count += 1
    else:
        list_frames.append(df.iloc[i])
        list_frames.append(count + 1)
        count = 0

# Random Sample Frames
video_data = [(list_frames[i], list_frames[i+1]) for i in range(0, len(list_frames), 2)]
all_frames = []
for video, count in video_data:
    all_frames.extend([(video, i) for i in range(count)])

sampled_frames = random.sample(all_frames, 400)
sorted_frames = sorted(sampled_frames, key=lambda x: (x[0], x[1])) # Sort the sampled frames by video name and then frame index

# Optional: display grouped results for easier reading
grouped = defaultdict(list)
for video, frame in sorted_frames:
    grouped[video].append(frame)
for video, frames in grouped.items():
    print(f"{video} ({len(frames)} frames): {sorted(frames)}")

