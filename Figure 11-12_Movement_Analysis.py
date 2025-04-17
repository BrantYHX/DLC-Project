import re
from itertools import groupby
from operator import itemgetter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pup_bodyparts = ["pupleftear","puprightear"]

def load_csv_and_split(csv_path):
    # Load the CSV file
    df = pd.read_csv(csv_path)

    # Select columns that correspond to pup body parts and coordinate types ('x' or 'y')
    condition = (df.iloc[1].isin(pup_bodyparts)) & (df.iloc[2].isin(['x', 'y']))
    selected_columns = df.columns[condition]
    df = df[selected_columns]

    # Each individual occupies 4 columns (e.g., nose-x, nose-y, ear-x, ear-y)
    cols_per_individual = 4
    num_individuals = int(len(df.columns) / cols_per_individual)
    individual_dfs = []

    # Split the data into separate DataFrames for each individual
    for i in range(num_individuals):
        start_col = i * cols_per_individual
        end_col = (i + 1) * cols_per_individual
        ind_data = df.iloc[:, start_col:end_col].copy()
        individual_dfs.append(ind_data)

    return individual_dfs

def compute_average_position(df):
    result = []
    df = df[3:]  # Skip header rows
    df = df.astype(float)  # Convert values to float

    # Loop through each frame
    for i in range(len(df)):
        row = df.iloc[i]
        coords = []
        # Collect valid (x, y) coordinate pairs
        for j in range(0, len(row), 2):
            x, y = row.iloc[j], row.iloc[j + 1]
            if pd.notna(x) and pd.notna(y):
                coords.append((x, y))
        # Compute the average position for that frame
        if len(coords) >= 1:
            mean_x = np.mean([pt[0] for pt in coords])
            mean_y = np.mean([pt[1] for pt in coords])
            result.append([mean_x, mean_y])
        else:
            result.append([np.nan, np.nan])

    return pd.DataFrame(result, columns=["avg_x", "avg_y"])


def compute_velocity(avg_df, velocity_threshold=100, window_size=5):
    # Step 1: Mark positions as NaN if the computed velocity is too high
    for i in range(1, len(avg_df)):
        x1, y1 = avg_df.loc[i - 1, "avg_x"], avg_df.loc[i - 1, "avg_y"]
        x2, y2 = avg_df.loc[i, "avg_x"], avg_df.loc[i, "avg_y"]
        if pd.notna(x1) and pd.notna(y1) and pd.notna(x2) and pd.notna(y2):
            dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            velocity = dist * 14 / 21.74  # Convert pixels/frame to cm/s
            if velocity > velocity_threshold:
                avg_df.loc[i, ["avg_x", "avg_y"]] = np.nan

    # Step 2: Apply rolling average to smooth the data
    smooth_df = avg_df.rolling(window=window_size, center=True, min_periods=1).mean()

    # Step 3: Recalculate velocity on the smoothed data
    final_velocity_list = [np.nan]  # First frame has no velocity
    for i in range(1, len(smooth_df)):
        x1, y1 = smooth_df.loc[i - 1, "avg_x"], smooth_df.loc[i - 1, "avg_y"]
        x2, y2 = smooth_df.loc[i, "avg_x"], smooth_df.loc[i, "avg_y"]
        if pd.notna(x1) and pd.notna(y1) and pd.notna(x2) and pd.notna(y2):
            dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            velocity = dist * 14 / 21.74
            if velocity < 100:  # Ignore unrealistically high speeds
                final_velocity_list.append(velocity)
        else:
            final_velocity_list.append(np.nan)

    return final_velocity_list

def filter_velocity_track(velocity_list, threshold=2.5, min_duration=10):
    '''Filter out movement that pass the threshold'''
    velocity_array = np.array(velocity_list)

    # Create a mask for velocities above the threshold
    mask = velocity_array > threshold
    idx = np.where(mask)[0]  # Indices where condition is met

    # Initialize output with all NaNs
    filtered_velocity = np.full_like(velocity_array, np.nan, dtype=float)
    segment_lengths = []

    # Group consecutive indices into movement segments
    for k, g in groupby(enumerate(idx), lambda x: x[0] - x[1]):
        group = list(map(itemgetter(1), g))
        if len(group) >= min_duration:  # Only keep segments long enough
            filtered_velocity[group] = velocity_array[group]
            segment_lengths.append(len(group))  # Record segment length

    return filtered_velocity, segment_lengths


csv_paths = [r"D:\huddle_analysis\video_20240526_131000 (YY)DLC_resnet50_DLC_Main_ProjectApr1shuffle1_100000_el.csv",
             r"D:\huddle_analysis\video_20240526_151000 (N E')DLC_resnet50_IR conditionJun30shuffle1_100000_el.csv",
             r"D:\huddle_analysis\video_20240527_131000 (N E')DLC_resnet50_IR conditionJun30shuffle1_100000_el.csv",
             r"D:\huddle_analysis\video_20240527_151000 (Y E')DLC_resnet50_IR conditionJun30shuffle1_100000_el.csv",
             r"D:\huddle_analysis\video_20240529_181000 (YY)DLC_resnet50_DLC_Main_ProjectApr1shuffle1_100000_el.csv",
             r"D:\huddle_analysis\video_20240529_191000 (N)DLC_resnet50_DLC_Main_ProjectApr1shuffle1_100000_el.csv",
             r"D:\huddle_analysis\video_20240531_111000 (N E)DLC_resnet50_DLC_Main_ProjectApr1shuffle1_100000_el.csv",
             r"D:\huddle_analysis\video_20240531_171000 (N E)DLC_resnet50_IR conditionJun30shuffle1_100000_el.csv",
             r"D:\huddle_analysis\video_20240601_161000 (YY E)DLC_resnet50_IR conditionJun30shuffle1_100000_el.csv",
             r"D:\huddle_analysis\video_20240601_191000 (N E)DLC_resnet50_IR conditionJun30shuffle1_100000_el.csv",
             r"D:\huddle_analysis\video_20240602_131000 (N E)DLC_resnet50_DLC_Main_ProjectApr1shuffle1_100000_el.csv",
             r"D:\huddle_analysis\video_20240602_201000 (N E)DLC_resnet50_DLC_Main_ProjectApr1shuffle1_100000_el.csv",
             r"D:\huddle_analysis\video_20240603_111000 (YY blurred)DLC_resnet50_DLC_Main_ProjectApr1shuffle1_100000_el.csv",
             r"D:\huddle_analysis\video_20240603_181000 (YYY blurred E)DLC_resnet50_IR conditionJun30shuffle1_100000_el.csv",
             r"D:\huddle_analysis\video_20240604_111000 (N E)DLC_resnet50_DLC_Main_ProjectApr1shuffle1_100000_el.csv",
             r"D:\huddle_analysis\video_20240604_191000 (N E)DLC_resnet50_DLC_Main_ProjectApr1shuffle1_100000_el.csv",
             r"D:\huddle_analysis\video_20240605_121000 (N E)DLC_resnet50_DLC_Main_ProjectApr1shuffle1_100000_el.csv",
             r"D:\huddle_analysis\video_20240605_171000 (Y E)DLC_resnet50_DLC_Main_ProjectApr1shuffle1_100000_el.csv",
             r"D:\huddle_analysis\video_20240606_151000 (N E)DLC_resnet50_DLC_Main_ProjectApr1shuffle1_100000_el.csv",
             r"D:\huddle_analysis\video_20240606_171000 (N E)DLC_resnet50_DLC_Main_ProjectApr1shuffle1_100000_el.csv",
             r"D:\huddle_analysis\video_20240607_181000 (N E)DLC_resnet50_DLC_Main_ProjectApr1shuffle1_100000_el.csv",
             r"D:\huddle_analysis\video_20240607_141000 (N E)DLC_resnet50_IR conditionJun30shuffle1_100000_el.csv",
             r"D:\huddle_analysis\video_20240608_201000 (N E)DLC_resnet50_IR conditionJun30shuffle1_100000_el.csv",
             r"D:\huddle_analysis\video_20240608_201000 (N E)DLC_resnet50_IR conditionJun30shuffle1_100000_el.csv",
             r"D:\huddle_analysis\video_20240609_141000 (N E)DLC_dlcrnetms5_IR conditionJun30shuffle1_100000_el.csv",
             r"D:\huddle_analysis\video_20240609_131000 (N E)DLC_resnet50_IR conditionJun30shuffle1_100000_el.csv"
             ]

# Put all velocity into a list
all_segment_list = []
for csv_path in csv_paths:

    match = re.search(r"video_2024(\d{4})", csv_path)
    if match:
        mmdd = match.group(1)

    individual_dfs = load_csv_and_split(csv_path)
    velocity_list = []  # Contain several lists of velocities for each individual
    segment_list = []
    for df in individual_dfs:
        avg_df = compute_average_position(df)
        velocity = compute_velocity(avg_df)
        valid_length = np.sum(~np.isnan(velocity))
        print(f"valid_length:{valid_length}")
        velocity, segment_lengths = filter_velocity_track(velocity, threshold=3, min_duration=14)
        print(f"Track segment lengths: {segment_lengths}")
        velocity_list.append(velocity)
        total_lengths = sum(segment_lengths)
        if total_lengths > 0:
            segment_list.append(total_lengths)
    print(len(velocity_list)) # number of individual in the video
    all_segment_list.append([mmdd, segment_list])
    print(all_segment_list)

    # num_individuals = len(velocity_list)
    # cols = 4
    # rows = int(np.ceil(num_individuals / cols))
    # plt.figure(figsize=(cols * 4, rows * 3))
    # for i in range(num_individuals):
    #     plt.subplot(rows, cols, i + 1)
    #     plt.plot(velocity_list[i], color='dodgerblue')
    #     plt.title(f"Individual {i + 1}", fontsize=10)
    #     plt.xlabel("Frame", fontsize=9)
    #     plt.ylabel("Velocity (cm/s)", fontsize=9)
    #     if np.any(~np.isnan(velocity_list[i])):
    #         plt.ylim(0, np.nanmax(velocity_list[i]) * 1.2)
    #     else:
    #         plt.ylim(0, 1)  # default ylim when all data are NaN
    #     plt.grid(True, linestyle='--', alpha=0.3)
    # plt.tight_layout(rect=[0, 0, 1, 0.95])
    # plt.suptitle("Frame-wise Velocity for Each Individual", fontsize=14)
    # plt.show()

records = []
for day, segments in all_segment_list:
    for segment in segments:
        records.append({'mmdd': day, 'segment_length': segment})
df = pd.DataFrame(records)
df.to_csv("Movement_data_3_14.csv", index=False)


# Plot movement across days with different Duration thresholds
velocity_list = [
        r"D:\DeepLabCut\velocity_ratio_summary_3_6.csv",
        r"D:\DeepLabCut\velocity_ratio_summary_3_10.csv",
        r"D:\DeepLabCut\velocity_ratio_summary_3_14.csv",
]

fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
days = ["P10", "P12", "P14", "P15", "P16", "P17", "P18", "P19", "P20", "P21", "P22", "P23"]

for i, velocity in enumerate(velocity_list):
    df = pd.read_csv(velocity)
    df['segment_length'] = df['segment_length'] * 100/8397  # calculate the proportion of movement in the video
    daily_mean = df.groupby('mmdd')['segment_length'].mean()
    daily_sem = df.groupby('mmdd')['segment_length'].sem()
    axs[i].errorbar(x=days, y=daily_mean, yerr=daily_sem, fmt='-o')
    axs[i].set_ylabel('Moving time (%)',fontsize=12)
    threshold = i* 4 + 6
    axs[i].set_title(f'Duration Threshold: {threshold} frames ({threshold/14:.2f} s)', fontsize=12)
    axs[i].grid(True, linestyle='--', alpha=0.3)
axs[-1].set_xlabel('Postnatal Day',fontsize=12)
fig.suptitle('Proportion of Moving Time Across Postnatal Days with Different Duration Thresholds', fontsize=16)
plt.tight_layout()
plt.show()


# Plot movement across days with different velocity thresholds
velocity_list = [
        r"D:\DeepLabCut\velocity_ratio_summary_2_10.csv",
        r"D:\DeepLabCut\velocity_ratio_summary_3_10.csv",
        r"D:\DeepLabCut\velocity_ratio_summary_4_10.csv",
]

fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
days = ["P10", "P12", "P14", "P15", "P16", "P17", "P18", "P19", "P20", "P21", "P22", "P23"]

for i, velocity in enumerate(velocity_list):
    df = pd.read_csv(velocity)
    df['segment_length'] = df['segment_length'] * 100/8397
    daily_mean = df.groupby('mmdd')['segment_length'].mean()
    daily_sem = df.groupby('mmdd')['segment_length'].sem()
    axs[i].errorbar(x=days, y=daily_mean, yerr=daily_sem, fmt='-o')
    axs[i].set_ylabel('Moving time (%)',fontsize=12)
    axs[i].set_title(f'Velocity Threshold: {i + 2} cm/s', fontsize=12)
    axs[i].grid(True, linestyle='--', alpha=0.3)
axs[-1].set_xlabel('Postnatal Day',fontsize=12)
fig.suptitle('Proportion of Moving Time Across Postnatal Days with Different Velocity Thresholds', fontsize=16)
plt.tight_layout()
plt.show()