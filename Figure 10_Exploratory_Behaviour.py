from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os
import glob
from matplotlib.widgets import RectangleSelector
from scipy.stats import shapiro, ttest_ind, f_oneway, levene
from statsmodels.stats.multitest import multipletests

pup_bodyparts = ["pupleftear","puprightear"]
single_bodyparts = ["leftear", "rightear", "shoulder", "spine1", "spine2"]

def load_and_split_data(csv_path):
    """
    Load a CSV file and split it into separate DataFrames:
    one for the mother and one for each pup.

    Parameters:
        csv_path (str): Path to the CSV file.

    Returns:
        list: A list of DataFrames, first for mother, followed by each pup.
    """
    df = pd.read_csv(csv_path)
    split_dfs = []

    # Extract mother data (single_bodyparts)
    condition = (df.iloc[1].isin(single_bodyparts)) & (df.iloc[2].isin(['x', 'y']))
    selected_columns = df.columns[condition]
    df_single = df[selected_columns]
    split_dfs.append(df_single)

    # Extract pup data (multiple individuals)
    condition = (df.iloc[1].isin(pup_bodyparts)) & (df.iloc[2].isin(['x', 'y']))
    selected_columns = df.columns[condition]
    df_pup = df[selected_columns]

    cols_per_individual = 4
    num_individuals = int(len(df_pup.columns) / 4)

    for i in range(num_individuals):
        start_col = i * cols_per_individual
        end_col = (i + 1) * cols_per_individual
        ind_data = df_pup.iloc[:, start_col:end_col].copy()
        split_dfs.append(ind_data)

    return split_dfs


def compute_average_position(df):
    """
    Compute the average x and y position for each frame,
    ignoring NaNs and using available body parts.

    Parameters:
        df (DataFrame): DataFrame with body part coordinates.

    Returns:
        DataFrame: A DataFrame with columns ["avg_x", "avg_y"] per frame.
    """
    result = []
    df = df[3:]
    df = df.astype(float)

    for i in range(len(df)):
        row = df.iloc[i]
        coords = []
        for j in range(0, len(row), 2):
            x, y = row.iloc[j], row.iloc[j + 1]
            if pd.notna(x) and pd.notna(y):
                coords.append((x, y))

        if len(coords) >= 1:
            mean_x = np.mean([pt[0] for pt in coords])
            mean_y = np.mean([pt[1] for pt in coords])
            result.append([mean_x, mean_y])
        else:
            result.append([np.nan, np.nan])

    return pd.DataFrame(result, columns=["avg_x", "avg_y"])


def on_select(eclick, erelease):
    """
    Callback function to record the coordinates of the rectangle
    drawn by the user during interactive selection.

    Parameters:
        eclick: Mouse click event.
        erelease: Mouse release event.
    """
    global box_coords
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata
    box_coords = (x1, y1, x2, y2)


def find_huddle_coordinate(file):
    """
    Open the corresponding image for the video, allow the user to draw a rectangle
    around the huddle, and return the coordinates.

    Parameters:
        file (str): Path to the CSV file used to identify the video name.

    Returns:
        tuple or None: (x1, y1, x2, y2) of the selected box or None if not found.
    """
    path = file
    match = re.search(r'video_\d{8}_\d{6}', path)
    image_name = match.group(0)
    folder_path = r"D:\huddle_analysis"
    matching_files = glob.glob(os.path.join(folder_path, f"{image_name}.*"))
    image_files = [f for f in matching_files if f.lower().endswith('.png')]

    if image_files:
        image_path = image_files[0]
        img = plt.imread(image_path)
        fig, ax = plt.subplots()
        ax.imshow(img)
        rect_selector = RectangleSelector(
            ax, on_select, useblit=True,
            button=[1],
            minspanx=5, minspany=5,
            spancoords='pixels', interactive=True
        )
        plt.show()

        if box_coords:
            print("Coordinates:", box_coords)
            return box_coords
    else:
        print("image not found")


def is_in_huddle(avg_df, huddle_coordinates):
    """
    Determine if the average position in each frame is inside the huddle region.

    Parameters:
        avg_df (DataFrame): DataFrame with avg_x and avg_y per frame.
        huddle_coordinates (tuple): (x1, y1, x2, y2) of the huddle box.

    Returns:
        list: For each frame, "Inside", "Outside", or "NAN".
    """
    result = []
    x1, y1, x2, y2 = huddle_coordinates
    for i in range(len(avg_df)):
        row = avg_df.iloc[i]
        if (row[0] >= min(x1, x2)) and (row[0] <= max(x1, x2)) and (row[1] >= min(y1, y2)) and (row[1] <= max(y1, y2)):
            result.append("Inside")
        elif (row[0] >= 0 and row[1] >= 0):
            result.append("Outside")
        else:
            result.append("NAN")
    return result


def count_pups_outside(huddle_list):
    """
    Count the number of pups that are outside the huddle in each frame.

    Parameters:
        huddle_list (list of lists): Each sublist contains "Inside"/"Outside"/"NAN" per frame.

    Returns:
        list: Number of pups outside the huddle per frame.
    """
    pup_statuses = huddle_list[1:]  # exclude the mother
    num_frames = len(pup_statuses[0])
    count_per_frame = []

    for frame_idx in range(num_frames):
        count = sum(1 for pup in pup_statuses if pup[frame_idx] == "Outside")
        count_per_frame.append(count)

    return count_per_frame


def plot_pups_outside(count_list):
    """
    Plot the number of pups outside the huddle over time (frame-by-frame).

    Parameters:
        count_list (list): Number of pups outside per frame.
    """
    frames = np.arange(len(count_list))
    plt.figure(figsize=(12, 4))
    plt.plot(frames, count_list, color='crimson', linewidth=1.5)
    plt.xlabel("Frame")
    plt.ylabel("Number of Pups Outside Huddle")
    plt.title("Number of Rat Pups Outside Huddle Over Time")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_huddle_raster(huddle_list):
    """
    Create a raster plot showing when each individual is outside the huddle.

    Parameters:
        huddle_list (list of lists): Each sublist contains "Inside"/"Outside"/"NAN" per frame.
    """
    num_animals = len(huddle_list)
    num_frames = len(huddle_list[0])
    fig, ax = plt.subplots(figsize=(12, 0.6 * num_animals))

    for i, animal_status in enumerate(huddle_list):
        in_huddle = [j for j, val in enumerate(animal_status) if val == "Outside"]
        ax.vlines(in_huddle, i + 0.4, i + 1.4, color='dodgerblue', lw=1)

    y_labels = ["Mum"] + [f"Pup {i}" for i in range(1, num_animals)]
    ax.set_yticks(np.arange(1, num_animals + 1))
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Individuals")
    ax.set_title("Outside the Huddle")
    ax.set_xlim(0, num_frames)
    plt.tight_layout()
    plt.show()


def time_outside_huddle(csv_path):
    """
    Full pipeline for computing and visualizing the amount of time pups spend
    outside the huddle in a video.

    Parameters:
        csv_path (str): Path to the CSV file with tracking data.

    Returns:
        float: Total time (in seconds) that pups are outside the huddle.
    """
    split_dfs = load_and_split_data(csv_path)
    coordinate = find_huddle_coordinate(csv_path)
    huddle = []

    for df in split_dfs:
        avg_df = compute_average_position(df)
        result = is_in_huddle(avg_df, coordinate)
        huddle.append(result)

    count_list = count_pups_outside(huddle)
    time_outside_per_pup = sum(count_list) / 14  # convert frames to seconds
    plot_huddle_raster(huddle)
    return time_outside_per_pup


csv_paths = [r"D:\huddle_analysis\video_20240526_131000 (YY)DLC_resnet50_DLC_Main_ProjectApr1shuffle1_100000_el.csv",
             r"D:\huddle_analysis\video_20240526_151000 (N E')DLC_resnet50_IR conditionJun30shuffle1_100000_el.csv",
             r"D:\huddle_analysis\video_20240527_131000 (N E')DLC_resnet50_IR conditionJun30shuffle1_100000_el.csv",
             r"D:\huddle_analysis\video_20240527_151000 (Y E')DLC_resnet50_IR conditionJun30shuffle1_100000_el.csv",
             r"D:\huddle_analysis\video_20240529_161000 (YY)DLC_resnet50_DLC_Main_ProjectApr1shuffle1_100000_el.csv",
             r"D:\huddle_analysis\video_20240529_181000 (YY)DLC_resnet50_DLC_Main_ProjectApr1shuffle1_100000_el.csv",
             r"D:\huddle_analysis\video_20240529_191000 (N)DLC_resnet50_DLC_Main_ProjectApr1shuffle1_100000_el.csv",
             r"D:\huddle_analysis\video_20240531_111000 (N E)DLC_resnet50_DLC_Main_ProjectApr1shuffle1_100000_el.csv",
             r"D:\huddle_analysis\video_20240531_121000 (YYY)DLC_resnet50_DLC_Main_ProjectApr1shuffle1_100000_el.csv",
             r"D:\huddle_analysis\video_20240531_171000 (N E)DLC_resnet50_IR conditionJun30shuffle1_100000_el.csv",
             r"D:\huddle_analysis\video_20240531_181000 (Y)DLC_resnet50_DLC_Main_ProjectApr1shuffle1_100000_el.csv",
             r"D:\huddle_analysis\video_20240601_141000 (Y E)DLC_resnet50_IR conditionJun30shuffle1_100000_el.csv",
             r"D:\huddle_analysis\video_20240601_161000 (YY E)DLC_resnet50_IR conditionJun30shuffle1_100000_el.csv",
             r"D:\huddle_analysis\video_20240601_191000 (N E)DLC_resnet50_IR conditionJun30shuffle1_100000_el.csv",
             r"D:\huddle_analysis\video_20240601_211000 (YY E)DLC_resnet50_DLC_Main_ProjectApr1shuffle1_100000_el.csv",
             r"D:\huddle_analysis\video_20240602_131000 (N E)DLC_resnet50_DLC_Main_ProjectApr1shuffle1_100000_el.csv",
             r"D:\huddle_analysis\video_20240602_171000 (YYY blurred E)DLC_resnet50_DLC_Main_ProjectApr1shuffle1_100000_el.csv",
             r"D:\huddle_analysis\video_20240602_201000 (N E)DLC_resnet50_DLC_Main_ProjectApr1shuffle1_100000_el.csv",
             r"D:\huddle_analysis\video_20240602_211000 (N E)DLC_resnet50_IR conditionJun30shuffle1_100000_el.csv",
             r"D:\huddle_analysis\video_20240603_111000 (YY blurred)DLC_resnet50_DLC_Main_ProjectApr1shuffle1_100000_el.csv",
             r"D:\huddle_analysis\video_20240603_181000 (YYY blurred E)DLC_resnet50_IR conditionJun30shuffle1_100000_el.csv",
             r"D:\huddle_analysis\video_20240603_211000 (YYY blurred)DLC_resnet50_DLC_Main_ProjectApr1shuffle1_100000_el.csv",
             r"D:\huddle_analysis\video_20240604_111000 (N E)DLC_resnet50_DLC_Main_ProjectApr1shuffle1_100000_el.csv",
             r"D:\huddle_analysis\video_20240604_131000 (N E)DLC_resnet50_DLC_Main_ProjectApr1shuffle1_100000_el.csv",
             r"D:\huddle_analysis\video_20240604_151000 (YY blurred E)DLC_resnet50_DLC_Main_ProjectApr1shuffle1_100000_el.csv",
             r"D:\huddle_analysis\video_20240604_161000 (N E)DLC_resnet50_DLC_Main_ProjectApr1shuffle1_100000_el.csv",
             r"D:\huddle_analysis\video_20240604_171000 (N E)DLC_resnet50_IR conditionJun30shuffle1_100000_el.csv",
             r"D:\huddle_analysis\video_20240604_191000 (N E)DLC_resnet50_DLC_Main_ProjectApr1shuffle1_100000_el.csv",
             r"D:\huddle_analysis\video_20240604_211000 (YY E)DLC_resnet50_IR conditionJun30shuffle1_100000_el.csv",
             r"D:\huddle_analysis\video_20240605_121000 (N E)DLC_resnet50_DLC_Main_ProjectApr1shuffle1_100000_el.csv",
             r"D:\huddle_analysis\video_20240605_151000 (N E)DLC_resnet50_DLC_Main_ProjectApr1shuffle1_100000_el.csv",
             r"D:\huddle_analysis\video_20240605_161000 (N E)DLC_resnet50_DLC_Main_ProjectApr1shuffle1_100000_el.csv",
             r"D:\huddle_analysis\video_20240605_171000 (Y E)DLC_resnet50_DLC_Main_ProjectApr1shuffle1_100000_el.csv",
             r"D:\huddle_analysis\video_20240605_211000 (N E)DLC_resnet50_IR conditionJun30shuffle1_100000_el.csv",
             ]

# Store the date (mmdd) and the total time pups spent outside the huddle on that date
data = []
for csv_path in csv_paths:
    match = re.search(r"video_2024(\d{4})", csv_path)  # Extract MMDD from filename
    if match:
        mmdd = match.group(1)
        time = time_outside_huddle(csv_path)  # Total time outside the huddle (in seconds)
        data.append([mmdd, time])

# Create a DataFrame to store results
df = pd.DataFrame(data, columns=["mmdd", "time"])
print(df)

# Calculate the average time outside per pup per video for each day
grouped = df.groupby("mmdd")["time"].sum()  # Sum across all videos on that day
standardized = grouped / (8 * 11)  # Normalize by 8 pups and 11 videos per day
result_df = standardized.reset_index()
result_df.columns = ["mmdd", "time_per_pup_per_video"]
print(result_df)

# Save both raw and averaged results
df.to_csv("Huddle_analysis_data.csv", index=False)
result_df.to_csv("daily_avg_outside_per_pup_per_video.csv", index=False)

# Reload the saved file (if needed later)
df = pd.read_csv("D:\DeepLabCut\Huddle_analysis_data.csv")
print(df)

# Compute daily mean and SEM (standard error of the mean)
daily_mean = df.groupby('mmdd')['time'].mean()
daily_sem = df.groupby('mmdd')['time'].sem()

# ----------- Statistical Testing -----------

# Shapiro-Wilk Test for normality within each day group
split_dfs = [g for _, g in df.groupby('mmdd')]
for i, g in enumerate(split_dfs):
    if len(g) < 3:
        print(f"Group {i+1} ({g['mmdd'].iloc[0]}): Not enough data for Shapiro-Wilk Test")
        continue
    stat, p = shapiro(g['time'])
    print(f"Group {i+1} ({g['mmdd'].iloc[0]}): Shapiro-Wilk stat={stat:.4f}, p={p:.4f}")

# Levene's Test for homogeneity of variance across groups
stat, p = levene(*[group['time'] for group in split_dfs])
print(f"Levene test for homogeneity of variance: p = {p}")

# One-way ANOVA to compare means across days
f_stat, p_anova = f_oneway(*(g['time'] for g in split_dfs if len(g) >= 3))
print(f"ANOVA: F = {f_stat:.4f}, p = {p_anova:.4f}")

# ----------- Post-hoc Pairwise Testing (Adjacent Days) -----------

# Ensure the days are sorted chronologically
sorted_days = sorted(df['mmdd'].unique())
p_values = []
pairs = []

# Perform pairwise t-tests between adjacent days
for i in range(len(sorted_days) - 1):
    day1 = sorted_days[i]
    day2 = sorted_days[i + 1]
    group1 = df[df['mmdd'] == day1]['time']
    group2 = df[df['mmdd'] == day2]['time']
    t_stat, p_val = ttest_ind(group1, group2, equal_var=True)
    p_values.append(p_val)
    pairs.append((day1, day2))

# Apply Bonferroni correction to control for multiple comparisons
reject, pvals_corrected, _, _ = multipletests(p_values, method='bonferroni')

# Print post-hoc test results
for i in range(len(pairs)):
    print(
        f"{pairs[i][0]} vs {pairs[i][1]}: raw p = {p_values[i]:.4f}, "
        f"corrected p = {pvals_corrected[i]:.4f}, significant = {reject[i]}"
    )

# ----------- Visualization -----------

# Plot average time outside the huddle with SEM over postnatal days
plt.figure(figsize=(10, 6))
plt.errorbar(
    x=["P9", "P10", "P12", "P14", "P15", "P16", "P17", "P18", "P19"],  # Replace with actual labels as needed
    y=daily_mean,
    yerr=daily_sem,
    fmt='-o'
)
plt.title('The Average Time Pups Explore Outside Huddle over Postnatal Days', fontsize=14)
plt.ylabel('Average time outside huddle per pup (s)', fontsize=12)
plt.xlabel('Postnatal Day', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()
