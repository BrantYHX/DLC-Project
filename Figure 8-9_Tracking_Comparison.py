import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu, shapiro

bodyparts = ["pupleftear","puprightear","pupshoulder","pupsnout","puptailbase"]


def load_csv_and_split(csv_path):
    """
    Load a DeepLabCut CSV and split it into individual DataFrames per animal.

    Each DataFrame will contain only the 'x' and 'y' columns for one tracked body part.
    It also removes the first three metadata rows and resets the index.

    Parameters:
        csv_path (str): Path to the CSV file.

    Returns:
        list: List of DataFrames, each for one tracked animal.
    """
    df = pd.read_csv(csv_path)

    # Select only the x/y columns for the specified bodyparts
    condition = (df.iloc[1].isin(bodyparts)) & (df.iloc[2].isin(['x', 'y']))
    selected_columns = df.columns[condition]
    df = df[selected_columns]

    cols_per_individual = 2  # x and y per individual
    num_individuals = int(len(df.columns) / cols_per_individual)
    individual_dfs = []

    for i in range(num_individuals):
        start_col = i * cols_per_individual
        end_col = (i + 1) * cols_per_individual
        ind_data = df.iloc[:, start_col:end_col].copy()

        # Remove header rows, convert to float, and rename columns
        ind_data = ind_data[3:]
        ind_data.columns = ['x', 'y']
        ind_data = ind_data.astype(float)
        ind_data = ind_data.reset_index(drop=True)

        individual_dfs.append(ind_data)

    return individual_dfs


def compute_step_size(df):
    """
    Compute step size (displacement) between consecutive frames.

    Parameters:
        df (DataFrame): Contains 'x' and 'y' coordinates for a single individual.

    Returns:
        list: List of step sizes (in cm), one per frame.
    """
    step_sizes = [np.nan]  # First frame has no previous frame to compare

    for i in range(1, len(df)):
        x1, y1 = df.loc[i - 1, "x"], df.loc[i - 1, "y"]
        x2, y2 = df.loc[i, "x"], df.loc[i, "y"]

        # Only compute if both current and previous coordinates are valid
        if pd.notna(x1) and pd.notna(y1) and pd.notna(x2) and pd.notna(y2):
            dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) / 21.74  # pixels to cm
            step_sizes.append(dist)
        else:
            step_sizes.append(np.nan)

    return step_sizes


def compute_outlier_proportion(step_sizes):
    """
    Compute the proportion of steps that are considered outliers.

    A step is an outlier if it's larger than 400/14 ≈ 28.6 cm.

    Parameters:
        step_sizes (list): Step sizes in cm.

    Returns:
        float: Proportion of outlier steps.
    """
    step_series = pd.Series(step_sizes).dropna()
    threshold = 400 / 14  # cm
    proportion = (step_series > threshold).sum() / len(step_series)
    return proportion


def get_outlier_proportions(individuals):
    """
    Compute outlier step proportions for a list of individuals.

    Parameters:
        individuals (list): List of DataFrames, each for one individual.

    Returns:
        list: Proportion of outlier steps per individual.
    """
    proportions = []
    for individual in individuals:
        step_sizes = compute_step_size(individual)
        proportion = compute_outlier_proportion(step_sizes)
        proportions.append(proportion)
    return proportions


def plot_group_comparison(data1, data2, group_names=["Before", "After"], ylabel="Outlier Count"):
    """
    Plot a bar chart comparing two groups using a Mann-Whitney U test.

    Parameters:
        data1 (list): Values from group 1.
        data2 (list): Values from group 2.
        group_names (list): Labels for the two groups.
        ylabel (str): Y-axis label for the plot.
    """
    # Perform statistical test
    stat, p = mannwhitneyu(data1, data2, alternative="greater")
    print(f"Mann-Whitney U test p-value: {p:.4f}")

    # Combine data for plotting
    df = pd.DataFrame({
        "Outlier Count": data1 + data2,
        "Group": [group_names[0]] * len(data1) + [group_names[1]] * len(data2)
    })

    # Create the bar plot
    plt.figure(figsize=(7, 6))
    ax = sns.barplot(
        x="Group", y="Outlier Count", data=df,
        errorbar='se',
        capsize=0.15,
        palette=["lightgray", "dodgerblue"]
    )

    # Annotate significance line and stars
    y_max = max(data1 + data2)
    margin = 0.01
    bar_y = y_max
    text_y = y_max
    plt.ylim(0, y_max + margin)
    plt.plot([0, 1], [bar_y, bar_y], color='black', linewidth=1.5)

    # Significance marker
    if p < 0.001:
        sig = "***"
    elif p < 0.01:
        sig = "**"
    elif p < 0.05:
        sig = "*"
    else:
        sig = "n.s."
    plt.text(0.5, text_y, sig, ha='center', va='bottom', fontsize=14)

    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.tight_layout()
    plt.show()


# ------------------ Analysis Pipeline ------------------

file_pairs = [
    # (before_filtered, after_original) for each video
    (
    r"C:\Users\HAOXUAN YIN\Desktop\Test_Auto_Track\video_20240609_131000 (N E)DLC_dlcrnetms5_IR conditionJun30shuffle1_100000_el_filtered.csv",
    r"C:\Users\HAOXUAN YIN\Desktop\Movement_analysis\video_20240609_131000 (N E)DLC_resnet50_IR conditionJun30shuffle1_100000_el.csv"),
    (
    r"C:\Users\HAOXUAN YIN\Desktop\Test_Auto_Track\video_20240607_141000 (N E)DLC_dlcrnetms5_IR conditionJun30shuffle1_100000_el_filtered.csv",
    r"C:\Users\HAOXUAN YIN\Desktop\Movement_analysis\video_20240607_141000 (N E)DLC_resnet50_IR conditionJun30shuffle1_100000_el.csv"),
    (
    r"C:\Users\HAOXUAN YIN\Desktop\Test_Auto_Track\video_20240608_201000 (N E)DLC_dlcrnetms5_IR conditionJun30shuffle1_100000_el_filtered.csv",
    r"C:\Users\HAOXUAN YIN\Desktop\Movement_analysis\video_20240608_201000 (N E)DLC_resnet50_IR conditionJun30shuffle1_100000_el.csv"),
    (
    r"C:\Users\HAOXUAN YIN\Desktop\Test_Auto_Track\video_20240609_141000 (N E)DLC_dlcrnetms5_IR conditionJun30shuffle1_100000_el_filtered.csv",
    r"C:\Users\HAOXUAN YIN\Desktop\Movement_analysis\video_20240609_141000 (N E)DLC_dlcrnetms5_IR conditionJun30shuffle1_100000_el.csv")
]

all_before = []  # Store outlier proportions from auto-tracking
all_after = []  # Store outlier proportions from manual correction

# Loop through each pair and calculate proportions
for before_path, after_path in file_pairs:
    individuals_before = load_csv_and_split(before_path)
    individuals_after = load_csv_and_split(after_path)

    proportions_before = get_outlier_proportions(individuals_before)
    proportions_after = get_outlier_proportions(individuals_after)

    all_before.extend(proportions_before)
    all_after.extend(proportions_after)

# Normality test for each group
_, p1 = shapiro(all_before)
_, p2 = shapiro(all_after)
print(f"Shapiro-Wilk p for group1: {p1:.4f}")
print(f"Shapiro-Wilk p for group2: {p2:.4f}")

# Visual comparison and statistical test
plot_group_comparison(
    all_before, all_after,
    group_names=["Automatic", "Manual"],
    ylabel="Proportion of Jumps"
)
