import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import levene
from scipy.stats import kruskal

def load_and_filter_data(csv_path, test_frames=[]):
    """
    Load RMSE values from a DLC evaluation CSV, select test frames, and apply IQR filtering.

    Parameters:
        csv_path (str): Path to the DLC dist_*.csv file.
        test_frames (list): Frame indices used as test data.

    Returns:
        np.ndarray: Filtered RMSE values (converted from pixels to cm).
    """
    df = pd.read_csv(csv_path)

    # Select columns that represent RMSE
    condition = (df.iloc[2].isin(["rmse"]))
    selected_columns = df.columns[condition]
    df = df[selected_columns]

    # Drop header and keep only test frames
    df = df[3:]
    df = df.iloc[test_frames]
    df = df.astype(float)

    # Flatten and remove NaNs
    values = df.to_numpy().flatten()
    values = values[~np.isnan(values)]

    # Apply IQR filtering to remove outliers
    Q1 = np.percentile(values, 25)
    Q3 = np.percentile(values, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_values = values[(values >= lower_bound) & (values <= upper_bound)]

    # Convert from pixels to cm
    filtered_values = filtered_values / 21.74

    return filtered_values

csv_paths = [r"D:\DLC_100_frames-Brant-2025-04-02\evaluation-results\iteration-0\DLC_100_framesApr2-trainset60shuffle1\dist_50000.csv",
             r"D:\DLC_400_frames-Brant-2025-04-03\evaluation-results\iteration-0\DLC_400_framesApr3-trainset80shuffle1\dist_50000.csv",
             r"D:\DLC_Main_Project-Brant-2025-04-01\evaluation-results\iteration-0\DLC_Main_ProjectApr1-trainset90shuffle1\dist_50000.csv"]

test_frames_list = [[176, 127,  29, 132, 111, 123,  65, 169, 193,  72,  97,  22,  41,
       108,  42, 112, 105, 168,   4, 159,  66, 165, 102, 186,   7,  80,
       166, 118, 170, 145,  74,  13, 131, 119,  32,  30, 116,  86,   5,
        63, 177,  10,  45,  88,  53,   9,  67,  55,  92,  56, 143,  98,
       130,  18, 142,  59,  94, 198,  82,  15,  91,  31,  24, 139, 126,
        68,  40, 140, 183, 106, 101, 124,  62, 115, 146,  77,   0, 147,
       181, 162], [ 25,  57, 272, 289, 368, 185, 313, 305,  65,   9, 218, 339, 336,
       389,  71, 198, 312,  77, 348, 147, 152, 109,  83, 376,  61, 245,
       291,  16, 188, 393, 146, 151, 137,  84, 250, 375, 288,  35,  47,
       140, 308, 283, 159, 164, 172,  10, 284, 257, 327,  20, 277, 387,
       124, 343,  87, 359, 201, 314,  68, 355,  48,  51,  43, 244, 300,
       181,   1, 187, 153, 297,  38,  90, 184, 337,  60, 278, 183, 379,
       161, 239, 258], [ 92, 519, 394, 707, 147, 524,  50,  72,  25,  36, 373, 121, 193,
       288, 142, 447, 312, 267, 103, 223, 362, 497, 168, 293, 541, 345,
        63, 146, 533, 337, 152, 427, 587, 357,  49, 615,  42, 407, 797,
        55, 653,  43, 351, 511, 280, 714, 522, 311, 686, 241, 626, 641,
       775, 584, 722, 255, 538,  99, 395, 777, 217, 473, 662, 335, 551,
       261, 328, 365,  66, 692,   4, 527, 718, 225, 744, 140,  73,   5,
        58, 423]]

# Load and filter data for all models
values_list = []
for i in range(len(csv_paths)):
    output = load_and_filter_data(csv_paths[i], test_frames_list[i])
    values_list.append(output)

# Check the result type
print(type(values_list[0]))  # Should be numpy.ndarray

# ------------------------- Statistical Testing -------------------------

# Levene's test for equal variance
stat, p = levene(*values_list)
print(f"Levene's test for homogeneity of variance: p = {p:.4f}")

# Kruskal-Wallis test (non-parametric ANOVA)
stat, p = kruskal(*values_list)
print(f"Kruskal-Wallis H-test: H = {stat:.2f}, p = {p:.4f}")

# Pairwise Mann-Whitney U tests
pairs = [(0, 1), (0, 2), (1, 2)]
labels = ["200 frames", "400 frames", "800 frames"]
comparisons = []

print("Pairwise Mann-Whitney U test results:")
for i, j in pairs:
    stat, p_val = mannwhitneyu(values_list[i], values_list[j], alternative='two-sided')
    comparisons.append((i, j, p_val))
    print(f"{labels[i]} vs {labels[j]}: p = {p_val:.4f}")

# Sort comparisons by p-value for later annotation
comparisons_sorted = sorted(comparisons, key=lambda x: x[2], reverse=True)

# ------------------------- Visualization -------------------------

# Create boxplot comparing models
plt.figure(figsize=(10, 6))
sns.boxplot(
    data=values_list, palette="Set2", width=0.5,
    meanprops={"marker": "o", "markerfacecolor": "black",
               "markeredgecolor": "black", "markersize": 5}
)
plt.xticks(ticks=range(len(values_list)), labels=labels, fontsize=12)
plt.title('Models trained with Different Number of Frames', fontsize=14)
plt.xlabel('Total Number of Frames', fontsize=12)
plt.ylabel('Errors on Test Frames (cm)', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Helper function to add significance lines
def add_significance_bar(x1, x2, y, height=0.02, text="*"):
    plt.plot([x1, x1, x2, x2], [y, y + height, y + height, y], lw=1.5, color='black')
    plt.text((x1 + x2) / 2, y + height + 0.01, text, ha='center', va='bottom', fontsize=12)

# Determine Y-position for significance lines
y_max = max([max(v) for v in values_list])
base = y_max + 0.05
pos_dict = {
    (1, 2): base + 0.0,
    (0, 1): base + 0.1,
    (0, 2): base + 0.2
}

# Add significance annotations
for (x1, x2, p_val) in comparisons:
    key = tuple(sorted([x1, x2]))
    y = pos_dict[key]
    if p_val < 0.001:
        text = "***"
    elif p_val < 0.01:
        text = "**"
    elif p_val < 0.05:
        text = "*"
    else:
        text = "ns"
    add_significance_bar(x1, x2, y, text=text)

plt.tight_layout()
plt.show()