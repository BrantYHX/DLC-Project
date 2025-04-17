import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu, shapiro
from scipy.stats import kruskal

csv_path = r"D:\DLC_Main_Project-Brant-2025-04-01\evaluation-results\iteration-0\DLC_Main_ProjectApr1-trainset90shuffle1\dist_100000.csv"
bodyparts = ["pupleftear", "puprightear", "pupshoulder", "pupsnout", "puptailbase"]
test_frames = [92, 519, 394, 707, 147, 524, 50, 72, 25, 36, 373, 121, 193, 288, 142, 447, 312,
               267, 103, 223, 362, 497, 168, 293, 541, 345, 63, 146, 533, 337, 152, 427, 587,
               357, 49, 615, 42, 407, 797, 55, 653, 43, 351, 511, 280, 714, 522, 311, 686, 241,
               626, 641, 775, 584, 722, 255, 538, 99, 395, 777, 217, 473, 662, 335, 551, 261,
               328, 365, 66, 692, 4, 527, 718, 225, 744, 140, 73, 5, 58, 423]

df = pd.read_csv(csv_path)
value_list = []

# Loop over each bodypart to extract and compare RMSE for test and train sets
for part in bodyparts:
    single_part_list = []

    # Select columns matching the bodypart and RMSE
    condition = ((df.iloc[1].isin([part])) & ((df.iloc[2].isin(["rmse"]))))
    selected_columns = df.columns[condition]
    df_selected = df[selected_columns]
    df_selected = df_selected[3:]  # Skip the header rows

    # Split indices into test and train
    all_indices = set(range(len(df_selected)))
    test_indices = set(test_frames)
    non_test_indices = list(all_indices - test_indices)

    # Get values for test and train sets
    df_test = df_selected.iloc[test_frames]
    df_train = df_selected.iloc[non_test_indices]

    for sub_df in [df_test, df_train]:
        values = sub_df.to_numpy(dtype=float).flatten()
        values = values[~np.isnan(values)]

        # Remove outliers using IQR
        Q1 = np.percentile(values, 25)
        Q3 = np.percentile(values, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        filtered_values = (values[(values >= lower_bound) & (values <= upper_bound)]) / 21.74  # Convert px to cm

        single_part_list.append(filtered_values[~np.isnan(filtered_values)])

    value_list.append(single_part_list)

# -------------------- Statistical Tests --------------------

# Shapiro-Wilk test for normality (per bodypart, test vs train)
for i, part in enumerate(bodyparts):
    stat_train, p_train = shapiro(value_list[i][1])  # Train set
    stat_test, p_test = shapiro(value_list[i][0])   # Test set
    print(f"{part} - Train Normality p = {p_train:.4f}")
    print(f"{part} - Test Normality p = {p_test:.4f}")

# Kruskal-Wallis test across bodyparts (Train)
train_data_across_parts = [value_list[i][1] for i in range(len(bodyparts))]
H_stat_train, p_kw_train = kruskal(*train_data_across_parts)
print(f"Kruskal-Wallis Test (Train): H = {H_stat_train:.2f}, p = {p_kw_train:.4f}")

# Kruskal-Wallis test across bodyparts (Test)
test_data_across_parts = [value_list[i][0] for i in range(len(bodyparts))]
H_stat_test, p_kw_test = kruskal(*test_data_across_parts)
print(f"Kruskal-Wallis Test (Test): H = {H_stat_test:.2f}, p = {p_kw_test:.4f}")

# Mann-Whitney U test: Train vs Test per bodypart
for i, part in enumerate(bodyparts):
    stat, p = mannwhitneyu(value_list[i][1], value_list[i][0])
    print(f"{part} - Mann-Whitney U Test: p = {p:.4f}")

# -------------------- Violin Plot --------------------

# Prepare data for plotting
rmse_values = []
bodypart_labels = []
set_labels = []

for i, part in enumerate(bodyparts):
    for value in value_list[i][1]:  # Train
        rmse_values.append(value)
        bodypart_labels.append(part)
        set_labels.append("Train")
    for value in value_list[i][0]:  # Test
        rmse_values.append(value)
        bodypart_labels.append(part)
        set_labels.append("Test")

# Print median test error per bodypart
for i in range(len(bodyparts)):
    print(f"Median error for {bodyparts[i]}: {np.median(value_list[i][0]):.2f} cm")

# Create DataFrame for seaborn violin plot
df_plot = pd.DataFrame({
    "RMSE": rmse_values,
    "Bodypart": bodypart_labels,
    "Set": set_labels
})

# Plot the RMSE distribution as violin plots
plt.figure(figsize=(9, 9))
sns.violinplot(
    data=df_plot,
    y="Bodypart", x="RMSE",
    hue="Set", split=True,
    inner="quartile", linewidth=1.2
)
plt.title("Error Distribution for Each Bodypart of Pup (Test vs Train)", fontsize=17)
plt.ylabel("Body Part", fontsize=15)
plt.xlabel("Error (cm)", fontsize=15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(title="Dataset", fontsize=14, title_fontsize=15)
plt.tight_layout()
plt.show()
