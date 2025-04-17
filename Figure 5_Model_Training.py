import matplotlib.pyplot as plt
import pickle

import numpy as np
import pandas as pd

def plot_training_loss(ax):
    """
       Plot training loss over iterations from DeepLabCut's learning_stats.csv.

       Parameters:
           ax (matplotlib.axes.Axes): The axis to draw the loss curve on.
       """
    iterations = []
    losses = []

    with open(r"D:\IR condition-Brant-2025\dlc-models\iteration-0\IR conditionJun30-trainset95shuffle1\train\learning_stats.csv", 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = {}
            for pair in line.split(', '):
                key, value = pair.split(': ')
                data[key] = value
            iterations.append(int(data['iteration']))
            losses.append(float(data['loss']))

    ax.plot(iterations, losses, color='black')
    ax.scatter(iterations, losses, color='black')
    ax.set_xlabel("Iterations",fontsize = 13)
    ax.set_ylabel("Loss",fontsize = 13)
    ax.set_title("Model training loss", fontsize = 15)



test_frames_list = [ 92, 519, 394, 707, 147, 524,  50,  72,  25,  36, 373, 121, 193,
       288, 142, 447, 312, 267, 103, 223, 362, 497, 168, 293, 541, 345,
        63, 146, 533, 337, 152, 427, 587, 357,  49, 615,  42, 407, 797,
        55, 653,  43, 351, 511, 280, 714, 522, 311, 686, 241, 626, 641,
       775, 584, 722, 255, 538,  99, 395, 777, 217, 473, 662, 335, 551,
       261, 328, 365,  66, 692,   4, 527, 718, 225, 744, 140,  73,   5,
        58, 423]
csv_paths = [r"D:\DLC_Main_Project-Brant-2025-04-01\evaluation-results\iteration-0\DLC_Main_ProjectApr1-trainset90shuffle1\dist_10000.csv",
             r"D:\DLC_Main_Project-Brant-2025-04-01\evaluation-results\iteration-0\DLC_Main_ProjectApr1-trainset90shuffle1\dist_20000.csv",
             r"D:\DLC_Main_Project-Brant-2025-04-01\evaluation-results\iteration-0\DLC_Main_ProjectApr1-trainset90shuffle1\dist_30000.csv",
             r"D:\DLC_Main_Project-Brant-2025-04-01\evaluation-results\iteration-0\DLC_Main_ProjectApr1-trainset90shuffle1\dist_40000.csv",
             r"D:\DLC_Main_Project-Brant-2025-04-01\evaluation-results\iteration-0\DLC_Main_ProjectApr1-trainset90shuffle1\dist_50000.csv",
             r"D:\DLC_Main_Project-Brant-2025-04-01\evaluation-results\iteration-0\DLC_Main_ProjectApr1-trainset90shuffle1\dist_60000.csv",
             r"D:\DLC_Main_Project-Brant-2025-04-01\evaluation-results\iteration-0\DLC_Main_ProjectApr1-trainset90shuffle1\dist_70000.csv",
             r"D:\DLC_Main_Project-Brant-2025-04-01\evaluation-results\iteration-0\DLC_Main_ProjectApr1-trainset90shuffle1\dist_80000.csv",
             r"D:\DLC_Main_Project-Brant-2025-04-01\evaluation-results\iteration-0\DLC_Main_ProjectApr1-trainset90shuffle1\dist_90000.csv",
             r"D:\DLC_Main_Project-Brant-2025-04-01\evaluation-results\iteration-0\DLC_Main_ProjectApr1-trainset90shuffle1\dist_100000.csv"]
def load_and_filter(csv_path, test_frames = []):
    """
        Load and split DeepLabCut evaluation CSV into filtered RMSE values for train and test sets.

        Parameters:
            csv_path (str): Path to dist_*.csv file.
            test_frames (list): List of test frame indices.

        Returns:
            tuple: (test_rmse_values, train_rmse_values), both in centimeters.
        """
    df = pd.read_csv(csv_path)
    condition = ((df.iloc[0].isin(["pup1", "pup2", "pup3", "pup4", "pup5", "pup6", "pup7", "pup8"])) & (
    (df.iloc[2].isin(["rmse"]))))
    selected_columns = df.columns[condition]
    df = df[selected_columns]
    df = df[3:]

    # Compute the filtered data for the test frames
    df_test = df.iloc[test_frames]
    df_test = df_test.astype(float)
    values_test = df_test.to_numpy()
    values_test = values_test[~np.isnan(values_test)]
    filtered_values_test = values_test / 21.74  # converting pixel to cm

    # Do the same thing for the train frames
    total_num_frames = 800  # 如果你知道总帧数是 800，填这个
    train_frames = list(set(range(total_num_frames)) - set(test_frames_list))
    df_train = df.iloc[train_frames]
    df_train = df_train.astype(float)
    values_train = df_train.to_numpy()
    values_train = values_train[~np.isnan(values_train)]
    filtered_values_train = values_train / 21.74  # converting pixel to cm

    return filtered_values_test, filtered_values_train

values_list_test = []
values_list_train = []
for i in range(len(csv_paths)):
    output_test, output_train = load_and_filter(csv_paths[i],test_frames_list)
    values_list_test.append(output_test)
    values_list_train.append(output_train)
medians_test = [np.median(rmse) for rmse in values_list_test]
medians_train = [np.median(rmse) for rmse in values_list_train]


def plot_median_rmse(ax):
    """
        Plot median RMSE across training iterations for train and test sets.

        Parameters:
            ax (matplotlib.axes.Axes): The axis to draw the plot on.
        """

    iteration_labels = [10000, 20000, 30000, 40000, 50000,
                        60000, 70000, 80000, 90000, 100000]

    ax.plot(iteration_labels, medians_train, marker='o', linestyle='-', color='dimgray', linewidth=2, label='Train')
    ax.plot(iteration_labels, medians_test, marker='o', linestyle='-', color='royalblue', linewidth=2, label='Test')
    ax.set_title("Median Error Across Iterations", fontsize=15)
    ax.set_xlabel("Iterations", fontsize=13)
    ax.set_ylabel("Median RMSE (cm)", fontsize=13)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(title="Dataset", fontsize=11, title_fontsize=12, loc="upper right")
    ax.set_facecolor("#f9f9f9")

fig, axs = plt.subplots(1, 2, figsize=(12, 5))
plot_training_loss(axs[0])
plot_median_rmse(axs[1])
for ax, label in zip(axs.flat, ["A", "B"]): ax.text(-0.1, 1.1, label, transform=ax.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')
plt.show()

