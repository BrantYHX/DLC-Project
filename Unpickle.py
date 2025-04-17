import pickle
import pandas as pd

file_path = r"D:\DLC_400_frames-Brant-2025-04-03\training-datasets\iteration-0\UnaugmentedDataSet_DLC_400_framesApr3\Documentation_data-DLC_400_frames_80shuffle1.pickle"
with open(file_path, "rb") as file:
    data = pickle.load(file)

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)

print(data)
