### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import pathlib as p

### External Imports ###
import pandas as pd
from sklearn.model_selection import KFold

### Internal Imports ###
from paths import pc_paths
from paths import hpc_paths
from helpers import utils as u

#########################



def resample_dataset(raw_data_path, resampled_data_path, csv_path, output_shape=(256, 256, 256)):
    dataframe = pd.read_csv(csv_path)
    for i, row in dataframe.iterrows():
        if i < 325:
            continue
        
        input_path = row['Input Path']
        ground_truth_path = row['Ground-Truth Path']

        input, _ = u.read_image(pc_paths.raw_data_path / input_path)
        ground_truth, _ = u.read_image(pc_paths.raw_data_path / ground_truth_path)

        resampled_input = u.resample_image(input, output_shape)
        resampled_ground_truth = u.resample_image(ground_truth, output_shape)

        print(f"Case: {i+1}/{len(dataframe)}")
        print(f"Input shape: {input.shape}")
        print(f"Ground-truth shape: {ground_truth.shape}")
        print(f"Min/Max Input: {input.min()}/{input.max()}")
        print(f"Min/Max GT: {ground_truth.min()}/{ground_truth.max()}")

        print(f"Resampled Input shape: {resampled_input.shape}")
        print(f"Resampled Ground-truth shape: {resampled_ground_truth.shape}")
        print(f"Resampled Min/Max Input: {resampled_input.min()}/{resampled_input.max()}")
        print(f"Resampled Min/Max GT: {resampled_ground_truth.min()}/{resampled_ground_truth.max()}")

        u.save_image(resampled_input, resampled_data_path / input_path)
        u.save_image(resampled_ground_truth, resampled_data_path / ground_truth_path)

if __name__ == "__main__":
    raw_data_path = pc_paths.raw_data_path
    resampled_data_path = pc_paths.fast_data_path / "Resampled_256_256_256"
    csv_path = pc_paths.csv_path / "dataset.csv"
    resample_dataset(raw_data_path, resampled_data_path, csv_path, output_shape=(256, 256, 256))
    pass