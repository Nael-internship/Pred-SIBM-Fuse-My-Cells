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




def resample_dataset(raw_data_path, resampled_data_path, csv_path, desired_spacing=(0.8, 0.8, 0.8)):
    dataframe = pd.read_csv(csv_path)
    for i, row in dataframe.iterrows(): 
        input_path = row['Input Path']
        ground_truth_path = row['Ground-Truth Path']

        input, metadata = u.read_image(raw_data_path / input_path)
        ground_truth, metadata = u.read_image(raw_data_path / ground_truth_path)

        desired_spacing_z, desired_spacing_y, desired_spacing_x = desired_spacing
        original_shape_z, original_shape_y, original_shape_x = ground_truth.shape[0], ground_truth.shape[1], ground_truth.shape[2]
        original_spacing_z, original_spacing_y, original_spacing_x = metadata['physical_size_z'], metadata['physical_size_y'], metadata['physical_size_x']
        desired_shape_z, desired_shape_y, desired_shape_x = int(original_shape_z * original_spacing_z / desired_spacing_z), int(original_shape_y * original_spacing_y / desired_spacing_y), int(original_shape_x * original_spacing_x / desired_spacing_x)
        output_shape = (desired_shape_z, desired_shape_y, desired_shape_x)

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

        print()

if __name__ == "__main__":
    ### 0.8 x 0.8 x 0.8 ###
    raw_data_path = hpc_paths.data_path / "RAW2"
    resampled_data_path = hpc_paths.data_path / "Resampled_08_08_08"
    if not os.path.isdir(resampled_data_path):
        os.makedirs(resampled_data_path)
    csv_path = hpc_paths.csv_path / "dataset.csv"
    resample_dataset(raw_data_path, resampled_data_path, csv_path, desired_spacing=(0.8, 0.8, 0.8))

    ### 1.0 x 1.0 x 1.0 ###
    raw_data_path = hpc_paths.data_path / "RAW2"
    resampled_data_path = hpc_paths.data_path / "Resampled_10_10_10"
    if not os.path.isdir(resampled_data_path):
        os.makedirs(resampled_data_path)
    csv_path = hpc_paths.csv_path / "dataset.csv"
    resample_dataset(raw_data_path, resampled_data_path, csv_path, desired_spacing=(1.0, 1.0, 1.0))

    ### 0.6 x 0.6 x 0.6 ###
    raw_data_path = hpc_paths.data_path / "RAW2"
    resampled_data_path = hpc_paths.data_path / "Resampled_06_06_06"
    if not os.path.isdir(resampled_data_path):
        os.makedirs(resampled_data_path)
    csv_path = hpc_paths.csv_path / "dataset.csv"
    resample_dataset(raw_data_path, resampled_data_path, csv_path, desired_spacing=(0.6, 0.6, 0.6))



    pass