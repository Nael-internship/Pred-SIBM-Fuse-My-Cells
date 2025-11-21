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

#########################


def parse_dataset(output_csv_path):
    dataset_path = pc_paths.raw_data_path
    all_files = os.listdir(dataset_path)
    fused_files = sorted([file for file in all_files if 'fused' in file and "tif" in file])
    angle_files = sorted([file for file in all_files if 'angle' in file and "tif" in file])

    print(f"Number of Fused files: {(len(fused_files))}")
    print(f"Number of Angle files: {(len(angle_files))}")
    print(fused_files[0:5])
    print(angle_files[0:5])

    dataset = []
    for i in range(len(fused_files)):
        to_append = (angle_files[i], fused_files[i])
        dataset.append(to_append)

    dataframe = pd.DataFrame(dataset, columns=['Input Path', 'Ground-Truth Path'])
    if not os.path.isdir(os.path.dirname(output_csv_path)):
        os.makedirs(os.path.dirname(output_csv_path))
    dataframe.to_csv(output_csv_path, index=False)    


def split_dataframe(input_csv_path, output_splits_path, num_folds=5, seed=1234):
    if not os.path.isdir(os.path.dirname(output_splits_path)):
        os.makedirs(os.path.dirname(output_splits_path))
    dataframe = pd.read_csv(input_csv_path)
    print(f"Dataset size: {len(dataframe)}")
    kf = KFold(n_splits=num_folds, shuffle=True)
    folds = kf.split(dataframe)
    for fold in range(num_folds):
        train_index, test_index = next(folds)
        current_training_dataframe = dataframe.loc[train_index]
        current_validation_dataframe = dataframe.loc[test_index]
        print(f"Fold {fold + 1} Training dataset size: {len(current_training_dataframe)}")
        print(f"Fold {fold + 1} Validation dataset size: {len(current_validation_dataframe)}")
        training_csv_path = output_splits_path / f"training_fold_{fold+1}.csv"
        validation_csv_path = output_splits_path / f"val_fold_{fold+1}.csv"
        current_training_dataframe.to_csv(training_csv_path, index=False)
        current_validation_dataframe.to_csv(validation_csv_path, index=False)





if __name__ == "__main__":
    csv_path = pc_paths.csv_path / "dataset.csv"
    # parse_dataset(csv_path)
    output_splits_path = pc_paths.csv_path
    split_dataframe(csv_path, output_splits_path)



    pass