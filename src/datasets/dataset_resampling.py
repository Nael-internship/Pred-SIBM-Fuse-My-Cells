### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import pathlib as p

### External Imports ###
import pandas as pd

### Internal Imports ###

#########################


def prepare_data(dataset_path, dataframe_path, mode="training"):
    dataframe = pd.read_csv(dataframe_path)
    dataset = []
    for idx in range(len(dataframe)):
        row = dataframe.iloc[idx]
        volume_path = dataset_path / row['Input Path']
        gt_path = dataset_path / row['Ground-Truth Path']
        if mode == "training":
            to_append = {"image": volume_path, "gt": gt_path}
        else:
            to_append = {"image": volume_path, "gt": gt_path, "input_path": row['Input Path']}
        dataset.append(to_append)
    return dataset



def run():
    pass

if __name__ == "__main__":
    run()