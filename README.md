# FuseMyCells `🔬+🤖 = 2×🔬`
France-BioImaging's Fuse My Cells challenge

Link to the challenge for more information :
* [fusemycells.grand-challenge.org](https://fusemycells.grand-challenge.org/)
* [france-bioimaging.org announcement](https://france-bioimaging.org/announcement/france-bioimaging-challenge-is-back-fuse-my-cells/)


## Results

* [test_phase/leaderboard](https://fusemycells.grand-challenge.org/evaluation/test_phase/leaderboard/)
  * closed 07/03/2025 (initial date 28/02/2025)
* [evaluation/leaderboard](https://fusemycells.grand-challenge.org/evaluation/evaluation/leaderboard/)
  * closed 17/03/2025

## Usage

### Prepare dataset

1. Download the dataset zip files un put all of them in
a folder. The following instruction need to be executed
from this folder.
2. Run script `01_unzip.py`
    * Notes: the scripts for data prepration are located
      in the `data` folder.
3. Run script `02_tif_to_hdf5.py`

Now, the zip files have been extracted to an `images` folder
and then put in a single file named `FuseMyCells.hdf5`.

### Run an evaluation

```
usage: eval.py [-h] [--use-gpu] --method {gaussian_filter,denoise_wavelet,denoise_tv_bregman} [--args ARGS [ARGS ...]]
               [--dataset DATASET] [--crop-data]
eval.py: error: the following arguments are required: --method
```

`python eval.py --method gaussian_filter --args sigma=0.5 --dataset FuseMyCells.hdf5`

## Method

Taking the [docker_template](https://seafile.lirmm.fr/d/233a5a399c8544dfb41a/) given by the organizer as a start point.

```python
from scipy import ndimage
if metadata['channel'] == 'nucleus':
    image_predict = ndimage.gaussian_filter(image_input, 0.442)
else:
    image_predict = ndimage.gaussian_filter(image_input, 0.5)
```

The filter sigma values have been manually selected from evaluation on the training dataset.
The evaluation process is done using `eval.py` and in our case usage of the `run.sh` script.


## Changelog

#### 27/03/2025

* Update README (cleaning for final version)
* Add code for the docker

#### 07/03/2025

* Add result used to specify methods
* Update README (add method)

#### 28/02/2025

* Adding evaluation of method script
  * usage for classical computer vision methods
* Update README (put my exp results in the idea section)

#### 17/01/2025

* Working on data acquisition and preprocessing
  * script to unzip all the data at once
  * script to convert all the images into a single HDF5 file for easier load
