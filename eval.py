import argparse
import os

import h5py
import numpy as np
import skimage
import scipy

from data.normalization import percentile_normalization

# Methods
method_mapping = {
    'gaussian_filter':    scipy.ndimage.gaussian_filter,
    'denoise_wavelet':    skimage.restoration.denoise_wavelet,
    'denoise_tv_bregman': skimage.restoration.denoise_tv_bregman,
}

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--use-gpu', action='store_true',
                    help='Flag to indicate to not hide the GPU with CUDA_VISIBLE_DEVICES')
parser.add_argument('--method', type=str, required=True,
                    choices=method_mapping.keys(),
                    help='Name of the method to use')
parser.add_argument('--args', type=str, nargs='+',
                    help='List of key=value arguments for the method')
parser.add_argument('--dataset', type=str, default='FuseMyCells.hdf5')
parser.add_argument('--crop-data', action='store_true',
                    help='Flag to indicate to crop the dataset to only eval on the center')
args = parser.parse_args()

if not args.use_gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# TensorFlow import may change depending on CUDA visibility
import tensorflow as tf

# Process arguments
method = method_mapping.get(args.method)
args_dict = {}
for item in args.args:
    key, value = item.split('=')
    args_dict[key] = float(value)  # Assuming values are floats

dataset_filename = args.dataset
dataset = h5py.File(dataset_filename, 'r')

def metric(x, y):
    return float(tf.image.ssim(percentile_normalization(x),
                               percentile_normalization(y),
                               10))

scores = {'all': []}
for k1 in dataset:
    scores[k1] = []

    for k2 in dataset[k1]:
        if args.crop_data:
            z, y, x = dataset[k1][k2].attrs['angle_shape']
            mid_z = z // 4
            mid_y = y // 4
            mid_x = x // 4
            image_input = dataset[k1][k2]['angle'][mid_z:3 * mid_z, mid_y:3 * mid_y, mid_x:3 * mid_x]
            image_truth = dataset[k1][k2]['fused'][mid_z:3 * mid_z, mid_y:3 * mid_y, mid_x:3 * mid_x]
        else:
            image_input = dataset[k1][k2]['angle']
            image_truth = dataset[k1][k2]['fused']

        base_score = metric(image_input, image_truth)
        method_score = metric(method(image_input, **args_dict), image_truth)

        scores['all'].append((method_score - base_score) / (1 - base_score))
        scores[k1].append((method_score - base_score) / (1 - base_score))

print(f">>>>> {str(method.__name__)},"
      f"{' '.join(args.args)},"
      f"{np.mean(scores['all'])},"
      f"{np.mean(scores['nucleus'])},"
      f"{np.mean(scores['membrane'])}")
