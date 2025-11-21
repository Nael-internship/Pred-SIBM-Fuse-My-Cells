import os
import numpy as np
import h5py
import image


def compare_dicts(dict1, dict2):
    differences = {}

    # Find keys that are in dict1 but not in dict2
    for key in dict1:
        if key not in dict2:
            differences[key] = (dict1[key], None)

    # Find keys that are in dict2 but not in dict1
    for key in dict2:
        if key not in dict1:
            differences[key] = (None, dict2[key])

    # Find keys that are in both dicts but have different values
    for key in dict1:
        if key in dict2 and dict1[key] != dict2[key]:
            differences[key] = (dict1[key], dict2[key])

    return differences


with h5py.File('FuseMyCells.hdf5', 'a') as hdf:
    for organ in ['nucleus', 'membrane']:

        if organ not in hdf:
            print(f'Create group {organ}.')
            group = hdf.create_group(organ)
        else:
            print(f'Get group {organ} (already exists).')
            group = hdf[organ]

        for i in range(500):
            filenameX = f'images/image_{i}_{organ}_angle.tif'
            filenameY = f'images/image_{i}_{organ}_fused.tif'
            if os.path.isfile(filenameX) and os.path.isfile(filenameY):
                print(f'images/image_{i}_{organ}')

                image_key = f'image_{i}'
                if image_key in group:
                    print(f'Skipping {image_key} (already exists).')
                    continue

                X, X_meta = image.imread(filenameX, metadata=True)
                Y, Y_meta = image.imread(filenameY, metadata=True)

                if not X_meta == Y_meta:
                    # Create cases for known errors in data
                    if (i in range(213, 400)) and organ =='nucleus':
                        X_meta = Y_meta
                    # Exception
                    else:
                        print(compare_dicts(X_meta, Y_meta))
                        raise ValueError

                sample = group.create_group(f'image_{i}')
                for k in X_meta.keys():
                    sample.attrs[k] = X_meta[k]

                # self computed attributes
                sample.attrs['angle_min']  = np.min(X)
                sample.attrs['angle_max']  = np.max(X)
                sample.attrs['angle_mean'] = np.mean(X)
                sample.attrs['angle_median'] = np.median(X)
                sample.attrs['angle_std']  = np.std(X)
                sample.attrs['angle_shape'] = X.shape

                sample.attrs['fused_min']  = np.min(Y)
                sample.attrs['fused_max']  = np.max(Y)
                sample.attrs['fused_mean'] = np.mean(Y)
                sample.attrs['fused_median'] = np.median(Y)
                sample.attrs['fused_std']  = np.std(Y)
                sample.attrs['fused_shape'] = Y.shape

                dsX = sample.create_dataset(f'angle', data=X)
                dsY = sample.create_dataset(f'fused', data=Y)

                hdf.flush()
