# script to unzip the dataset images
import os
import zipfile

def unzip(filepath):
    fpath, fname = os.path.split(filepath)
    if not zipfile.is_zipfile(filepath):
        raise NotImplementedError

    print('UNZIP', filepath, 'TO', fpath)
    with zipfile.ZipFile(filepath, 'r') as my_zip:
        # extract everything
        my_zip.extractall(fpath + 'images/')


unzip('image_0-20.zip')
for i in range(20, 400, 20):
    unzip(f'image_{i+1}-{i+20}.zip')
