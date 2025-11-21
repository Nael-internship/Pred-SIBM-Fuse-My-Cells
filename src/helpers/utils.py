### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import pathlib as p

### External Imports ###
import numpy as np
import torch as tc
import tifffile
from skimage.transform import resize
from monai import data
from monai import transforms as mtr

### Internal Imports ###

#########################


def read_image(path):
    with tifffile.TiffFile(path) as tif:
        metadata = tif.imagej_metadata
        image_data = tif.asarray()
    return image_data, metadata


def save_image(image, path):
    tifffile.imwrite(
        path,
        image,
        bigtiff=True,
        tile=(128, 128),
    )

def resample_image(array, output_shape):
    """
    Resample the input image array to the given output shape using interpolation.

    Parameters:
    - array (np.ndarray): Input image array.
    - output_shape (tuple): Desired output shape (height, width).

    Returns:
    - resampled_array (np.ndarray): Resampled image array.
    """

    resampled_array = resize(array, output_shape, mode='reflect', anti_aliasing=True, preserve_range=True)
    return resampled_array


def percentile_normalization(image, pmin=2, pmax=99.8, axis=None):
    """
    Compute a percentile normalization for the given image.
    Parameters:
    - image (array): array (2D or 3D) of the image file.
    - pmin  (int or float): the minimal percentage for the percentiles to compute.
                            Values must be between 0 and 100 inclusive.
    - pmax  (int or float): the maximal percentage for the percentiles to compute.
                            Values must be between 0 and 100 inclusive.
    - axis : Axis or axes along which the percentiles are computed.
             The default (=None) is to compute it along a flattened version of the array.
    - dtype (dtype): type of the wanted percentiles (uint16 by default)

    Returns:
    Normalized image (np.ndarray): An array containing the normalized image.
    """

    if not (np.isscalar(pmin) and np.isscalar(pmax) and 0 <= pmin < pmax <= 100 ):
        raise ValueError("Invalid values for pmin and pmax")

    low_percentile = np.percentile(image, pmin, axis = axis, keepdims = True)
    high_percentile = np.percentile(image, pmax, axis = axis, keepdims = True)

    if low_percentile == high_percentile:
        print(f"Same min {low_percentile} and high {high_percentile}, image may be empty")
        return image

    return (image - low_percentile) / (high_percentile - low_percentile)



class TifReader(data.image_reader.ImageReader):
    def __init__(self):
        super().__init__()

    def verify_suffix(self, filename) -> bool:
        return True

    def read(self, data, **kwargs):
        image, _ = read_image(data[0])
        return image[np.newaxis, :, :, :]

    def get_data(self, img) -> tuple[np.ndarray, dict]:
        return img, {}

    def _get_meta_dict(self, img) -> dict:
        """
        Get the all the metadata of the image and convert to dict type.
        Args:
            img: a PIL Image object loaded from an image file.

        """
        return {}

    def _get_spatial_shape(self, img):
        """
        Get the spatial shape of image data, it doesn't contain the channel dim.
        Args:
            img: a PIL Image object loaded from an image file.
        """
        return np.asarray((img.shape[0], img.shape[1]))




class TifReaderUsingMeta(data.image_reader.ImageReader):
    def __init__(self, desired_spacing=(0.6, 0.6, 0.6)):
        super().__init__()
        self.desired_spacing = desired_spacing

    def verify_suffix(self, filename) -> bool:
        return True

    def read(self, data, **kwargs):
        image, metadata = read_image(data[0])
        desired_spacing_z, desired_spacing_y, desired_spacing_x = self.desired_spacing
        original_shape_z, original_shape_y, original_shape_x = image.shape[0], image.shape[1], image.shape[2]
        original_spacing_z, original_spacing_y, original_spacing_x = metadata['physical_size_z'], metadata['physical_size_y'], metadata['physical_size_x']
        desired_shape_z, desired_shape_y, desired_shape_x = int(original_shape_z * original_spacing_z / desired_spacing_z), int(original_shape_y * original_spacing_y / desired_spacing_y), int(original_shape_x * original_spacing_x / desired_spacing_x)
        output_shape = (desired_shape_z, desired_shape_y, desired_shape_x)
        image = resample_image(image, output_shape)
        return image[np.newaxis, :, :, :]

    def get_data(self, img) -> tuple[np.ndarray, dict]:
        return img, {}

    def _get_meta_dict(self, img) -> dict:
        """
        Get the all the metadata of the image and convert to dict type.
        Args:
            img: a PIL Image object loaded from an image file.

        """
        return {}

    def _get_spatial_shape(self, img):
        """
        Get the spatial shape of image data, it doesn't contain the channel dim.
        Args:
            img: a PIL Image object loaded from an image file.
        """
        return np.asarray((img.shape[0], img.shape[1]))














class TifReaderUsingMetaTC(data.image_reader.ImageReader):
    def __init__(self, desired_spacing=(0.6, 0.6, 0.6)):
        super().__init__()
        self.desired_spacing = desired_spacing

    def verify_suffix(self, filename) -> bool:
        return True

    def read(self, data, **kwargs):
        image, metadata = read_image(data[0])
        desired_spacing_z, desired_spacing_y, desired_spacing_x = self.desired_spacing
        original_shape_z, original_shape_y, original_shape_x = image.shape[0], image.shape[1], image.shape[2]
        original_spacing_z, original_spacing_y, original_spacing_x = metadata['physical_size_z'], metadata['physical_size_y'], metadata['physical_size_x']
        desired_shape_z, desired_shape_y, desired_shape_x = int(original_shape_z * original_spacing_z / desired_spacing_z), int(original_shape_y * original_spacing_y / desired_spacing_y), int(original_shape_x * original_spacing_x / desired_spacing_x)
        output_shape = (desired_shape_z, desired_shape_y, desired_shape_x)
        
        with tc.no_grad():
            image_tc = tc.from_numpy(image).unsqueeze(0)
            resampled_image_tc = mtr.Resize(spatial_shape=output_shape)(image_tc)
        image = resampled_image_tc.detach().cpu().numpy()
        return image

    def get_data(self, img) -> tuple[np.ndarray, dict]:
        return img, {}

    def _get_meta_dict(self, img) -> dict:
        """
        Get the all the metadata of the image and convert to dict type.
        Args:
            img: a PIL Image object loaded from an image file.

        """
        return {}

    def _get_spatial_shape(self, img):
        """
        Get the spatial shape of image data, it doesn't contain the channel dim.
        Args:
            img: a PIL Image object loaded from an image file.
        """
        return np.asarray((img.shape[0], img.shape[1]))
