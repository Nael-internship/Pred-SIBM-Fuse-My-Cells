import tifffile

def tifffile_imread(filename, metadata=False):
    with tifffile.TiffFile(filename) as tiff:
        image_data = tiff.asarray()
        if not metadata:
            return image_data
        metadata = tiff.imagej_metadata

        return image_data, metadata


# function aliases
imread = tifffile_imread
tifffile_imwrite = tifffile.imwrite
imwrite = tifffile_imwrite
