"""

The following is a simple example algorithm inference
where the images predicted are only full black images.

"""

print(" START IMPORT ")

import os
from os import listdir, mkdir
from os.path import basename, isdir, join
from pathlib import Path

import numpy as np
import tifffile

# from tools import percentile_normalization


print(" END IMPORT ")

INPUT_PATH = Path("/input/images/fluorescence-lightsheet-3D-microscopy")
print(f" INPUT_PATH IS   " + str(INPUT_PATH))
os.system("ls -l " + str(INPUT_PATH))

OUTPUT_PATH = Path("/output")
if not isdir(join(OUTPUT_PATH,"images")):
    mkdir(join(OUTPUT_PATH, "images"))

OUTPUT_PATH = Path("/output/images")
if not isdir(join(OUTPUT_PATH, "fused-fluorescence-lightsheet-3D-microscopy")):
    mkdir(join(OUTPUT_PATH, "fused-fluorescence-lightsheet-3D-microscopy"))

OUTPUT_PATH = Path("/output/images/fused-fluorescence-lightsheet-3D-microscopy")
print(" OUTPUT IS  " + str(OUTPUT_PATH))

RESOURCE_PATH = Path("resources") #WEIGHTS NEED TO BE PUT IN RESOURCE_PATH
print(" RESOURCE_PATH IS   " + str(RESOURCE_PATH))
os.system("ls -l " + str(RESOURCE_PATH))


def run():
    print(" LOAD NETWORK ")
    # model = mynetwork
    # weight_file = join(RESOURCE_PATH, your_model.keras")
    # model.load_weights(weight_file)

    print(f" LIST IMAGES IN  {INPUT_PATH} ")

    for input_file_name in listdir(INPUT_PATH):
        if input_file_name.endswith("tiff") or input_file_name.endswith("tif"):
            print(" --> Predict " + input_file_name)
            image_input, metadata = read_image(join(INPUT_PATH,input_file_name))

            # Prediction
            # image_predict = np.zeros(image_input.shape, dtype = np.uint16) # model.predict()
            # V1
            '''
            image_predict = np.copy(image_input)
            '''
            # V2 & 3
            '''
            from scipy import ndimage
            image_predict = ndimage.gaussian_filter(image_input, 0.5)
            # image_predict = ndimage.gaussian_filter(image_input, 0.48)
            '''
            # V4
            '''
            import skimage.restoration
            image_predict = skimage.restoration.denoise_wavelet(image_input, 11.0)
            '''
            # V5
            '''
            import skimage.morphology
            image_predict = skimage.morphology.closing(image_input, skimage.morphology.ball(1.0))
            '''
            # V6 / VF
            from scipy import ndimage
            if metadata['channel'] == 'nucleus':
                image_predict = ndimage.gaussian_filter(image_input, 0.442)
            else:
                image_predict = ndimage.gaussian_filter(image_input, 0.5)

            save_image(location = join(OUTPUT_PATH, basename(input_file_name)),
                       array = image_predict,
                       metadata = metadata
                       )

    print(" --> LIST OUTPUT IMAGES IN "+str(OUTPUT_PATH))

    for output_images in listdir(OUTPUT_PATH):
        print(" --> FOUND "+str(output_images))
    return 0



def standardize_metadata(metadata : dict):
    key_map = {
        "spacing": ["spacing"],
        "PhysicalSizeX": ["PhysicalSizeX", "physicalsizex", "physical_size_x"],
        "PhysicalSizeY": ["PhysicalSizeY", "physicalsizey", "physical_size_y"],
        "PhysicalSizeZ": ["PhysicalSizeZ", "physicalsizez", "physical_size_z"],
        "unit": ["unit"],
        "axes": ["axes"],
        "channel": ["channel"],
        "shape": ["shape"],
        "study": ["study"],
    }

    # Normalize metadata by looking up possible keys
    standardized_metadata = {}
    for standard_key, possible_keys in key_map.items():
        for key in possible_keys:
            if key in metadata:
                standardized_metadata[standard_key] = metadata[key]
                break  # Stop once we find the first available key

    return standardized_metadata



def read_image(location): # WARNING IMAGE DATA EN ZYX
    import tifffile
    # Read the TIFF file and get the image and metadata
    with tifffile.TiffFile(location) as tif:

        image_data = tif.asarray() # Extract image array data

        if tif.shaped_metadata is not None:
            shp_metadata = tif.shaped_metadata[0]
            metadata = standardize_metadata(shp_metadata)

            return image_data, metadata
        else:
            if tif.imagej_metadata is not None:
                shape = list(image_data.shape)
                imgj_metadata = tif.imagej_metadata
                imgj_metadata['shape'] = shape
                metadata = standardize_metadata(imgj_metadata)

                return image_data, metadata

            else:
                metadata = tif.pages[0].tags['ImageDescription'].value
                print(f"error loading metadata: {metadata}, type of object : {type(metadata)}")



def save_image(*, location, array, metadata):

    PhysicalSizeX = metadata['PhysicalSizeX']
    PhysicalSizeY = metadata['PhysicalSizeY']
    tifffile.imwrite(
        location,
        array,
        bigtiff=True, #Keep it for 3D images
        resolution=(1. / PhysicalSizeX, 1. / PhysicalSizeY),
        metadata=metadata,
        tile=(128, 128),
        )


if __name__ == "__main__":
    raise SystemExit(run())
