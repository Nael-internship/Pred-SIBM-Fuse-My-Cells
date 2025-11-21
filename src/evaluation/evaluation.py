### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from typing import Union
import pathlib
import time

### External Imports ###
import numpy as np
import pandas as pd
import torch as tc
import torchio as tio
from scipy import ndimage as nd

### MONAI Imports ###
from monai import transforms as mtr
from monai.inferers import sliding_window_inference
from monai.data import Dataset, list_data_collate, DataLoader

### Internal Imports ###
from paths import pc_paths as p
from helpers import objective_functions as of
from datasets import dataset_patch as dp
from datasets import dataset_resampling as ds
from networks import runet
from helpers import utils as u
########################


def load_checkpoint(model, checkpoint_path):
    state_dict = tc.load(checkpoint_path, weights_only=False)['state_dict']
    new_state_dict = {}
    for key in state_dict.keys():
        new_state_dict[key.replace("model.", "", 1)] = state_dict[key]
    model.load_state_dict(new_state_dict)
    return model


def load_double_checkpoint(model_resampled, model_patch, checkpoint_path):
    state_dict = tc.load(checkpoint_path, weights_only=False)['state_dict']
    ### Model Resampled ###
    resampled_state_dict = {}
    for key in state_dict.keys():
        if "resampled_model" in key:
            resampled_state_dict[key.replace("resampled_model.", "", 1)] = state_dict[key]
    model_resampled.load_state_dict(resampled_state_dict)
    ### Model Patch ###
    patch_state_dict = {}
    for key in state_dict.keys():
        if "patch_model" in key:
            patch_state_dict[key.replace("patch_model.", "", 1)] = state_dict[key]
    model_patch.load_state_dict(patch_state_dict)
    return model_resampled, model_patch







def evaluation_identity(data_path, input_csv_path, output_csv_path, model=None, device = "cuda:0"):
    data = ds.prepare_data(data_path, input_csv_path, mode="testing")
    batch_size = 1
    num_workers = 8
    transforms = mtr.Compose([
        mtr.LoadImaged(keys=['image', 'gt'], reader=u.TifReader, image_only=True),
        mtr.NormalizeIntensityd(keys=["image", "gt"]),
    ])

    dataset = Dataset(data=data, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=list_data_collate, shuffle=False)

    results = []
    with tc.no_grad():
        for idx, batch in enumerate(dataloader):
            print(f"Case: {idx + 1} / {len(dataset)}")
            input_data, ground_truth = batch["image"], batch['gt']
            original_shape = input_data.shape[2:]
            output = input_data.clone()
            output = output.cpu()
            print(f"Upsampled shape: {output.shape}")
            mae = of.mean_absolute_error(output, ground_truth)
            mse = of.mean_squared_error(output, ground_truth)
            pcc = -of.pearson_correlation_coefficient(output, ground_truth)
            cd = -of.cosine_distance(output, ground_truth)
            ssim_transform = mtr.Resize(spatial_size=(original_shape[0] // 2, original_shape[1] // 2, original_shape[2] // 2))
            ssim = -of.structural_similarity_index_measure(ssim_transform(output[0]).unsqueeze(0).cuda(), ssim_transform(ground_truth[0]).unsqueeze(0).cuda())
            print(f"MAE: {mae:.4f}, SSIM: {ssim:.4f}, MSE: {mse:.4f}, PCC: {pcc:.4f}, CD: {cd:.4f}")
            path = str(batch["input_path"][0])
            to_append = (path, mae.item(), ssim.item(), mse.item(), pcc.item(), cd.item())
            results.append(to_append)

    dataframe = pd.DataFrame(results, columns=['Path', 'MAE', 'SSIM', 'MSE', 'PCC', 'CD'])
    dataframe.to_csv(output_csv_path, index=False)













def evaluation_resampled(data_path, input_csv_path, output_csv_path, model, device = "cuda:0"):
    data = ds.prepare_data(data_path, input_csv_path, mode="testing")
    patch_size = (256, 256, 256)
    batch_size = 1
    num_workers = 8
    model = model.to(device)
    model.eval()

    transforms = mtr.Compose([
        mtr.LoadImaged(keys=['image', 'gt'], reader=u.TifReader, image_only=True),
        mtr.NormalizeIntensityd(keys=["image", "gt"]),
    ])

    dataset = Dataset(data=data, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=list_data_collate, shuffle=False)

    results = []
    with tc.no_grad():
        for idx, batch in enumerate(dataloader):
            print(f"Case: {idx + 1} / {len(dataset)}")
            input_data, ground_truth = batch["image"], batch['gt']
            original_shape = input_data.shape[2:]
            downsampling_transform = mtr.Resize(spatial_size=patch_size)
            upsampling_transform = mtr.Resize(spatial_size=(original_shape[0], original_shape[1], original_shape[2]))
            print(f"Input shape: {input_data.shape}")
            input_data = downsampling_transform(input_data[0]).unsqueeze(0)
            print(f"Downsampled shape: {input_data.shape}")
            input_data = input_data.to(device)
            output = model(input_data)
            output = upsampling_transform(output[0]).unsqueeze(0)
            output = output.cpu()
            print(f"Upsampled shape: {output.shape}")
            mae = of.mean_absolute_error(output, ground_truth)
            mse = of.mean_squared_error(output, ground_truth)
            pcc = -of.pearson_correlation_coefficient(output, ground_truth)
            cd = -of.cosine_distance(output, ground_truth)
            ssim_transform = mtr.Resize(spatial_size=(original_shape[0] // 2, original_shape[1] // 2, original_shape[2] // 2))
            ssim = -of.structural_similarity_index_measure(ssim_transform(output[0]).unsqueeze(0).cuda(), ssim_transform(ground_truth[0]).unsqueeze(0).cuda())
            print(f"MAE: {mae:.4f}, SSIM: {ssim:.4f}, MSE: {mse:.4f}, PCC: {pcc:.4f}, CD: {cd:.4f}")
            path = str(batch["input_path"][0])
            to_append = (path, mae.item(), ssim.item(), mse.item(), pcc.item(), cd.item())
            results.append(to_append)

    dataframe = pd.DataFrame(results, columns=['Path', 'MAE', 'SSIM', 'MSE', 'PCC', 'CD'])
    dataframe.to_csv(output_csv_path, index=False)














def evaluation_patch_based(data_path, input_csv_path, output_csv_path, model, device="cuda:0"):
    data = dp.prepare_data(data_path, input_csv_path, mode="testing")
    patch_size = (256, 256, 256)
    batch_size = 1
    num_workers = 8
    model = model.to(device)
    model.eval()

    transforms = mtr.Compose([
        mtr.LoadImaged(keys=['image', 'gt'], reader=u.TifReader, image_only=True),
        mtr.NormalizeIntensityd(keys=["image", "gt"]),
    ])

    dataset = Dataset(data=data, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=list_data_collate, shuffle=False)

    results = []
    with tc.no_grad():
        for idx, batch in enumerate(dataloader):
            print(f"Case: {idx + 1} / {len(dataset)}")
            input_data, ground_truth = batch["image"], batch['gt']
            original_shape = input_data.shape[2:]
            print(f"Input shape: {input_data.shape}")
            # input_data, ground_truth = input_data.to(device), ground_truth.to(device)
            b_t = time.time()
            output = sliding_window_inference(input_data, patch_size, batch_size, model.forward, mode='constant', sw_device=device, device="cpu")
            e_t = time.time()
            print(f"Inference time: {e_t - b_t} seconds.")
            mae = of.mean_absolute_error(output, ground_truth)
            mse = of.mean_squared_error(output, ground_truth)
            pcc = -of.pearson_correlation_coefficient(output, ground_truth)
            cd = -of.cosine_distance(output, ground_truth)
            ssim_transform = mtr.Resize(spatial_size=(original_shape[0] // 2, original_shape[1] // 2, original_shape[2] // 2))
            ssim = -of.structural_similarity_index_measure(ssim_transform(output[0]).unsqueeze(0).cuda(), ssim_transform(ground_truth[0]).unsqueeze(0).cuda())
            print(f"MAE: {mae:.4f}, SSIM: {ssim:.4f}, MSE: {mse:.4f}, PCC: {pcc:.4f}, CD: {cd:.4f}")
            path = str(batch["input_path"][0])
            to_append = (path, mae.item(), ssim.item(), mse.item(), pcc.item(), cd.item())
            results.append(to_append)

    dataframe = pd.DataFrame(results, columns=['Path', 'MAE', 'SSIM', 'MSE', 'PCC', 'CD'])
    dataframe.to_csv(output_csv_path, index=False)




















def evaluation_doublestep(data_path, input_csv_path, output_csv_path, resampled_model, patch_model, device="cuda:0", use_weights=False):
    data = dp.prepare_data(data_path, input_csv_path, mode="testing")
    patch_size = (224, 224, 224)
    batch_size = 1
    num_workers = 8
    resampled_model = resampled_model.to(device)
    resampled_model.eval()
    patch_model = patch_model.to(device)
    patch_model.eval()

    transforms = mtr.Compose([
        mtr.LoadImaged(keys=['image', 'gt'], reader=u.TifReaderUsingMeta, image_only=True),
        mtr.NormalizeIntensityd(keys=["image", "gt"]),
    ])

    dataset = Dataset(data=data, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=list_data_collate, shuffle=False)

    results = []
    with tc.no_grad():
        for idx, batch in enumerate(dataloader):
            print(f"Case: {idx + 1} / {len(dataset)}")
            input_data, ground_truth = batch["image"], batch['gt']
            original_shape = input_data.shape[2:]
            print(f"Input shape: {input_data.shape}")


            downsampling_transform = mtr.Resize(spatial_size=patch_size)
            upsampling_transform = mtr.Resize(spatial_size=(original_shape[0], original_shape[1], original_shape[2]))
            image_resampled_tc = downsampling_transform(input_data[0]).unsqueeze(0)
            print(f"Resampled Image Shape: {image_resampled_tc.shape}")
            output_resampled_tc = resampled_model(image_resampled_tc.to(device)).cpu()
            print(f"Resampled Output Shape: {output_resampled_tc.shape}")
            output_upsampled_tc = upsampling_transform(output_resampled_tc[0]).unsqueeze(0)
            print(f"Upsampled Output Shape: {output_upsampled_tc.shape}")
            if original_shape[0] <= patch_size[0] or original_shape[1] <= patch_size[1] or original_shape[2] <= patch_size[2]:
                output_patches_tc = output_upsampled_tc
                print(f"Second step not necessary due to low resolution.")
            else:
                image_patches_tc = tc.cat((input_data, output_upsampled_tc), dim=1)
                output_patches_tc = sliding_window_inference(image_patches_tc, patch_size, batch_size, patch_model.forward, mode='constant', sw_device=device, device="cpu")
                print(f"Patch Output Shape: {output_patches_tc.shape}")

            output = output_patches_tc.cpu().detach()

            if use_weights:
                output = mtr.NormalizeIntensity()(output[0]).unsqueeze(0)
                difference = np.abs(input_data.cpu().numpy() - output.cpu().numpy())[0, 0, :, :, :]
                difference_resampled = u.resample_image(difference, (128, 128, 128))
                difference_resampled = nd.maximum_filter(difference_resampled, size=(9, 9, 9))
                difference_resampled = nd.gaussian_filter(difference_resampled, sigma=3)
                difference_resampled = u.percentile_normalization(difference_resampled, pmin=2, pmax=98)
                difference_resampled = np.clip(difference_resampled, 0, 1)
                difference_upsampled = u.resample_image(difference_resampled, difference.shape)
                difference_upsampled = tc.tensor(difference_upsampled).unsqueeze(0).unsqueeze(0)
                output = input_data * (1 - difference_upsampled) + output * difference_upsampled

            mae = of.mean_absolute_error(output, ground_truth)
            mse = of.mean_squared_error(output, ground_truth)
            pcc = -of.pearson_correlation_coefficient(output, ground_truth)
            cd = -of.cosine_distance(output, ground_truth)
            ssim_transform = mtr.Resize(spatial_size=(original_shape[0] // 2, original_shape[1] // 2, original_shape[2] // 2))
            ssim = -of.structural_similarity_index_measure(ssim_transform(output[0]).unsqueeze(0).cuda(), ssim_transform(ground_truth[0]).unsqueeze(0).cuda())
            print(f"MAE: {mae:.4f}, SSIM: {ssim:.4f}, MSE: {mse:.4f}, PCC: {pcc:.4f}, CD: {cd:.4f}")
            path = str(batch["input_path"][0])
            to_append = (path, mae.item(), ssim.item(), mse.item(), pcc.item(), cd.item())
            results.append(to_append)

    dataframe = pd.DataFrame(results, columns=['Path', 'MAE', 'SSIM', 'MSE', 'PCC', 'CD'])
    dataframe.to_csv(output_csv_path, index=False)



        




def run():
    pass






if __name__ == "__main__":
    run()
