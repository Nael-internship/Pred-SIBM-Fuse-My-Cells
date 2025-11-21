### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from typing import Union
import pathlib

### External Imports ###
import numpy as np
import pandas as pd
import torch as tc
import torchio as tio

### MONAI Imports ###
from monai import transforms as mtr

### Internal Imports ###
from paths import hpc_paths as p
from helpers import objective_functions as of
from datasets import dataset_patch as dp
from networks import runet
from helpers import utils as u
########################



def load_checkpoint_without_discriminator(model_resampled, model_patch, checkpoint_path):
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






# NOTE - Only best experiments are kept here



def exp_mae_4(fold=1):
    """
    """
    ### Dataset Params ###
    training_dataset_path = p.data_path / "Resampled_06_06_06"
    validation_dataset_path = p.data_path / "Resampled_06_06_06"
    training_csv_path = p.csv_path / f"training_fold_{fold}.csv"
    validation_csv_path = p.csv_path / f"val_fold_{fold}.csv"

    ### Prepare Data ###
    training_data = dp.prepare_data(training_dataset_path, training_csv_path)
    validation_data = dp.prepare_data(validation_dataset_path, validation_csv_path)

    ### Prepare Loading & Augmentation ###
    samples_per_volume = 1
    patch_size = (224, 224, 224)
    batch_size = 1
    num_workers = 4
    training_transforms = mtr.Compose([
        mtr.LoadImaged(keys=['image', 'gt'], reader=u.TifReader, image_only=True),
        mtr.NormalizeIntensityd(keys=["image", "gt"]),
        mtr.RandAxisFlipd(keys=['image', 'gt'], prob=0.5),
        mtr.RandRotate90d(keys=['image', 'gt'], prob=0.5),
        mtr.SpatialPadd(keys=["image", "gt"], spatial_size=patch_size, method="symmetric", mode='constant'),
    ])

    validation_transforms = mtr.Compose([
        mtr.LoadImaged(keys=['image', 'gt'], reader=u.TifReader, image_only=True),
        mtr.NormalizeIntensityd(keys=["image", "gt"]),
        mtr.SpatialPadd(keys=["image", "gt"], spatial_size=patch_size, method="symmetric", mode='constant'),
    ])

    ### General Parameters ###
    experiment_name = f"FMC_RUNet_DoubleStepAdv_224_06_MAE_Fold{fold}"
    learning_rate = 0.001
    save_step = 50
    to_load_checkpoint_path = None
    number_of_images_to_log = 3
    objective_function = of.mean_absolute_error
    objective_function_params = {}
    optimizer_weight_decay = 0.005
    echo = True

    accelerator = 'gpu'
    devices = [0, 1, 2, 3]
    logger = None
    callbacks = None
    max_epochs = 801
    precision = "bf16-mixed"
    strategy = "ddp_find_unused_parameters_true"
    deterministic = False

    use_initial_transform = False

    ### Declare Models ###
    config_resampled = runet.config_224()
    model_resampled = runet.RUNet(**config_resampled)

    config_patch = runet.config_224_ss()
    model_patch = runet.RUNet(**config_patch)

    config_discriminator = runet.config_224_disc()
    model_discriminator = runet.RUNetDiscriminator(**config_discriminator)

    ### Load Checkpoint ###
    checkpoints = {1: 'epoch=299_general.ckpt'}
    checkpoint_path = p.checkpoints_path / f"FMC_RUNet_DoubleStep_256_MAE_Fold{fold}" / checkpoints[int(fold)]
    model_resampled, model_patch = load_checkpoint_without_discriminator(model_resampled, model_patch, checkpoint_path)

    ### Lightning Parameters ###
    lighting_params = dict()
    lighting_params['accelerator'] = accelerator
    lighting_params['devices'] = devices
    lighting_params['logger'] = logger
    lighting_params['callbacks'] = callbacks
    lighting_params['max_epochs'] = max_epochs
    lighting_params['precision'] = precision
    lighting_params['strategy'] = strategy
    lighting_params['deterministic'] = deterministic

    ### Parse Parameters ###
    training_params = dict()
    ### General params
    training_params['experiment_name'] = experiment_name
    training_params['patch_model'] = model_patch
    training_params['resampled_model'] = model_resampled
    training_params['discriminator_model'] = model_discriminator
    training_params['learning_rate'] = learning_rate
    training_params['to_load_checkpoint_path'] = to_load_checkpoint_path
    training_params['save_step'] = save_step
    training_params['number_of_images_to_log'] = number_of_images_to_log
    training_params['echo'] = echo
    training_params['num_iterations'] = max_epochs
    training_params['lightning_params'] = lighting_params

    training_params['training_data'] = training_data
    training_params['validation_data'] = validation_data
    training_params['training_transforms'] = training_transforms
    training_params['validation_transforms'] = validation_transforms
    training_params['num_workers'] = num_workers
    training_params['batch_size'] = batch_size
    training_params['patch_size'] = patch_size
    training_params['samples_per_volume'] = samples_per_volume
    training_params['use_initial_transform'] = use_initial_transform
    
    ### Cost functions and params
    training_params['objective_function'] = objective_function
    training_params['objective_function_params'] = objective_function_params
    training_params['optimizer_weight_decay'] = optimizer_weight_decay
    training_params['lightning_params'] = lighting_params

    ########################################
    return training_params




