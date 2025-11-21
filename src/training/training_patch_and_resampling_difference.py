### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import pathlib
from itertools import chain
from typing import Iterable, Callable, Union

### External Imports ###
import torch as tc
import lightning as pl
import matplotlib.pyplot as plt
from torchvision import transforms as tv
import PIL

### MONAI Imports ###
from monai import transforms as mtr
from monai.data import Dataset, list_data_collate, DataLoader

### Internal Imports ###
from visualization import volumetric as vol
from helpers import objective_functions as of

#########################




class LightningModule(pl.LightningModule):
    def __init__(self, training_params : dict, lightning_params : dict):
        super().__init__()
        ### Models ###
        self.resampled_model : tc.nn.Module = training_params['resampled_model']
        self.patch_model : tc.nn.Module = training_params['patch_model']
        ### General Params ###
        self.training_params = training_params
        self.learning_rate : float = training_params['learning_rate']
        self.optimizer_weight_decay : float = training_params['optimizer_weight_decay']
        self.log_image_iters : Iterable[int] = training_params['log_image_iters']
        self.number_of_images_to_log : int = training_params['number_of_images_to_log']
        self.echo : bool = training_params['echo']

        ### Objective Functions ###
        self.objective_function : Callable = training_params['objective_function']
        self.objective_function_params : dict = training_params['objective_function_params']
        self.patch_size : tuple = training_params['patch_size']
        self.samples_per_volume : int = training_params['samples_per_volume']
        self.use_initial_transform : int = training_params['use_initial_transform']

    def forward(self, x):
        output = self.resampled_model(x)
        return output
    
    def configure_optimizers(self):
        optimizer = tc.optim.AdamW(chain(self.resampled_model.parameters(), self.patch_model.parameters()), self.learning_rate, weight_decay=self.optimizer_weight_decay)
        scheduler = {
            "scheduler": tc.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, patience=10),
            "frequency": 1,
            "interval": "epoch",
            "monitor": "Loss/Validation/loss",
        }
        dict = {'optimizer': optimizer, "lr_scheduler": scheduler}
        return dict
    
    def training_step(self, batch, batch_idx):
        ### Get Batch ###
        input_data, ground_truth = batch["image"], batch['gt']
        original_shape = input_data.shape[2:]
        with tc.no_grad():
            downsampling_transform = mtr.Resize(spatial_size=self.patch_size)
            upsampling_transform = mtr.Resize(spatial_size=(original_shape[0], original_shape[1], original_shape[2]))
            input_resampled = downsampling_transform(input_data[0]).unsqueeze(0)
            ground_truth_resampled = downsampling_transform(ground_truth[0]).unsqueeze(0)
            ground_truth_resampled = tc.abs(ground_truth_resampled - input_resampled)
            ground_truth_resampled = (ground_truth_resampled - tc.min(ground_truth_resampled)) / (tc.max(ground_truth_resampled) - tc.min(ground_truth_resampled))
        output_resampled = self.resampled_model(input_resampled)
        with tc.no_grad():
            output_upsampled = upsampling_transform(output_resampled[0]).unsqueeze(0)
            patches = tc.cat((input_data, output_upsampled, ground_truth), dim=1)
            patch_transform = mtr.RandSpatialCropSamples(roi_size=self.patch_size, random_size=False, num_samples=self.samples_per_volume)
            epatches = tc.stack(patch_transform(patches[0]))
            input_patch = epatches[:, 0:2, :, :, :]
            difference_patch = epatches[:, 1:2, :, :, :]
            ground_truth_patch = epatches[:, 2:, :, :, :]
        output_patch = self.patch_model(input_patch)
        output_patch = (1 - difference_patch) * input_patch[:, 0:1, :, :, :] + difference_patch * output_patch

        loss_resampled = self.objective_function(output_resampled, ground_truth_resampled, **self.objective_function_params)
        loss_patch = self.objective_function(output_patch, ground_truth_patch, **self.objective_function_params)

        loss = (loss_resampled + loss_patch) / 2.0

        mae_resampled = of.mean_absolute_error(output_resampled, ground_truth_resampled)
        cd_resampled = of.cosine_distance(output_resampled, ground_truth_resampled)
        pcc_resampled = of.pearson_correlation_coefficient(output_resampled, ground_truth_resampled)
        ssim_resampled = -of.structural_similarity_index_measure(output_resampled, ground_truth_resampled)

        mae_patch = of.mean_absolute_error(output_patch, ground_truth_patch)
        cd_patch = of.cosine_distance(output_patch, ground_truth_patch)
        pcc_patch = of.pearson_correlation_coefficient(output_patch, ground_truth_patch)
        ssim_patch = -of.structural_similarity_index_measure(output_patch, ground_truth_patch)

        ### Log Losses ###
        self.log("Loss/Training/loss", loss, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)
        self.log("Loss/Training/loss_resampled", loss_resampled, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)
        self.log("Loss/Training/loss_patch", loss_patch, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)

        self.log("Loss/Training/mae_resampled", mae_resampled, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)
        self.log("Loss/Training/cd_resampled", cd_resampled, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)
        self.log("Loss/Training/pcc_resampled", pcc_resampled, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)
        self.log("Loss/Training/ssim_resampled", ssim_resampled, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)

        self.log("Loss/Training/mae_patch", mae_patch, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)
        self.log("Loss/Training/cd_patch", cd_patch, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)
        self.log("Loss/Training/pcc_patch", pcc_patch, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)
        self.log("Loss/Training/ssim_patch", ssim_patch, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)

        if self.current_epoch in self.log_image_iters and batch_idx < self.number_of_images_to_log:
            self.log_images(input_resampled[:, 0:1, :, :, :], output_resampled, ground_truth_resampled, batch_idx, "Training_Resampled")
            self.log_images(input_patch[:, 0:1, :, :, :], output_patch, ground_truth_patch, batch_idx, "Training_Patch")
        return loss
 
                        
    def validation_step(self, batch, batch_idx):
        ### Get Batch ###
        input_data, ground_truth = batch["image"], batch['gt']
        original_shape = input_data.shape[2:]
        with tc.no_grad():
            downsampling_transform = mtr.Resize(spatial_size=self.patch_size)
            upsampling_transform = mtr.Resize(spatial_size=(original_shape[0], original_shape[1], original_shape[2]))
            input_resampled = downsampling_transform(input_data[0]).unsqueeze(0)
            ground_truth_resampled = downsampling_transform(ground_truth[0]).unsqueeze(0)
            ground_truth_resampled = tc.abs(ground_truth_resampled - input_resampled)
            ground_truth_resampled = (ground_truth_resampled - tc.min(ground_truth_resampled)) / (tc.max(ground_truth_resampled) - tc.min(ground_truth_resampled))
        output_resampled = self.resampled_model(input_resampled)
        with tc.no_grad():
            output_upsampled = upsampling_transform(output_resampled[0]).unsqueeze(0)
            patches = tc.cat((input_data, output_upsampled, ground_truth), dim=1)
            patch_transform = mtr.RandSpatialCropSamples(roi_size=self.patch_size, random_size=False, num_samples=self.samples_per_volume)
            epatches = tc.stack(patch_transform(patches[0]))
            input_patch = epatches[:, 0:2, :, :, :]
            difference_patch = epatches[:, 1:2, :, :, :]
            ground_truth_patch = epatches[:, 2:, :, :, :]
        output_patch = self.patch_model(input_patch)
        output_patch = (1 - difference_patch) * input_patch[:, 0:1, :, :, :] + difference_patch * output_patch

        loss_resampled = self.objective_function(output_resampled, ground_truth_resampled, **self.objective_function_params)
        loss_patch = self.objective_function(output_patch, ground_truth_patch, **self.objective_function_params)

        loss = (loss_resampled + loss_patch) / 2.0

        mae_resampled = of.mean_absolute_error(output_resampled, ground_truth_resampled)
        cd_resampled = of.cosine_distance(output_resampled, ground_truth_resampled)
        pcc_resampled = of.pearson_correlation_coefficient(output_resampled, ground_truth_resampled)
        ssim_resampled = -of.structural_similarity_index_measure(output_resampled, ground_truth_resampled)

        mae_patch = of.mean_absolute_error(output_patch, ground_truth_patch)
        cd_patch = of.cosine_distance(output_patch, ground_truth_patch)
        pcc_patch = of.pearson_correlation_coefficient(output_patch, ground_truth_patch)
        ssim_patch = -of.structural_similarity_index_measure(output_patch, ground_truth_patch)

        ### Log Losses ###
        self.log("Loss/Validation/loss", loss, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)
        self.log("Loss/Validation/loss_resampled", loss_resampled, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)
        self.log("Loss/Validation/loss_patch", loss_patch, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)

        self.log("Loss/Validation/mae_resampled", mae_resampled, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)
        self.log("Loss/Validation/cd_resampled", cd_resampled, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)
        self.log("Loss/Validation/pcc_resampled", pcc_resampled, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)
        self.log("Loss/Validation/ssim_resampled", ssim_resampled, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)

        self.log("Loss/Validation/mae_patch", mae_patch, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)
        self.log("Loss/Validation/cd_patch", cd_patch, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)
        self.log("Loss/Validation/pcc_patch", pcc_patch, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)
        self.log("Loss/Validation/ssim_patch", ssim_patch, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)

        if self.current_epoch in self.log_image_iters and batch_idx < self.number_of_images_to_log:
            self.log_images(input_resampled[:, 0:1, :, :, :], output_resampled, ground_truth_resampled, batch_idx, "Validation_Resampled")
            self.log_images(input_patch[:, 0:1, :, :, :], output_patch, ground_truth_patch, batch_idx, "Validation_Patch")
 
    def log_images(self, input_data, output, ground_truth, i, mode) -> None:
        initial_ssim = -of.structural_similarity_index_measure(input_data, ground_truth)
        output_ssim = -of.structural_similarity_index_measure(output, ground_truth)
        buf = vol.show_volumes_2d(input_data[0].to(tc.float32), ground_truth[0].to(tc.float32), output[0].to(tc.float32), spacing=(1, 1, 1), return_buffer=True, suptitle=f"Initial SSIM: {initial_ssim:.4f}, Output SSIM: {output_ssim:.4f}", names=["Input", "Ground-Truth", "Output"], dpi=100, show=False)
        image = PIL.Image.open(buf)
        image = tv.ToTensor()(image).unsqueeze(0)[0]
        title = f"{mode}. Case: {i}, Epoch: {str(self.current_epoch)}"
        self.logger.experiment.add_image(title, image, 0)
        plt.close('all')



class LightningDataModule(pl.LightningDataModule):
    def __init__(self,
                training_data,
                validation_data,
                training_transforms,
                validation_transforms,
                num_workers,
                batch_size):
        super().__init__()
        self.training_data = training_data
        self.validation_data = validation_data
        self.training_transforms = training_transforms
        self.validation_transforms = validation_transforms
        self.num_workers = num_workers
        self.batch_size = batch_size

    def prepare_data(self):
        pass

    def setup(self, stage):
        self.training_dataset = Dataset(data=self.training_data, transform=self.training_transforms)
        self.validation_dataset = Dataset(data=self.validation_data, transform=self.validation_transforms)

    def train_dataloader(self):
        return DataLoader(self.training_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=list_data_collate, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=1, num_workers=self.num_workers, collate_fn=list_data_collate)

class Trainer():
    def __init__(self, **training_params : dict):
        ### General params
        self.training_params = training_params
        self.lightning_params = training_params['lightning_params']
        self.checkpoints_path : Union[str, pathlib.Path] = training_params['checkpoints_path']
        self.to_load_checkpoint_path : Union[str, pathlib.Path, None] = training_params['to_load_checkpoint_path']
        if self.to_load_checkpoint_path is None:
            self.module = LightningModule(self.training_params, self.lightning_params)
        else:
            self.load_checkpoint()
    
        self.trainer = pl.Trainer(**self.lightning_params)
        self.data_module = LightningDataModule(
                training_params['training_data'],
                training_params['validation_data'],
                training_params['training_transforms'],
                training_params['validation_transforms'],
                training_params['num_workers'],
                training_params['batch_size'],
        )

    def save_checkpoint(self) -> None:
        self.trainer.save_checkpoint(pathlib.Path(self.checkpoints_path) / "Last_Iteration")

    def load_checkpoint(self) -> None:
        self.module = LightningModule.load_from_checkpoint(self.to_load_checkpoint_path, training_params=self.training_params, lightning_params=self.lightning_params) 

    def run(self) -> None:
        self.trainer.fit(self.module, self.data_module)
        self.save_checkpoint()