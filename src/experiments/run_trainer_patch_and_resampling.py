### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import argparse
import pathlib

### External Imports ###
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import ModelCheckpoint

### Internal Imports ###
from paths import pc_paths as p
from training import training_patch_and_resampling
from experiments.pc_experiments import experiments as exp
########################


def initialize(training_params):
    experiment_name = training_params['experiment_name']
    num_iterations = training_params['num_iterations']
    save_step = training_params['save_step']
    checkpoints_path = os.path.join(p.checkpoints_path, experiment_name)
    log_image_iters = list(range(0, num_iterations, save_step))
    pathlib.Path(checkpoints_path).mkdir(parents=True, exist_ok=True)
    log_dir = os.path.join(p.logs_path, experiment_name)
    pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
    general_checkpoint = ModelCheckpoint(dirpath=checkpoints_path, filename='{epoch}_general', every_n_epochs=save_step, save_top_k=-1)
    best_loss_checkpoint = ModelCheckpoint(dirpath=checkpoints_path, filename='{epoch}_loss', save_top_k=1, mode='min', monitor="Loss/Validation/loss")
    best_loss_checkpoint_resampled = ModelCheckpoint(dirpath=checkpoints_path, filename='{epoch}_loss_resampled', save_top_k=1, mode='min', monitor="Loss/Validation/loss_resampled")
    best_loss_checkpoint_patch = ModelCheckpoint(dirpath=checkpoints_path, filename='{epoch}_loss_patch', save_top_k=1, mode='min', monitor="Loss/Validation/loss_patch")

    best_mae_checkpoint_resampled = ModelCheckpoint(dirpath=checkpoints_path, filename='{epoch}_mae_resampled', save_top_k=1, mode='min', monitor="Loss/Validation/mae_resampled")
    best_cd_checkpoint_resampled = ModelCheckpoint(dirpath=checkpoints_path, filename='{epoch}_cd_resampled', save_top_k=1, mode='min', monitor="Loss/Validation/cd_resampled")
    best_pcc_checkpoint_resampled = ModelCheckpoint(dirpath=checkpoints_path, filename='{epoch}_pcc_resampled', save_top_k=1, mode='min', monitor="Loss/Validation/pcc_resampled")
    best_ssim_checkpoint_resampled = ModelCheckpoint(dirpath=checkpoints_path, filename='{epoch}_ssim_resampled', save_top_k=1, mode='max', monitor="Loss/Validation/ssim_resampled")

    best_mae_checkpoint_patch = ModelCheckpoint(dirpath=checkpoints_path, filename='{epoch}_mae_patch', save_top_k=1, mode='min', monitor="Loss/Validation/mae_patch")
    best_cd_checkpoint_patch = ModelCheckpoint(dirpath=checkpoints_path, filename='{epoch}_cd_patch', save_top_k=1, mode='min', monitor="Loss/Validation/cd_patch")
    best_pcc_checkpoint_patch = ModelCheckpoint(dirpath=checkpoints_path, filename='{epoch}_pcc_patch', save_top_k=1, mode='min', monitor="Loss/Validation/pcc_patch")
    best_ssim_checkpoint_patch = ModelCheckpoint(dirpath=checkpoints_path, filename='{epoch}_ssim_patch', save_top_k=1, mode='max', monitor="Loss/Validation/ssim_patch")

    logger = pl_loggers.TensorBoardLogger(save_dir=log_dir, name=experiment_name)
    training_params['lightning_params']['logger'] = logger
    training_params['lightning_params']['callbacks'] = [general_checkpoint, best_loss_checkpoint, best_loss_checkpoint_resampled, best_loss_checkpoint_patch,
                                                         best_mae_checkpoint_resampled, best_cd_checkpoint_resampled, best_pcc_checkpoint_resampled, best_ssim_checkpoint_resampled,
                                                         best_mae_checkpoint_patch, best_cd_checkpoint_patch, best_pcc_checkpoint_patch, best_ssim_checkpoint_patch]  
    training_params['checkpoints_path'] = checkpoints_path
    training_params['log_image_iters'] = log_image_iters
    return training_params

def run_training(training_params):
    training_params = initialize(training_params)
    trainer = training_patch_and_resampling.Trainer(**training_params)
    trainer.run()


if __name__ == "__main__":
    pass