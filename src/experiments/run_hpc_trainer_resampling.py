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
from paths import hpc_paths as p
from training import training_resampling, training_patch
from experiments.hpc_experiments import experiments_resampled, experiments_patch_based
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
    best_mae_checkpoint = ModelCheckpoint(dirpath=checkpoints_path, filename='{epoch}_mae', save_top_k=1, mode='min', monitor="Loss/Validation/mae")
    best_cd_checkpoint = ModelCheckpoint(dirpath=checkpoints_path, filename='{epoch}_cd', save_top_k=1, mode='min', monitor="Loss/Validation/cd")
    best_pcc_checkpoint = ModelCheckpoint(dirpath=checkpoints_path, filename='{epoch}_pcc', save_top_k=1, mode='min', monitor="Loss/Validation/pcc")
    best_ssim_checkpoint = ModelCheckpoint(dirpath=checkpoints_path, filename='{epoch}_ssim', save_top_k=1, mode='max', monitor="Loss/Validation/ssim")
    logger = pl_loggers.TensorBoardLogger(save_dir=log_dir, name=experiment_name)
    training_params['lightning_params']['logger'] = logger
    training_params['lightning_params']['callbacks'] = [general_checkpoint, best_loss_checkpoint, best_mae_checkpoint, best_cd_checkpoint, best_pcc_checkpoint, best_ssim_checkpoint]   
    training_params['checkpoints_path'] = checkpoints_path
    training_params['log_image_iters'] = log_image_iters
    return training_params

def run_training(training_params, benchmark):
    training_params = initialize(training_params)
    if benchmark == "exp_resampled":
        trainer = training_resampling.Trainer(**training_params)
    elif benchmark == "exp_patch":
        trainer = training_patch.Trainer(**training_params)
    else:
        raise ValueError("Unsupported benchmark.")
    trainer.run()

def run_named_experiment(function_name, benchmark, fold=1):
    if benchmark == "exp_resampled":
        config = getattr(experiments_resampled, function_name)(fold)
    elif benchmark == "exp_patch":
        config = getattr(experiments_patch_based, function_name)(fold)
    else:
        raise ValueError("Unsupported benchmark.")
    run_training(config, benchmark)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', default=None)
    parser.add_argument('--benchmark', default=None)
    parser.add_argument('--fold', default=None)
    args = parser.parse_args()
    function_name = args.experiment
    print(f"Function name: {function_name}")
    benchmark = args.benchmark
    print(f"Benchmark: {benchmark}")
    fold = args.fold
    print(f"Fold: {fold}")
    run_named_experiment(function_name, benchmark, fold)