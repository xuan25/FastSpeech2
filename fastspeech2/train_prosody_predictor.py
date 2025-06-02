import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from .config import DatasetConfig, DatasetFeaturePropertiesConfig, ModelConfig, TrainConfig, TrainOptimizerConfig
from .dataset.data_models import DataBatch, DataBatchTorch, DatasetFeatureStats
from .model.prosody_predictor import ProsodyPredictor, ProsodyPredictorOutput
from .model.loss import ProsodyPredictorLossResult
from .model.optimizer import ScheduledOptim
from .utils.model import get_param_num
from .utils.tools import log_prosody_predictor
from .model.loss import ProsodyPredictorLoss
from .dataset.dataset import DatasetSplit, OriginalDatasetWithSentiment

from .evaluate_prosody_predictor import evaluate

def get_model_train(model_config: ModelConfig,
              dataset_feature_properties_config: DatasetFeaturePropertiesConfig,
              dataset_feature_stats: DatasetFeatureStats,
              device, 
              train_optimizer_config: TrainOptimizerConfig,
              ckpt_path: str|None = None,
              ) -> tuple[ProsodyPredictor, ScheduledOptim, int]:

    model = ProsodyPredictor(model_config, dataset_feature_properties_config, dataset_feature_stats).to(device)
    
    ckpt: dict = {}
    if ckpt_path:
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"])

    # init_lr = np.power(model_config.transformer_config.encoder_hidden, -0.5)
    scheduled_optim = ScheduledOptim(
        model.parameters(), train_optimizer_config, 0
    )
    if ckpt_path:
        scheduled_optim.load_state_dict(ckpt["optimizer"])
    model.train()

    training_steps = 0
    if ckpt_path:
        training_steps = ckpt["training_stats"]["steps"]

    return model, scheduled_optim, training_steps

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_ckpt", type=str, default=None)
    parser.add_argument(
        "-d",
        "--dataset_config",
        type=str,
        # required=True,
        help="path to dataset.yaml",
        default="config/LibriTTS/dataset.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, 
        # required=True, 
        help="path to model.yaml",
        default="config/LibriTTS/model.yaml",
    )
    parser.add_argument(
        "-t", "--train_config", type=str, 
        # required=True, 
        help="path to train.yaml",
        default="config/LibriTTS/train.yaml",
    )
    parser.add_argument(
        "-o", "--output_dir", type=str,
        # required=True,
        help="path to output directory",
        default="output/default",
    )
    args = parser.parse_args()

    restore_ckpt = args.restore_ckpt
    dataset_config = DatasetConfig.load_from_yaml(args.dataset_config)
    model_config = ModelConfig.load_from_yaml(args.model_config)
    train_config = TrainConfig.load_from_yaml(args.train_config)
    output_dir = args.output_dir
    ckpt_output_dir = os.path.join(output_dir, train_config.output_config.ckpt_dir_name)
    log_output_dir = os.path.join(output_dir, train_config.output_config.log_dir_name)

    dataset_feature_stats = DatasetFeatureStats.from_json(
        dataset_config.path_config.stats_file,
        dataset_config.path_config.speaker_map_file,
    )

    git_revision = None
    try:
        from .utils.git import get_git_revision_hash
        git_revision = get_git_revision_hash()
    except RuntimeError:
        print("Git repository not found. Skipping git revision logging.")

    # dump configs to output directory
    os.makedirs(output_dir, exist_ok=True)
    config = {
        "dataset": dataset_config.to_dict(),
        "model": model_config.to_dict(),
        "train": train_config.to_dict(),
        "git_revision": git_revision,
    }
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        f.write(json.dumps(config, indent=4))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    print("Prepare training ...")

    # Get dataset
    dataset = OriginalDatasetWithSentiment(
        dataset_path_config=dataset_config.path_config,
        dataset_preprocessing_config=dataset_config.preprocessing_config,
        split=DatasetSplit.TRAIN,
    )
    batch_size = train_config.step_config.batch_size
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
        num_workers=8
    )

    # Prepare model
    model_raw, optimizer, training_steps = get_model_train(
        model_config,
        dataset_config.feature_properties_config,
        dataset_feature_stats,
        device,
        train_config.optimizer_config,
        ckpt_path=restore_ckpt,
    )
    model = nn.DataParallel(model_raw)
    num_param = get_param_num(model)
    Loss = ProsodyPredictorLoss(
        dataset_config.feature_properties_config
    ).to(device)
    print("Number of Prosody Predictor Parameters:", num_param)

    # Init logger
    train_log_path = os.path.join(log_output_dir, "train")
    val_log_path = os.path.join(log_output_dir, "val")

    for p in [ ckpt_output_dir, train_log_path, val_log_path ]:
        os.makedirs(p, exist_ok=True)
    
    train_logger = SummaryWriter(train_log_path)
    val_logger = SummaryWriter(val_log_path)

    # Training
    step = training_steps + 1
    epoch = training_steps // len(loader) + 1
    grad_acc_step = train_config.optimizer_config.grad_acc_step
    grad_clip_thresh = train_config.optimizer_config.grad_clip_thresh
    total_step = train_config.step_config.total_step
    log_step = train_config.step_config.log_step
    save_step = train_config.step_config.save_step
    val_step = train_config.step_config.val_step

    total_step_bar = tqdm(total=total_step, desc="Training", position=0, dynamic_ncols=True)
    total_step_bar.n = training_steps
    total_step_bar.update()

    while True:
        epoch_bar = tqdm(total=len(loader), desc="Epoch {}".format(epoch), position=1, dynamic_ncols=True)
        for batch in loader:

            batch: DataBatch = batch
            batch_torch: DataBatchTorch = batch.to_torch(device)

            # Forward
            output: ProsodyPredictorOutput = model(batch_torch)

            # Cal Loss
            losses: ProsodyPredictorLossResult = Loss(batch_torch, output)
            total_loss = losses.total_loss

            # Backward
            total_loss: torch.Tensor = total_loss / grad_acc_step   # temporary fix for type hinting
            total_loss.backward()
            if step % grad_acc_step == 0:
                # Clipping gradients to avoid gradient explosion
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)

                # Update weights
                optimizer: ScheduledOptim = optimizer
                optimizer.step()
                optimizer.zero_grad()

            if step % log_step == 0:
                # losses = [l.item() for l in losses]
                message1 = "Step {}/{}, ".format(step, total_step)
                message2 = "Total Loss: {:.4f}, Pitch Loss: {:.4f}, Energy Loss: {:.4f}, Duration Loss: {:.4f}".format(
                    losses.total_loss.item(),
                    losses.pitch_loss.item(),
                    losses.energy_loss.item(),
                    losses.duration_loss.item(),
                )

                with open(os.path.join(train_log_path, "log.txt"), "a", encoding="utf-8") as f:
                    f.write(message1 + message2 + "\n")

                total_step_bar.write(message1 + message2)

                log_prosody_predictor(train_logger, step, losses=losses)

            if step % val_step == 0:
                model.eval()
                message = evaluate(
                    model,
                    step,
                    batch_size,
                    dataset_config,
                    val_logger,
                    device)
                with open(os.path.join(val_log_path, "log.txt"), "a", encoding="utf-8") as f:
                    f.write(message + "\n")
                total_step_bar.write(message)

                model.train()

            if step % save_step == 0:
                ckpt_payload = {
                    "model": model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "training_stats":
                    {
                        "steps": step,
                    },
                    "configs": {
                        "dataset_config": dataset_config,
                        "model_config": model_config,
                        "train_config": train_config,
                    },
                    "dataset_feature_stats": dataset_feature_stats,
                }
                    
                torch.save(
                    ckpt_payload,
                    os.path.join(
                        ckpt_output_dir,
                        "{}.pth".format(step),
                    )
                )

            if step == total_step:
                quit()
            step += 1
            total_step_bar.update(1)

            epoch_bar.update(1)
        epoch += 1


if __name__ == "__main__":
    
    main()
