import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from .config import DatasetConfig, ModelConfig, TrainConfig
from .dataset.data_models import DataBatch, DataBatchTorch, DatasetFeatureStats
from .model.fastspeech2 import FastSpeech2, FastSpeech2Output
from .model.loss import FastSpeech2LossResult
from .model.optimizer import ScheduledOptim
from .utils.model import get_model_train, get_vocoder, get_param_num
from .utils.tools import log, synth_one_sample
from .model import FastSpeech2Loss
from .dataset.dataset import DatasetSplit, OriginalDatasetWithSentiment

from .evaluate import evaluate

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
    Loss = FastSpeech2Loss(
        dataset_config.feature_properties_config
    ).to(device)
    print("Number of FastSpeech2 Parameters:", num_param)

    # Load vocoder
    vocoder = get_vocoder(model_config.vocoder_config, device)

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
    synth_step = train_config.step_config.synth_step
    val_step = train_config.step_config.val_step

    total_step_bar = tqdm(total=total_step, desc="Training", position=0)
    total_step_bar.n = training_steps
    total_step_bar.update()

    while True:
        epoch_bar = tqdm(total=len(loader), desc="Epoch {}".format(epoch), position=1)
        for batch in loader:

            batch: DataBatch = batch
            batch_torch: DataBatchTorch = batch.to_torch(device)

            # Forward
            output: FastSpeech2Output = model(batch_torch)

            # Cal Loss
            losses: FastSpeech2LossResult = Loss(batch_torch, output)
            total_loss = losses.total_loss

            # Backward
            total_loss: torch.Tensor = total_loss / grad_acc_step   # temporary fix for type hinting
            total_loss.backward()
            if step % grad_acc_step == 0:
                # Clipping gradients to avoid gradient explosion
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)

                # Update weights
                optimizer: ScheduledOptim = optimizer
                optimizer.step_and_update_lr()
                optimizer.zero_grad()

            if step % log_step == 0:
                # losses = [l.item() for l in losses]
                message1 = "Step {}/{}, ".format(step, total_step)
                message2 = "Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Pitch Loss: {:.4f}, Energy Loss: {:.4f}, Duration Loss: {:.4f}".format(
                    losses.total_loss.item(),
                    losses.mel_loss.item(),
                    losses.postnet_mel_loss.item(),
                    losses.pitch_loss.item(),
                    losses.energy_loss.item(),
                    losses.duration_loss.item(),
                )

                with open(os.path.join(train_log_path, "log.txt"), "a") as f:
                    f.write(message1 + message2 + "\n")

                total_step_bar.write(message1 + message2)

                log(train_logger, step, losses=losses)

            if step % synth_step == 0:
                fig, wav_reconstruction, wav_prediction, tag = synth_one_sample(
                    batch_torch,
                    output,
                    vocoder,
                    model_config.vocoder_config,
                    dataset_feature_stats,
                    dataset_config.feature_properties_config,
                )
                log(
                    train_logger,
                    fig=fig,
                    tag="Training/step_{}_{}".format(step, tag),
                )
                sampling_rate = dataset_config.feature_properties_config.sampling_rate
                log(
                    train_logger,
                    audio=wav_reconstruction,
                    sampling_rate=sampling_rate,
                    tag="Training/step_{}_{}_reconstructed".format(step, tag),
                )
                log(
                    train_logger,
                    audio=wav_prediction,
                    sampling_rate=sampling_rate,
                    tag="Training/step_{}_{}_synthesized".format(step, tag),
                )

            if step % val_step == 0:
                model.eval()
                message = evaluate(
                    model,
                    step,
                    batch_size,
                    dataset_config,
                    model_config.vocoder_config,
                    dataset_feature_stats,
                    dataset_config.feature_properties_config,
                    val_logger,
                    vocoder,device)
                with open(os.path.join(val_log_path, "log.txt"), "a") as f:
                    f.write(message + "\n")
                total_step_bar.write(message)

                model.train()

            if step % save_step == 0:
                torch.save(
                    {
                        "model": model.module.state_dict(),
                        "optimizer": optimizer._optimizer.state_dict(),
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
                    },
                    os.path.join(
                        ckpt_output_dir,
                        "{}.pth".format(step),
                    ),
                )

            if step == total_step:
                quit()
            step += 1
            total_step_bar.update(1)

            epoch_bar.update(1)
        epoch += 1


if __name__ == "__main__":
    
    main()
