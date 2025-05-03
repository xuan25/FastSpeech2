import argparse
import os

import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# from fastspeech2.dataset import Dataset
from fastspeech2.dataset.data_models import DataBatch, DataBatchTorch, DatasetFeatureStats
from fastspeech2.model.fastspeech2 import FastSpeech2, FastSpeech2Output
from fastspeech2.model.loss import FastSpeech2LossResult
from fastspeech2.model.optimizer import ScheduledOptim
from fastspeech2.utils.model import get_model, get_vocoder, get_param_num
from fastspeech2.utils.tools import log, synth_one_sample
from fastspeech2.model import FastSpeech2Loss
# from fastspeech2.dataset import Dataset
from dataset import DatasetSplit, OriginalDatasetWithSentiment

# from fastspeech2.evaluate import evaluate
from evaluate import evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args, configs):
    print("Prepare training ...")

    # preprocess_config, model_config, train_config = configs

    import test
    dataset_config, model_config, train_config = test.get_test_configs()
    restore_step = 0


    

    dataset_feature_stats = DatasetFeatureStats.from_json(
        dataset_config.path_config.stats_file,
        dataset_config.path_config.speaker_map_file,
    )

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
    model, optimizer = get_model(
        model_config,
        dataset_config.feature_properties_config,
        dataset_feature_stats,
        device,
        train_config.optimizer_config,
        True
    )
    model = nn.DataParallel(model)
    num_param = get_param_num(model)
    Loss = FastSpeech2Loss(
        dataset_config.feature_properties_config
    ).to(device)
    print("Number of FastSpeech2 Parameters:", num_param)

    # Load vocoder
    vocoder = get_vocoder(model_config.vocoder_config, device)

    # Init logger
    for p in [ train_config.path_config.ckpt_path, train_config.path_config.log_path, train_config.path_config.result_path ]:
        os.makedirs(p, exist_ok=True)
    train_log_path = os.path.join(train_config.path_config.log_path, "train")
    val_log_path = os.path.join(train_config.path_config.log_path, "val")
    os.makedirs(train_log_path, exist_ok=True)
    os.makedirs(val_log_path, exist_ok=True)
    train_logger = SummaryWriter(train_log_path)
    val_logger = SummaryWriter(val_log_path)

    # Training
    step = restore_step + 1
    epoch = 1
    grad_acc_step = train_config.optimizer_config.grad_acc_step
    grad_clip_thresh = train_config.optimizer_config.grad_clip_thresh
    total_step = train_config.step_config.total_step
    log_step = train_config.step_config.log_step
    save_step = train_config.step_config.save_step
    synth_step = train_config.step_config.synth_step
    val_step = train_config.step_config.val_step

    total_step_bar = tqdm(total=total_step, desc="Training", position=0)
    total_step_bar.n = restore_step
    total_step_bar.update()

    while True:
        epoch_bar = tqdm(total=len(loader), desc="Epoch {}".format(epoch), position=1)
        for batch in loader:

            batch: DataBatch = batch
            batch_torch: DataBatchTorch = batch.to_torch(device)

            # Forward
            model: FastSpeech2 = model  # temporary fix for type hinting
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
                    vocoder)
                with open(os.path.join(val_log_path, "log.txt"), "a") as f:
                    f.write(message + "\n")
                total_step_bar.write(message)

                model.train()

            if step % save_step == 0:
                torch.save(
                    {
                        "model": model.module.state_dict(),
                        "optimizer": optimizer._optimizer.state_dict(),
                    },
                    os.path.join(
                        train_config.path_config.ckpt_path,
                        "{}.pth.tar".format(step),
                    ),
                )

            if step == total_step:
                quit()
            step += 1
            total_step_bar.update(1)

            epoch_bar.update(1)
        epoch += 1


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--restore_step", type=int, default=0)
    # parser.add_argument(
    #     "-p",
    #     "--preprocess_config",
    #     type=str,
    #     # required=True,
    #     help="path to preprocess.yaml",
    #     default="config/LibriTTS/preprocess.yaml",
    # )
    # parser.add_argument(
    #     "-m", "--model_config", type=str, 
    #     # required=True, 
    #     help="path to model.yaml",
    #     default="config/LibriTTS/model.yaml",
    # )
    # parser.add_argument(
    #     "-t", "--train_config", type=str, 
    #     # required=True, 
    #     help="path to train.yaml",
    #     default="config/LibriTTS/train.yaml",
    # )
    # args = parser.parse_args()

    # # Read Config
    # preprocess_config = yaml.load(
    #     open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    # )
    # model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    # train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    # configs = (preprocess_config, model_config, train_config)

    # main(args, configs)
    main(None, None)
