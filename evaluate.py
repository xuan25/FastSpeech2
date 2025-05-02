import argparse
import os

import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader

from fastspeech2.dataset.data_models import DataBatch, DataBatchTorch
from fastspeech2.model.data_models import FastSpeech2LossResult
from fastspeech2.model.fastspeech2 import FastSpeech2Output
from fastspeech2.utils.model import get_model, get_vocoder
from fastspeech2.utils.tools import log, synth_one_sample
from fastspeech2.model import FastSpeech2Loss
# from dataset import Dataset
from dataset import OriginalDatasetWithSentiment



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model, step, configs, stats_file, logger=None, vocoder=None):
    preprocess_config, model_config, train_config = configs

    # Get dataset
    dataset = OriginalDatasetWithSentiment(
        meta_file="data/augmented_data/LibriTTS-original/val.txt",
        speaker_map_file="data/augmented_data/LibriTTS-original/speakers.json",
        feature_dir="data/augmented_data/LibriTTS-original",
        sentiment_file="data/original/LibriTTS/sentiment_scores_libri-tts.csv",
    )
    # dataset = SentimentBalancedDataset(
    #     meta_file="preprocessed_data/LibriTTS/val.txt",
    #     sentiment_path="sentiment/sentiment_scores_libri-tts.csv",
    #     feature_dir="preprocessed_data/LibriTTS",
    #     speaker_map_file="preprocessed_data/LibriTTS/speakers.json",
    #     text_cleaners=["english_cleaners"],
    # )
    batch_size = train_config["optimizer"]["batch_size"]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
        num_workers=8
    )

    # Get loss function
    Loss = FastSpeech2Loss(preprocess_config, model_config).to(device)

    # Evaluation

    loss_sums = FastSpeech2LossResult(
        total_loss=torch.tensor(0.0),
        mel_loss=torch.tensor(0.0),
        postnet_mel_loss=torch.tensor(0.0),
        pitch_loss=torch.tensor(0.0),
        energy_loss=torch.tensor(0.0),
        duration_loss=torch.tensor(0.0),
    )
    
    for batch in loader:
        batch: DataBatch = batch
        batch_torch: DataBatchTorch = batch.to_torch(device)
        with torch.no_grad():
            # Forward
            output: FastSpeech2Output = model(batch_torch)

            # Cal Loss
            losses: FastSpeech2LossResult = Loss(batch_torch, output)

            loss_sums.total_loss += losses.total_loss.item() * batch_torch.batch_size
            loss_sums.mel_loss += losses.mel_loss.item() * batch_torch.batch_size
            loss_sums.postnet_mel_loss += losses.postnet_mel_loss.item() * batch_torch.batch_size
            loss_sums.pitch_loss += losses.pitch_loss.item() * batch_torch.batch_size
            loss_sums.energy_loss += losses.energy_loss.item() * batch_torch.batch_size
            loss_sums.duration_loss += losses.duration_loss.item() * batch_torch.batch_size


    loss_means = FastSpeech2LossResult(
        total_loss=loss_sums.total_loss / len(dataset),
        mel_loss=loss_sums.mel_loss / len(dataset),
        postnet_mel_loss=loss_sums.postnet_mel_loss / len(dataset),
        pitch_loss=loss_sums.pitch_loss / len(dataset),
        energy_loss=loss_sums.energy_loss / len(dataset),
        duration_loss=loss_sums.duration_loss / len(dataset),
    )

    message = (
        "Validation Step {}, Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Pitch Loss: {:.4f}, Energy Loss: {:.4f}, Duration Loss: {:.4f}".format(
            step,
            loss_means.total_loss.item(),
            loss_means.mel_loss.item(),
            loss_means.postnet_mel_loss.item(),
            loss_means.pitch_loss.item(),
            loss_means.energy_loss.item(),
            loss_means.duration_loss.item(),
        )
    )

    if logger is not None:
        fig, wav_reconstruction, wav_prediction, tag = synth_one_sample(
            batch_torch,
            output,
            vocoder,
            model_config,
            stats_file,
            preprocess_config,
        )

        log(logger, step, losses=loss_means)
        log(
            logger,
            fig=fig,
            tag="Validation/step_{}_{}".format(step, tag),
        )
        sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
        log(
            logger,
            audio=wav_reconstruction,
            sampling_rate=sampling_rate,
            tag="Validation/step_{}_{}_reconstructed".format(step, tag),
        )
        log(
            logger,
            audio=wav_prediction,
            sampling_rate=sampling_rate,
            tag="Validation/step_{}_{}_synthesized".format(step, tag),
        )

    return message


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=30000)
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    )
    args = parser.parse_args()

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    # Get model
    model = get_model(args, configs, device, train=False).to(device)

    stats_file = os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json") # TODO: do not hardcode this
    message = evaluate(model, args.restore_step, configs, stats_file)
    print(message)