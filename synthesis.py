import argparse
import json
import os
from string import punctuation

import numpy as np
import torch
from torch.utils.data import Dataset
import tqdm
import yaml

import re

from fastspeech2.dataset.data_models import DataBatch, DataBatchTorch
from fastspeech2.model import FastSpeech2
from fastspeech2.model.optimizer import ScheduledOptim
from fastspeech2.text import text_to_sequence
from g2p_en import G2p

from fastspeech2.utils.model import get_vocoder

from torch.utils.data import DataLoader

from fastspeech2.utils.tools import synth_samples
from dataset import TextOnlyDatasetWithSentiment


def get_model(ckpt_path, configs, device):
    (preprocess_config, model_config, train_config) = configs

    model = FastSpeech2(preprocess_config, model_config).to(device)
    if ckpt_path:
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"])

    model.eval()
    model.requires_grad_ = False
    return model

def synthesize(model, configs, vocoder, batchs, control_values, device, stats_file, output_dir):
    preprocess_config, model_config, train_config = configs
    pitch_control, energy_control, duration_control = control_values

    for batch in tqdm.tqdm(batchs, desc="[Decoding]", dynamic_ncols=True):
        batch: DataBatch = batch
        batch_torch: DataBatchTorch = batch.to_torch(device)
        with torch.no_grad():
            # Forward
            output = model(
                batch_torch,
                p_control=pitch_control,
                e_control=energy_control,
                d_control=duration_control
            )
            synth_samples(
                batch,
                output,
                vocoder,
                model_config,
                preprocess_config,
                stats_file,
                output_dir,
            )

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_path", 
        type=str, 
        # default="output_archive/onehot_encoder/ckpt/LibriTTS/30000.pth.tar"
        # default="output_archive/original/ckpt/LibriTTS/30000.pth.tar"
        default="output/ckpt/LibriTTS/4000.pth.tar"
    )
    parser.add_argument(
        "--meta_file",
        type=str,
        default="data/augmented_data/LibriTTS-original/val.txt",
        help="path to a source file with format like LibriTTS dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        # default="synth_output_val/onehot_encoder",
        # default="synth_output_val/original",
        default="output/synth_val",
        help="path to a source file with format like LibriTTS dataset",
    )
    parser.add_argument(
        "--speaker_config",
        type=str,
        # required=True,
        default="data/augmented_data/LibriTTS-original/speakers.json",
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        # required=True,
        default="config/LibriTTS/preprocess.yaml",
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", 
        type=str, 
        # required=True, 
        help="path to model.yaml",
        default="config/LibriTTS/model.yaml",
    )
    parser.add_argument(
        "-t", "--train_config", 
        type=str, 
        # required=True, 
        help="path to train.yaml",
        default="config/LibriTTS/train.yaml",
    )
    parser.add_argument(
        "--pitch_control",
        type=float,
        default=1.0,
        help="control the pitch of the whole utterance, larger value for higher pitch",
    )
    parser.add_argument(
        "--energy_control",
        type=float,
        default=1.0,
        help="control the energy of the whole utterance, larger value for larger volume",
    )
    parser.add_argument(
        "--duration_control",
        type=float,
        default=1.0,
        help="control the speed of the whole utterance, larger value for slower speaking rate",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    # Get model
    model = get_model(args.ckpt_path, configs, device)

    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    # Get dataset
    dataset = TextOnlyDatasetWithSentiment(
        meta_file=args.meta_file,
        speaker_map_file=args.speaker_config,
        sentiment_file="data/original/LibriTTS/sentiment_scores_libri-tts.csv"
        )

    batchs = DataLoader(
        dataset,
        batch_size=32,
        num_workers=16,
        collate_fn=dataset.collate_fn,
    )

    control_values = args.pitch_control, args.energy_control, args.duration_control

    stats_file = os.path.join(
        preprocess_config["path"]["preprocessed_path"], "stats.json"
    )   # TODO: do not hardcode this

    output_dir: str = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    synthesize(model,configs, vocoder, batchs, control_values, device, stats_file, output_dir)

if __name__ == "__main__":
    main()

    print("done!")
