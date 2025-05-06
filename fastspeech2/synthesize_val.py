import argparse
import os

import torch
import tqdm
from torch.utils.data import DataLoader

from .config import (DatasetConfig, DatasetFeaturePropertiesConfig,
                     ModelConfig, ModelVocoderConfig)
from .dataset.data_models import DataBatch, DataBatchTorch, DatasetFeatureStats
from .dataset.dataset import DatasetSplit, TextOnlyDatasetWithSentiment
from .utils.model import get_model_infer, get_vocoder
from .utils.tools import synth_samples

def synthesize(model, vocoder, vocoder_config: ModelVocoderConfig, feature_properties_config: DatasetFeaturePropertiesConfig, batchs, control_values, device, stats: DatasetFeatureStats, output_dir):
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
                batch_torch,
                output,
                vocoder,
                vocoder_config,
                feature_properties_config,
                stats,
                output_dir,
            )

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_path", 
        type=str, 
        default="output/ckpt/LibriTTS/4000.pth.tar"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/synth_val",
        help="path to a source file with format like LibriTTS dataset",
    )
    parser.add_argument(
        "-d",
        "--dataset_config",
        type=str,
        # required=True, 
        default="config/LibriTTS/dataset.yaml",
        help="path to dataset.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", 
        type=str, 
        # required=True, 
        help="path to model.yaml",
        default="config/LibriTTS/model.yaml",
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

    ckpt_path = args.ckpt_path
    output_dir = args.output_dir
    control_values = args.pitch_control, args.energy_control, args.duration_control
    dataset_config = DatasetConfig.load_from_yaml(args.dataset_config)
    model_config = ModelConfig.load_from_yaml(args.model_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_feature_stats = DatasetFeatureStats.from_json(
        dataset_config.path_config.stats_file,
        dataset_config.path_config.speaker_map_file
    )

    # Get model
    model = get_model_infer(ckpt_path, model_config, dataset_config.feature_properties_config, dataset_feature_stats, device)

    # Load vocoder
    vocoder = get_vocoder(model_config.vocoder_config, device)

    # Get dataset
    dataset = TextOnlyDatasetWithSentiment(
        dataset_config.path_config,
        dataset_config.preprocessing_config,
        DatasetSplit.VAL
    )

    batchs = DataLoader(
        dataset,
        batch_size=32,
        num_workers=16,
        collate_fn=dataset.collate_fn,
    )

    output_dir = "output/synth_val"

    os.makedirs(output_dir, exist_ok=True)
    synthesize(model, vocoder, model_config.vocoder_config, dataset_config.feature_properties_config, batchs, control_values, device, dataset_feature_stats, output_dir)

if __name__ == "__main__":
    main()

    print("done!")
