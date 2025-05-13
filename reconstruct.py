import argparse
import os

import torch
import tqdm
from torch.utils.data import DataLoader

from fastspeech2.model.data_models import FastSpeech2Output

from fastspeech2.config import (DatasetConfig, DatasetFeaturePropertiesConfig,
                     ModelConfig, ModelVocoderConfig)
from fastspeech2.dataset.data_models import DataBatch, DataBatchTorch, DatasetFeatureStats
from fastspeech2.dataset.dataset import DatasetSplit, OriginalDatasetWithSentiment, TextOnlyDatasetWithSentiment
from fastspeech2.utils.model import get_model_infer, get_vocoder
from fastspeech2.utils.tools import get_mask_from_lengths, synth_samples

def synthesize(vocoder, vocoder_config: ModelVocoderConfig, feature_properties_config: DatasetFeaturePropertiesConfig, batchs, control_values, device, stats: DatasetFeatureStats, output_dir):
    pitch_control, energy_control, duration_control = control_values

    for batch in tqdm.tqdm(batchs, desc="[Decoding]", dynamic_ncols=True):
        batch: DataBatch = batch
        batch_torch: DataBatchTorch = batch.to_torch(device)

        assert batch_torch.mels is not None, "Mel spectrograms are None"
        assert batch_torch.pitches is not None, "Pitch predictions are None"
        assert batch_torch.energies is not None, "Energy predictions are None"
        assert batch_torch.durations is not None, "Duration predictions are None"
        assert batch_torch.mel_lens is not None, "Text lengths are None"


        text_masks = get_mask_from_lengths(batch_torch.text_lens, batch_torch.text_len_max)
        mel_masks = get_mask_from_lengths(batch_torch.mel_lens, batch_torch.mel_len_max)
        
        with torch.no_grad():
            # Forward
            output = FastSpeech2Output(
                output=batch_torch.mels,
                postnet_output=batch_torch.mels,
                pitch_predictions=batch_torch.pitches,
                energy_predictions=batch_torch.energies,
                log_duration_predictions=torch.log(batch_torch.durations),
                duration_rounded=batch_torch.durations,
                text_masks=text_masks,
                mel_masks=mel_masks,
                text_lens=batch_torch.text_lens,
                mel_lens=batch_torch.mel_lens,
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
        "--output_dir",
        type=str,
        default="output/reconstruct/synth_val",
        help="output directory for synthesized samples",
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
        default="config/LibriTTS/model.yaml",
        help="path to model.yaml",
    )
    parser.add_argument(
        "--data_split",
        type=str,
        default="val",
        choices=["train", "val"],
        help="data split to use for synthesis",
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
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="batch size for synthesis",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    control_values = args.pitch_control, args.energy_control, args.duration_control
    dataset_config = DatasetConfig.load_from_yaml(args.dataset_config)
    model_config = ModelConfig.load_from_yaml(args.model_config)
    data_split_str = args.data_split
    data_split = DatasetSplit[data_split_str.upper()]

    batch_size = args.batch_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_feature_stats = DatasetFeatureStats.from_json(
        dataset_config.path_config.stats_file,
        dataset_config.path_config.speaker_map_file
    )

    # Load vocoder
    vocoder = get_vocoder(model_config.vocoder_config, device)

    # Get dataset
    dataset = OriginalDatasetWithSentiment(
        dataset_config.path_config,
        dataset_config.preprocessing_config,
        data_split
    )

    batchs = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=16,
        collate_fn=dataset.collate_fn,
    )

    os.makedirs(output_dir, exist_ok=True)
    synthesize(vocoder, model_config.vocoder_config, dataset_config.feature_properties_config, batchs, control_values, device, dataset_feature_stats, output_dir)

if __name__ == "__main__":
    main()

    print("done!")
