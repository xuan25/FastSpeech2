import argparse
import csv
import os

import torch
import tqdm
from torch.utils.data import DataLoader

from fastspeech2.config import (
    DatasetConfig,
)
from fastspeech2.dataset.data_models import DataBatch, DataBatchTorch
from fastspeech2.dataset.dataset import DatasetSplit, DatasetWithLabel

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        # default="output/default/pred",
        default="output/prosody_predictor_gt/gt_MELD/pred",
        help="path to a source file with format like LibriTTS dataset",
    )
    parser.add_argument(
        "-d",
        "--dataset_config",
        type=str,
        # required=True, 
        # default="config/LibriTTS/dataset.yaml",
        default="config/LibriTTS/dataset_emotion.yaml",
        help="path to dataset.yaml",
    )
    parser.add_argument(
        "--data_split",
        type=str,
        default="val",
        choices=["train", "val"],
        help="data split to use for synthesis",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="batch size for synthesis",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    dataset_config = DatasetConfig.load_from_yaml(args.dataset_config)
    data_split_str = args.data_split
    data_split = DatasetSplit[data_split_str.upper()]

    batch_size = args.batch_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get dataset
    dataset = DatasetWithLabel(
        dataset_config.path_config,
        dataset_config.preprocessing_config,
        data_split
    )

    batchs = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=dataset.collate_fn,
    )

    os.makedirs(output_dir, exist_ok=True)

    # emotions = [
    #     "anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"
    # ]

    # pitches_neg = []
    # energies_neg = []
    # durations_neg = []

    # pitches_neu = []
    # energies_neu = []
    # durations_neu = []

    # pitches_pos = []
    # energies_pos = []
    # durations_pos = []

    with open(os.path.join(output_dir, f"0.csv"), "w", encoding="utf-8", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["data_id", "phone_idx", "phone", "pitch", "energy", "duration", "label"])

        for batch in tqdm.tqdm(batchs, desc="[Decoding]", dynamic_ncols=True):
            batch: DataBatch = batch
            batch_torch: DataBatchTorch = batch.to_torch(device)
            with torch.no_grad():

                for i, sample in enumerate(batch):
                    sample_id = sample.data_id
                    for j, phone in enumerate(sample.text):
                        assert batch_torch.pitches is not None, "Pitches should not be None"
                        assert batch_torch.energies is not None, "Energies should not be None"
                        assert batch_torch.durations is not None, "Durations should not be None"
                        pitch = batch_torch.pitches[i, j].item()
                        energy = batch_torch.energies[i, j].item()
                        duration = batch_torch.durations[i, j].item()

                        from fastspeech2.text.symbols import symbols
                        phone_alphabet = symbols[phone]
                        csv_writer.writerow([sample_id, j, phone_alphabet, pitch, energy, duration, sample.label])

if __name__ == "__main__":
    main()

    print("done!")
