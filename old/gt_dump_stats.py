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
from fastspeech2.dataset.dataset import DatasetSplit, OriginalDatasetWithSentiment

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/default/pred",
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
    dataset = OriginalDatasetWithSentiment(
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

    pitches_neg = []
    energies_neg = []
    durations_neg = []

    pitches_neu = []
    energies_neu = []
    durations_neu = []

    pitches_pos = []
    energies_pos = []
    durations_pos = []

    with open(os.path.join(output_dir, f"pred_{data_split_str}.csv"), "w", encoding="utf-8", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["data_id", "phone_idx", "phone", "pitch", "energy", "duration", "sentiment"])

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

                        if sample.sentiment == 0:
                            pitches_neg.append(pitch)
                            energies_neg.append(energy)
                            durations_neg.append(duration)
                        elif sample.sentiment == 1:
                            pitches_neu.append(pitch)
                            energies_neu.append(energy)
                            durations_neu.append(duration)
                        elif sample.sentiment == 2:
                            pitches_pos.append(pitch)
                            energies_pos.append(energy)
                            durations_pos.append(duration)

                        from fastspeech2.text.symbols import symbols
                        phone_alphabet = symbols[phone]
                        csv_writer.writerow([sample_id, j, phone_alphabet, pitch, energy, duration, sample.sentiment])
        
        pitch_mean_neg = sum(pitches_neg) / len(pitches_neg)
        energy_mean_neg = sum(energies_neg) / len(energies_neg)
        duration_mean_neg = sum(durations_neg) / len(durations_neg)   

        pitch_mean_neu = sum(pitches_neu) / len(pitches_neu)
        energy_mean_neu = sum(energies_neu) / len(energies_neu)
        duration_mean_neu = sum(durations_neu) / len(durations_neu)

        pitch_mean_pos = sum(pitches_pos) / len(pitches_pos)
        energy_mean_pos = sum(energies_pos) / len(energies_pos)
        duration_mean_pos = sum(durations_pos) / len(durations_pos)

        pitch_all = pitches_neg + pitches_neu + pitches_pos
        energy_all = energies_neg + energies_neu + energies_pos
        duration_all = durations_neg + durations_neu + durations_pos

        pitch_mean_all = sum(pitch_all) / len(pitch_all)
        energy_mean_all = sum(energy_all) / len(energy_all)
        duration_mean_all = sum(duration_all) / len(duration_all)

        with open(os.path.join(output_dir, f"stats_{data_split_str}.csv"), "w", encoding="utf-8", newline="") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(["pitch_mean_all", "pitch_mean_neg", "pitch_mean_neu", "pitch_mean_pos",
                                 "energy_mean_all", "energy_mean_neg", "energy_mean_neu", "energy_mean_pos",
                                 "duration_mean_all", "duration_mean_neg", "duration_mean_neu", "duration_mean_pos"])
            csv_writer.writerow([pitch_mean_all, pitch_mean_neg, pitch_mean_neu, pitch_mean_pos,
                                 energy_mean_all, energy_mean_neg, energy_mean_neu, energy_mean_pos,
                                 duration_mean_all, duration_mean_neg, duration_mean_neu, duration_mean_pos])

if __name__ == "__main__":
    main()

    print("done!")
