import csv
import enum
import json
import os

import numpy as np
from torch.utils.data import Dataset
import tqdm

from ..config import DatasetPathConfig, DatasetPreprocessingConfig
from .data_models import DataBatch, DataSample
from ..text import text_to_sequence

def load_sentiment(file):
    with open(file, newline='', encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)

        headers = next(reader)

        label_names = headers[1:]

        data_ids = []
        scores = []
        for row in reader:
            data_ids.append(row[0])
            scores.append([float(label) for label in row[1:]])

        labels = np.argmax(np.array(scores), axis=1)

    return data_ids, labels, label_names

class DatasetSplit(enum.Enum):
    TRAIN = "train"
    VAL = "val"

class OriginalDatasetWithSentiment(Dataset):
    def __init__(
        self, dataset_path_config: DatasetPathConfig, dataset_preprocessing_config: DatasetPreprocessingConfig, split: DatasetSplit, sort=False
    ):
        self.feature_dir = dataset_path_config.feature_dir
        self.text_cleaners = dataset_preprocessing_config.text_cleaners
        
        meta_file = None
        if split == DatasetSplit.TRAIN:
            meta_file = dataset_path_config.meta_file_train
        elif split == DatasetSplit.VAL:
            meta_file = dataset_path_config.meta_file_val
        else:
            raise ValueError(f"Unknown split: {split}")
        
        self.basename, self.speaker, self.text, self.raw_text = self.process_meta(
            meta_file
        )
        with open(dataset_path_config.speaker_map_file) as f:
            self.speaker_map = json.load(f)
        self.sort = sort

        # Load sentiment labels if provided
        if dataset_path_config.sentiment_file:
            sent_data_ids, sent_labels, label_names = load_sentiment(dataset_path_config.sentiment_file)
            self.sentiment_map = {data_id: sent_label for data_id, sent_label in zip(sent_data_ids, sent_labels)}
        else:
            self.sentiment_map = None

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        raw_text = self.raw_text[idx]
        phone = np.array(text_to_sequence(self.text[idx], self.text_cleaners))
        mel_path = os.path.join(
            self.feature_dir,
            "mel",
            f"{speaker}",
            f"{basename}.npy",
        )
        mel = np.load(mel_path)
        pitch_path = os.path.join(
            self.feature_dir,
            "pitch",
            f"{speaker}",
            f"{basename}.npy",
        )
        pitch = np.load(pitch_path)
        energy_path = os.path.join(
            self.feature_dir,
            "energy",
            f"{speaker}",
            f"{basename}.npy",
        )
        energy = np.load(energy_path)
        duration_path = os.path.join(
            self.feature_dir,
            "duration",
            f"{speaker}",
            f"{basename}.npy",
        )
        duration = np.load(duration_path)

        # check for nan or inf and fix them
        if np.isnan(mel).any() or np.isinf(mel).any():
            mel = np.nan_to_num(mel)
            tqdm.tqdm.write(f"Fixed nan values in mel for {basename}")
        if np.isnan(pitch).any() or np.isinf(pitch).any():
            pitch = np.nan_to_num(pitch)
            tqdm.tqdm.write(f"Fixed nan values in pitch for {basename}")
        if np.isnan(energy).any() or np.isinf(energy).any():
            energy = np.nan_to_num(energy)
            tqdm.tqdm.write(f"Fixed nan values in energy for {basename}")
        if np.isnan(duration).any() or np.isinf(duration).any():
            duration = np.nan_to_num(duration)
            tqdm.tqdm.write(f"Fixed nan values in duration for {basename}")

        sentiment_label = self.sentiment_map.get(basename, -1) if self.sentiment_map else None

        if sentiment_label == -1:
            raise ValueError(f"Sentiment label not found for {basename}")

        sample = DataSample(
            data_id=basename,
            speaker=speaker_id,
            text=phone,
            raw_text=raw_text,
            mel=mel,
            pitch=pitch,
            energy=energy,
            duration=duration,
            sentiment=sentiment_label
        )

        return sample

    def process_meta(self, filename):
        with open(
            filename, "r", encoding="utf-8"
        ) as f:
            name = []
            speaker = []
            text = []
            raw_text = []
            for line in f.readlines():
                n, s, t, r = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
                text.append(t)
                raw_text.append(r)
            return name, speaker, text, raw_text

    def collate_fn(self, data_samples):
        batch = DataBatch(data_samples, sort=self.sort)
        return batch


def load_meta(filename):
    with open(
        filename, "r", encoding="utf-8"
    ) as f:
        name = []
        speaker = []
        text = []
        raw_text = []
        for line in f.readlines():
            n, s, t, r = line.strip("\n").split("|")
            name.append(n)
            speaker.append(s)
            text.append(t)
            raw_text.append(r)
        return name, speaker, text, raw_text

class TextOnlyDatasetWithSentiment(Dataset):
    def __init__(
        self, dataset_path_config: DatasetPathConfig, dataset_preprocessing_config: DatasetPreprocessingConfig, split: DatasetSplit
    ):

        self.text_cleaners = dataset_preprocessing_config.text_cleaners

        meta_file = None
        if split == DatasetSplit.TRAIN:
            meta_file = dataset_path_config.meta_file_train
        elif split == DatasetSplit.VAL:
            meta_file = dataset_path_config.meta_file_val
        else:
            raise ValueError(f"Unknown split: {split}")

        # load metadata
        self.data_ids, self.speakers, self.texts, self.raw_texts = load_meta(meta_file)

        with open(dataset_path_config.speaker_map_file, mode="r", encoding="utf-8") as f:
            self.speaker_map = json.load(f)

        # Load sentiment labels if provided
        if dataset_path_config.sentiment_file:
            sent_data_ids, sent_labels, label_names = load_sentiment(dataset_path_config.sentiment_file)
            self.sentiment_map = {data_id: sent_label for data_id, sent_label in zip(sent_data_ids, sent_labels)}
        else:
            self.sentiment_map = None

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        speaker = self.speakers[idx]
        speaker_id = self.speaker_map[speaker]
        raw_text = self.raw_texts[idx]
        phone = np.array(text_to_sequence(self.texts[idx], self.text_cleaners))

        sentiment_label = self.sentiment_map.get(data_id, -1) if self.sentiment_map else None

        sample = DataSample(
            data_id=data_id,
            speaker=speaker_id,
            text=phone,
            raw_text=raw_text,
            mel=None,
            pitch=None,
            energy=None,
            duration=None,
            sentiment=sentiment_label
        )

        return sample

    def collate_fn(self, data):
        
        data_batch = DataBatch(data, sort=True)
        return data_batch

def main():
    
    from torch.utils.data import DataLoader

    dataset_path_config=DatasetPathConfig(
        base_dir="../../config/LibriTTS",
        meta_file_train="../data/LibriTTS/train.txt",
        meta_file_val="../data/LibriTTS/val.txt",
        speaker_map_file="../data/LibriTTS/speakers.json",
        feature_dir="../data/LibriTTS",
        stats_file="../data/LibriTTS/stats.json",
        sentiment_file="../data/LibriTTS/sentiment_scores.csv",
    )
    dataset_preprocessing_config=DatasetPreprocessingConfig(
        lexicon_path="../../lexicon/librispeech-lexicon.txt",
        text_cleaners=["english_cleaners"],
    )

    dataset = OriginalDatasetWithSentiment(
        dataset_path_config=dataset_path_config,
        dataset_preprocessing_config=dataset_preprocessing_config,
        split=DatasetSplit.VAL,
        sort=True
    )
    
    for i in range(3):
        sample = dataset[i]
        # print(sample)
        print(f"Speaker: {sample.speaker}, Text: {sample.text.shape}, Sentiment: {sample.sentiment}")
        print(f"Raw Text: {sample.raw_text}")
        print(f"Mel shape: {sample.mel.shape}, Pitch shape: {sample.pitch.shape}, Energy shape: {sample.energy.shape}, Duration shape: {sample.duration.shape}") # type: ignore
        print("-" * 50)


    dataloader = DataLoader(dataset, batch_size=4, collate_fn=dataset.collate_fn)
    for batch in dataloader:
        batch: DataBatch = batch
        # print(batch)
        print(f"Batch size: {batch.batch_size}, Text lens: {batch.text_lens}, Mel lens: {batch.mel_lens}")
        print(f"Speakers: {batch.speakers}, Texts: {batch.texts.shape}")
        print("-" * 50)
        for sample in batch.data_samples:
            # print(sample)
            print(f"Speaker: {sample.speaker}, Text: {sample.text.shape}, Sentiment: {sample.sentiment}")
            print(f"Raw Text: {sample.raw_text}")
            print(f"Mel shape: {sample.mel.shape}, Pitch shape: {sample.pitch.shape}, Energy shape: {sample.energy.shape}, Duration shape: {sample.duration.shape}") # type: ignore
            print("-" * 50)

        break

if __name__ == "__main__":
    main()
