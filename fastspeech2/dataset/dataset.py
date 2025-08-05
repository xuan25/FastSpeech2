import csv
import enum
from functools import lru_cache
import io
import json
import os
from typing import IO

import numpy as np
from torch.utils.data import Dataset
import tqdm

from .datasetfs import DatasetFS

from ..config import DatasetPathConfig, DatasetPreprocessingConfig
from .data_models import DataBatch, DataSample
from ..text import text_to_sequence

def load_sentiment(file_stream: IO[bytes]) -> tuple[list[str], list[int], list[str]]:
    with io.TextIOWrapper(file_stream, encoding='utf-8') as csv_stream:
        reader = csv.reader(csv_stream)

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




class DatasetWithSentimentContrastive(Dataset):
    def __init__(
        self, dataset_path_config: DatasetPathConfig, dataset_preprocessing_config: DatasetPreprocessingConfig, split: DatasetSplit, sort=False
    ):
        
        dataset_fs = self.get_dataset_fs(dataset_path_config.base_dir)
        self.base_dir = dataset_path_config.base_dir

        self.feature_dir = dataset_path_config.feature_dir
        self.text_cleaners = dataset_preprocessing_config.text_cleaners
        
        meta_file = None
        if split == DatasetSplit.TRAIN:
            meta_file = dataset_path_config.meta_file_train
        elif split == DatasetSplit.VAL:
            meta_file = dataset_path_config.meta_file_val
        else:
            raise ValueError(f"Unknown split: {split}")
        
        with dataset_fs.open(meta_file) as f:
            self.basename, self.speaker, self.text, self.raw_text = self.process_meta(
                f
            )
        with dataset_fs.open(dataset_path_config.speaker_map_file) as f:
            self.speaker_map = json.load(f)
        self.sort = sort

        # Load sentiment labels if provided
        if dataset_path_config.sentiment_file:
            with dataset_fs.open(dataset_path_config.sentiment_file) as f:
                sent_data_ids, sent_labels, label_names = load_sentiment(f)
            self.sentiment_map = {data_id: sent_label for data_id, sent_label in zip(sent_data_ids, sent_labels)}
        else:
            self.sentiment_map = None

        self.sentiment_map_reverse: dict[int, list[int]] = {}
        if self.sentiment_map:
            for idx in range(len(self.basename)):
                sentiment_label = self.sentiment_map[self.basename[idx]]
                if sentiment_label not in self.sentiment_map_reverse:
                    self.sentiment_map_reverse[sentiment_label] = []
                self.sentiment_map_reverse[sentiment_label].append(idx)

    @lru_cache(maxsize=None)
    def get_dataset_fs(self, base_path):
        return DatasetFS(base_path)

    def __len__(self):
        return len(self.text)

    def __load_data_sample(self, idx: int) -> DataSample:
        dataset_fs = self.get_dataset_fs(self.base_dir)

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
        with dataset_fs.open(mel_path) as f:
            with io.BytesIO(f.read()) as buffer:
                mel = np.load(buffer)
        pitch_path = os.path.join(
            self.feature_dir,
            "pitch",
            f"{speaker}",
            f"{basename}.npy",
        )
        with dataset_fs.open(pitch_path) as f:
            with io.BytesIO(f.read()) as buffer:
                pitch = np.load(buffer)
        energy_path = os.path.join(
            self.feature_dir,
            "energy",
            f"{speaker}",
            f"{basename}.npy",
        )
        with dataset_fs.open(energy_path) as f:
            with io.BytesIO(f.read()) as buffer:
                energy = np.load(buffer)
        duration_path = os.path.join(
            self.feature_dir,
            "duration",
            f"{speaker}",
            f"{basename}.npy",
        )
        with dataset_fs.open(duration_path) as f:
            with io.BytesIO(f.read()) as buffer:
                duration = np.load(buffer)

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
    
    def __load_data_sample_text(self, idx: int) -> DataSample:
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        raw_text = self.raw_text[idx]
        phone = np.array(text_to_sequence(self.text[idx], self.text_cleaners))

        sentiment_label = self.sentiment_map.get(basename, -1) if self.sentiment_map else None

        if sentiment_label == -1:
            raise ValueError(f"Sentiment label not found for {basename}")

        sample = DataSample(
            data_id=basename,
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
    
    def __getitem__(self, idx) -> tuple[list[DataSample], np.ndarray]:

        sample: DataSample = self.__load_data_sample(idx)

        sample_neg1 = DataSample(
            data_id=sample.data_id,
            speaker=sample.speaker,
            text=sample.text,
            raw_text=sample.raw_text,
            mel=sample.mel,
            pitch=sample.pitch,
            energy=sample.energy,
            duration=sample.duration,
            sentiment=(sample.sentiment + 1) % 3 if sample.sentiment is not None else None # simple negative sample generation by adding 1 to sentiment label
        )

        sample_neg2 = DataSample(
            data_id=sample.data_id,
            speaker=sample.speaker,
            text=sample.text,
            raw_text=sample.raw_text,
            mel=sample.mel,
            pitch=sample.pitch,
            energy=sample.energy,
            duration=sample.duration,
            sentiment=(sample.sentiment + 2) % 3 if sample.sentiment is not None else None # simple negative sample generation by adding 2 to sentiment label
        )

        # sample 2 positive samples form the same sentiment class
        assert sample.sentiment is not None, "Sample sentiment must not be None for contrastive learning"
        sentiment_samples = self.sentiment_map_reverse[sample.sentiment]
        if len(sentiment_samples) < 3:
            raise ValueError(f"Not enough samples for sentiment {sample.sentiment} to create contrastive samples")
        
        sample_pos1 = self.__load_data_sample_text(np.random.choice(sentiment_samples))
        text_length_pos1 = min(sample_pos1.text.shape[0], sample.text.shape[0])
        sample_pos1.text = sample.text[:text_length_pos1]  # ensure same length for contrastive learning
        sample_pos1.duration = sample.duration[:text_length_pos1] if sample.duration is not None else None
        sample_pos1.pitch = sample.pitch[:text_length_pos1] if sample.pitch is not None else None
        sample_pos1.energy = sample.energy[:text_length_pos1] if sample.energy is not None else None
        mel_length_pos1 = np.sum(sample_pos1.duration) if sample_pos1.duration is not None else 0
        sample_pos1.mel = sample.mel[:mel_length_pos1] if sample.mel is not None else None

        sample_pos2 = self.__load_data_sample(np.random.choice(sentiment_samples))
        text_length_pos2 = min(sample_pos2.text.shape[0], sample.text.shape[0])
        sample_pos2.text = sample.text[:text_length_pos2]
        sample_pos2.duration = sample.duration[:text_length_pos2] if sample.duration is not None else None
        sample_pos2.pitch = sample.pitch[:text_length_pos2] if sample.pitch is not None else None
        sample_pos2.energy = sample.energy[:text_length_pos2] if sample.energy is not None else None
        mel_length_pos2 = np.sum(sample_pos2.duration) if sample_pos2.duration is not None else 0
        sample_pos2.mel = sample.mel[:mel_length_pos2] if sample.mel is not None else None

        # the first sample is the original sample, the second and third sample is negative contrastive samples
        mask = np.array([0, -1, -1, 1, 1], dtype=np.int8)  # 0 for original, -1 for negative samples

        return ([sample, sample_neg1, sample_neg2, sample_pos1, sample_pos2], mask)

    def process_meta(self, file_stream: IO[bytes]) -> tuple[list[str], list[str], list[str], list[str]]:
        with io.TextIOWrapper(file_stream, encoding="utf-8") as f:
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

    def collate_fn(self, data_samples_with_mask_batch: list[tuple[list[DataSample], np.ndarray]]):

        data_samples = []
        mask_arrays = []
        for data_samples_partial, masks_partial in data_samples_with_mask_batch:
            data_samples.extend(data_samples_partial)
            mask_arrays.append(masks_partial)

        batch = DataBatch(data_samples, sort=self.sort)

        mask = np.concatenate(mask_arrays, axis=0)
        mask = mask[batch.sample_idxs]  # reorder mask according to sample_idxs

        return batch, mask


class OriginalDatasetWithSentiment(Dataset):
    def __init__(
        self, dataset_path_config: DatasetPathConfig, dataset_preprocessing_config: DatasetPreprocessingConfig, split: DatasetSplit, sort=False
    ):
        
        dataset_fs = self.get_dataset_fs(dataset_path_config.base_dir)
        self.base_dir = dataset_path_config.base_dir

        self.feature_dir = dataset_path_config.feature_dir
        self.text_cleaners = dataset_preprocessing_config.text_cleaners
        
        meta_file = None
        if split == DatasetSplit.TRAIN:
            meta_file = dataset_path_config.meta_file_train
        elif split == DatasetSplit.VAL:
            meta_file = dataset_path_config.meta_file_val
        else:
            raise ValueError(f"Unknown split: {split}")
        
        with dataset_fs.open(meta_file) as f:
            self.basename, self.speaker, self.text, self.raw_text = self.process_meta(
                f
            )
        with dataset_fs.open(dataset_path_config.speaker_map_file) as f:
            self.speaker_map = json.load(f)
        self.sort = sort

        # Load sentiment labels if provided
        if dataset_path_config.sentiment_file:
            with dataset_fs.open(dataset_path_config.sentiment_file) as f:
                sent_data_ids, sent_labels, label_names = load_sentiment(f)
            self.sentiment_map = {data_id: sent_label for data_id, sent_label in zip(sent_data_ids, sent_labels)}
        else:
            self.sentiment_map = None

        
    @lru_cache(maxsize=None)
    def get_dataset_fs(self, base_path):
        return DatasetFS(base_path)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        dataset_fs = self.get_dataset_fs(self.base_dir)

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
        with dataset_fs.open(mel_path) as f:
            with io.BytesIO(f.read()) as buffer:
                mel = np.load(buffer)
        pitch_path = os.path.join(
            self.feature_dir,
            "pitch",
            f"{speaker}",
            f"{basename}.npy",
        )
        with dataset_fs.open(pitch_path) as f:
            with io.BytesIO(f.read()) as buffer:
                pitch = np.load(buffer)
        energy_path = os.path.join(
            self.feature_dir,
            "energy",
            f"{speaker}",
            f"{basename}.npy",
        )
        with dataset_fs.open(energy_path) as f:
            with io.BytesIO(f.read()) as buffer:
                energy = np.load(buffer)
        duration_path = os.path.join(
            self.feature_dir,
            "duration",
            f"{speaker}",
            f"{basename}.npy",
        )
        with dataset_fs.open(duration_path) as f:
            with io.BytesIO(f.read()) as buffer:
                duration = np.load(buffer)

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

    def process_meta(self, file_stream: IO[bytes]) -> tuple[list[str], list[str], list[str], list[str]]:
        with io.TextIOWrapper(file_stream, encoding="utf-8") as f:
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


class TextOnlyDatasetWithSentiment(Dataset):
    def __init__(
        self, dataset_path_config: DatasetPathConfig, dataset_preprocessing_config: DatasetPreprocessingConfig, split: DatasetSplit
    ):
        dataset_fs = self.get_dataset_fs(dataset_path_config.base_dir)
        self.text_cleaners = dataset_preprocessing_config.text_cleaners

        meta_file = None
        if split == DatasetSplit.TRAIN:
            meta_file = dataset_path_config.meta_file_train
        elif split == DatasetSplit.VAL:
            meta_file = dataset_path_config.meta_file_val
        else:
            raise ValueError(f"Unknown split: {split}")

        # load metadata
        self.data_ids, self.speakers, self.texts, self.raw_texts = self.process_meta(dataset_fs.open(meta_file))

        with dataset_fs.open(dataset_path_config.speaker_map_file) as f:
            self.speaker_map = json.load(f)

        # Load sentiment labels if provided
        if dataset_path_config.sentiment_file:
            with dataset_fs.open(dataset_path_config.sentiment_file) as f:
                sent_data_ids, sent_labels, label_names = load_sentiment(f)
            self.sentiment_map = {data_id: sent_label for data_id, sent_label in zip(sent_data_ids, sent_labels)}
        else:
            self.sentiment_map = None

    def process_meta(self, file_stream: IO[bytes]) -> tuple[list[str], list[str], list[str], list[str]]:
        with io.TextIOWrapper(file_stream, encoding="utf-8") as f:
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
    
    @lru_cache(maxsize=None)
    def get_dataset_fs(self, base_path):
        return DatasetFS(base_path)

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

    dataset_path_config = DatasetPathConfig.from_dict(
        base_dir="config/LibriTTS",
        config_dict={
            "base_dir": "../../data/LibriTTS.tar",
            "meta_file_train": "train.txt",
            "meta_file_val": "val.txt",
            "speaker_map_file": "speakers.json",
            "feature_dir": "",
            "stats_file": "stats.json",
            # "sentiment_file": "sentiment_scores.csv",
            "sentiment_file": "../overrides/sentiment_scores_override_all_neu.csv",
        })
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


    dataloader = DataLoader(dataset, batch_size=4, num_workers=2, collate_fn=dataset.collate_fn)
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
