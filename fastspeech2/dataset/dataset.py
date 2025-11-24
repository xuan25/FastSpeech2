import csv
import enum
from functools import lru_cache
import io
import json
import os
from pathlib import Path
from typing import IO

import numpy as np
from torch.utils.data import Dataset
import tqdm

from .datasetfs import DatasetFS

from ..config import DatasetPathConfig, DatasetPreprocessingConfig
from .data_models import DataBatch, DataSample
from ..text import text_to_sequence

def load_labels(file_stream: IO[bytes]) -> tuple[list[str], list[int], list[str]]:
    with io.TextIOWrapper(file_stream, encoding='utf-8') as csv_stream:
        reader = csv.DictReader(csv_stream)

        headers = reader.fieldnames

        assert headers is not None, "Labels file must have headers"

        label_names = headers[1:]

        data_ids = []
        scores = []
        for row in reader:
            data_ids.append(row['basename'])
            scores.append([float(row[label]) for label in label_names])

        labels = np.argmax(np.array(scores), axis=1)

    return data_ids, labels, list(label_names)

class DatasetSplit(enum.Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"




class DatasetWithLabelContrastive(Dataset):
    def __init__(
        self, dataset_path_config: DatasetPathConfig, dataset_preprocessing_config: DatasetPreprocessingConfig, split: DatasetSplit, sort=False
    ):
        
        self.base_dir = dataset_path_config.base_dir

        self.feature_dir = dataset_path_config.feature_dir
        self.text_cleaners = dataset_preprocessing_config.text_cleaners

        dataset_fs = self.get_dataset_fs()
        
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

        # Load label if provided
        if dataset_path_config.label_file:
            with dataset_fs.open(dataset_path_config.label_file) as f:
                label_data_ids, label_labels, label_names = load_labels(f)
            self.label_map = {data_id: label_label for data_id, label_label in zip(label_data_ids, label_labels)}
        else:
            self.label_map = None


        # For label removed cases
        # filter out samples without labels and print a warning
        if self.label_map:
            num_excluded = 0
            filtered_basename = []
            filtered_speaker = []
            filtered_text = []
            filtered_raw_text = []
            for idx in range(len(self.basename)):
                if self.basename[idx] in self.label_map:
                    filtered_basename.append(self.basename[idx])
                    filtered_speaker.append(self.speaker[idx])
                    filtered_text.append(self.text[idx])
                    filtered_raw_text.append(self.raw_text[idx])
                else:
                    num_excluded += 1
            if num_excluded > 0:
                tqdm.tqdm.write(f"Warning: {num_excluded} samples do not have labels and will be ignored.")
            self.basename = filtered_basename
            self.speaker = filtered_speaker
            self.text = filtered_text
            self.raw_text = filtered_raw_text



        self.label_map_reverse: dict[int, list[int]] = {}
        if self.label_map:
            for idx in range(len(self.basename)):
                label = self.label_map[self.basename[idx]]
                if label not in self.label_map_reverse:
                    self.label_map_reverse[label] = []
                self.label_map_reverse[label].append(idx)

    @lru_cache(maxsize=1)
    def get_dataset_fs(self):
        return DatasetFS(self.base_dir)

    def __len__(self):
        return len(self.text)

    def __load_data_sample(self, idx: int) -> DataSample:
        dataset_fs = self.get_dataset_fs()

        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        raw_text = self.raw_text[idx]
        phone = np.array(text_to_sequence(self.text[idx], self.text_cleaners))
        mel_path = Path(
            self.feature_dir,
            "mel",
            f"{basename}.npy",
        ).as_posix()
        with dataset_fs.open(mel_path) as f:
            with io.BytesIO(f.read()) as buffer:
                mel: np.ndarray = np.load(buffer)
                mel = mel.astype(np.float32)
        pitch_path = Path(
            self.feature_dir,
            "pitch",
            f"{basename}.npy",
        ).as_posix()
        with dataset_fs.open(pitch_path) as f:
            with io.BytesIO(f.read()) as buffer:
                pitch: np.ndarray = np.load(buffer)
                pitch = pitch.astype(np.float32)
        energy_path = Path(
            self.feature_dir,
            "energy",
            f"{basename}.npy",
        ).as_posix()
        with dataset_fs.open(energy_path) as f:
            with io.BytesIO(f.read()) as buffer:
                energy: np.ndarray = np.load(buffer)
                energy = energy.astype(np.float32)
        duration_path = Path(
            self.feature_dir,
            "duration",
            f"{basename}.npy",
        ).as_posix()
        with dataset_fs.open(duration_path) as f:
            with io.BytesIO(f.read()) as buffer:
                duration: np.ndarray = np.load(buffer)
                duration = duration.astype(np.int64)

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

        label = self.label_map.get(basename, -1) if self.label_map else None

        if label == -1:
            raise ValueError(f"Label not found for {basename}")

        sample = DataSample(
            data_id=basename,
            speaker=speaker_id,
            text=phone,
            raw_text=raw_text,
            mel=mel,
            pitch=pitch,
            energy=energy,
            duration=duration,
            label=label,
            label_source=label,
        )
        return sample
    
    def __load_data_sample_text(self, idx: int) -> DataSample:
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        raw_text = self.raw_text[idx]
        phone = np.array(text_to_sequence(self.text[idx], self.text_cleaners))

        label = self.label_map.get(basename, -1) if self.label_map else None

        if label == -1:
            raise ValueError(f"Label not found for {basename}")

        sample = DataSample(
            data_id=basename,
            speaker=speaker_id,
            text=phone,
            raw_text=raw_text,
            mel=None,
            pitch=None,
            energy=None,
            duration=None,
            label=label,
            label_source=label,
        )
        return sample
    
    def __getitem__(self, idx) -> tuple[list[DataSample], np.ndarray]:

        sample: DataSample = self.__load_data_sample(idx)

        num_label_categories = len(self.label_map_reverse)

        # negative samples: samples from different label classes

        samples_neg = []

        for i in range(num_label_categories - 1):
            sample_neg =  DataSample(
                data_id=sample.data_id,
                speaker=sample.speaker,
                text=sample.text,
                raw_text=sample.raw_text,
                mel=sample.mel,
                pitch=sample.pitch,
                energy=sample.energy,
                duration=sample.duration,
                label=(sample.label + i + 1) % num_label_categories if sample.label is not None else None, # simple negative sample generation by adding 1 to label
                label_source=sample.label_source
            )
            samples_neg.append(sample_neg)

        # sample positive samples form the same label class
        assert sample.label is not None, "Sample label must not be None for contrastive learning"
        label_samples = self.label_map_reverse[sample.label]
        if len(label_samples) < 3:
            raise ValueError(f"Not enough samples for label {sample.label} to create contrastive samples")

        samples_pos = []
        for i in range(num_label_categories-1):
            sample_pos = self.__load_data_sample_text(np.random.choice(label_samples))
            text_length_pos = min(sample_pos.text.shape[0], sample.text.shape[0])
            sample_pos.text = sample_pos.text[:text_length_pos]  # ensure same length for contrastive learning, but keep the contrastive content
            sample_pos.duration = sample.duration[:text_length_pos] if sample.duration is not None else None
            sample_pos.pitch = sample.pitch[:text_length_pos] if sample.pitch is not None else None
            sample_pos.energy = sample.energy[:text_length_pos] if sample.energy is not None else None
            mel_length_pos = np.sum(sample_pos.duration) if sample_pos.duration is not None else 0
            sample_pos.mel = sample.mel[:mel_length_pos] if sample.mel is not None else None
            samples_pos.append(sample_pos)

        # the first sample is the original sample, the second and third sample is negative contrastive samples
        mask = np.array([0] + [-1] * (num_label_categories - 1) + [1] * (num_label_categories - 1), dtype=np.int8)  # 0 for original, -1 for negative samples, 1 for positive samples

        # return ([sample, sample_neg1, sample_neg2, sample_pos1, sample_pos2], mask)
        return ([sample] + samples_neg + samples_pos, mask)

    def process_meta(self, file_stream: IO[bytes]) -> tuple[list[str], list[str], list[str], list[str]]:
        with io.TextIOWrapper(file_stream, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            basename = []
            speaker = []
            text = []
            raw_text = []
            for line in reader:
                basename.append(line["basename"])
                speaker.append(line["speaker"])
                text.append(line["text"])
                raw_text.append(line["raw_text"])
            return basename, speaker, text, raw_text

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


class DatasetWithLabel(Dataset):
    def __init__(
        self, dataset_path_config: DatasetPathConfig, dataset_preprocessing_config: DatasetPreprocessingConfig, split: DatasetSplit, sort=False
    ):
        self.base_dir = dataset_path_config.base_dir

        self.feature_dir = dataset_path_config.feature_dir
        self.text_cleaners = dataset_preprocessing_config.text_cleaners
        
        dataset_fs = self.get_dataset_fs()

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

        # Load labels if provided
        if dataset_path_config.label_file:
            with dataset_fs.open(dataset_path_config.label_file) as f:
                label_data_ids, label_labels, label_names = load_labels(f)
            self.label_map = {data_id: label_label for data_id, label_label in zip(label_data_ids, label_labels)}
        else:
            self.label_map = None

        # For label removed cases
        if self.label_map:
            num_excluded = 0
            filtered_basename = []
            filtered_speaker = []
            filtered_text = []
            filtered_raw_text = []
            for idx in range(len(self.basename)):
                if self.basename[idx] in self.label_map:
                    filtered_basename.append(self.basename[idx])
                    filtered_speaker.append(self.speaker[idx])
                    filtered_text.append(self.text[idx])
                    filtered_raw_text.append(self.raw_text[idx])
                else:
                    num_excluded += 1
            if num_excluded > 0:
                tqdm.tqdm.write(f"Warning: {num_excluded} samples do not have labels and will be ignored.")
            self.basename = filtered_basename
            self.speaker = filtered_speaker
            self.text = filtered_text
            self.raw_text = filtered_raw_text

        
    @lru_cache(maxsize=None)
    def get_dataset_fs(self):
        return DatasetFS(self.base_dir)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        dataset_fs = self.get_dataset_fs()

        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        raw_text = self.raw_text[idx]
        phone = np.array(text_to_sequence(self.text[idx], self.text_cleaners))
        mel_path = Path(
            self.feature_dir,
            "mel",
            f"{basename}.npy",
        ).as_posix()
        with dataset_fs.open(mel_path) as f:
            with io.BytesIO(f.read()) as buffer:
                mel: np.ndarray = np.load(buffer)
                mel = mel.astype(np.float32)
        pitch_path = Path(
            self.feature_dir,
            "pitch",
            f"{basename}.npy",
        ).as_posix()
        with dataset_fs.open(pitch_path) as f:
            with io.BytesIO(f.read()) as buffer:
                pitch: np.ndarray = np.load(buffer)
                pitch = pitch.astype(np.float32)
        energy_path = Path(
            self.feature_dir,
            "energy",
            f"{basename}.npy",
        ).as_posix()
        with dataset_fs.open(energy_path) as f:
            with io.BytesIO(f.read()) as buffer:
                energy: np.ndarray = np.load(buffer)
                energy = energy.astype(np.float32)
        duration_path = Path(
            self.feature_dir,
            "duration",
            f"{basename}.npy",
        ).as_posix()
        with dataset_fs.open(duration_path) as f:
            with io.BytesIO(f.read()) as buffer:
                duration: np.ndarray = np.load(buffer)
                duration = duration.astype(np.int64)

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

        label = self.label_map.get(basename, -1) if self.label_map else None

        if label == -1:
            raise ValueError(f"Label not found for {basename}")

        sample = DataSample(
            data_id=basename,
            speaker=speaker_id,
            text=phone,
            raw_text=raw_text,
            mel=mel,
            pitch=pitch,
            energy=energy,
            duration=duration,
            label=label,
            label_source=label,
        )

        return sample

    def process_meta(self, file_stream: IO[bytes]) -> tuple[list[str], list[str], list[str], list[str]]:
        with io.TextIOWrapper(file_stream, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            basename = []
            speaker = []
            text = []
            raw_text = []
            for line in reader:
                basename.append(line["basename"])
                speaker.append(line["speaker"])
                text.append(line["text"])
                raw_text.append(line["raw_text"])
            return basename, speaker, text, raw_text

    def collate_fn(self, data_samples):
        batch = DataBatch(data_samples, sort=self.sort)
        return batch



class TextOnlyDatasetWithLabel(Dataset):
    def __init__(
        self, dataset_path_config: DatasetPathConfig, dataset_preprocessing_config: DatasetPreprocessingConfig, split: DatasetSplit
    ):
        self.base_dir = dataset_path_config.base_dir

        self.text_cleaners = dataset_preprocessing_config.text_cleaners

        dataset_fs = self.get_dataset_fs()

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

        # Load labels if provided
        if dataset_path_config.label_file:
            with dataset_fs.open(dataset_path_config.label_file) as f:
                label_data_ids, label_labels, label_names = load_labels(f)
            self.label_map = {data_id: label_label for data_id, label_label in zip(label_data_ids, label_labels)}
        else:
            self.label_map = None

        # For label removed cases
        if self.label_map:
            num_excluded = 0
            filtered_data_ids = []
            filtered_speakers = []
            filtered_texts = []
            filtered_raw_texts = []
            for idx in range(len(self.data_ids)):
                if self.data_ids[idx] in self.label_map:
                    filtered_data_ids.append(self.data_ids[idx])
                    filtered_speakers.append(self.speakers[idx])
                    filtered_texts.append(self.texts[idx])
                    filtered_raw_texts.append(self.raw_texts[idx])
                else:
                    num_excluded += 1
            if num_excluded > 0:
                tqdm.tqdm.write(f"Warning: {num_excluded} samples do not have labels and will be ignored.")
            self.data_ids = filtered_data_ids
            self.speakers = filtered_speakers
            self.texts = filtered_texts
            self.raw_texts = filtered_raw_texts

    def process_meta(self, file_stream: IO[bytes]) -> tuple[list[str], list[str], list[str], list[str]]:
        with io.TextIOWrapper(file_stream, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            basename = []
            speaker = []
            text = []
            raw_text = []
            for line in reader:
                basename.append(line["basename"])
                speaker.append(line["speaker"])
                text.append(line["text"])
                raw_text.append(line["raw_text"])
            return basename, speaker, text, raw_text
    
    @lru_cache(maxsize=None)
    def get_dataset_fs(self):
        return DatasetFS(self.base_dir)

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        speaker = self.speakers[idx]
        speaker_id = self.speaker_map[speaker]
        raw_text = self.raw_texts[idx]
        phone = np.array(text_to_sequence(self.texts[idx], self.text_cleaners))

        label = self.label_map.get(data_id, -1) if self.label_map else None

        sample = DataSample(
            data_id=data_id,
            speaker=speaker_id,
            text=phone,
            raw_text=raw_text,
            mel=None,
            pitch=None,
            energy=None,
            duration=None,
            label=label,
            label_source=label,
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
            # "label_file": "label_scores.csv",
            "label_file": "../overrides/label_scores_override_all_neu.csv",
        })
    dataset_preprocessing_config=DatasetPreprocessingConfig(
        lexicon_path="../../lexicon/librispeech-lexicon.txt",
        text_cleaners=["english_cleaners"],
    )

    dataset = DatasetWithLabel(
        dataset_path_config=dataset_path_config,
        dataset_preprocessing_config=dataset_preprocessing_config,
        split=DatasetSplit.VAL,
        sort=True
    )
    
    for i in range(3):
        sample = dataset[i]
        # print(sample)
        print(f"Speaker: {sample.speaker}, Text: {sample.text.shape}, Label: {sample.label}")
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
            print(f"Speaker: {sample.speaker}, Text: {sample.text.shape}, Label: {sample.label}")
            print(f"Raw Text: {sample.raw_text}")
            print(f"Mel shape: {sample.mel.shape}, Pitch shape: {sample.pitch.shape}, Energy shape: {sample.energy.shape}, Duration shape: {sample.duration.shape}") # type: ignore
            print("-" * 50)

        break

if __name__ == "__main__":
    main()
