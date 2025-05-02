import csv
import json
import logging
import math
import os
from typing import List

import numpy as np
from torch.utils.data import Dataset
import tqdm

from fastspeech2.text import text_to_sequence
from fastspeech2.utils.tools import pad_1D, pad_2D



class OriginalDataset(Dataset):
    def __init__(
        self, filename, preprocess_config, train_config, sort=False, drop_last=False
    ):
        self.dataset_name = preprocess_config["dataset"]
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
        self.batch_size = train_config["optimizer"]["batch_size"]

        self.basename, self.speaker, self.text, self.raw_text = self.process_meta(
            filename
        )
        with open(os.path.join(self.preprocessed_path, "speakers.json")) as f:
            self.speaker_map = json.load(f)
        self.sort = sort
        self.drop_last = drop_last

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        raw_text = self.raw_text[idx]
        phone = np.array(text_to_sequence(self.text[idx], self.cleaners))
        mel_path = os.path.join(
            self.preprocessed_path,
            "mel",
            f"{speaker}",
            f"{basename}.npy",
            # "{}-mel-{}.npy".format(speaker, basename),
        )
        mel = np.load(mel_path)
        pitch_path = os.path.join(
            self.preprocessed_path,
            "pitch",
            f"{speaker}",
            f"{basename}.npy",
            # "{}-pitch-{}.npy".format(speaker, basename),
        )
        pitch = np.load(pitch_path)
        energy_path = os.path.join(
            self.preprocessed_path,
            "energy",
            f"{speaker}",
            f"{basename}.npy",
            # "{}-energy-{}.npy".format(speaker, basename),
        )
        energy = np.load(energy_path)
        duration_path = os.path.join(
            self.preprocessed_path,
            "duration",
            f"{speaker}",
            f"{basename}.npy",
            # "{}-duration-{}.npy".format(speaker, basename),
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
            
        sample = {
            "id": basename,
            "speaker": speaker_id,
            "text": phone,
            "raw_text": raw_text,
            "mel": mel,
            "pitch": pitch,
            "energy": energy,
            "duration": duration,
        }

        return sample

    def process_meta(self, filename):
        with open(
            os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8"
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

    def reprocess(self, data, idxs):
        ids = [data[idx]["id"] for idx in idxs]
        speakers = [data[idx]["speaker"] for idx in idxs]
        texts = [data[idx]["text"] for idx in idxs]
        raw_texts = [data[idx]["raw_text"] for idx in idxs]
        mels = [data[idx]["mel"] for idx in idxs]
        pitches = [data[idx]["pitch"] for idx in idxs]
        energies = [data[idx]["energy"] for idx in idxs]
        durations = [data[idx]["duration"] for idx in idxs]

        text_lens = np.array([text.shape[0] for text in texts])
        mel_lens = np.array([mel.shape[0] for mel in mels])

        speakers = np.array(speakers)
        texts = pad_1D(texts)
        mels = pad_2D(mels)
        pitches = pad_1D(pitches)
        energies = pad_1D(energies)
        durations = pad_1D(durations)

        return (
            ids,
            raw_texts,
            speakers,
            texts,
            text_lens,
            max(text_lens),
            mels,
            mel_lens,
            max(mel_lens),
            pitches,
            energies,
            durations,
        )

    def collate_fn(self, data):
        data_size = len(data)

        if self.sort:
            len_arr = np.array([d["text"].shape[0] for d in data])
            idx_arr = np.argsort(-len_arr)
        else:
            idx_arr = np.arange(data_size)

        tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.batch_size) :]
        idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % self.batch_size)]
        idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()
        if not self.drop_last and len(tail) > 0:
            idx_arr += [tail.tolist()]

        output = list()
        for idx in idx_arr:
            output.append(self.reprocess(data, idx))

        return output

SENTIMENT_FILE = "../sentiment-balanced-dataset/data/original/LibriTTS/sentiment_scores_libri-tts.csv"

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

class OriginalDatasetWithSentiment(Dataset):
    def __init__(
        self, filename, preprocess_config, train_config, sort=False, drop_last=False, sentiment_file=SENTIMENT_FILE
    ):
        self.dataset_name = preprocess_config["dataset"]
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
        self.batch_size = train_config["optimizer"]["batch_size"]

        self.basename, self.speaker, self.text, self.raw_text = self.process_meta(
            filename
        )
        with open(os.path.join(self.preprocessed_path, "speakers.json")) as f:
            self.speaker_map = json.load(f)
        self.sort = sort
        self.drop_last = drop_last

        # Load sentiment labels if provided
        if sentiment_file:
            sent_data_ids, sent_labels, label_names = load_sentiment(sentiment_file)
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
        phone = np.array(text_to_sequence(self.text[idx], self.cleaners))
        mel_path = os.path.join(
            self.preprocessed_path,
            "mel",
            f"{speaker}",
            f"{basename}.npy",
            # "{}-mel-{}.npy".format(speaker, basename),
        )
        mel = np.load(mel_path)
        pitch_path = os.path.join(
            self.preprocessed_path,
            "pitch",
            f"{speaker}",
            f"{basename}.npy",
            # "{}-pitch-{}.npy".format(speaker, basename),
        )
        pitch = np.load(pitch_path)
        energy_path = os.path.join(
            self.preprocessed_path,
            "energy",
            f"{speaker}",
            f"{basename}.npy",
            # "{}-energy-{}.npy".format(speaker, basename),
        )
        energy = np.load(energy_path)
        duration_path = os.path.join(
            self.preprocessed_path,
            "duration",
            f"{speaker}",
            f"{basename}.npy",
            # "{}-duration-{}.npy".format(speaker, basename),
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
            
        sample = {
            "id": basename,
            "speaker": speaker_id,
            "text": phone,
            "raw_text": raw_text,
            "sentiment": sentiment_label if self.sentiment_map else None,
            "mel": mel,
            "pitch": pitch,
            "energy": energy,
            "duration": duration,
        }

        return sample

    def process_meta(self, filename):
        with open(
            os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8"
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

    def reprocess(self, data, idxs):
        ids = [data[idx]["id"] for idx in idxs]
        speakers = [data[idx]["speaker"] for idx in idxs]
        texts = [data[idx]["text"] for idx in idxs]
        raw_texts = [data[idx]["raw_text"] for idx in idxs]
        sentiment_labels = [data[idx]["sentiment"] for idx in idxs] if self.sentiment_map else None
        mels = [data[idx]["mel"] for idx in idxs]
        pitches = [data[idx]["pitch"] for idx in idxs]
        energies = [data[idx]["energy"] for idx in idxs]
        durations = [data[idx]["duration"] for idx in idxs]

        text_lens = np.array([text.shape[0] for text in texts])
        mel_lens = np.array([mel.shape[0] for mel in mels])

        speakers = np.array(speakers)
        texts = pad_1D(texts)
        if sentiment_labels is not None:
            sentiment_labels = np.array(sentiment_labels)
        mels = pad_2D(mels)
        pitches = pad_1D(pitches)
        energies = pad_1D(energies)
        durations = pad_1D(durations)

        return (
            ids,
            raw_texts,
            speakers,
            texts,
            text_lens,
            max(text_lens),
            sentiment_labels,
            mels,
            mel_lens,
            max(mel_lens),
            pitches,
            energies,
            durations,
        )

    def collate_fn(self, data):
        data_size = len(data)

        if self.sort:
            len_arr = np.array([d["text"].shape[0] for d in data])
            idx_arr = np.argsort(-len_arr)
        else:
            idx_arr = np.arange(data_size)

        tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.batch_size) :]
        idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % self.batch_size)]
        idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()
        if not self.drop_last and len(tail) > 0:
            idx_arr += [tail.tolist()]

        output = list()
        for idx in idx_arr:
            output.append(self.reprocess(data, idx))

        return output



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

class TextOnlyDataset(Dataset):
    def __init__(
        self, meta_file, speaker_map_file, text_cleaners = ["english_cleaners"]
    ):

        self.text_cleaners = text_cleaners

        # load metadata
        self.data_ids, self.speakers, self.texts, self.raw_texts = load_meta(meta_file)

        with open(speaker_map_file, mode="r", encoding="utf-8") as f:
            self.speaker_map = json.load(f)

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        speaker = self.speakers[idx]
        speaker_id = self.speaker_map[speaker]
        raw_text = self.raw_texts[idx]
        phone = np.array(text_to_sequence(self.texts[idx], self.text_cleaners))

        sample = {
            "id": data_id,
            "speaker": speaker_id,
            "text": phone,
            "raw_text": raw_text,
        }

        return sample

    def reprocess(self, data, idxs):
        ids = [data[idx]["id"] for idx in idxs]
        speakers = [data[idx]["speaker"] for idx in idxs]
        texts = [data[idx]["text"] for idx in idxs]
        raw_texts = [data[idx]["raw_text"] for idx in idxs]

        text_lens = np.array([text.shape[0] for text in texts])

        speakers = np.array(speakers)
        texts = pad_1D(texts)

        return (
            ids,
            raw_texts,
            speakers,
            texts,
            text_lens,
            max(text_lens),
        )

    def collate_fn(self, data):
        data_size = len(data)

        output = self.reprocess(data, np.arange(data_size))

        return output



class TextOnlyDatasetWithSentiment(Dataset):
    def __init__(
        self, meta_file, speaker_map_file, text_cleaners = ["english_cleaners"], sentiment_file=SENTIMENT_FILE
    ):

        self.text_cleaners = text_cleaners

        # load metadata
        self.data_ids, self.speakers, self.texts, self.raw_texts = load_meta(meta_file)

        with open(speaker_map_file, mode="r", encoding="utf-8") as f:
            self.speaker_map = json.load(f)

        # Load sentiment labels if provided
        if sentiment_file:
            sent_data_ids, sent_labels, label_names = load_sentiment(sentiment_file)
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

        sample = {
            "id": data_id,
            "speaker": speaker_id,
            "text": phone,
            "raw_text": raw_text,
            "sentiment": sentiment_label if self.sentiment_map else None,
        }

        return sample

    def reprocess(self, data, idxs):
        ids = [data[idx]["id"] for idx in idxs]
        speakers = [data[idx]["speaker"] for idx in idxs]
        texts = [data[idx]["text"] for idx in idxs]
        raw_texts = [data[idx]["raw_text"] for idx in idxs]
        sentiment_labels = [data[idx]["sentiment"] for idx in idxs] if self.sentiment_map else None

        text_lens = np.array([text.shape[0] for text in texts])

        speakers = np.array(speakers)
        texts = pad_1D(texts)
        if sentiment_labels is not None:
            sentiment_labels = np.array(sentiment_labels)

        sentiment_labels = np.array(sentiment_labels) if sentiment_labels is not None else None

        return (
            ids,
            raw_texts,
            speakers,
            texts,
            text_lens,
            max(text_lens),
            sentiment_labels,
        )

    def collate_fn(self, data):
        data_size = len(data)

        output = self.reprocess(data, np.arange(data_size))

        return output
