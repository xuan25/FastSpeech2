import json
import numpy as np
import numpy.typing as npt
import torch

# from ..utils.tools import pad_1D, pad_2D


def pad_1D(inputs, PAD=0):
    def pad_data(x, length, PAD):
        x_padded = np.pad(
            x, (0, length - x.shape[0]), mode="constant", constant_values=PAD
        )
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


def pad_2D(inputs, maxlen=None):
    def pad(x, max_len):
        PAD = 0
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")

        s = np.shape(x)[1]
        x_padded = np.pad(
            x, (0, max_len - np.shape(x)[0]), mode="constant", constant_values=PAD
        )
        return x_padded[:, :s]

    if not maxlen:
        maxlen = max(np.shape(x)[0] for x in inputs)

    output = np.stack([pad(x, maxlen) for x in inputs])

    return output

class DataSample:

    def __init__(self, data_id: str, speaker: int, text: npt.NDArray[np.intp], raw_text: str, mel: npt.NDArray[np.float_] | None, pitch: npt.NDArray[np.float_] | None, energy: npt.NDArray[np.float_] | None, duration: npt.NDArray[np.float_] | None, sentiment: int | None):
        self.data_id = data_id
        self.speaker = speaker
        self.text = text
        self.raw_text = raw_text
        self.mel = mel
        self.pitch = pitch
        self.energy = energy
        self.duration = duration
        self.sentiment = sentiment

    def __repr__(self):
        return f"DataSample(data_id={self.data_id}, speaker={self.speaker}, text={self.text}, raw_text={self.raw_text}, sentiment={self.sentiment}, mel={self.mel}, pitch={self.pitch}, energy={self.energy}, duration={self.duration})"

class DataBatch:
    def __init__(self, data_samples: list[DataSample], sort=True):

        self.data_samples = data_samples
        self.sort = sort

        self.batch_size = len(data_samples)

        if self.sort:
            sample_lens = np.array([sample.text.shape[0] for sample in data_samples])
            sample_idxs: list[int] = np.argsort(-sample_lens).tolist()
        else:
            sample_idxs: list[int] = np.arange(self.batch_size).tolist()

        self.text_lens: npt.NDArray[np.intp] = np.array([data_sample.text.shape[0] for data_sample in data_samples])
        self.mel_lens: npt.NDArray[np.intp] | None = np.array([data_sample.mel.shape[0] for data_sample in data_samples]) if data_samples[0].mel is not None else None # type: ignore

        self.text_len_max: int = max(self.text_lens)
        self.mel_len_max: int | None = max(self.mel_lens) if self.mel_lens is not None else None

        self.data_ids = [data_samples[idx].data_id for idx in sample_idxs]
        self.speakers: npt.NDArray[np.intp] = np.array([data_samples[idx].speaker for idx in sample_idxs])
        self.texts: npt.NDArray[np.intp] = pad_1D([data_samples[idx].text for idx in sample_idxs])
        self.raw_texts = [data_samples[idx].raw_text for idx in sample_idxs]
        self.mels: npt.NDArray[np.float_] | None = pad_2D([data_samples[idx].mel for idx in sample_idxs]) if data_samples[0].mel is not None else None
        self.pitches: npt.NDArray[np.float_] | None = pad_1D([data_samples[idx].pitch for idx in sample_idxs]) if data_samples[0].pitch is not None else None
        self.energies: npt.NDArray[np.float_] | None = pad_1D([data_samples[idx].energy for idx in sample_idxs]) if data_samples[0].energy is not None else None
        self.durations: npt.NDArray[np.float_] | None = pad_1D([data_samples[idx].duration for idx in sample_idxs]) if data_samples[0].duration is not None else None
        self.sentiments: npt.NDArray[np.intp] | None = np.array([data_samples[idx].sentiment for idx in sample_idxs]) if data_samples[0].sentiment is not None else None

    def __repr__(self):
        return f"DataBatch(data_ids={self.data_ids}, speakers={self.speakers}, texts={self.texts}, raw_texts={self.raw_texts}, mels={self.mels}, pitches={self.pitches}, energies={self.energies}, durations={self.durations})"
    
    def __len__(self):
        return self.batch_size
    
    def __getitem__(self, idx):
        if idx >= self.batch_size:
            raise IndexError("Index out of range")
        return self.data_samples[idx]
    
    def __iter__(self):
        for sample in self.data_samples:
            yield sample

    def to_torch(self, device):
        
        # self.speakers = torch.from_numpy(self.speakers).long().to(device)
        # self.texts = torch.from_numpy(self.texts).long().to(device)
        # self.text_lens = torch.from_numpy(self.text_lens).to(device)
        # self.mels = torch.from_numpy(self.mels).float().to(device) if self.mels is not None else None
        # self.mel_lens = torch.from_numpy(self.mel_lens).to(device) if self.mel_lens is not None else None
        # self.pitches = torch.from_numpy(self.pitches).float().to(device) if self.pitches is not None else None
        # self.energies = torch.from_numpy(self.energies).to(device) if self.energies is not None else None
        # self.durations = torch.from_numpy(self.durations).long().to(device) if self.durations is not None else None

        result = DataBatchTorch(self, device)

        return result
    
class DataBatchTorch:
    def __init__(self, data_batch: DataBatch, device: str|torch.device = "cpu"):

        self.data_samples = data_batch.data_samples
        self.sort = data_batch.sort
        self.batch_size = data_batch.batch_size

        self.data_ids = data_batch.data_ids
        self.speakers = torch.from_numpy(data_batch.speakers).long().to(device)
        self.texts = torch.from_numpy(data_batch.texts).long().to(device)
        self.raw_texts = data_batch.raw_texts
        self.mels = torch.from_numpy(data_batch.mels).float().to(device) if data_batch.mels is not None else None
        self.pitches = torch.from_numpy(data_batch.pitches).float().to(device) if data_batch.pitches is not None else None
        self.energies = torch.from_numpy(data_batch.energies).to(device) if data_batch.energies is not None else None
        self.durations = torch.from_numpy(data_batch.durations).long().to(device) if data_batch.durations is not None else None
        self.sentiments = torch.from_numpy(data_batch.sentiments).long().to(device) if data_batch.sentiments is not None else None
        self.text_lens = torch.from_numpy(data_batch.text_lens).to(device)
        self.mel_lens = torch.from_numpy(data_batch.mel_lens).to(device) if data_batch.mel_lens is not None else None
        self.text_len_max = data_batch.text_len_max
        self.mel_len_max = data_batch.mel_len_max
    
    def __repr__(self):
        return f"DataBatchTorch(data_ids={self.data_ids}, speakers={self.speakers}, texts={self.texts}, raw_texts={self.raw_texts}, mels={self.mels}, pitches={self.pitches}, energies={self.energies}, durations={self.durations})"
    
    def __len__(self):
        return self.batch_size
    
    def __getitem__(self, idx):
        if idx >= self.batch_size:
            raise IndexError("Index out of range")
        return self.data_samples[idx]
    
    def __iter__(self):
        for sample in self.data_samples:
            yield sample

class DatasetFeatureStats:

    def __init__(self, 
                 pitch_min: float, 
                 pitch_max: float, 
                 pitch_mean: float, 
                 pitch_std: float, 
                 energy_min: float, 
                 energy_max: float, 
                 energy_mean: float, 
                 energy_std: float, 
                 n_speakers: int
                ):
        self.pitch_min = pitch_min
        self.pitch_max = pitch_max
        self.pitch_mean = pitch_mean
        self.pitch_std = pitch_std
        self.energy_min = energy_min
        self.energy_max = energy_max
        self.energy_mean = energy_mean
        self.energy_std = energy_std
        self.n_speakers = n_speakers

    def __repr__(self):
        return f"DatasetStats(pitch_min={self.pitch_min}, pitch_max={self.pitch_max}, energy_min={self.energy_min}, energy_max={self.energy_max})"

    @classmethod
    def from_json(cls, stats_file: str, speaker_file: str):
        with open(stats_file, "r", encoding="utf-8") as f:
            stats = json.load(f)
            pitch_min, pitch_max, pitch_mean, pitch_std = stats["pitch"]
            energy_min, energy_max, energy_mean, energy_std = stats["energy"]
        with open(speaker_file, "r", encoding="utf-8") as f:
            n_speakers = len(json.load(f))
        return cls(
            pitch_min=pitch_min,
            pitch_max=pitch_max,
            pitch_mean=pitch_mean,
            pitch_std=pitch_std,
            energy_min=energy_min,
            energy_max=energy_max,
            energy_mean=energy_mean,
            energy_std=energy_std,
            n_speakers=n_speakers
        )
