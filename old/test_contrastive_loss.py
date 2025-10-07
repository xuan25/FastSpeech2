import numpy as np
import torch
from fastspeech2.config import DatasetFeaturePropertiesConfig
from fastspeech2.dataset.data_models import DataBatch, DataBatchTorch, DataSample
from fastspeech2.model.data_models import ProsodyPredictorOutput
from fastspeech2.model.loss import ProsodyPredictorContrastiveLoss
from fastspeech2.utils.tools import get_mask_from_lengths

config = DatasetFeaturePropertiesConfig.from_dict({
    "pitch_feature_level": "phoneme_level",
    "energy_feature_level": "phoneme_level",
    "n_mel_channels": 2,
    "max_wav_value": 32768.0,
    "sampling_rate": 22050,
    "stft_hop_length": 256,
    "num_sentiments": 3,
})

loss = ProsodyPredictorContrastiveLoss(config, 0, 0)

data_samples = [
    DataSample(
        data_id="sample1",
        speaker=0,
        text=np.array([1, 2, 3], dtype=np.intp),
        raw_text="Sample text 1",
        mel=np.array([[0, 1, 2, 3, 4, 5],[0, 1, 2, 3, 4, 5]]).T.astype(np.float64),
        pitch=np.array([0, 1, 2]).astype(np.float64),
        energy=np.array([0, 1, 2]).astype(np.float64),
        duration=np.array([0, 1, 2]).astype(np.float64),
        sentiment=0
    ),
    DataSample(
        data_id="sample2",
        speaker=1,
        text=np.array([4, 5], dtype=np.intp),
        raw_text="Sample text 2",
        mel=np.array([[6, 7, 8, 9],[6, 7, 8, 9]]).T.astype(np.float64),
        pitch=np.array([3, 4]).astype(np.float64),
        energy=np.array([3, 4]).astype(np.float64),
        duration=np.array([3, 4]).astype(np.float64),
        sentiment=1
    ),
    DataSample(
        data_id="sample3",
        speaker=2,
        text=np.array([6, 7, 8, 9, 10], dtype=np.intp),
        raw_text="Sample text 3",
        mel=np.array([[10, 11, 12, 13, 14, 15, 16, 17],[10, 11, 12, 13, 14, 15, 16, 17]]).T.astype(np.float64),
        pitch=np.array([6, 7, 8, 9, 10]).astype(np.float64),
        energy=np.array([6, 7, 8, 9, 10]).astype(np.float64),
        duration=np.array([6, 7, 8, 9, 10]).astype(np.float64),
        sentiment=2
    ),
]

data_batch = DataBatch(data_samples, sort=False)
data_batch_torch = DataBatchTorch(data_batch, device="cpu")

text_masks = get_mask_from_lengths(data_batch_torch.text_lens, data_batch_torch.text_len_max)
frame_masks = (
    get_mask_from_lengths(data_batch_torch.mel_lens, data_batch_torch.mel_len_max)
    if data_batch_torch.mel_lens is not None
    else None
)

assert data_batch_torch.pitches is not None, "Pitches should not be None"
assert data_batch_torch.energies is not None, "Energies should not be None"
assert data_batch_torch.durations is not None, "Durations should not be None"

pred = ProsodyPredictorOutput(
    pitch_predictions=data_batch_torch.pitches,
    energy_predictions=data_batch_torch.energies,
    log_duration_predictions=data_batch_torch.durations,
    duration_rounded=data_batch_torch.durations,
    text_masks=text_masks,
    text_lens=data_batch_torch.text_lens,
    frame_mask=frame_masks,
)

loss.forward(data_batch_torch, pred, contrastive_mask=torch.tensor([-1, 0, 1], dtype=torch.int8))
