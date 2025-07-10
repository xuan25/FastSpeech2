import sys
import os
# Adjust the import path to the project root (if necessary)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
from fastspeech2.config import DatasetFeaturePropertiesConfig
from fastspeech2.dataset.data_models import DataBatch, DataBatchTorch, DataSample
from fastspeech2.model.data_models import FastSpeech2Output
from fastspeech2.model.loss import FastSpeech2Loss

datasetFeaturePropertiesConfig = DatasetFeaturePropertiesConfig.from_dict(
    {
        "pitch_feature_level": "phoneme_level",
        "energy_feature_level": "phoneme_level",
        "n_mel_channels": 80,
        "max_wav_value": 32768.0,
        "sampling_rate": 22050,
        "stft_hop_length": 256,
        "num_sentiments": 3,
    }
)

loss = FastSpeech2Loss(datasetFeaturePropertiesConfig)

input_data = DataBatchTorch(
    DataBatch(
        [
            DataSample(
                data_id="1",
                speaker=0,
                text=np.array([1, 2, 3, 4, 5], dtype=np.int32),
                raw_text="hello world",
                mel=np.random.rand(5, 80).astype(np.float32),
                pitch=np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32),
                energy=np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32),
                duration=np.array([1, 1, 1, 1, 1], dtype=np.float32),
                sentiment=0,
            )
        ]
    )
)

predictions = FastSpeech2Output(
    output=torch.randn(1, 5, 80).float(),
    postnet_output=torch.randn(1, 5, 80).float(),
    pitch_predictions=torch.randn(1, 5).float(),
    energy_predictions=torch.randn(1, 5).float(),
    log_duration_predictions=torch.randn(1, 5).float(),
    duration_rounded=torch.randn(1, 5).int(),
    text_masks=torch.ones(1, 5).bool(),
    text_lens=torch.ones(1, 5).bool(),
    mel_masks=torch.ones(1, 5).bool(),
    mel_lens=torch.ones(1, 5).bool(),
)

res = loss.forward(input_data, predictions)

print(res)