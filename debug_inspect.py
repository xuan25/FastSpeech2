import argparse
import csv
import os

import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader

from fastspeech2.model.data_models import ProsodyPredictorOutput
from fastspeech2.model.prosody_predictor import ProsodyPredictor

from fastspeech2.dataset.datasetfs import DatasetFS

from fastspeech2.config import (
    DatasetConfig, 
    DatasetFeaturePropertiesConfig, 
    DatasetPathConfig, 
    DatasetPreprocessingConfig,
    LossConfig, 
    ModelConfig, 
    ModelGlobalConfig, 
    ModelTransformerConfig, 
    ModelVarianceEmbeddingConfig, 
    ModelVariancePredictorConfig, 
    ModelVocoderConfig, 
    TrainConfig, 
    TrainOutputConfig, 
    TrainOptimizerConfig,
    TrainStepConfig
)
from fastspeech2.dataset.data_models import DataBatch, DataBatchTorch, DatasetFeatureStats
from fastspeech2.dataset.dataset import DatasetSplit, TextOnlyDatasetWithSentiment

torch.serialization.add_safe_globals([
    np._core.multiarray.scalar, 
    np.dtype, np.dtypes.Float64DType, 
    # DatasetConfig, DatasetPathConfig, 
    # DatasetFeaturePropertiesConfig, 
    # DatasetPreprocessingConfig, 
    # ModelConfig, 
    # ModelTransformerConfig,
    # ModelVariancePredictorConfig,
    # ModelVarianceEmbeddingConfig,
    # ModelVocoderConfig,
    # ModelGlobalConfig,
    # TrainConfig,
    # TrainOutputConfig,
    # TrainOptimizerConfig,
    # TrainStepConfig,
    # LossConfig,
    # DatasetFeatureStats
    ])

def get_model_infer(ckpt_path, 
              model_config: ModelConfig,
              dataset_feature_properties_config: DatasetFeaturePropertiesConfig,
              dataset_feature_stats: DatasetFeatureStats, 
              device) -> ProsodyPredictor:

    model = ProsodyPredictor(model_config, dataset_feature_properties_config, dataset_feature_stats).to(device)
    if ckpt_path:
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"])

    model.eval()
    # model.requires_grad_ = False
    model.requires_grad_(False)
    return model

dataset_config = DatasetConfig.load_from_yaml("config/LibriTTS/dataset_sentiment.yaml")

dataset_fs = DatasetFS(dataset_config.path_config.base_dir)

with DatasetFS(dataset_config.path_config.base_dir) as dataset_fs:
    with dataset_fs.open(dataset_config.path_config.stats_file) as stats_stream, \
        dataset_fs.open(dataset_config.path_config.speaker_map_file) as speaker_stream:
        # Load dataset feature statistics
        dataset_feature_stats = DatasetFeatureStats.from_json(
            stats_stream,
            speaker_stream,
        )



model = get_model_infer(
    ckpt_path="output/prosody_predictor_contrastive/sentiment_input-0-overrideembedding0-B/ckpt/40000.pth",
    model_config=ModelConfig.load_from_yaml("config/LibriTTS/model_sentiment_input.yaml"),
    dataset_feature_properties_config=dataset_config.feature_properties_config,
    dataset_feature_stats=dataset_feature_stats,
    device="cpu"
)

print(model.encoder.sentiment_emb_input.weight)