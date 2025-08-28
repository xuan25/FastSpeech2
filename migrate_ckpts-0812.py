import torch

# SOURCE_DIR = "output/prosody_predictor_contrastive-old-0812"
# DEST_DIR = "output/prosody_predictor_contrastive"
SOURCE_DIR = "output/prosody_predictor-old-0812"
DEST_DIR = "output/prosody_predictor"

import numpy as np
from fastspeech2.config import (
    DatasetConfig, DatasetPathConfig, DatasetFeaturePropertiesConfig,
    DatasetPreprocessingConfig, ModelConfig, ModelTransformerConfig,
    ModelVariancePredictorConfig, ModelVarianceEmbeddingConfig,
    ModelVocoderConfig, ModelGlobalConfig, TrainConfig, TrainOutputConfig,
    TrainOptimizerConfig, TrainStepConfig, LossConfig
)
from fastspeech2.dataset.data_models import DatasetFeatureStats

torch.serialization.add_safe_globals([
    np._core.multiarray.scalar, 
    np.dtype, np.dtypes.Float64DType, 
    DatasetConfig, DatasetPathConfig, 
    DatasetFeaturePropertiesConfig, 
    DatasetPreprocessingConfig, 
    ModelConfig, 
    ModelTransformerConfig,
    ModelVariancePredictorConfig,
    ModelVarianceEmbeddingConfig,
    ModelVocoderConfig,
    ModelGlobalConfig,
    TrainConfig,
    TrainOutputConfig,
    TrainOptimizerConfig,
    TrainStepConfig,
    LossConfig,
    DatasetFeatureStats
    ])

def migrate_ckpt(old_ckpt_path: str, new_ckpt_path: str):
    old_ckpt = torch.load(old_ckpt_path)
    # Migrate configs

    # Patch
    # if train_config does not have a attribute of loss_config, create a new one
    # Error: 'TrainConfig' object has no attribute 'loss_config'
    train_config = old_ckpt["configs"]["train_config"]
    if not hasattr(train_config, 'loss_config'):
        train_config.loss_config = LossConfig(
            lambda_neg=0.0,
            lambda_pos=0.0
        )

    new_ckpt = {
        "model": old_ckpt["model"],
        "optimizer": old_ckpt["optimizer"],
        "training_stats":
        {
            "steps": old_ckpt["training_stats"]["steps"],
        },
        "configs": {
            "dataset_config": old_ckpt["configs"]["dataset_config"].to_dict(),
            "model_config": old_ckpt["configs"]["model_config"].to_dict(),
            "train_config": old_ckpt["configs"]["train_config"].to_dict(),
        },
        "dataset_feature_stats": old_ckpt["dataset_feature_stats"].to_dict(),
    }
    torch.save(new_ckpt, new_ckpt_path)

# walk through all checkpoints in the source directory
import os
for root, dirs, files in os.walk(SOURCE_DIR):
    for file in files:
        if file.endswith(".pth"):
            old_ckpt_path = os.path.join(root, file)
            relative_path = os.path.relpath(old_ckpt_path, SOURCE_DIR)
            new_ckpt_path = os.path.join(DEST_DIR, relative_path)
            print(f"Migrate {old_ckpt_path} to {new_ckpt_path}")
            os.makedirs(os.path.dirname(new_ckpt_path), exist_ok=True)
            migrate_ckpt(old_ckpt_path, new_ckpt_path)
print("Migration completed.")
