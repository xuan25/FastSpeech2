import argparse
import csv
import functools
import os

import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader

from .model.data_models import ProsodyPredictorOutput
from .model.prosody_predictor import ProsodyPredictor

from .dataset.datasetfs import DatasetFS

from .config import (
    DatasetConfig, 
    DatasetFeaturePropertiesConfig, 
    ModelConfig, 
)
from .dataset.data_models import DataBatch, DataBatchTorch, DatasetFeatureStats
from .dataset.dataset import DatasetSplit, TextOnlyDatasetWithLabel

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
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])

    model.eval()
    # model.requires_grad_ = False
    model.requires_grad_(False)
    return model

dataset_cache = {}
def get_dataset(dataset_config: DatasetConfig, data_split: DatasetSplit) -> TextOnlyDatasetWithLabel:
    key = (dataset_config, data_split)
    if key in dataset_cache:
        dataset = dataset_cache[key]
        if dataset.get_dataset_fs().tar is not None:
            return dataset
    
    dataset = TextOnlyDatasetWithLabel(
        dataset_config.path_config,
        dataset_config.preprocessing_config,
        data_split
    )
    dataset_cache[key] = dataset
    return dataset

def process(ckpt_path: str, output_path: str, dataset_config_path: str, model_config_path: str, data_split_name: str, pitch_control: float, energy_control: float, duration_control: float, batch_size: int, label_control: int):
    control_values = pitch_control, energy_control, duration_control
    dataset_config = DatasetConfig.load_from_yaml(dataset_config_path)
    model_config = ModelConfig.load_from_yaml(model_config_path)
    data_split = DatasetSplit[data_split_name.upper()]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get dataset
    dataset = get_dataset(dataset_config, data_split)
    dataset_fs = dataset.get_dataset_fs()

    with dataset_fs.open(dataset_config.path_config.stats_file) as stats_stream, \
        dataset_fs.open(dataset_config.path_config.speaker_map_file) as speaker_stream:
        # Load dataset feature statistics
        dataset_feature_stats = DatasetFeatureStats.from_json(
            stats_stream,
            speaker_stream,
        )

    # Get model
    model: ProsodyPredictor = get_model_infer(ckpt_path, model_config, dataset_config.feature_properties_config, dataset_feature_stats, device)

    batchs = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=dataset.collate_fn,
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    pitch_control, energy_control, duration_control = control_values

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["data_id", "phone_idx", "phone", "pitch", "energy", "duration", "label", "label_source"])

        for batch in tqdm.tqdm(batchs, desc="[Decoding]", dynamic_ncols=True, leave=False):
            batch: DataBatch = batch

            if label_control >= 0:
                for i, sample in enumerate(batch):
                    sample.label = label_control
                assert batch.labels is not None
                batch.labels = np.array([label_control] * batch.labels.shape[0], dtype=np.int64)

            batch_torch: DataBatchTorch = batch.to_torch(device)
            with torch.no_grad():
                # Forward
                output: ProsodyPredictorOutput = model(
                    batch_torch,
                    p_control=pitch_control,
                    e_control=energy_control,
                    d_control=duration_control
                )

                for i, sample in enumerate(batch):
                    sample_id = sample.data_id
                    for j, phone in enumerate(sample.text):
                        pitch = output.pitch_predictions[i, j].item()
                        energy = output.energy_predictions[i, j].item()
                        duration_log = output.log_duration_predictions[i, j]
                        duration = torch.exp(duration_log).item()  # Convert log duration to actual duration
                        from .text.symbols import symbols
                        phone_alphabet = symbols[phone]
                        csv_writer.writerow([sample_id, j, phone_alphabet, pitch, energy, duration, sample.label, sample.label_source])

    # dataset_fs.close()

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_path", 
        type=str, 
        default="output/default/ckpt/4000.pth"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="output/default/synth_val",
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
        "-m", "--model_config", 
        type=str, 
        # required=True, 
        help="path to model.yaml",
        default="config/LibriTTS/model.yaml",
    )
    parser.add_argument(
        "--data_split",
        type=str,
        default="val",
        choices=["train", "val"],
        help="data split to use for synthesis",
    )
    parser.add_argument(
        "--pitch_control",
        type=float,
        default=1.0,
        help="control the pitch of the whole utterance, larger value for higher pitch",
    )
    parser.add_argument(
        "--energy_control",
        type=float,
        default=1.0,
        help="control the energy of the whole utterance, larger value for larger volume",
    )
    parser.add_argument(
        "--duration_control",
        type=float,
        default=1.0,
        help="control the speed of the whole utterance, larger value for slower speaking rate",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="batch size for synthesis",
    )
    parser.add_argument(
        "--label_control",
        type=int,
        default=-1,
        help="control the label of the whole utterance, -1 for no control",
    )
    args = parser.parse_args()

    process(
        ckpt_path=args.ckpt_path,
        output_path=args.output_path,
        dataset_config_path=args.dataset_config,
        model_config_path=args.model_config,
        data_split_name=args.data_split,
        pitch_control=args.pitch_control,
        energy_control=args.energy_control,
        duration_control=args.duration_control,
        batch_size=args.batch_size,
        label_control=args.label_control,
    )

if __name__ == "__main__":
    main()

    print("done!")
