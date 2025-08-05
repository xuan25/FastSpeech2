import argparse
import csv
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
    DatasetPathConfig, 
    DatasetPreprocessingConfig, 
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
from .dataset.data_models import DataBatch, DataBatchTorch, DatasetFeatureStats
from .dataset.dataset import DatasetSplit, TextOnlyDatasetWithSentiment

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
    DatasetFeatureStats
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
    args = parser.parse_args()

    ckpt_path = args.ckpt_path
    output_path = args.output_path
    control_values = args.pitch_control, args.energy_control, args.duration_control
    dataset_config = DatasetConfig.load_from_yaml(args.dataset_config)
    model_config = ModelConfig.load_from_yaml(args.model_config)
    data_split_str = args.data_split
    data_split = DatasetSplit[data_split_str.upper()]

    batch_size = args.batch_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_fs = DatasetFS(dataset_config.path_config.base_dir)

    with DatasetFS(dataset_config.path_config.base_dir) as dataset_fs:
        with dataset_fs.open(dataset_config.path_config.stats_file) as stats_stream, \
            dataset_fs.open(dataset_config.path_config.speaker_map_file) as speaker_stream:
            # Load dataset feature statistics
            dataset_feature_stats = DatasetFeatureStats.from_json(
                stats_stream,
                speaker_stream,
            )

    # Get model
    model: ProsodyPredictor = get_model_infer(ckpt_path, model_config, dataset_config.feature_properties_config, dataset_feature_stats, device)

    # Get dataset
    dataset = TextOnlyDatasetWithSentiment(
        dataset_config.path_config,
        dataset_config.preprocessing_config,
        data_split
    )

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
        csv_writer.writerow(["data_id", "phone_idx", "phone", "pitch", "energy", "duration"])

        for batch in tqdm.tqdm(batchs, desc="[Decoding]", dynamic_ncols=True):
            batch: DataBatch = batch
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
                        csv_writer.writerow([sample_id, j, phone_alphabet, pitch, energy, duration])

if __name__ == "__main__":
    main()

    print("done!")
