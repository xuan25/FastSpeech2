import argparse
import torch
from torch.utils.data import DataLoader

from .config import DatasetConfig, DatasetFeaturePropertiesConfig, ModelConfig, ModelVocoderConfig, TrainConfig
from .dataset.data_models import DataBatch, DataBatchTorch, DatasetFeatureStats
from .model.data_models import FastSpeech2LossResult
from .model.fastspeech2 import FastSpeech2Output
from .utils.model import get_model_infer
from .utils.tools import log, synth_one_sample
from .model import FastSpeech2Loss
from .dataset.dataset import DatasetSplit, OriginalDatasetWithSentiment

def evaluate(model, step,
             batch_size,
             dataset_config: DatasetConfig,
             vocoder_config: ModelVocoderConfig,
             stats: DatasetFeatureStats,
             feature_properties_config: DatasetFeaturePropertiesConfig, 
             logger=None, vocoder=None, device="cpu"):

    # Get dataset
    dataset = OriginalDatasetWithSentiment(
        dataset_path_config=dataset_config.path_config,
        dataset_preprocessing_config=dataset_config.preprocessing_config,
        split=DatasetSplit.VAL,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
        num_workers=8
    )

    # Get loss function
    loss_func = FastSpeech2Loss(dataset_config.feature_properties_config).to(device)

    # Evaluation
    loss_sums = FastSpeech2LossResult(
        total_loss=torch.tensor(0.0),
        mel_loss=torch.tensor(0.0),
        postnet_mel_loss=torch.tensor(0.0),
        pitch_loss=torch.tensor(0.0),
        energy_loss=torch.tensor(0.0),
        duration_loss=torch.tensor(0.0),
    )
    
    for batch in loader:
        batch: DataBatch = batch
        batch_torch: DataBatchTorch = batch.to_torch(device)
        with torch.no_grad():
            # Forward
            output: FastSpeech2Output = model(batch_torch)

            # Cal Loss
            losses: FastSpeech2LossResult = loss_func(batch_torch, output)

            loss_sums.total_loss += losses.total_loss.item() * batch_torch.batch_size
            loss_sums.mel_loss += losses.mel_loss.item() * batch_torch.batch_size
            loss_sums.postnet_mel_loss += losses.postnet_mel_loss.item() * batch_torch.batch_size
            loss_sums.pitch_loss += losses.pitch_loss.item() * batch_torch.batch_size
            loss_sums.energy_loss += losses.energy_loss.item() * batch_torch.batch_size
            loss_sums.duration_loss += losses.duration_loss.item() * batch_torch.batch_size


    loss_means = FastSpeech2LossResult(
        total_loss=loss_sums.total_loss / len(dataset),
        mel_loss=loss_sums.mel_loss / len(dataset),
        postnet_mel_loss=loss_sums.postnet_mel_loss / len(dataset),
        pitch_loss=loss_sums.pitch_loss / len(dataset),
        energy_loss=loss_sums.energy_loss / len(dataset),
        duration_loss=loss_sums.duration_loss / len(dataset),
    )

    message = (
        "Validation Step {}, Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Pitch Loss: {:.4f}, Energy Loss: {:.4f}, Duration Loss: {:.4f}".format(
            step,
            loss_means.total_loss.item(),
            loss_means.mel_loss.item(),
            loss_means.postnet_mel_loss.item(),
            loss_means.pitch_loss.item(),
            loss_means.energy_loss.item(),
            loss_means.duration_loss.item(),
        )
    )

    if logger is not None:
        fig, wav_reconstruction, wav_prediction, tag = synth_one_sample(
            batch_torch,
            output,
            vocoder,
            vocoder_config,
            stats,
            feature_properties_config,
        )

        log(logger, step, losses=loss_means)
        log(
            logger,
            fig=fig,
            tag="Validation/step_{}_{}".format(step, tag),
        )
        sampling_rate = feature_properties_config.sampling_rate
        log(
            logger,
            audio=wav_reconstruction,
            sampling_rate=sampling_rate,
            tag="Validation/step_{}_{}_reconstructed".format(step, tag),
        )
        log(
            logger,
            audio=wav_prediction,
            sampling_rate=sampling_rate,
            tag="Validation/step_{}_{}_synthesized".format(step, tag),
        )

    return message

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_path", 
        # required=True, 
        type=int, 
        default=None
    )
    parser.add_argument(
        "-d",
        "--dataset_config",
        type=str,
        required=True,
        help="path to dataset.yaml",
    )
    parser.add_argument(
        "-m", "--model_config",
        type=str,
        required=True,
        help="path to model.yaml"
    )
    parser.add_argument(
        "-t",
        "--train_config",
        type=str,
        required=True,
        help="path to train.yaml"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = args.ckpt_path
    dataset_config = DatasetConfig.load_from_yaml(args.dataset_config)
    model_config = ModelConfig.load_from_yaml(args.model_config)
    train_config = TrainConfig.load_from_yaml(args.train_config)

    dataset_stats = DatasetFeatureStats.from_json(
        dataset_config.path_config.stats_file,
        dataset_config.path_config.speaker_map_file,
    )

    model = get_model_infer(
        ckpt_path,
        model_config,
        dataset_config.feature_properties_config,
        dataset_stats,
        device=device
    )

    message = evaluate(
        model,
        ckpt_path,
        train_config.step_config.batch_size,
        dataset_config,
        model_config.vocoder_config,
        dataset_stats,
        dataset_config.feature_properties_config,
        device=device
    )

    print(message)

if __name__ == "__main__":
    main()
