import argparse
from functools import lru_cache
import torch
from torch.utils.data import DataLoader

from .dataset.datasetfs import DatasetFS

from .config import DatasetConfig, DatasetFeaturePropertiesConfig, LossConfig, ModelConfig, TrainConfig
from .dataset.data_models import DataBatch, DataBatchTorch, DatasetFeatureStats
from .model.data_models import ProsodyPredictorContrastiveLossResult, ProsodyPredictorLossResult
from .model.prosody_predictor import ProsodyPredictor, ProsodyPredictorOutput
from .utils.tools import log_prosody_predictor, log_prosody_predictor_contrastive
from .model.loss import ProsodyPredictorContrastiveLoss, ProsodyPredictorLoss
from .dataset.dataset import DatasetSplit, DatasetWithLabelContrastive, DatasetWithLabelContrastive


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

@lru_cache(maxsize=None)
def get_dataset_loader(
    dataset_config: DatasetConfig,
    batch_size: int,
    device: str | torch.device = "cpu",
) -> tuple[DatasetWithLabelContrastive | DatasetWithLabelContrastive, DataLoader]:
    # Get dataset
    dataset = DatasetWithLabelContrastive(
        dataset_path_config=dataset_config.path_config,
        dataset_preprocessing_config=dataset_config.preprocessing_config,
        split=DatasetSplit.VAL,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
        num_workers=0
    )

    return dataset, loader

def evaluate(model: ProsodyPredictor | torch.nn.DataParallel[ProsodyPredictor], step: int,
             batch_size: int,
             dataset_config: DatasetConfig,
             loss_config: LossConfig,
             logger=None, device: str | torch.device="cpu"):

    # Get dataset
    # dataset = OriginalDatasetWithSentiment(
    #     dataset_path_config=dataset_config.path_config,
    #     dataset_preprocessing_config=dataset_config.preprocessing_config,
    #     split=DatasetSplit.VAL,
    # )

    # loader = DataLoader(
    #     dataset,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     collate_fn=dataset.collate_fn,
    #     num_workers=2
    # )

    dataset, loader = get_dataset_loader(dataset_config, batch_size, device)

    # Get loss function
    loss_func = ProsodyPredictorContrastiveLoss(dataset_config.feature_properties_config, 
                                                loss_config.lambda_neg, loss_config.lambda_pos).to(device)

    # Evaluation
    loss_sums = ProsodyPredictorContrastiveLossResult(
        pitch_loss_std=torch.tensor(0.0, device=device),
        energy_loss_std=torch.tensor(0.0, device=device),
        duration_loss_std=torch.tensor(0.0, device=device),
        pitch_loss_neg=torch.tensor(0.0, device=device),
        energy_loss_neg=torch.tensor(0.0, device=device),
        duration_loss_neg=torch.tensor(0.0, device=device),
        pitch_loss_pos=torch.tensor(0.0, device=device),
        energy_loss_pos=torch.tensor(0.0, device=device),
        duration_loss_pos=torch.tensor(0.0, device=device),
        pitch_loss=torch.tensor(0.0, device=device),
        energy_loss=torch.tensor(0.0, device=device),
        duration_loss=torch.tensor(0.0, device=device),
        total_loss=torch.tensor(0.0, device=device),
    )

    


    from .model.loss import ProsodyPredictorWassersteinLoss
    
    @lru_cache(maxsize=None)
    def get_wasserstein_loss_f(dataset, batch_size, dataset_config) -> ProsodyPredictorWassersteinLoss:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=dataset.collate_fn,
            num_workers=0
        )
        wasserstein_loss_f = ProsodyPredictorWassersteinLoss(
            dataset_config.feature_properties_config,
            loader).to(device)
        return wasserstein_loss_f
    
    wasserstein_loss_f = get_wasserstein_loss_f(dataset, batch_size, dataset_config)
    wasserstein_loss_f.reset()




    
    batch_torch: DataBatchTorch | None = None
    output: ProsodyPredictorOutput | None = None
    for batch, contrastive_mask in loader:
        batch: DataBatch = batch
        batch_torch = batch.to_torch(device)

        contrastive_mask: torch.Tensor = torch.from_numpy(contrastive_mask).to(device)

        with torch.no_grad():
            # Forward
            output = model(batch_torch)

            assert output is not None

            # Cal Loss
            losses: ProsodyPredictorContrastiveLossResult = loss_func(batch_torch, output, contrastive_mask)

            batch_size_std = (contrastive_mask == 0).sum().item()
            # batch_size_neg = (contrastive_mask == -1).sum().item()
            # batch_size_pos = (contrastive_mask == 1).sum().item()

            # Accumulate loss
            loss_sums.pitch_loss_std += losses.pitch_loss_std.item() * batch_size_std
            loss_sums.energy_loss_std += losses.energy_loss_std.item() * batch_size_std
            loss_sums.duration_loss_std += losses.duration_loss_std.item() * batch_size_std
            loss_sums.pitch_loss_neg += losses.pitch_loss_neg.item() * batch_size_std
            loss_sums.energy_loss_neg += losses.energy_loss_neg.item() * batch_size_std
            loss_sums.duration_loss_neg += losses.duration_loss_neg.item() * batch_size_std
            loss_sums.pitch_loss_pos += losses.pitch_loss_pos.item() * batch_size_std
            loss_sums.energy_loss_pos += losses.energy_loss_pos.item() * batch_size_std
            loss_sums.duration_loss_pos += losses.duration_loss_pos.item() * batch_size_std
            loss_sums.pitch_loss += losses.pitch_loss.item() * batch_size_std
            loss_sums.energy_loss += losses.energy_loss.item() * batch_size_std
            loss_sums.duration_loss += losses.duration_loss.item() * batch_size_std
            loss_sums.total_loss += losses.total_loss.item() * batch_size_std




            wasserstein_loss_f.update(batch_torch, output)


    loss_means = ProsodyPredictorContrastiveLossResult(
        pitch_loss_std=loss_sums.pitch_loss_std / len(dataset),
        energy_loss_std=loss_sums.energy_loss_std / len(dataset),
        duration_loss_std=loss_sums.duration_loss_std / len(dataset),
        pitch_loss_neg=loss_sums.pitch_loss_neg / len(dataset),
        energy_loss_neg=loss_sums.energy_loss_neg / len(dataset),
        duration_loss_neg=loss_sums.duration_loss_neg / len(dataset),
        pitch_loss_pos=loss_sums.pitch_loss_pos / len(dataset),
        energy_loss_pos=loss_sums.energy_loss_pos / len(dataset),
        duration_loss_pos=loss_sums.duration_loss_pos / len(dataset),
        pitch_loss=loss_sums.pitch_loss / len(dataset),
        energy_loss=loss_sums.energy_loss / len(dataset),
        duration_loss=loss_sums.duration_loss / len(dataset),
        total_loss=loss_sums.total_loss / len(dataset),
    )




    negative_distance_nll_pitch, negative_distance_nll_energy, negative_distance_nll_duration = wasserstein_loss_f.compute_negative_distance_nll()

    if logger is not None:
        logger.add_scalar("Loss/ndnll_pitch", negative_distance_nll_pitch, step)
        logger.add_scalar("Loss/ndnll_energy", negative_distance_nll_energy, step)
        logger.add_scalar("Loss/ndnll_duration", negative_distance_nll_duration, step)




    message = (
        "Validation Step {}, Total Loss: {:.4f}, Pitch Loss: {:.4f}, Energy Loss: {:.4f}, Duration Loss: {:.4f}".format(
            step,
            loss_means.total_loss.item(),
            loss_means.pitch_loss.item(),
            loss_means.energy_loss.item(),
            loss_means.duration_loss.item(),
        )
    )

    if logger is not None:
        log_prosody_predictor_contrastive(logger, step, losses=loss_means)

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

    with DatasetFS(dataset_config.path_config.base_dir) as dataset_fs:
        with dataset_fs.open(dataset_config.path_config.stats_file) as stats_stream, \
            dataset_fs.open(dataset_config.path_config.speaker_map_file) as speaker_stream:
            # Load dataset feature statistics
            dataset_stats = DatasetFeatureStats.from_json(
                stats_stream,
                speaker_stream,
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
        loss_config=train_config.loss_config,
        device=device
    )

    print(message)

if __name__ == "__main__":
    main()
