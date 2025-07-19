import torch
import torch.nn as nn

from ..config import DatasetFeaturePropertiesConfig

from ..dataset.data_models import DataBatchTorch
from .fastspeech2 import FastSpeech2Output
from .data_models import FastSpeech2LossResult, ProsodyPredictorContrastiveLossResult, ProsodyPredictorLossResult, ProsodyPredictorOutput

class FastSpeech2Loss(nn.Module):
    """ FastSpeech2 Loss """

    def __init__(self, dataset_feature_properties_config: DatasetFeaturePropertiesConfig):
        super(FastSpeech2Loss, self).__init__()

        self.pitch_feature_level = dataset_feature_properties_config.pitch_feature_level
        self.energy_feature_level = dataset_feature_properties_config.energy_feature_level

        # TODO: refactor this to use a config enum
        assert self.pitch_feature_level in [
            "phoneme_level",
            "frame_level",
        ], f"Invalid pitch feature level: {self.pitch_feature_level}"

        assert self.energy_feature_level in [
            "phoneme_level",
            "frame_level",
        ], f"Invalid energy feature level: {self.energy_feature_level}"

        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()

    def forward(self, inputs: DataBatchTorch, predictions: FastSpeech2Output) -> FastSpeech2LossResult:
        # (
        #     mel_targets,
        #     _,
        #     _,
        #     pitch_targets,
        #     energy_targets,
        #     duration_targets,
        # ) = inputs[6:]

        assert inputs.mels is not None, "mel_targets is None"
        assert inputs.pitches is not None, "pitch_targets is None"
        assert inputs.energies is not None, "energy_targets is None"
        assert inputs.durations is not None, "duration_targets is None"

        mel_targets = inputs.mels
        pitch_targets = inputs.pitches
        energy_targets = inputs.energies
        duration_targets = inputs.durations

        # (
        #     mel_predictions,
        #     postnet_mel_predictions,
        #     pitch_predictions,
        #     energy_predictions,
        #     log_duration_predictions,
        #     _,
        #     src_masks,
        #     mel_masks,
        #     _,
        #     _,
        # ) = predictions

        mel_predictions = predictions.output
        postnet_mel_predictions = predictions.postnet_output
        pitch_predictions = predictions.pitch_predictions
        energy_predictions = predictions.energy_predictions
        log_duration_predictions = predictions.log_duration_predictions
        text_masks = predictions.text_masks
        mel_masks = predictions.mel_masks
        
        
        text_masks = ~text_masks
        mel_masks = ~mel_masks
        log_duration_targets = torch.log(duration_targets.float() + 1)
        mel_targets = mel_targets[:, : mel_masks.shape[1], :]
        mel_masks = mel_masks[:, :mel_masks.shape[1]]

        log_duration_targets.requires_grad = False
        pitch_targets.requires_grad = False
        energy_targets.requires_grad = False
        mel_targets.requires_grad = False

        if self.pitch_feature_level == "phoneme_level":
            pitch_predictions = pitch_predictions.masked_select(text_masks)
            pitch_targets = pitch_targets.masked_select(text_masks)
        elif self.pitch_feature_level == "frame_level":
            pitch_predictions = pitch_predictions.masked_select(mel_masks)
            pitch_targets = pitch_targets.masked_select(mel_masks)

        if self.energy_feature_level == "phoneme_level":
            energy_predictions = energy_predictions.masked_select(text_masks)
            energy_targets = energy_targets.masked_select(text_masks)
        if self.energy_feature_level == "frame_level":
            energy_predictions = energy_predictions.masked_select(mel_masks)
            energy_targets = energy_targets.masked_select(mel_masks)

        log_duration_predictions = log_duration_predictions.masked_select(text_masks)
        log_duration_targets = log_duration_targets.masked_select(text_masks)

        mel_predictions = mel_predictions.masked_select(mel_masks.unsqueeze(-1))
        postnet_mel_predictions = postnet_mel_predictions.masked_select(
            mel_masks.unsqueeze(-1)
        )
        mel_targets = mel_targets.masked_select(mel_masks.unsqueeze(-1))

        mel_loss = self.mae_loss(mel_predictions, mel_targets)
        postnet_mel_loss = self.mae_loss(postnet_mel_predictions, mel_targets)

        pitch_loss = self.mse_loss(pitch_predictions, pitch_targets)
        energy_loss = self.mse_loss(energy_predictions, energy_targets)
        duration_loss = self.mse_loss(log_duration_predictions, log_duration_targets)

        total_loss = (
            mel_loss + postnet_mel_loss + duration_loss + pitch_loss + energy_loss
        )

        result = FastSpeech2LossResult(
            total_loss=total_loss,
            mel_loss=mel_loss,
            postnet_mel_loss=postnet_mel_loss,
            pitch_loss=pitch_loss,
            energy_loss=energy_loss,
            duration_loss=duration_loss,
        )

        return result


class ProsodyPredictorContrastiveLoss(nn.Module):
    """ FastSpeech2 Loss """

    def __init__(self, dataset_feature_properties_config: DatasetFeaturePropertiesConfig, lambda_neg: float = 0.03, lambda_pos: float = 1.0):
        super(ProsodyPredictorContrastiveLoss, self).__init__()

        self.lambda_neg = lambda_neg
        self.lambda_pos = lambda_pos

        self.pitch_feature_level = dataset_feature_properties_config.pitch_feature_level
        self.energy_feature_level = dataset_feature_properties_config.energy_feature_level

        # TODO: refactor this to use a config enum
        assert self.pitch_feature_level in [
            "phoneme_level",
            "frame_level",
        ], f"Invalid pitch feature level: {self.pitch_feature_level}"

        assert self.energy_feature_level in [
            "phoneme_level",
            "frame_level",
        ], f"Invalid energy feature level: {self.energy_feature_level}"

        self.mse_loss = nn.MSELoss(reduction='none')
        self.mae_loss = nn.L1Loss(reduction='none')

    def forward(self, inputs: DataBatchTorch, predictions: ProsodyPredictorOutput, contrastive_mask: torch.Tensor) -> ProsodyPredictorContrastiveLossResult:
        assert inputs.pitches is not None, "pitch_targets is None"
        assert inputs.energies is not None, "energy_targets is None"
        assert inputs.durations is not None, "duration_targets is None"

        pitch_targets = inputs.pitches
        energy_targets = inputs.energies
        duration_targets = inputs.durations

        pitch_predictions = predictions.pitch_predictions
        energy_predictions = predictions.energy_predictions
        log_duration_predictions = predictions.log_duration_predictions
        text_masks = predictions.text_masks
        frame_masks = predictions.frame_masks
        
        text_masks = ~text_masks
        log_duration_targets = torch.log(duration_targets.float() + 1)

        log_duration_targets.requires_grad = False
        pitch_targets.requires_grad = False
        energy_targets.requires_grad = False

        # expand mask from [batch_size] to [batch_size, text_len] and then apply masked_select so wa can get a phone-level mask
        contrastive_mask_expanded_text_level = contrastive_mask.unsqueeze(-1).expand(-1, text_masks.shape[1])
        contrastive_mask_expanded_text_level = contrastive_mask_expanded_text_level.masked_select(text_masks)
        # expand mask from [batch_size] to [batch_size, mel_len] and then apply masked_select so we can get a frame-level mask
        if frame_masks is not None:
            contrastive_mask_expanded_frame_level = contrastive_mask.unsqueeze(-1).expand(-1, frame_masks.shape[1])
            contrastive_mask_expanded_frame_level = contrastive_mask_expanded_frame_level.masked_select(~frame_masks)
        else:
            contrastive_mask_expanded_frame_level = None

        if self.pitch_feature_level == "phoneme_level":
            pitch_contrastive_mask = contrastive_mask_expanded_text_level
            pitch_predictions = pitch_predictions.masked_select(text_masks)
            pitch_targets = pitch_targets.masked_select(text_masks)
        elif self.pitch_feature_level == "frame_level":
            assert frame_masks is not None, "frame_masks is None for frame level pitch feature"
            pitch_contrastive_mask = contrastive_mask_expanded_frame_level 
            pitch_predictions = pitch_predictions.masked_select(~frame_masks)
            pitch_targets = pitch_targets.masked_select(~frame_masks)
        else:
            raise ValueError(f"Invalid pitch feature level: {self.pitch_feature_level}")

        if self.energy_feature_level == "phoneme_level":
            energy_contrastive_mask = contrastive_mask_expanded_text_level
            energy_predictions = energy_predictions.masked_select(text_masks)
            energy_targets = energy_targets.masked_select(text_masks)
        elif self.energy_feature_level == "frame_level":
            assert frame_masks is not None, "frame_masks is None for frame level energy feature"
            energy_contrastive_mask = contrastive_mask_expanded_frame_level
            energy_predictions = energy_predictions.masked_select(~frame_masks)
            energy_targets = energy_targets.masked_select(~frame_masks)
        else:
            raise ValueError(f"Invalid energy feature level: {self.energy_feature_level}")

        duration_contrastive_mask = contrastive_mask_expanded_text_level
        log_duration_predictions = log_duration_predictions.masked_select(text_masks)
        log_duration_targets = log_duration_targets.masked_select(text_masks)

        pitch_loss_raw = self.mse_loss(pitch_predictions, pitch_targets)
        energy_loss_raw = self.mse_loss(energy_predictions, energy_targets)
        duration_loss_raw = self.mse_loss(log_duration_predictions, log_duration_targets)

        # mask 0: std_loss, mask -1: negative contrastive loss, mask 1: positive contrastive loss

        pitch_loss_std = torch.mean(pitch_loss_raw[pitch_contrastive_mask == 0], dim=0)
        energy_loss_std = torch.mean(energy_loss_raw[energy_contrastive_mask == 0], dim=0)
        duration_loss_std = torch.mean(duration_loss_raw[duration_contrastive_mask == 0], dim=0)

        pitch_loss_neg = torch.mean(pitch_loss_raw[pitch_contrastive_mask == -1], dim=0)
        energy_loss_neg = torch.mean(energy_loss_raw[energy_contrastive_mask == -1], dim=0)
        duration_loss_neg = torch.mean(duration_loss_raw[duration_contrastive_mask == -1], dim=0)

        pitch_loss_pos = torch.mean(pitch_loss_raw[pitch_contrastive_mask == 1], dim=0)
        energy_loss_pos = torch.mean(energy_loss_raw[energy_contrastive_mask == 1], dim=0)
        duration_loss_pos = torch.mean(duration_loss_raw[duration_contrastive_mask == 1], dim=0)

        pitch_loss_std = torch.nan_to_num(pitch_loss_std, nan=0.0)
        energy_loss_std = torch.nan_to_num(energy_loss_std, nan=0.0)
        duration_loss_std = torch.nan_to_num(duration_loss_std, nan=0.0)

        pitch_loss_neg = torch.nan_to_num(pitch_loss_neg, nan=0.0)
        energy_loss_neg = torch.nan_to_num(energy_loss_neg, nan=0.0)
        duration_loss_neg = torch.nan_to_num(duration_loss_neg, nan=0.0)

        pitch_loss_pos = torch.nan_to_num(pitch_loss_pos, nan=0.0)
        energy_loss_pos = torch.nan_to_num(energy_loss_pos, nan=0.0)
        duration_loss_pos = torch.nan_to_num(duration_loss_pos, nan=0.0)

        pitch_loss = pitch_loss_std + -self.lambda_neg * pitch_loss_neg + self.lambda_pos * pitch_loss_pos
        energy_loss = energy_loss_std + -self.lambda_neg * energy_loss_neg + self.lambda_pos * energy_loss_pos
        duration_loss = duration_loss_std + -self.lambda_neg * duration_loss_neg + self.lambda_pos * duration_loss_pos

        total_loss = (
            duration_loss + pitch_loss + energy_loss
        )

        result = ProsodyPredictorContrastiveLossResult(
            pitch_loss_std=pitch_loss_std,
            energy_loss_std=energy_loss_std,
            duration_loss_std=duration_loss_std,
            pitch_loss_neg=pitch_loss_neg,
            energy_loss_neg=energy_loss_neg,
            duration_loss_neg=duration_loss_neg,
            pitch_loss_pos=pitch_loss_pos,
            energy_loss_pos=energy_loss_pos,
            duration_loss_pos=duration_loss_pos,
            pitch_loss=pitch_loss,
            energy_loss=energy_loss,
            duration_loss=duration_loss,
            total_loss=total_loss,
        )

        return result
        
class ProsodyPredictorLoss(nn.Module):
    """ FastSpeech2 Loss """

    def __init__(self, dataset_feature_properties_config: DatasetFeaturePropertiesConfig):
        super(ProsodyPredictorLoss, self).__init__()

        self.pitch_feature_level = dataset_feature_properties_config.pitch_feature_level
        self.energy_feature_level = dataset_feature_properties_config.energy_feature_level

        # TODO: refactor this to use a config enum
        assert self.pitch_feature_level in [
            "phoneme_level",
            "frame_level",
        ], f"Invalid pitch feature level: {self.pitch_feature_level}"

        assert self.energy_feature_level in [
            "phoneme_level",
            "frame_level",
        ], f"Invalid energy feature level: {self.energy_feature_level}"

        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()

    def forward(self, inputs: DataBatchTorch, predictions: ProsodyPredictorOutput) -> ProsodyPredictorLossResult:
        assert inputs.pitches is not None, "pitch_targets is None"
        assert inputs.energies is not None, "energy_targets is None"
        assert inputs.durations is not None, "duration_targets is None"

        pitch_targets = inputs.pitches
        energy_targets = inputs.energies
        duration_targets = inputs.durations

        pitch_predictions = predictions.pitch_predictions
        energy_predictions = predictions.energy_predictions
        log_duration_predictions = predictions.log_duration_predictions
        text_masks = predictions.text_masks
        frame_masks = predictions.frame_masks
        
        text_masks = ~text_masks
        log_duration_targets = torch.log(duration_targets.float() + 1)

        log_duration_targets.requires_grad = False
        pitch_targets.requires_grad = False
        energy_targets.requires_grad = False

        if self.pitch_feature_level == "phoneme_level":
            pitch_predictions = pitch_predictions.masked_select(text_masks)
            pitch_targets = pitch_targets.masked_select(text_masks)
        elif self.pitch_feature_level == "frame_level":
            assert frame_masks is not None, "frame_masks is None for frame level pitch feature"
            pitch_predictions = pitch_predictions.masked_select(~frame_masks)
            pitch_targets = pitch_targets.masked_select(~frame_masks)

        if self.energy_feature_level == "phoneme_level":
            energy_predictions = energy_predictions.masked_select(text_masks)
            energy_targets = energy_targets.masked_select(text_masks)
        elif self.energy_feature_level == "frame_level":
            assert frame_masks is not None, "frame_masks is None for frame level energy feature"
            energy_predictions = energy_predictions.masked_select(~frame_masks)
            energy_targets = energy_targets.masked_select(~frame_masks)

        log_duration_predictions = log_duration_predictions.masked_select(text_masks)
        log_duration_targets = log_duration_targets.masked_select(text_masks)

        pitch_loss = self.mse_loss(pitch_predictions, pitch_targets)
        energy_loss = self.mse_loss(energy_predictions, energy_targets)
        duration_loss = self.mse_loss(log_duration_predictions, log_duration_targets)

        total_loss = (
            duration_loss + pitch_loss + energy_loss
        )

        result = ProsodyPredictorLossResult(
            total_loss=total_loss,
            pitch_loss=pitch_loss,
            energy_loss=energy_loss,
            duration_loss=duration_loss,
        )

        return result
