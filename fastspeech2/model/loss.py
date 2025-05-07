import torch
import torch.nn as nn

from ..config import DatasetFeaturePropertiesConfig

from ..dataset.data_models import DataBatchTorch
from .fastspeech2 import FastSpeech2Output
from .data_models import FastSpeech2LossResult
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
