from typing import Any
import torch
import torch.nn as nn

from ..config import DatasetFeaturePropertiesConfig

from ..dataset.data_models import DataBatchTorch
from .fastspeech2 import DataBatch, FastSpeech2Output
from .data_models import FastSpeech2LossResult, ProsodyPredictorContrastiveLossResult, ProsodyPredictorLossResult, ProsodyPredictorOutput


import numpy as np
import numpy.typing as npt

class ProsodyPredictorWassersteinLoss(nn.Module):
    """ ProsodyPredictorWassersteinLoss """

    def __init__(self, dataset_feature_properties_config: DatasetFeaturePropertiesConfig, dataloader: torch.utils.data.DataLoader):
        super(ProsodyPredictorWassersteinLoss, self).__init__()

        assert dataset_feature_properties_config.num_emotions > 0, "num_emotions must be greater than 0, sentiment not implemented yet"

        num_categories = dataset_feature_properties_config.num_emotions

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

        assert self.pitch_feature_level == "phoneme_level", "ProsodyPredictorWassersteinLoss only supports phoneme level pitch feature"
        assert self.energy_feature_level == "phoneme_level", "ProsodyPredictorWassersteinLoss only supports phoneme level energy feature"


        assert num_categories > 1, "num_categories must be greater than 1"


        self.num_categories = num_categories
        # load anchor points for wasserstein distance calculation

        # anchors_duration: list[list[float]] = []
        # anchors_pitch: list[list[float]] = []
        # anchors_energy: list[list[float]] = []

        # for sample_idx in range(num_categories):
        #     anchors_duration.append([])
        #     anchors_pitch.append([])
        #     anchors_energy.append([])
        
        self.reset()

        labels_gt_li = []
        durations_gt_li = []
        pitches_gt_li = []
        energies_gt_li = []

        for batch, contrastive_mask in dataloader:
            batch: DataBatch = batch
            assert batch.pitches is not None, "pitch_targets is None"
            assert batch.energies is not None, "energy_targets is None"
            assert batch.durations is not None, "duration_targets is None"
            assert batch.emotions is not None, "emotions is None"

            # mask all gt samples as anchors
            batch_mask_gt: npt.NDArray[np.intp] = contrastive_mask == 0
            batch_durations_gt: npt.NDArray[np.float64] = batch.durations[batch_mask_gt]
            batch_pitches_gt: npt.NDArray[np.float64] = batch.pitches[batch_mask_gt]
            batch_energies_gt: npt.NDArray[np.float64] = batch.energies[batch_mask_gt]
            batch_labels_gt: npt.NDArray[np.intp] = batch.emotions[batch_mask_gt]
            
            # process each gt sample in the anchers of this batch
            for sample_idx in range(batch_labels_gt.shape[0]):

                text_len: npt.NDArray[np.intp] = batch.text_lens[sample_idx]

                label: npt.NDArray[np.intp] = batch_labels_gt[sample_idx]
                duration: npt.NDArray[np.float64] = batch_durations_gt[sample_idx][:text_len]
                pitch: npt.NDArray[np.intp] = batch_pitches_gt[sample_idx][:text_len]
                energy: npt.NDArray[np.intp] = batch_energies_gt[sample_idx][:text_len]

                labels_gt_li.extend([label.item()] * text_len)
                durations_gt_li.extend(duration)
                pitches_gt_li.extend(pitch)
                energies_gt_li.extend(energy)

        labels_gt: npt.NDArray[np.int64] = np.array(labels_gt_li)
        durations_gt: npt.NDArray[np.float64] = np.array(durations_gt_li)
        pitches_gt: npt.NDArray[np.float64] = np.array(pitches_gt_li)
        energies_gt: npt.NDArray[np.float64] = np.array(energies_gt_li)

        # convert lists to tensors: list[num_categorise] of tensors[num_samples_flattened]
        self.anchors_duration: list[torch.Tensor] = []
        self.anchors_pitch: list[torch.Tensor] = []
        self.anchors_energy: list[torch.Tensor] = []

        # TODO: expose device as a parameter
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for i in range(num_categories):

            anchor_mask: npt.NDArray[np.bool] = np.array(labels_gt) == i

            anchor_durations_gt = durations_gt[anchor_mask]
            anchor_pitches_gt = pitches_gt[anchor_mask]
            anchor_energies_gt = energies_gt[anchor_mask]

            self.anchors_duration.append(torch.tensor(anchor_durations_gt).to(device))
            self.anchors_pitch.append(torch.tensor(anchor_pitches_gt).to(device))
            self.anchors_energy.append(torch.tensor(anchor_energies_gt).to(device))
    
    def reset(self):
        self.tgt_durations: list[list[torch.Tensor]] = []
        self.tgt_pitches: list[list[torch.Tensor]] = []
        self.tgt_energies:list[list[torch.Tensor]] = []

        for i in range(self.num_categories):
            self.tgt_durations.append([])
            self.tgt_pitches.append([])
            self.tgt_energies.append([])

    def update(self, inputs: DataBatchTorch, predictions: ProsodyPredictorOutput) -> Any:

        labels = inputs.emotions

        assert labels is not None, "labels is None"

        for i in range(labels.shape[0]):
            label = labels[i]
            pred_pitch = predictions.pitch_predictions[i]
            pred_energy = predictions.energy_predictions[i]
            pred_duration = predictions.log_duration_predictions[i]

            self.tgt_durations[label].append(pred_duration)
            self.tgt_pitches[label].append(pred_pitch)
            self.tgt_energies[label].append(pred_energy)

    def negative_distance_nll(self, pred_vals:torch.Tensor, anchors_vals:list[torch.Tensor], tgt_anchor_idx:int) -> torch.Tensor:
        # softmax score
        from scipy.stats import wasserstein_distance

        scores = []
        for i in range(len(anchors_vals)):
            if len(anchors_vals[i]) > 0 and len(pred_vals) > 0:
                anchors_val = anchors_vals[i]
                dist = wasserstein_distance(anchors_val.cpu(), pred_vals.cpu())
                scores.append(-dist)
            else:
                # distances.append(float('inf'))
                raise ValueError("No anchor points for category {}".format(i))
        
        scores = torch.tensor(scores, device=pred_vals.device)

        ndnnl = -torch.log(torch.exp(scores[tgt_anchor_idx]) / torch.sum(torch.exp(scores)))

        return ndnnl

    # def compute_soft_distance(self) -> Any:
    #     from scipy.stats import wasserstein_distance
    #     pitch_loss = 0.0
    #     energy_loss = 0.0
    #     duration_loss = 0.0

    #     for i in range(self.num_categories):
    #         if len(self.anchors_duration[i]) > 0 and len(self.tgt_durations[i]) > 0:
    #             duration_loss += wasserstein_distance(self.anchors_duration[i], self.tgt_durations[i])
    #         if len(self.anchors_pitch[i]) > 0 and len(self.tgt_pitches[i]) > 0:
    #             pitch_loss += wasserstein_distance(self.anchors_pitch[i], self.tgt_pitches[i])
    #         if len(self.anchors_energy[i]) > 0 and len(self.tgt_energies[i]) > 0:
    #             energy_loss += wasserstein_distance(self.anchors_energy[i], self.tgt_energies[i])

    #     return pitch_loss, energy_loss, duration_loss

    def compute_negative_distance_nll(self) -> tuple[float, float, float]:

        ndnll_duration_mean = 0.0
        ndnll_pitch_mean = 0.0
        ndnll_energy_mean = 0.0

        for i in range(self.num_categories):

            pred_durations = torch.concat(self.tgt_durations[i])
            ndnll_duration = self.negative_distance_nll(pred_durations, self.anchors_duration, i)
            ndnll_duration_mean += ndnll_duration.item() / self.num_categories

            pred_pitches = torch.concat(self.tgt_pitches[i])
            ndnll_pitch = self.negative_distance_nll(pred_pitches, self.anchors_pitch, i)
            ndnll_pitch_mean += ndnll_pitch.item() / self.num_categories

            pred_energies = torch.concat(self.tgt_energies[i])
            ndnll_energy = self.negative_distance_nll(pred_energies, self.anchors_energy, i)
            ndnll_energy_mean += ndnll_energy.item() / self.num_categories

        return ndnll_pitch_mean, ndnll_energy_mean, ndnll_duration_mean
        



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

    def __init__(self, dataset_feature_properties_config: DatasetFeaturePropertiesConfig, lambda_neg: float, lambda_pos: float):
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
        contrastive_mask_expanded_text_level_selected = contrastive_mask_expanded_text_level.masked_select(text_masks)
        # expand mask from [batch_size] to [batch_size, mel_len] and then apply masked_select so we can get a frame-level mask
        if frame_masks is not None:
            contrastive_mask_expanded_frame_level = contrastive_mask.unsqueeze(-1).expand(-1, frame_masks.shape[1])
            contrastive_mask_expanded_frame_level_selected = contrastive_mask_expanded_frame_level.masked_select(~frame_masks)
        else:
            contrastive_mask_expanded_frame_level_selected = None

        if self.pitch_feature_level == "phoneme_level":
            pitch_contrastive_mask = contrastive_mask_expanded_text_level_selected
            pitch_predictions = pitch_predictions.masked_select(text_masks)
            pitch_targets = pitch_targets.masked_select(text_masks)
        elif self.pitch_feature_level == "frame_level":
            assert frame_masks is not None, "frame_masks is None for frame level pitch feature"
            pitch_contrastive_mask = contrastive_mask_expanded_frame_level_selected
            pitch_predictions = pitch_predictions.masked_select(~frame_masks)
            pitch_targets = pitch_targets.masked_select(~frame_masks)
        else:
            raise ValueError(f"Invalid pitch feature level: {self.pitch_feature_level}")

        if self.energy_feature_level == "phoneme_level":
            energy_contrastive_mask = contrastive_mask_expanded_text_level_selected
            energy_predictions = energy_predictions.masked_select(text_masks)
            energy_targets = energy_targets.masked_select(text_masks)
        elif self.energy_feature_level == "frame_level":
            assert frame_masks is not None, "frame_masks is None for frame level energy feature"
            energy_contrastive_mask = contrastive_mask_expanded_frame_level_selected
            energy_predictions = energy_predictions.masked_select(~frame_masks)
            energy_targets = energy_targets.masked_select(~frame_masks)
        else:
            raise ValueError(f"Invalid energy feature level: {self.energy_feature_level}")

        duration_contrastive_mask = contrastive_mask_expanded_text_level_selected
        log_duration_predictions = log_duration_predictions.masked_select(text_masks)
        log_duration_targets = log_duration_targets.masked_select(text_masks)

        pitch_loss_raw = self.mse_loss(pitch_predictions, pitch_targets)
        energy_loss_raw = self.mse_loss(energy_predictions, energy_targets)
        duration_loss_raw = self.mse_loss(log_duration_predictions, log_duration_targets)

        # mask 0: std_loss, mask -1: negative contrastive loss, mask 1: positive contrastive loss
        pitch_loss_raw_std = pitch_loss_raw[pitch_contrastive_mask == 0]
        energy_loss_raw_std = energy_loss_raw[energy_contrastive_mask == 0]
        duration_loss_raw_std = duration_loss_raw[duration_contrastive_mask == 0]
        pitch_loss_std = torch.mean(pitch_loss_raw_std, dim=0)
        energy_loss_std = torch.mean(energy_loss_raw_std, dim=0)
        duration_loss_std = torch.mean(duration_loss_raw_std, dim=0)

        pitch_loss_raw_neg = pitch_loss_raw[pitch_contrastive_mask == -1]
        energy_loss_raw_neg = energy_loss_raw[energy_contrastive_mask == -1]
        duration_loss_raw_neg = duration_loss_raw[duration_contrastive_mask == -1]
        pitch_loss_neg = torch.mean(pitch_loss_raw_neg, dim=0)
        energy_loss_neg = torch.mean(energy_loss_raw_neg, dim=0)
        duration_loss_neg = torch.mean(duration_loss_raw_neg, dim=0)

        pitch_loss_raw_pos = pitch_loss_raw[pitch_contrastive_mask == 1]
        energy_loss_raw_pos = energy_loss_raw[energy_contrastive_mask == 1]
        duration_loss_raw_pos = duration_loss_raw[duration_contrastive_mask == 1]
        pitch_loss_pos = torch.mean(pitch_loss_raw_pos, dim=0)
        energy_loss_pos = torch.mean(energy_loss_raw_pos, dim=0)
        duration_loss_pos = torch.mean(duration_loss_raw_pos, dim=0)

        pitch_loss_std = torch.nan_to_num(pitch_loss_std, nan=0.0)
        energy_loss_std = torch.nan_to_num(energy_loss_std, nan=0.0)
        duration_loss_std = torch.nan_to_num(duration_loss_std, nan=0.0)

        pitch_loss_neg = torch.nan_to_num(pitch_loss_neg, nan=0.0)
        energy_loss_neg = torch.nan_to_num(energy_loss_neg, nan=0.0)
        duration_loss_neg = torch.nan_to_num(duration_loss_neg, nan=0.0)

        pitch_loss_pos = torch.nan_to_num(pitch_loss_pos, nan=0.0)
        energy_loss_pos = torch.nan_to_num(energy_loss_pos, nan=0.0)
        duration_loss_pos = torch.nan_to_num(duration_loss_pos, nan=0.0)

        # Contrastive 1
        # pitch_loss = pitch_loss_std + -self.lambda_neg * pitch_loss_neg + self.lambda_pos * pitch_loss_pos
        # energy_loss = energy_loss_std + -self.lambda_neg * energy_loss_neg + self.lambda_pos * energy_loss_pos
        # duration_loss = duration_loss_std + -self.lambda_neg * duration_loss_neg + self.lambda_pos * duration_loss_pos

        # Contrastive 2
        # pitch_loss = pitch_loss_std + (self.lambda_pos * pitch_loss_pos) / (self.lambda_neg * pitch_loss_neg)
        # energy_loss = energy_loss_std + (self.lambda_pos * energy_loss_pos) / (self.lambda_neg * energy_loss_neg)
        # duration_loss = duration_loss_std + (self.lambda_pos * duration_loss_pos) / (self.lambda_neg * duration_loss_neg)

        # Contrastive 3
        # pitch_loss = pitch_loss_std + self.lambda_pos * torch.log(torch.exp(pitch_loss_pos) / torch.exp(pitch_loss_neg))
        # energy_loss = energy_loss_std + self.lambda_pos * torch.log(torch.exp(energy_loss_pos) / torch.exp(energy_loss_neg))
        # duration_loss = duration_loss_std + self.lambda_pos * torch.log(torch.exp(duration_loss_pos) / torch.exp(duration_loss_neg))

        # Contrastive 4
        # pitch_loss = torch.log(torch.exp(pitch_loss_std) / torch.exp(pitch_loss_neg))
        # energy_loss = torch.log(torch.exp(energy_loss_std) / torch.exp(energy_loss_neg))
        # duration_loss = torch.log(torch.exp(duration_loss_std) / torch.exp(duration_loss_neg))

        # Contrastive 5
        pitch_loss = pitch_loss_std + self.lambda_pos * -torch.log(torch.exp(-pitch_loss_pos) / (torch.exp(-pitch_loss_pos) + torch.exp(-pitch_loss_neg)))
        energy_loss = energy_loss_std + self.lambda_pos * -torch.log(torch.exp(-energy_loss_pos) / (torch.exp(-energy_loss_pos) + torch.exp(-energy_loss_neg)))
        duration_loss = duration_loss_std + self.lambda_pos * -torch.log(torch.exp(-duration_loss_pos) / (torch.exp(-duration_loss_pos) + torch.exp(-duration_loss_neg)))



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

    # def forward(self, inputs: DataBatchTorch, predictions: ProsodyPredictorOutput, contrastive_mask: torch.Tensor) -> ProsodyPredictorContrastiveLossResult:
    #     assert inputs.pitches is not None, "pitch_targets is None"
    #     assert inputs.energies is not None, "energy_targets is None"
    #     assert inputs.durations is not None, "duration_targets is None"

    #     pitch_targets = inputs.pitches
    #     energy_targets = inputs.energies
    #     duration_targets = inputs.durations

    #     pitch_predictions = predictions.pitch_predictions
    #     energy_predictions = predictions.energy_predictions
    #     log_duration_predictions = predictions.log_duration_predictions
    #     text_masks = predictions.text_masks
    #     frame_masks = predictions.frame_masks
        
    #     text_masks = ~text_masks
    #     log_duration_targets = torch.log(duration_targets.float() + 1)

    #     log_duration_targets.requires_grad = False
    #     pitch_targets.requires_grad = False
    #     energy_targets.requires_grad = False


    #     assert self.pitch_feature_level == "phoneme_level", "ProsodyPredictorContrastiveLoss only supports phoneme level pitch feature"
    #     assert self.energy_feature_level == "phoneme_level", "ProsodyPredictorContrastiveLoss only supports phoneme level energy feature"

    #     # loss std duration
    #     contrastive_mask_std = contrastive_mask == 0

    #     contrastive_std_text_mask = text_masks[contrastive_mask_std]
    #     contrastive_std_log_duration_targets = log_duration_targets[contrastive_mask_std]
    #     contrastive_std_duration_pred = log_duration_predictions[contrastive_mask_std]

    #     contrastive_std_log_duration_targets_selected = contrastive_std_log_duration_targets.masked_select(contrastive_std_text_mask)
    #     contrastive_std_duration_pred_selected = contrastive_std_duration_pred.masked_select(contrastive_std_text_mask)

    #     contrastive_std_loss_duration = self.mse_loss(contrastive_std_duration_pred_selected, contrastive_std_log_duration_targets_selected)
    #     contrastive_std_loss_duration_mean = torch.mean(contrastive_std_loss_duration, dim=0)

    #     # loss neg duration
    #     contrastive_mask_neg = contrastive_mask == -1

    #     contrastive_neg_text_mask = text_masks[contrastive_mask_neg]
    #     contrastive_neg_log_duration_targets = log_duration_targets[contrastive_mask_neg]
    #     contrastive_neg_duration_pred = log_duration_predictions[contrastive_mask_neg]

    #     contrastive_neg_log_duration_targets_selected = contrastive_neg_log_duration_targets.masked_select(contrastive_neg_text_mask)
    #     contrastive_neg_duration_pred_selected = contrastive_neg_duration_pred.masked_select(contrastive_neg_text_mask)

    #     contrastive_neg_loss_duration = self.mse_loss(contrastive_neg_duration_pred_selected, contrastive_neg_log_duration_targets_selected)
    #     contrastive_neg_loss_duration_mean = torch.mean(contrastive_neg_loss_duration, dim=0)

    #     # loss pos duration
    #     contrastive_mask_pos = contrastive_mask == 1

    #     contrastive_pos_text_mask = text_masks[contrastive_mask_pos]
    #     contrastive_pos_log_duration_targets = log_duration_targets[contrastive_mask_pos]
    #     contrastive_pos_duration_pred = log_duration_predictions[contrastive_mask_pos]

    #     contrastive_pos_log_duration_targets_selected = contrastive_pos_log_duration_targets.masked_select(contrastive_pos_text_mask)
    #     contrastive_pos_duration_pred_selected = contrastive_pos_duration_pred.masked_select(contrastive_pos_text_mask)

    #     contrastive_pos_loss_duration = self.mse_loss(contrastive_pos_duration_pred_selected, contrastive_pos_log_duration_targets_selected)
    #     contrastive_pos_loss_duration_mean = torch.mean(contrastive_pos_loss_duration, dim=0)


    #     # loss std pitch
    #     contrastive_mask_std = contrastive_mask == 0
    #     contrastive_std_text_mask = text_masks[contrastive_mask_std]
    #     contrastive_std_pitch_targets = pitch_targets[contrastive_mask_std]
    #     contrastive_std_pitch_predictions = pitch_predictions[contrastive_mask_std]

    #     contrastive_std_pitch_targets_selected = contrastive_std_pitch_targets.masked_select(contrastive_std_text_mask)
    #     contrastive_std_pitch_predictions_selected = contrastive_std_pitch_predictions.masked_select(contrastive_std_text_mask)

    #     contrastive_std_loss_pitch = self.mse_loss(contrastive_std_pitch_predictions_selected, contrastive_std_pitch_targets_selected)
    #     contrastive_std_loss_pitch_mean = torch.mean(contrastive_std_loss_pitch, dim=0)
        
    #     # loss neg pitch
    #     contrastive_mask_neg = contrastive_mask == -1

    #     contrastive_neg_text_mask = text_masks[contrastive_mask_neg]
    #     contrastive_neg_pitch_targets = pitch_targets[contrastive_mask_neg]
    #     contrastive_neg_pitch_predictions = pitch_predictions[contrastive_mask_neg]

    #     contrastive_neg_pitch_targets_selected = contrastive_neg_pitch_targets.masked_select(contrastive_neg_text_mask)
    #     contrastive_neg_pitch_predictions_selected = contrastive_neg_pitch_predictions.masked_select(contrastive_neg_text_mask)

    #     contrastive_neg_loss_pitch = self.mse_loss(contrastive_neg_pitch_predictions_selected, contrastive_neg_pitch_targets_selected)
    #     contrastive_neg_loss_pitch_mean = torch.mean(contrastive_neg_loss_pitch, dim=0)

    #     # loss pos pitch
    #     contrastive_mask_pos = contrastive_mask == 1

    #     contrastive_pos_text_mask = text_masks[contrastive_mask_pos]
    #     contrastive_pos_pitch_targets = pitch_targets[contrastive_mask_pos]
    #     contrastive_pos_pitch_predictions = pitch_predictions[contrastive_mask_pos]

    #     contrastive_pos_pitch_targets_selected = contrastive_pos_pitch_targets.masked_select(contrastive_pos_text_mask)
    #     contrastive_pos_pitch_predictions_selected = contrastive_pos_pitch_predictions.masked_select(contrastive_pos_text_mask)

    #     contrastive_pos_loss_pitch = self.mse_loss(contrastive_pos_pitch_predictions_selected, contrastive_pos_pitch_targets_selected)
    #     contrastive_pos_loss_pitch_mean = torch.mean(contrastive_pos_loss_pitch, dim=0)

    #     # loss std energy
    #     contrastive_mask_std = contrastive_mask == 0

    #     contrastive_std_text_mask = text_masks[contrastive_mask_std]
    #     contrastive_std_energy_targets = energy_targets[contrastive_mask_std]
    #     contrastive_std_energy_predictions = energy_predictions[contrastive_mask_std]

    #     contrastive_std_energy_targets_selected = contrastive_std_energy_targets.masked_select(contrastive_std_text_mask)
    #     contrastive_std_energy_predictions_selected = contrastive_std_energy_predictions.masked_select(contrastive_std_text_mask)

    #     contrastive_std_loss_energy = self.mse_loss(contrastive_std_energy_predictions_selected, contrastive_std_energy_targets_selected)
    #     contrastive_std_loss_energy_mean = torch.mean(contrastive_std_loss_energy, dim=0)

    #     # loss neg energy
    #     contrastive_mask_neg = contrastive_mask == -1

    #     contrastive_neg_text_mask = text_masks[contrastive_mask_neg]
    #     contrastive_neg_energy_targets = energy_targets[contrastive_mask_neg]
    #     contrastive_neg_energy_predictions = energy_predictions[contrastive_mask_neg]

    #     contrastive_neg_energy_targets_selected = contrastive_neg_energy_targets.masked_select(contrastive_neg_text_mask)
    #     contrastive_neg_energy_predictions_selected = contrastive_neg_energy_predictions.masked_select(contrastive_neg_text_mask)

    #     contrastive_neg_loss_energy = self.mse_loss(contrastive_neg_energy_predictions_selected, contrastive_neg_energy_targets_selected)
    #     contrastive_neg_loss_energy_mean = torch.mean(contrastive_neg_loss_energy, dim=0)

    #     # loss pos energy
    #     contrastive_mask_pos = contrastive_mask == 1

    #     contrastive_pos_text_mask = text_masks[contrastive_mask_pos]
    #     contrastive_pos_energy_targets = energy_targets[contrastive_mask_pos]
    #     contrastive_pos_energy_predictions = energy_predictions[contrastive_mask_pos]

    #     contrastive_pos_energy_targets_selected = contrastive_pos_energy_targets.masked_select(contrastive_pos_text_mask)
    #     contrastive_pos_energy_predictions_selected = contrastive_pos_energy_predictions.masked_select(contrastive_pos_text_mask)

    #     contrastive_pos_loss_energy = self.mse_loss(contrastive_pos_energy_predictions_selected, contrastive_pos_energy_targets_selected)
    #     contrastive_pos_loss_energy_mean = torch.mean(contrastive_pos_loss_energy, dim=0)

    #     # combine losses

    #     pitch_loss = contrastive_std_loss_pitch_mean + -self.lambda_neg * contrastive_neg_loss_pitch_mean + self.lambda_pos * contrastive_pos_loss_pitch_mean
    #     energy_loss = contrastive_std_loss_energy_mean + -self.lambda_neg * contrastive_neg_loss_energy_mean + self.lambda_pos * contrastive_pos_loss_energy_mean
    #     duration_loss = contrastive_std_loss_duration_mean + -self.lambda_neg * contrastive_neg_loss_duration_mean + self.lambda_pos * contrastive_pos_loss_duration_mean

    #     total_loss = (
    #         duration_loss + pitch_loss + energy_loss
    #     )

    #     result = ProsodyPredictorContrastiveLossResult(
    #         pitch_loss_std=contrastive_std_loss_pitch_mean,
    #         energy_loss_std=contrastive_std_loss_energy_mean,
    #         duration_loss_std=contrastive_std_loss_duration_mean,
    #         pitch_loss_neg=contrastive_neg_loss_pitch_mean,
    #         energy_loss_neg=contrastive_neg_loss_energy_mean,
    #         duration_loss_neg=contrastive_neg_loss_duration_mean,
    #         pitch_loss_pos=contrastive_pos_loss_pitch_mean,
    #         energy_loss_pos=contrastive_pos_loss_energy_mean,
    #         duration_loss_pos=contrastive_pos_loss_duration_mean,
    #         pitch_loss=pitch_loss,
    #         energy_loss=energy_loss,
    #         duration_loss=duration_loss,
    #         total_loss=total_loss,
    #     )

    #     return result
    




class ProsodyPredictorContrastiveLoss2(nn.Module):
    """ FastSpeech2 Loss """

    def __init__(self, dataset_feature_properties_config: DatasetFeaturePropertiesConfig, lambda_neg: float, lambda_pos: float):
        super(ProsodyPredictorContrastiveLoss2, self).__init__()

        self.lambda_neg = lambda_neg
        self.lambda_pos = lambda_pos

        self.loss_func = ProsodyPredictorLoss(dataset_feature_properties_config)

    def forward(self, inputs_std: DataBatchTorch, predictions_std: ProsodyPredictorOutput,
                inputs_neg: DataBatchTorch, predictions_neg: ProsodyPredictorOutput,
                inputs_pos: DataBatchTorch, predictions_pos: ProsodyPredictorOutput,
                ) -> ProsodyPredictorContrastiveLossResult:
        
        assert inputs_std.pitches is not None, "pitch_targets is None"
        assert inputs_std.energies is not None, "energy_targets is None"
        assert inputs_std.durations is not None, "duration_targets is None"

        assert inputs_neg.pitches is not None, "neg pitch_targets is None"
        assert inputs_neg.energies is not None, "neg energy_targets is None"
        assert inputs_neg.durations is not None, "neg duration_targets is None"

        assert inputs_pos.pitches is not None, "pos pitch_targets is None"
        assert inputs_pos.energies is not None, "pos energy_targets is None"
        assert inputs_pos.durations is not None, "pos duration_targets is None"

        # Calculate losses for standard inputs
        result_std = self.loss_func(inputs_std, predictions_std)
        pitch_loss_std = result_std.pitch_loss
        energy_loss_std = result_std.energy_loss
        duration_loss_std = result_std.duration_loss

        # Calculate losses for negative inputs
        result_neg = self.loss_func(inputs_neg, predictions_neg)
        pitch_loss_neg = result_neg.pitch_loss
        energy_loss_neg = result_neg.energy_loss
        duration_loss_neg = result_neg.duration_loss

        # Calculate losses for positive inputs
        result_pos = self.loss_func(inputs_pos, predictions_pos)
        pitch_loss_pos = result_pos.pitch_loss
        energy_loss_pos = result_pos.energy_loss
        duration_loss_pos = result_pos.duration_loss

        # Combine losses
        # pitch_loss = pitch_loss_std + -self.lambda_neg * pitch_loss_neg + self.lambda_pos * pitch_loss_pos
        # energy_loss = energy_loss_std + -self.lambda_neg * energy_loss_neg + self.lambda_pos * energy_loss_pos
        # duration_loss = duration_loss_std + -self.lambda_neg * duration_loss_neg + self.lambda_pos * duration_loss_pos

        pitch_loss = pitch_loss_std + (self.lambda_pos * pitch_loss_pos) / (pitch_loss_neg)
        energy_loss = energy_loss_std + (self.lambda_pos * energy_loss_pos) / (energy_loss_neg)
        duration_loss = duration_loss_std + (self.lambda_pos * duration_loss_pos) / (duration_loss_neg)

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
