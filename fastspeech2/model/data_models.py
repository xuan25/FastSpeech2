import torch

class FastSpeech2Output:

    def __init__(self, output: torch.Tensor, postnet_output: torch.Tensor, pitch_predictions: torch.Tensor, energy_predictions: torch.Tensor, log_duration_predictions: torch.Tensor, duration_rounded: torch.Tensor, text_masks: torch.Tensor, mel_masks: torch.Tensor, text_lens: torch.Tensor, mel_lens: torch.Tensor):
        self.output = output
        self.postnet_output = postnet_output
        self.pitch_predictions = pitch_predictions
        self.energy_predictions = energy_predictions
        self.log_duration_predictions = log_duration_predictions
        self.duration_rounded = duration_rounded
        self.text_masks = text_masks
        self.mel_masks = mel_masks
        self.text_lens = text_lens
        self.mel_lens = mel_lens

class FastSpeech2LossResult:
    def __init__(self, total_loss: torch.Tensor, mel_loss: torch.Tensor, postnet_mel_loss: torch.Tensor, pitch_loss: torch.Tensor, energy_loss: torch.Tensor, duration_loss: torch.Tensor):
        self.total_loss = total_loss
        self.mel_loss = mel_loss
        self.postnet_mel_loss = postnet_mel_loss
        self.pitch_loss = pitch_loss
        self.energy_loss = energy_loss
        self.duration_loss = duration_loss

    def __repr__(self):
        return f"LossResult(total_loss={self.total_loss}, mel_loss={self.mel_loss}, postnet_mel_loss={self.postnet_mel_loss}, pitch_loss={self.pitch_loss}, energy_loss={self.energy_loss}, duration_loss={self.duration_loss})"


class ProsodyPredictorOutput:

    def __init__(self, pitch_predictions: torch.Tensor, energy_predictions: torch.Tensor, log_duration_predictions: torch.Tensor, duration_rounded: torch.Tensor, text_masks: torch.Tensor, text_lens: torch.Tensor, frame_mask: torch.Tensor | None = None):
        self.pitch_predictions = pitch_predictions
        self.energy_predictions = energy_predictions
        self.log_duration_predictions = log_duration_predictions
        self.duration_rounded = duration_rounded
        self.text_masks = text_masks
        self.text_lens = text_lens
        self.frame_masks = frame_mask

class ProsodyPredictorLossResult:
    def __init__(self, total_loss: torch.Tensor, pitch_loss: torch.Tensor, energy_loss: torch.Tensor, duration_loss: torch.Tensor):
        self.total_loss = total_loss
        self.pitch_loss = pitch_loss
        self.energy_loss = energy_loss
        self.duration_loss = duration_loss

    def __repr__(self):
        return f"LossResult(total_loss={self.total_loss}, pitch_loss={self.pitch_loss}, energy_loss={self.energy_loss}, duration_loss={self.duration_loss})"

class ProsodyPredictorContrastiveLossResult:
    def __init__(self, 
                 pitch_loss_std: torch.Tensor, energy_loss_std: torch.Tensor, duration_loss_std: torch.Tensor,
                 pitch_loss_neg: torch.Tensor, energy_loss_neg: torch.Tensor, duration_loss_neg: torch.Tensor,
                 pitch_loss_pos: torch.Tensor, energy_loss_pos: torch.Tensor, duration_loss_pos: torch.Tensor,
                 pitch_loss: torch.Tensor, energy_loss: torch.Tensor, duration_loss: torch.Tensor,
                 total_loss: torch.Tensor):
        self.pitch_loss_std = pitch_loss_std
        self.energy_loss_std = energy_loss_std
        self.duration_loss_std = duration_loss_std
        self.pitch_loss_neg = pitch_loss_neg
        self.energy_loss_neg = energy_loss_neg
        self.duration_loss_neg = duration_loss_neg
        self.pitch_loss_pos = pitch_loss_pos
        self.energy_loss_pos = energy_loss_pos
        self.duration_loss_pos = duration_loss_pos
        self.pitch_loss = pitch_loss
        self.energy_loss = energy_loss
        self.duration_loss = duration_loss
        self.total_loss = total_loss

    def __repr__(self):
        return (
            f"ProsodyPredictorLossResult(pitch_loss_std={self.pitch_loss_std}, "
            f"energy_loss_std={self.energy_loss_std}, "
            f"duration_loss_std={self.duration_loss_std}, "
            f"pitch_loss_neg={self.pitch_loss_neg}, "
            f"energy_loss_neg={self.energy_loss_neg}, "
            f"duration_loss_neg={self.duration_loss_neg}, "
            f"pitch_loss_pos={self.pitch_loss_pos}, "
            f"energy_loss_pos={self.energy_loss_pos}, "
            f"duration_loss_pos={self.duration_loss_pos}, "
            f"pitch_loss={self.pitch_loss}, "
            f"energy_loss={self.energy_loss}, "
            f"duration_loss={self.duration_loss}, "
            f"total_loss={self.total_loss})"
        )