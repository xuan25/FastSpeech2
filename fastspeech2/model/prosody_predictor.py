import torch.nn as nn

from ..model.data_models import ProsodyPredictorOutput

from ..utils.tools import get_mask_from_lengths

from ..dataset.data_models import DataBatchTorch, DatasetFeatureStats
from .modules import VarianceAdaptor
from ..transformer.Models import Encoder

from ..config import DatasetFeaturePropertiesConfig, ModelConfig

class ProsodyPredictor(nn.Module):
    """ Prosody Predictor """

    def __init__(self, 
                 model_config: ModelConfig,
                 dataset_feature_properties_config: DatasetFeaturePropertiesConfig,
                 dataset_feature_stats: DatasetFeatureStats,
        ):
        super(ProsodyPredictor, self).__init__()
        self.model_config = model_config

        assert model_config.global_config.sentiment_mode in ["input", "input_concat", "input_translate", "input_translate2", "after_encoder", "before_prosodic_predictors", None], \
            f"Invalid sentiment mode: {model_config.global_config.sentiment_mode}. Expected one of ['input', 'input_concat', 'input_translate', 'input_translate2', 'after_encoder', 'before_prosodic_predictors', None]."
        assert model_config.global_config.emotion_mode in ["input", "input_translate2", None], \
            f"Invalid emotion mode: {model_config.global_config.emotion_mode}. Expected one of ['input', 'input_translate2', None]."

        self.encoder = Encoder(
            model_config.transformer_config,
            model_config.global_config.max_seq_len, 
            model_config.global_config.sentiment_mode,
            dataset_feature_properties_config.num_sentiments if model_config.global_config.sentiment_mode in ["input", "input_translate", "input_translate2", "input_concat"] else None,
            model_config.global_config.emotion_mode,
            dataset_feature_properties_config.num_emotions if model_config.global_config.emotion_mode in ["input", "input_translate2"] else None
        )
        self.variance_adaptor = VarianceAdaptor(
            model_config.transformer_config.encoder_hidden, 
            model_config.variance_embedding_config,
            model_config.variance_predictor_config,
            dataset_feature_properties_config,
            dataset_feature_stats,
            model_config.global_config.sentiment_mode)

        self.speaker_emb = None
        if model_config.global_config.multi_speaker:
            self.speaker_emb = nn.Embedding(
                dataset_feature_stats.n_speakers,
                model_config.transformer_config.encoder_hidden,
            )

        self.sentiment_emb_after_encoder = None
        if model_config.global_config.sentiment_mode == "after_encoder":
            self.sentiment_emb_after_encoder = nn.Embedding(
                dataset_feature_properties_config.num_sentiments,
                model_config.transformer_config.encoder_hidden,
            )

    def forward(
        self,
        batch: DataBatchTorch,
        p_control: float = 1.0,
        e_control: float = 1.0,
        d_control: float = 1.0,
    ) -> ProsodyPredictorOutput:
        text_masks = get_mask_from_lengths(batch.text_lens, batch.text_len_max)
        frame_masks = (
            get_mask_from_lengths(batch.mel_lens, batch.mel_len_max)
            if batch.mel_lens is not None
            else None
        )

        output = self.encoder(
            batch.texts,
            text_masks,
            batch.sentiments,
            batch.sentiments_source,
            batch.emotions,
            batch.emotions_source,
        )

        if self.speaker_emb is not None:
            output = output + self.speaker_emb(batch.speakers).unsqueeze(1).expand(
                -1, batch.text_len_max, -1
            )

        if self.sentiment_emb_after_encoder is not None:
            sentiment_embedded = self.sentiment_emb_after_encoder(batch.sentiments).unsqueeze(1).expand(
                -1, batch.text_len_max, -1
            )
            output = output + sentiment_embedded

        (
            output,
            pitch_predictions,
            energy_predictions,
            log_duration_predictions,
            duration_rounded,
            frame_lens,
            frame_masks,
        ) = self.variance_adaptor(
            output,
            text_masks,
            frame_masks,
            batch.mel_len_max,
            batch.pitches,
            batch.energies,
            batch.durations,
            p_control,
            e_control,
            d_control,
            batch.sentiments,
        )

        output = ProsodyPredictorOutput(
            pitch_predictions=pitch_predictions,
            energy_predictions=energy_predictions,
            log_duration_predictions=log_duration_predictions,
            duration_rounded=duration_rounded,
            text_masks=text_masks,
            text_lens=batch.text_lens,
            frame_mask=frame_masks,
        )

        return output

