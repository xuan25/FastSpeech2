import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import DatasetFeaturePropertiesConfig, ModelConfig, ModelGlobalConfig, ModelTransformerConfig


from ..dataset.data_models import DataBatch, DataBatchTorch, DatasetFeatureStats
from ..transformer import Encoder, Decoder, PostNet
from .modules import VarianceAdaptor
from ..utils.tools import get_mask_from_lengths
from .data_models import FastSpeech2Output

class FastSpeech2(nn.Module):
    """ FastSpeech2 """

    def __init__(self, 
                 model_config: ModelConfig,
                 dataset_feature_properties_config: DatasetFeaturePropertiesConfig,
                 dataset_feature_stats: DatasetFeatureStats,
        ):
        super(FastSpeech2, self).__init__()
        self.model_config = model_config

        assert model_config.global_config.sentiment_mode in ["input", "after_encoder", "before_prosodic_predictors", None]

        self.encoder = Encoder(
            model_config.transformer_config, 
            model_config.global_config.max_seq_len,
            model_config.global_config.sentiment_mode,
            dataset_feature_properties_config.num_sentiments if model_config.global_config.sentiment_mode == "input" else None)
        self.variance_adaptor = VarianceAdaptor(
            model_config.transformer_config.encoder_hidden, 
            model_config.variance_embedding_config,
            model_config.variance_predictor_config,
            dataset_feature_properties_config,
            dataset_feature_stats,
            model_config.global_config.sentiment_mode)
        self.decoder = Decoder(model_config.transformer_config, model_config.global_config.max_seq_len)
        self.mel_linear = nn.Linear(
            model_config.transformer_config.decoder_hidden,
            dataset_feature_properties_config.n_mel_channels,
        )
        self.postnet = PostNet()

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
    ):
        text_masks = get_mask_from_lengths(batch.text_lens, batch.text_len_max)
        mel_masks = (
            get_mask_from_lengths(batch.mel_lens, batch.mel_len_max)
            if batch.mel_lens is not None
            else None
        )

        output = self.encoder(batch.texts, text_masks)

        if self.speaker_emb is not None:
            output = output + self.speaker_emb(batch.speakers).unsqueeze(1).expand(
                -1, batch.text_len_max, -1
            )

        if self.sentiment_emb_after_encoder is not None:
            output = output + self.sentiment_emb_after_encoder(batch.sentiments).unsqueeze(1).expand(
                -1, batch.text_len_max, -1
            )

        (
            output,
            pitch_predictions,
            energy_predictions,
            log_duration_predictions,
            duration_rounded,
            mel_lens,
            mel_masks,
        ) = self.variance_adaptor(
            output,
            text_masks,
            mel_masks,
            batch.mel_len_max,
            batch.pitches,
            batch.energies,
            batch.durations,
            p_control,
            e_control,
            d_control,
            batch.sentiments if self.sentiment_emb_after_encoder is not None else None,
        )

        output, mel_masks = self.decoder(output, mel_masks)
        output = self.mel_linear(output)

        postnet_output = self.postnet(output) + output

        output = FastSpeech2Output(
            output=output,
            postnet_output=postnet_output,
            pitch_predictions=pitch_predictions,
            energy_predictions=energy_predictions,
            log_duration_predictions=log_duration_predictions,
            duration_rounded=duration_rounded,
            text_masks=text_masks,
            mel_masks=mel_masks,
            text_lens=batch.text_lens,
            mel_lens=mel_lens,
        )

        return output
