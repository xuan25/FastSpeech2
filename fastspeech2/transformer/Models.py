import torch
import torch.nn as nn
import numpy as np

from ..config import ModelTransformerConfig

from . import Constants
from .Layers import FFTBlock
from ..text.symbols import symbols


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    """ Sinusoid position encoding table """

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array(
        [get_posi_angle_vec(pos_i) for pos_i in range(n_position)]
    )

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.0

    return torch.FloatTensor(sinusoid_table)


class Encoder(nn.Module):
    """ Encoder """

    def __init__(self, config_transformer: ModelTransformerConfig, max_seq_len: int, sentiment_mode: str | None, num_sentiments: int | None, emotion_mode: str | None = None, num_emotions: int | None = None):
        super(Encoder, self).__init__()

        n_position = max_seq_len + 1
        n_src_vocab = len(symbols) + 1
        d_word_vec = config_transformer.encoder_hidden
        n_layers = config_transformer.encoder_layer
        n_head = config_transformer.encoder_head
        d_k = d_v = (
            config_transformer.encoder_hidden
            // config_transformer.encoder_head
        )
        d_model = config_transformer.encoder_hidden
        d_inner = config_transformer.conv_filter_size
        kernel_size = config_transformer.conv_kernel_size
        dropout = config_transformer.encoder_dropout

        self.max_seq_len = max_seq_len
        self.d_model = d_model

        self.sentiment_mode = sentiment_mode
        self.num_sentiments = num_sentiments

        self.emotion_mode = emotion_mode
        self.num_emotions = num_emotions

        if sentiment_mode == "input_concat":
            assert num_sentiments is not None, "num_sentiments must be provided for input_concat sentiment mode"
            d_word_vec -= num_sentiments

        self.src_word_emb = nn.Embedding(
            n_src_vocab, d_word_vec, padding_idx=Constants.PAD
        )
        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, d_word_vec).unsqueeze(0),
            requires_grad=False,
        )

        self.layer_stack = nn.ModuleList(
            [
                FFTBlock(
                    d_model, n_head, d_k, d_v, d_inner, kernel_size, dropout=dropout
                )
                for _ in range(n_layers)
            ]
        )

        self.sentiment_emb_input = None
        # self.sentiment_emb_input_from = None
        # self.sentiment_emb_input_to = None

        if sentiment_mode == "input":
            assert num_sentiments is not None, "num_sentiments must be provided for input sentiment mode"
            self.sentiment_emb_input = nn.Embedding(
                num_sentiments,
                d_word_vec,
            )
            # self.sentiment_emb_input = nn.Embedding(
            #     num_sentiments,
            #     d_word_vec,
            #     _weight=torch.zeros(num_sentiments, d_word_vec),
            #     _freeze=True
            # )
            # print(f"Warning: Initialized sentiment embedding with zeros for {num_sentiments} sentiments.")
        if sentiment_mode == "input_translate":
            assert num_sentiments is not None, "num_sentiments must be provided for input_translate sentiment mode"
            # self.sentiment_emb_input_from = nn.Embedding(
            #     num_sentiments,
            #     d_word_vec,
            # )
            # self.sentiment_emb_input_to = nn.Embedding(
            #     num_sentiments,
            #     d_word_vec,
            # )
            num_sentiment_transforms = num_sentiments * num_sentiments
            self.sentiment_emb_input = nn.Embedding(
                num_sentiment_transforms,
                d_word_vec,
            )
            # self.sentiment_emb_input.weight.data.normal_(0, 0.1)
        if sentiment_mode == "input_translate2":
            assert num_sentiments is not None, "num_sentiments must be provided for input_translate2 sentiment mode"
            # self.sentiment_emb_input_from = nn.Embedding(
            #     num_sentiments,
            #     d_word_vec,
            # )
            # self.sentiment_emb_input_to = nn.Embedding(
            #     num_sentiments,
            #     d_word_vec,
            # )
            num_sentiment_transforms = num_sentiments * num_sentiments
            self.sentiment_emb_input = nn.Embedding(
                num_sentiment_transforms,
                d_word_vec,
            )
            # self.sentiment_emb_input.weight.data.normal_(0, 0.1)
        if sentiment_mode == "input_concat":
            assert num_sentiments is not None, "num_sentiments must be provided for input_concat sentiment mode"
            orthogonal_weights = torch.eye(num_sentiments)
            self.sentiment_emb_input = nn.Embedding(
                num_sentiments,
                num_sentiments,
            )
            self.sentiment_emb_input.weight.data = orthogonal_weights

        if emotion_mode == "input":
            assert num_emotions is not None, "num_emotions must be provided for input emotion mode"
            self.emotion_emb_input = nn.Embedding(
                num_emotions,
                d_word_vec,
            )
        if emotion_mode == "input_translate2":
            assert num_emotions is not None, "num_emotions must be provided for input_translate2 emotion mode"
            # num_emotion_transforms = num_emotions * num_emotions
            num_emotion_transforms = num_emotions
            self.emotion_emb_input = nn.Embedding(
                num_emotion_transforms,
                d_word_vec,
            )

    def forward(self, src_seq: torch.Tensor, mask: torch.Tensor, sentiments: torch.Tensor | None = None, sentiments_source: torch.Tensor | None = None, emotions: torch.Tensor | None = None, emotions_source: torch.Tensor | None = None, return_attns=False):

        enc_slf_attn_list = []
        batch_size, max_len = src_seq.shape[0], src_seq.shape[1]

        # -- Prepare masks
        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)

        # -- Forward
        if not self.training and src_seq.shape[1] > self.max_seq_len:
            enc_output = self.src_word_emb(src_seq) + get_sinusoid_encoding_table(
                src_seq.shape[1], self.d_model
            )[: src_seq.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(
                src_seq.device
            )
        else:
            enc_output = self.src_word_emb(src_seq) + self.position_enc[
                :, :max_len, :
            ].expand(batch_size, -1, -1)
        
        if self.sentiment_mode == "input":
            assert self.sentiment_emb_input is not None, "sentiment_emb_input must be provided for input sentiment mode"
            sentiment_emb = self.sentiment_emb_input(
                sentiments
            ).unsqueeze(1).expand(batch_size, max_len, -1)
            enc_output = enc_output + sentiment_emb
        elif self.sentiment_mode == "input_translate":
            # assert self.sentiment_emb_input_from is not None and self.sentiment_emb_input_to is not None, "sentiment_emb_input_from and sentiment_emb_input_to must be provided for input_translate sentiment mode"
            # sentiment_emb_from = self.sentiment_emb_input_from(
            #     sentiments_source
            # ).unsqueeze(1).expand(batch_size, max_len, -1)
            # sentiment_emb_to = self.sentiment_emb_input_to(
            #     sentiments
            # ).unsqueeze(1).expand(batch_size, max_len, -1)
            # enc_output = enc_output + (sentiment_emb_to - sentiment_emb_from)

            assert self.sentiment_emb_input is not None, "sentiment_emb_input must be provided for input_translate sentiment mode"
            assert sentiments_source is not None, "sentiments_source must be provided for input_translate sentiment mode"
            assert sentiments is not None, "sentiments must be provided for input_translate sentiment mode"
            assert self.num_sentiments is not None, "num_sentiments must be provided for input_translate sentiment mode"

            sentiment_transform_id = sentiments_source * self.num_sentiments + sentiments

            sentiment_emb = self.sentiment_emb_input(
                sentiment_transform_id
            ).unsqueeze(1).expand(batch_size, max_len, -1)
            enc_output = enc_output + sentiment_emb

        elif self.sentiment_mode == "input_translate2":
            assert self.sentiment_emb_input is not None, "sentiment_emb_input must be provided for input_translate2 sentiment mode"
            assert sentiments_source is not None, "sentiments_source must be provided for input_translate2 sentiment mode"
            assert sentiments is not None, "sentiments must be provided for input_translate2 sentiment mode"
            assert self.num_sentiments is not None, "num_sentiments must be provided for input_translate2 sentiment mode"
            
            sentiment_emb_source = self.sentiment_emb_input(
                sentiments_source
            )[:, :self.sentiment_emb_input.embedding_dim // 2]
            sentiment_emb_target = self.sentiment_emb_input(
                sentiments
            )[:, self.sentiment_emb_input.embedding_dim // 2:]

            sentiment_emb = torch.cat((sentiment_emb_source, sentiment_emb_target), dim=-1
                                      ).unsqueeze(1).expand(batch_size, max_len, -1)

            enc_output = enc_output + sentiment_emb
            
        elif self.sentiment_mode == "input_concat":
            assert self.sentiment_emb_input is not None, "sentiment_emb_input must be provided for input_concat sentiment mode"
            sentiment_emb = self.sentiment_emb_input(
                sentiments
            ).unsqueeze(1).expand(batch_size, max_len, -1)
            enc_output = torch.cat((enc_output, sentiment_emb), dim=-1)

        if self.emotion_mode == "input":
            assert self.emotion_emb_input is not None, "emotion_emb_input must be provided for input emotion mode"
            emotion_emb = self.emotion_emb_input(
                emotions
            ).unsqueeze(1).expand(batch_size, max_len, -1)
            enc_output = enc_output + emotion_emb

        elif self.emotion_mode == "input_translate2":
            assert self.emotion_emb_input is not None, "emotion_emb_input must be provided for input_translate2 emotion mode"
            assert emotions_source is not None, "emotions_source must be provided for input_translate2 emotion mode"
            assert emotions is not None, "emotions must be provided for input_translate2 emotion mode"
            assert self.num_emotions is not None, "num_emotions must be provided for input_translate2 emotion mode"
            
            emotion_emb_source = self.emotion_emb_input(
                emotions_source
            )[:, :self.emotion_emb_input.embedding_dim // 2]
            emotion_emb_target = self.emotion_emb_input(
                emotions
            )[:, self.emotion_emb_input.embedding_dim // 2:]

            emotion_emb = torch.cat((emotion_emb_source, emotion_emb_target), dim=-1
                                      ).unsqueeze(1).expand(batch_size, max_len, -1)

            enc_output = enc_output + emotion_emb


        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output, mask=mask, slf_attn_mask=slf_attn_mask
            )
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        return enc_output


class Decoder(nn.Module):
    """ Decoder """

    def __init__(self, config_transformer: ModelTransformerConfig, max_seq_len: int):
        super(Decoder, self).__init__()

        n_position = max_seq_len + 1
        d_word_vec = config_transformer.decoder_hidden
        n_layers = config_transformer.decoder_layer
        n_head = config_transformer.decoder_head
        d_k = d_v = (
            config_transformer.decoder_hidden
            // config_transformer.decoder_head
        )
        d_model = config_transformer.decoder_hidden
        d_inner = config_transformer.conv_filter_size
        kernel_size = config_transformer.conv_kernel_size
        dropout = config_transformer.decoder_dropout

        self.max_seq_len = max_seq_len
        self.d_model = d_model

        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, d_word_vec).unsqueeze(0),
            requires_grad=False,
        )

        self.layer_stack = nn.ModuleList(
            [
                FFTBlock(
                    d_model, n_head, d_k, d_v, d_inner, kernel_size, dropout=dropout
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, enc_seq: torch.Tensor, mask: torch.Tensor, return_attns=False):

        dec_slf_attn_list = []
        batch_size, max_len = enc_seq.shape[0], enc_seq.shape[1]

        # -- Forward
        if not self.training and enc_seq.shape[1] > self.max_seq_len:
            # -- Prepare masks
            slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
            dec_output = enc_seq + get_sinusoid_encoding_table(
                enc_seq.shape[1], self.d_model
            )[: enc_seq.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(
                enc_seq.device
            )
        else:
            max_len = min(max_len, self.max_seq_len)

            # -- Prepare masks
            slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
            dec_output = enc_seq[:, :max_len, :] + self.position_enc[
                :, :max_len, :
            ].expand(batch_size, -1, -1)
            mask = mask[:, :max_len]
            slf_attn_mask = slf_attn_mask[:, :, :max_len]

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn = dec_layer(
                dec_output, mask=mask, slf_attn_mask=slf_attn_mask
            )
            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]

        return dec_output, mask
