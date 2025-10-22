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

    def __init__(self, config_transformer: ModelTransformerConfig, max_seq_len: int, label_embedding_mode: str | None, num_label_categories: int | None):
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

        self.label_embedding_mode = label_embedding_mode
        self.num_label_categories = num_label_categories

        if label_embedding_mode == "input_concat":
            assert num_label_categories is not None, "num_label_categories must be provided for input_concat label_embedding_mode"
            d_word_vec -= num_label_categories

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

        if label_embedding_mode == "input":
            assert num_label_categories is not None, "num_label_categories must be provided for input label_embedding_mode"
            self.label_emb_input = nn.Embedding(
                num_label_categories,
                d_word_vec,
            )
            # self.label_emb_input = nn.Embedding(
            #     num_label_categories,
            #     d_word_vec,
            #     _weight=torch.zeros(num_label_categories, d_word_vec),
            #     _freeze=True
            # )
            # print(f"Warning: Initialized label embedding with zeros for {num_label_categories} labels.")
        if label_embedding_mode == "input_translate":
            assert num_label_categories is not None, "num_label_categories must be provided for input_translate label_embedding_mode"
            # self.label_emb_input_from = nn.Embedding(
            #     num_label_categories,
            #     d_word_vec,
            # )
            # self.label_emb_input_to = nn.Embedding(
            #     num_label_categories,
            #     d_word_vec,
            # )
            num_label_transforms = num_label_categories * num_label_categories
            self.label_emb_input = nn.Embedding(
                num_label_transforms,
                d_word_vec,
            )
            # self.label_emb_input.weight.data.normal_(0, 0.1)
        if label_embedding_mode == "input_translate2":
            assert num_label_categories is not None, "num_label_categories must be provided for input_translate2 label_embedding_mode"
            # self.label_emb_input_from = nn.Embedding(
            #     num_label_categories,
            #     d_word_vec,
            # )
            # self.label_emb_input_to = nn.Embedding(
            #     num_label_categories,
            #     d_word_vec,
            # )
            num_label_transforms = num_label_categories * num_label_categories
            self.label_emb_input = nn.Embedding(
                num_label_transforms,
                d_word_vec,
            )
            # self.label_emb_input.weight.data.normal_(0, 0.1)
        if label_embedding_mode == "input_concat":
            assert num_label_categories is not None, "num_label_categories must be provided for input_concat label_embedding_mode"
            orthogonal_weights = torch.eye(num_label_categories)
            self.label_emb_input = nn.Embedding(
                num_label_categories,
                num_label_categories,
            )
            self.label_emb_input.weight.data = orthogonal_weights

    def forward(self, src_seq: torch.Tensor, mask: torch.Tensor, labels: torch.Tensor | None = None, labels_source: torch.Tensor | None = None, return_attns=False):

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
        
        if self.label_embedding_mode == "input":
            assert self.label_emb_input is not None, "label_emb_input must be provided for input label_embedding_mode"
            label_emb = self.label_emb_input(
                labels
            ).unsqueeze(1).expand(batch_size, max_len, -1)
            enc_output = enc_output + label_emb
        elif self.label_embedding_mode == "input_translate":
            # assert self.label_emb_input_from is not None and self.label_emb_input_to is not None, "label_emb_input_from and label_emb_input_to must be provided for input_translate label_embedding_mode"
            # label_emb_from = self.label_emb_input_from(
            #     labels_source
            # ).unsqueeze(1).expand(batch_size, max_len, -1)
            # label_emb_to = self.label_emb_input_to(
            #     labels
            # ).unsqueeze(1).expand(batch_size, max_len, -1)
            # enc_output = enc_output + (label_emb_to - label_emb_from)

            assert self.label_emb_input is not None, "label_emb_input must be provided for input_translate label_embedding_mode"
            assert labels_source is not None, "labels_source must be provided for input_translate label_embedding_mode"
            assert labels is not None, "labels must be provided for input_translate label_embedding_mode"
            assert self.num_label_categories is not None, "num_label_categories must be provided for input_translate label_embedding_mode"

            label_transform_id = labels_source * self.num_label_categories + labels

            label_emb = self.label_emb_input(
                label_transform_id
            ).unsqueeze(1).expand(batch_size, max_len, -1)
            enc_output = enc_output + label_emb

        elif self.label_embedding_mode == "input_translate2":
            assert self.label_emb_input is not None, "label_emb_input must be provided for input_translate2 label_embedding_mode"
            assert labels_source is not None, "labels_source must be provided for input_translate2 label_embedding_mode"
            assert labels is not None, "labels must be provided for input_translate2 label_embedding_mode"
            assert self.num_label_categories is not None, "num_label_categories must be provided for input_translate2 label_embedding_mode"

            label_emb_source = self.label_emb_input(
                labels_source
            )[:, :self.label_emb_input.embedding_dim // 2]
            label_emb_target = self.label_emb_input(
                labels
            )[:, self.label_emb_input.embedding_dim // 2:]

            label_emb = torch.cat((label_emb_source, label_emb_target), dim=-1
                                   ).unsqueeze(1).expand(batch_size, max_len, -1)

            enc_output = enc_output + label_emb
            
        elif self.label_embedding_mode == "input_concat":
            assert self.label_emb_input is not None, "label_emb_input must be provided for input_concat label_embedding_mode"
            label_emb = self.label_emb_input(
                labels
            ).unsqueeze(1).expand(batch_size, max_len, -1)
            enc_output = torch.cat((enc_output, label_emb), dim=-1)

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
