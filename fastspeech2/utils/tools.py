import os
import json

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
from scipy.io import wavfile
from matplotlib import pyplot as plt

from ..config import DatasetFeaturePropertiesConfig, ModelVocoderConfig

from ..dataset.data_models import DataBatchTorch, DatasetFeatureStats
from ..model.data_models import FastSpeech2Output, FastSpeech2LossResult


matplotlib.use("Agg")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# def to_device(data, device):
#     if len(data) == 12:
#         (
#             ids,
#             raw_texts,
#             speakers,
#             texts,
#             src_lens,
#             max_src_len,
#             mels,
#             mel_lens,
#             max_mel_len,
#             pitches,
#             energies,
#             durations,
#         ) = data

#         speakers = torch.from_numpy(speakers).long().to(device)
#         texts = torch.from_numpy(texts).long().to(device)
#         src_lens = torch.from_numpy(src_lens).to(device)
#         mels = torch.from_numpy(mels).float().to(device)
#         mel_lens = torch.from_numpy(mel_lens).to(device)
#         pitches = torch.from_numpy(pitches).float().to(device)
#         energies = torch.from_numpy(energies).to(device)
#         durations = torch.from_numpy(durations).long().to(device)

#         return (
#             ids,
#             raw_texts,
#             speakers,
#             texts,
#             src_lens,
#             max_src_len,
#             mels,
#             mel_lens,
#             max_mel_len,
#             pitches,
#             energies,
#             durations,
#         )

#     if len(data) == 6:
#         (ids, raw_texts, speakers, texts, src_lens, max_src_len) = data

#         speakers = torch.from_numpy(speakers).long().to(device)
#         texts = torch.from_numpy(texts).long().to(device)
#         src_lens = torch.from_numpy(src_lens).to(device)

#         return (ids, raw_texts, speakers, texts, src_lens, max_src_len)


def log(
    logger, step=None, losses: FastSpeech2LossResult | None=None, fig=None, audio=None, sampling_rate=22050, tag=""
):
    if losses is not None:
        logger.add_scalar("Loss/total_loss", losses.total_loss, step)
        logger.add_scalar("Loss/mel_loss", losses.mel_loss, step)
        logger.add_scalar("Loss/mel_postnet_loss", losses.postnet_mel_loss, step)
        logger.add_scalar("Loss/pitch_loss", losses.pitch_loss, step)
        logger.add_scalar("Loss/energy_loss", losses.energy_loss, step)
        logger.add_scalar("Loss/duration_loss", losses.duration_loss, step)

    if fig is not None:
        logger.add_figure(tag, fig)

    if audio is not None:
        logger.add_audio(
            tag,
            audio / max(abs(audio)),
            sample_rate=sampling_rate,
        )


def get_mask_from_lengths(lengths, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(device)
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

    return mask


def expand(values, durations):
    out = list()
    for value, d in zip(values, durations):
        out += [value] * max(0, int(d))
    return np.array(out)


def synth_one_sample(targets: DataBatchTorch, predictions: FastSpeech2Output, vocoder, vocoder_config: ModelVocoderConfig, stats: DatasetFeatureStats, feature_properties_config: DatasetFeaturePropertiesConfig):

    assert targets.mels is not None, "targets.mels is None"
    assert targets.durations is not None, "targets.durations is None"
    assert targets.pitches is not None, "targets.pitches is None"
    assert targets.energies is not None, "targets.energies is None"

    basename = targets.data_ids[0]
    src_len = predictions.text_lens[0].item()
    mel_len = predictions.mel_lens[0].item()
    mel_target = targets.mels[0, :mel_len].detach().transpose(0, 1)
    mel_prediction = predictions.postnet_output[0, :mel_len].detach().transpose(0, 1)
    duration = targets.durations[0, :src_len].detach().cpu().numpy()
    if feature_properties_config.pitch_feature_level == "phoneme_level":       # TODO: need to be refactored
        pitch = targets.pitches[0, :src_len].detach().cpu().numpy()
        pitch = expand(pitch, duration)
    else:
        pitch = targets.pitches[0, :mel_len].detach().cpu().numpy()
    if feature_properties_config.energy_feature_level == "phoneme_level":      # TODO: need to be refactored
        energy = targets.energies[0, :src_len].detach().cpu().numpy()
        energy = expand(energy, duration)
    else:
        energy = targets.energies[0, :mel_len].detach().cpu().numpy()

    fig = plot_mel(
        [
            (mel_prediction.cpu().numpy(), pitch, energy),
            (mel_target.cpu().numpy(), pitch, energy),
        ],
        stats,
        ["Synthetized Spectrogram", "Ground-Truth Spectrogram"],
    )

    if vocoder is not None:
        from .model import vocoder_infer

        wav_reconstruction = vocoder_infer(
            mel_target.unsqueeze(0),
            vocoder,
            vocoder_config.model,
            feature_properties_config.max_wav_value,
        )[0]
        wav_prediction = vocoder_infer(
            mel_prediction.unsqueeze(0),
            vocoder,
            vocoder_config.model,
            feature_properties_config.max_wav_value,
        )[0]
    else:
        wav_reconstruction = wav_prediction = None

    return fig, wav_reconstruction, wav_prediction, basename


def synth_samples(targets: DataBatchTorch, predictions: FastSpeech2Output, vocoder, vocoder_config: ModelVocoderConfig, feature_properties_config: DatasetFeaturePropertiesConfig, stats: DatasetFeatureStats, output_dir: str):
    for i in range(targets.batch_size):
        basename = targets.data_ids[i]
        src_len = predictions.text_lens[i].item()
        mel_len = predictions.mel_lens[i].item()
        mel_prediction = predictions.postnet_output[i, :mel_len].detach().transpose(0, 1)
        duration = predictions.duration_rounded[i, :src_len].detach().cpu().numpy()
        if feature_properties_config.pitch_feature_level == "phoneme_level":       # TODO: need to be refactored
            pitch = predictions.pitch_predictions[i, :src_len].detach().cpu().numpy()
            pitch = expand(pitch, duration)
        else:
            pitch = predictions.pitch_predictions[i, :mel_len].detach().cpu().numpy()
        if feature_properties_config.energy_feature_level == "phoneme_level":      # TODO: need to be refactored
            energy = predictions.energy_predictions[i, :src_len].detach().cpu().numpy()
            energy = expand(energy, duration)
        else:
            energy = predictions.energy_predictions[i, :mel_len].detach().cpu().numpy()

        fig = plot_mel(
            [
                (mel_prediction.cpu().numpy(), pitch, energy),
            ],
            stats,
            ["Synthetized Spectrogram"],
        )
        plt.savefig(os.path.join(output_dir, "{}.pdf".format(basename)))
        plt.close()

    from .model import vocoder_infer

    mel_predictions = predictions.postnet_output.transpose(1, 2)
    lengths = predictions.mel_lens * feature_properties_config.stft_hop_length
    wav_predictions = vocoder_infer(
        mel_predictions, vocoder, vocoder_config.model, feature_properties_config.max_wav_value, lengths=lengths
    )

    sampling_rate = feature_properties_config.sampling_rate
    for wav, basename in zip(wav_predictions, targets.data_ids):
        wavfile.write(os.path.join(output_dir, "{}.wav".format(basename)), sampling_rate, wav)


def plot_mel(data, stats: DatasetFeatureStats, titles):
    fig, axes = plt.subplots(len(data), 1, squeeze=False)
    if titles is None:
        titles = [None for i in range(len(data))]
    pitch_min, pitch_max, pitch_mean, pitch_std, energy_min, energy_max = stats.pitch_min, stats.pitch_max, stats.pitch_mean, stats.pitch_std, stats.energy_min, stats.energy_max
    pitch_min = pitch_min * pitch_std + pitch_mean
    pitch_max = pitch_max * pitch_std + pitch_mean

    def add_axis(fig, old_ax):
        ax = fig.add_axes(old_ax.get_position(), anchor="W")
        ax.set_facecolor("None")
        return ax

    for i in range(len(data)):
        mel, pitch, energy = data[i]
        pitch = pitch * pitch_std + pitch_mean
        axes[i][0].imshow(mel, origin="lower")
        axes[i][0].set_aspect(2.5, adjustable="box")
        axes[i][0].set_ylim(0, mel.shape[0])
        axes[i][0].set_title(titles[i], fontsize="medium")
        axes[i][0].tick_params(labelsize="x-small", left=False, labelleft=False)
        axes[i][0].set_anchor("W")

        ax1 = add_axis(fig, axes[i][0])
        ax1.plot(pitch, color="tomato")
        ax1.set_xlim(0, mel.shape[1])
        ax1.set_ylim(0, pitch_max)
        ax1.set_ylabel("F0", color="tomato")
        ax1.tick_params(
            labelsize="x-small", colors="tomato", bottom=False, labelbottom=False
        )

        ax2 = add_axis(fig, axes[i][0])
        ax2.plot(energy, color="darkviolet")
        ax2.set_xlim(0, mel.shape[1])
        ax2.set_ylim(energy_min, energy_max)
        ax2.set_ylabel("Energy", color="darkviolet")
        ax2.yaxis.set_label_position("right")
        ax2.tick_params(
            labelsize="x-small",
            colors="darkviolet",
            bottom=False,
            labelbottom=False,
            left=False,
            labelleft=False,
            right=True,
            labelright=True,
        )

    return fig


# def pad_1D(inputs, PAD=0):
#     def pad_data(x, length, PAD):
#         x_padded = np.pad(
#             x, (0, length - x.shape[0]), mode="constant", constant_values=PAD
#         )
#         return x_padded

#     max_len = max((len(x) for x in inputs))
#     padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])

#     return padded


# def pad_2D(inputs, maxlen=None):
#     def pad(x, max_len):
#         PAD = 0
#         if np.shape(x)[0] > max_len:
#             raise ValueError("not max_len")

#         s = np.shape(x)[1]
#         x_padded = np.pad(
#             x, (0, max_len - np.shape(x)[0]), mode="constant", constant_values=PAD
#         )
#         return x_padded[:, :s]

#     if maxlen:
#         output = np.stack([pad(x, maxlen) for x in inputs])
#     else:
#         max_len = max(np.shape(x)[0] for x in inputs)
#         output = np.stack([pad(x, max_len) for x in inputs])

#     return output


# def pad(input_ele, mel_max_length=None):
#     if mel_max_length:
#         max_len = mel_max_length
#     else:
#         max_len = max([input_ele[i].size(0) for i in range(len(input_ele))])

#     out_list = list()
#     for i, batch in enumerate(input_ele):
#         if len(batch.shape) == 1:
#             one_batch_padded = F.pad(
#                 batch, (0, max_len - batch.size(0)), "constant", 0.0
#             )
#         elif len(batch.shape) == 2:
#             one_batch_padded = F.pad(
#                 batch, (0, 0, 0, max_len - batch.size(0)), "constant", 0.0
#             )
#         out_list.append(one_batch_padded)
#     out_padded = torch.stack(out_list)
#     return out_padded
