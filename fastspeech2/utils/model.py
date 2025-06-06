import os
import json

import torch
import numpy as np

from ..dataset.data_models import DatasetFeatureStats

from ..config import DatasetFeaturePropertiesConfig, ModelConfig, ModelVocoderConfig, TrainOptimizerConfig

from .. import hifigan
from ..model import FastSpeech2

def get_model_infer(ckpt_path, 
              model_config: ModelConfig,
              dataset_feature_properties_config: DatasetFeaturePropertiesConfig,
              dataset_feature_stats: DatasetFeatureStats, 
              device) -> FastSpeech2:

    model = FastSpeech2(model_config, dataset_feature_properties_config, dataset_feature_stats).to(device)
    if ckpt_path:
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"])

    model.eval()
    # model.requires_grad_ = False
    model.requires_grad_(False)
    return model

def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param


def get_vocoder(vocoder_config: ModelVocoderConfig, device) -> object | hifigan.Generator:
    name = vocoder_config.model
    speaker = vocoder_config.speaker

    if name == "MelGAN":
        if speaker == "LJSpeech":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "linda_johnson"
            )
        elif speaker == "universal":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "multi_speaker"
            )
        else:
            raise ValueError("Unknown MelGAN speaker: {}".format(speaker))
        vocoder.mel2wav.eval() # type: ignore
        vocoder.mel2wav.to(device) # type: ignore
    elif name == "HiFi-GAN":
        hifigan_dir = os.path.dirname(hifigan.__file__)
        with open(os.path.join(hifigan_dir, "config", "config.json"), "r", encoding="utf-8") as f:
            vocoder_attr = json.load(f)
        vocoder_attr = hifigan.AttrDict(vocoder_attr)
        vocoder = hifigan.Generator(vocoder_attr)
        if speaker == "LJSpeech":
            ckpt = torch.load(os.path.join(hifigan_dir, "ckpt", "generator_LJSpeech.pth.tar"))
        elif speaker == "universal":
            ckpt = torch.load(os.path.join(hifigan_dir, "ckpt", "generator_universal.pth.tar"))
        else:
            raise ValueError("Unknown HIFIGAN speaker: {}".format(speaker))
        vocoder.load_state_dict(ckpt["generator"])
        vocoder.eval()
        vocoder.remove_weight_norm()
        vocoder.to(device)
    else:
        raise ValueError("Unknown vocoder: {}".format(name))

    return vocoder


def vocoder_infer(mels, vocoder, model_name, max_wav_value, lengths=None):
    with torch.no_grad():
        if model_name == "MelGAN":
            wavs = vocoder.inverse(mels / np.log(10))
        elif model_name == "HiFi-GAN":
            wavs = vocoder(mels).squeeze(1)
        else:
            raise ValueError("Unknown vocoder: {}".format(model_name))

    wavs = (
        wavs.cpu().numpy() * max_wav_value
    ).astype("int16")
    wavs = [wav for wav in wavs]

    for i in range(len(mels)):
        if lengths is not None:
            wavs[i] = wavs[i][: lengths[i]]

    return wavs
