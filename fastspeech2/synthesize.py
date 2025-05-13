import argparse
import os
import re
from string import punctuation

import numpy as np
import torch

from fastspeech2.text import text_to_sequence
from g2p_en import G2p

from .config import DatasetConfig, ModelConfig
from .dataset.data_models import DataBatch, DataBatchTorch, DataSample, DatasetFeatureStats
from .utils.model import get_model_infer, get_vocoder
from .utils.tools import synth_samples

def load_lexicon(lex_path):
    lexicon = {}
    with open(lex_path, "r", encoding="utf-8") as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon


def preprocess_english(text, lexicon_path, clearer_names):
    text = text.rstrip(punctuation)
    lexicon = load_lexicon(lexicon_path)

    g2p = G2p()
    phones = []
    words = re.split(r"([,;.\-\?\!\s+])", text)
    for w in words:
        if w.lower() in lexicon:
            phones += lexicon[w.lower()]
        else:
            phones += list(filter(lambda p: p != " ", g2p(w)))
    phones = "{" + "}{".join(phones) + "}"
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    phones = phones.replace("}{", " ")

    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones, clearer_names
        )
    )

    return np.array(sequence)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="raw text to synthesize, for single-sentence mode only",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="en",
        help="language of the text, for single-sentence mode only",
    )
    parser.add_argument(
        "--speaker_id",
        type=int,
        default=0,
        help="speaker ID for multi-speaker synthesis",
    )
    parser.add_argument(
        "--sentiment_id",
        type=int,
        default=0,
        help="sentiment ID for multi-sentiment synthesis",
    )
    parser.add_argument(
        "--ckpt_path", 
        type=str, 
        default="output/default/ckpt/4000.pth.tar"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/default/synth_val",
        help="path to a source file with format like LibriTTS dataset",
    )
    parser.add_argument(
        "-d",
        "--dataset_config",
        type=str,
        # required=True, 
        default="config/LibriTTS/dataset.yaml",
        help="path to dataset.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", 
        type=str, 
        # required=True, 
        help="path to model.yaml",
        default="config/LibriTTS/model.yaml",
    )
    parser.add_argument(
        "--pitch_control",
        type=float,
        default=1.0,
        help="control the pitch of the whole utterance, larger value for higher pitch",
    )
    parser.add_argument(
        "--energy_control",
        type=float,
        default=1.0,
        help="control the energy of the whole utterance, larger value for larger volume",
    )
    parser.add_argument(
        "--duration_control",
        type=float,
        default=1.0,
        help="control the speed of the whole utterance, larger value for slower speaking rate",
    )
    args = parser.parse_args()

    raw_text = args.text
    lang = args.lang
    speaker_id = args.speaker_id
    sentiment_id = args.sentiment_id

    ckpt_path = args.ckpt_path
    output_dir = args.output_dir
    control_values = args.pitch_control, args.energy_control, args.duration_control
    dataset_config = DatasetConfig.load_from_yaml(args.dataset_config)
    model_config = ModelConfig.load_from_yaml(args.model_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_feature_stats = DatasetFeatureStats.from_json(
        dataset_config.path_config.stats_file,
        dataset_config.path_config.speaker_map_file
    )

    # prepare text
    if lang == "en":
        text = np.array([preprocess_english(raw_text, dataset_config.preprocessing_config.lexicon_path, dataset_config.preprocessing_config.text_cleaners)])
    else:
        raise ValueError("Language not supported: {}".format(lang))

    # Get model
    model = get_model_infer(ckpt_path, model_config, dataset_config.feature_properties_config, dataset_feature_stats, device)

    # Load vocoder
    vocoder = get_vocoder(model_config.vocoder_config, device)

    data_sample = DataSample(
        data_id=raw_text,
        speaker=speaker_id,
        text=text,
        raw_text=raw_text,
        mel=None,
        pitch=None,
        energy=None,
        duration=None,
        sentiment=sentiment_id,
    )

    batch = DataBatch([data_sample])   # create a batch with a single sample

    os.makedirs(output_dir, exist_ok=True)

    pitch_control, energy_control, duration_control = control_values
    with torch.no_grad():
        batch: DataBatch = batch
        batch_torch: DataBatchTorch = batch.to_torch(device)
        # Forward
        output = model(
            batch_torch,
            p_control=pitch_control,
            e_control=energy_control,
            d_control=duration_control
        )
        synth_samples(
            batch_torch,
            output,
            vocoder,
            model_config.vocoder_config,
            dataset_config.feature_properties_config,
            dataset_feature_stats,
            output_dir,
        )

if __name__ == "__main__":
    main()

    print("done!")
