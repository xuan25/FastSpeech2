import csv
import tqdm
from fastspeech2.predict_batch_prosody_predictor import process

LABEL_INDEX_FILE = "data/expresso_style.csv"
# LABEL_INDEX_FILE = "data/expresso_style_no_whisper.csv"

def get_labels_from_index_file(label_index_file):
    with open(label_index_file, 'r') as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames is not None, "CSV file must have header"
        labels = reader.fieldnames[1:]  # Skip the first column which is assumed to be 'basename' or similar
    return labels

label_names = get_labels_from_index_file(LABEL_INDEX_FILE)

# MODEL_CKPT_DIR = "expresso/style/prosody_predictor/embedding_input/contrastive6_0.1"
# MODEL_CKPT_DIR = "expresso/style/prosody_predictor/embedding_input/contrastive5_0.1"
# MODEL_CKPT_DIR = "expresso/style/prosody_predictor/embedding_input/default"
# MODEL_CKPT_DIR = "expresso/style/prosody_predictor/embedding_input/contrastive6_0.1_0.1"
# MODEL_CKPT_DIR = "expresso/style/prosody_predictor/embedding_input/contrastive5_0.1_0.1"

# MODEL_CKPT_DIR = "expresso/style/prosody_predictor/embedding_input/contrastive6_0.1_no_whisper"
# MODEL_CKPT_DIR = "expresso/style/prosody_predictor/embedding_input/contrastive5_0.1_no_whisper"
# MODEL_CKPT_DIR = "expresso/style/prosody_predictor/embedding_input/contrastive6_0.1_0.1_no_whisper"
# MODEL_CKPT_DIR = "expresso/style/prosody_predictor/embedding_input/contrastive5_0.1_0.1_no_whisper"
# MODEL_CKPT_DIR = "expresso/style/prosody_predictor/embedding_input/default_no_whisper"

MODEL_CKPT_DIR = "expresso/style/prosody_predictor/embedding_input/contrastive7_0"
MODEL_CKPT_DIR = "expresso/style/prosody_predictor/embedding_input/contrastive7_0.1"
MODEL_CKPT_DIR = "expresso/style/prosody_predictor/embedding_input/contrastive7_0.1_0.1"

DATA_SPLIT_NAME = "val"

CONFIGS = [
    {
        "ckpt_path": f"output/{MODEL_CKPT_DIR}/ckpt/{model_ckpt_num}.pth",
        "output_path": f"output/{MODEL_CKPT_DIR}/pred/{DATA_SPLIT_NAME}/{label_name}/{model_ckpt_num}.csv",
        "dataset_config_path": "config/dataset_expresso_style.yaml",
        # "dataset_config_path": "config/dataset_expresso_style_no_whisper.yaml",
        "model_config_path": "config/model_label_input.yaml",
        "data_split_name": DATA_SPLIT_NAME,
        "pitch_control": 1.0,
        "energy_control": 1.0,
        "duration_control": 1.0,
        "batch_size": 16,
        "label_control": label_control,
    }
    for model_ckpt_num in [
        i for i in range(1000, 400000, 1000)
    ]
    for label_control, label_name in [
        (-1, "default"),
    ] + [(i, label_names[i]) for i in range(len(label_names))]
]
print(CONFIGS)

if __name__ == "__main__":
    for config in tqdm.tqdm(CONFIGS, desc="Processing configurations", dynamic_ncols=True, leave=False):
        process(**config)

    print("Processing completed for all configurations.")
