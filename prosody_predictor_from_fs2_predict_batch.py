import tqdm
from fastspeech2.predict_batch_prosody_predictor_from_fs2 import process


CONFIGS = [

    {
        "ckpt_path": "output/fastspeech2/default/ckpt/50000.pth",
        "output_path": "output/fastspeech2/default/pred/val.csv",
        "dataset_config_path": "config/LibriTTS/dataset.yaml",
        "model_config_path": "config/LibriTTS/model.yaml",
        "data_split_name": "val",
        "pitch_control": 1.0,
        "energy_control": 1.0,
        "duration_control": 1.0,
        "batch_size": 16
    },
]

if __name__ == "__main__":
    for config in tqdm.tqdm(CONFIGS, desc="Processing configurations"):
        process(**config)

    print("Processing completed for all configurations.")
