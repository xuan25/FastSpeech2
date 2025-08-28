import tqdm
from fastspeech2.predict_batch_prosody_predictor import process


CONFIGS = [


    {
        "ckpt_path": "output/prosody_predictor/sentiment_input_overrideembedding0/ckpt/40000.pth",
        "output_path": "output/prosody_predictor/sentiment_input_overrideembedding0/pred/val.csv",
        "dataset_config_path": "config/LibriTTS/dataset_sentiment.yaml",
        "model_config_path": "config/LibriTTS/model_sentiment_input.yaml",
        "data_split_name": "val",
        "pitch_control": 1.0,
        "energy_control": 1.0,
        "duration_control": 1.0,
        "batch_size": 16
    },
    {
        "ckpt_path": "output/prosody_predictor/sentiment_input_overrideembedding0/ckpt/40000.pth",
        "output_path": "output/prosody_predictor/sentiment_input_overrideembedding0/pred/val_neg.csv",
        "dataset_config_path": "config/LibriTTS/dataset_sentiment_override_all_neg.yaml",
        "model_config_path": "config/LibriTTS/model_sentiment_input.yaml",
        "data_split_name": "val",
        "pitch_control": 1.0,
        "energy_control": 1.0,
        "duration_control": 1.0,
        "batch_size": 16
    },
    {
        "ckpt_path": "output/prosody_predictor/sentiment_input_overrideembedding0/ckpt/40000.pth",
        "output_path": "output/prosody_predictor/sentiment_input_overrideembedding0/pred/val_neu.csv",
        "dataset_config_path": "config/LibriTTS/dataset_sentiment_override_all_neu.yaml",
        "model_config_path": "config/LibriTTS/model_sentiment_input.yaml",
        "data_split_name": "val",
        "pitch_control": 1.0,
        "energy_control": 1.0,
        "duration_control": 1.0,
        "batch_size": 16
    },
    {
        "ckpt_path": "output/prosody_predictor/sentiment_input_overrideembedding0/ckpt/40000.pth",
        "output_path": "output/prosody_predictor/sentiment_input_overrideembedding0/pred/val_pos.csv",
        "dataset_config_path": "config/LibriTTS/dataset_sentiment_override_all_pos.yaml",
        "model_config_path": "config/LibriTTS/model_sentiment_input.yaml",
        "data_split_name": "val",
        "pitch_control": 1.0,
        "energy_control": 1.0,
        "duration_control": 1.0,
        "batch_size": 16
    },


    # {
    #     "ckpt_path": "output/prosody_predictor/sentiment_input/ckpt/40000.pth",
    #     "output_path": "output/prosody_predictor/sentiment_input/pred/val.csv",
    #     "dataset_config_path": "config/LibriTTS/dataset_sentiment.yaml",
    #     "model_config_path": "config/LibriTTS/model_sentiment_input.yaml",
    #     "data_split_name": "val",
    #     "pitch_control": 1.0,
    #     "energy_control": 1.0,
    #     "duration_control": 1.0,
    #     "batch_size": 16
    # },
    # {
    #     "ckpt_path": "output/prosody_predictor/sentiment_input/ckpt/40000.pth",
    #     "output_path": "output/prosody_predictor/sentiment_input/pred/val_neg.csv",
    #     "dataset_config_path": "config/LibriTTS/dataset_sentiment_override_all_neg.yaml",
    #     "model_config_path": "config/LibriTTS/model_sentiment_input.yaml",
    #     "data_split_name": "val",
    #     "pitch_control": 1.0,
    #     "energy_control": 1.0,
    #     "duration_control": 1.0,
    #     "batch_size": 16
    # },
    # {
    #     "ckpt_path": "output/prosody_predictor/sentiment_input/ckpt/40000.pth",
    #     "output_path": "output/prosody_predictor/sentiment_input/pred/val_neu.csv",
    #     "dataset_config_path": "config/LibriTTS/dataset_sentiment_override_all_neu.yaml",
    #     "model_config_path": "config/LibriTTS/model_sentiment_input.yaml",
    #     "data_split_name": "val",
    #     "pitch_control": 1.0,
    #     "energy_control": 1.0,
    #     "duration_control": 1.0,
    #     "batch_size": 16
    # },
    # {
    #     "ckpt_path": "output/prosody_predictor/sentiment_input/ckpt/40000.pth",
    #     "output_path": "output/prosody_predictor/sentiment_input/pred/val_pos.csv",
    #     "dataset_config_path": "config/LibriTTS/dataset_sentiment_override_all_pos.yaml",
    #     "model_config_path": "config/LibriTTS/model_sentiment_input.yaml",
    #     "data_split_name": "val",
    #     "pitch_control": 1.0,
    #     "energy_control": 1.0,
    #     "duration_control": 1.0,
    #     "batch_size": 16
    # },


    # {
    #     "ckpt_path": "output/prosody_predictor/default-B/ckpt/40000.pth",
    #     "output_path": "output/prosody_predictor/default-B/pred/val.csv",
    #     "dataset_config_path": "config/LibriTTS/dataset.yaml",
    #     "model_config_path": "config/LibriTTS/model.yaml",
    #     "data_split_name": "val",
    #     "pitch_control": 1.0,
    #     "energy_control": 1.0,
    #     "duration_control": 1.0,
    #     "batch_size": 16
    # },


    # {
    #     "ckpt_path": "output/prosody_predictor_contrastive/sentiment_input-0-overrideembedding0-B/ckpt/40000.pth",
    #     "output_path": "output/prosody_predictor_contrastive/sentiment_input-0-overrideembedding0-B/pred/val.csv",
    #     "dataset_config_path": "config/LibriTTS/dataset_sentiment.yaml",
    #     "model_config_path": "config/LibriTTS/model_sentiment_input.yaml",
    #     "data_split_name": "val",
    #     "pitch_control": 1.0,
    #     "energy_control": 1.0,
    #     "duration_control": 1.0,
    #     "batch_size": 16
    # },
    # {
    #     "ckpt_path": "output/prosody_predictor_contrastive/sentiment_input-0-overrideembedding0-B/ckpt/40000.pth",
    #     "output_path": "output/prosody_predictor_contrastive/sentiment_input-0-overrideembedding0-B/pred/val_neg.csv",
    #     "dataset_config_path": "config/LibriTTS/dataset_sentiment_override_all_neg.yaml",
    #     "model_config_path": "config/LibriTTS/model_sentiment_input.yaml",
    #     "data_split_name": "val",
    #     "pitch_control": 1.0,
    #     "energy_control": 1.0,
    #     "duration_control": 1.0,
    #     "batch_size": 16
    # },
    # {
    #     "ckpt_path": "output/prosody_predictor_contrastive/sentiment_input-0-overrideembedding0-B/ckpt/40000.pth",
    #     "output_path": "output/prosody_predictor_contrastive/sentiment_input-0-overrideembedding0-B/pred/val_neu.csv",
    #     "dataset_config_path": "config/LibriTTS/dataset_sentiment_override_all_neu.yaml",
    #     "model_config_path": "config/LibriTTS/model_sentiment_input.yaml",
    #     "data_split_name": "val",
    #     "pitch_control": 1.0,
    #     "energy_control": 1.0,
    #     "duration_control": 1.0,
    #     "batch_size": 16
    # },
    # {
    #     "ckpt_path": "output/prosody_predictor_contrastive/sentiment_input-0-overrideembedding0-B/ckpt/40000.pth",
    #     "output_path": "output/prosody_predictor_contrastive/sentiment_input-0-overrideembedding0-B/pred/val_pos.csv",
    #     "dataset_config_path": "config/LibriTTS/dataset_sentiment_override_all_pos.yaml",
    #     "model_config_path": "config/LibriTTS/model_sentiment_input.yaml",
    #     "data_split_name": "val",
    #     "pitch_control": 1.0,
    #     "energy_control": 1.0,
    #     "duration_control": 1.0,
    #     "batch_size": 16
    # },


    # {
    #     "ckpt_path": "output/prosody_predictor_contrastive/sentiment_input-0-overrideembedding0/ckpt/40000.pth",
    #     "output_path": "output/prosody_predictor_contrastive/sentiment_input-0-overrideembedding0/pred/val.csv",
    #     "dataset_config_path": "config/LibriTTS/dataset_sentiment.yaml",
    #     "model_config_path": "config/LibriTTS/model_sentiment_input.yaml",
    #     "data_split_name": "val",
    #     "pitch_control": 1.0,
    #     "energy_control": 1.0,
    #     "duration_control": 1.0,
    #     "batch_size": 16
    # },
    # {
    #     "ckpt_path": "output/prosody_predictor_contrastive/sentiment_input-0-overrideembedding0/ckpt/40000.pth",
    #     "output_path": "output/prosody_predictor_contrastive/sentiment_input-0-overrideembedding0/pred/val_neg.csv",
    #     "dataset_config_path": "config/LibriTTS/dataset_sentiment_override_all_neg.yaml",
    #     "model_config_path": "config/LibriTTS/model_sentiment_input.yaml",
    #     "data_split_name": "val",
    #     "pitch_control": 1.0,
    #     "energy_control": 1.0,
    #     "duration_control": 1.0,
    #     "batch_size": 16
    # },
    # {
    #     "ckpt_path": "output/prosody_predictor_contrastive/sentiment_input-0-overrideembedding0/ckpt/40000.pth",
    #     "output_path": "output/prosody_predictor_contrastive/sentiment_input-0-overrideembedding0/pred/val_neu.csv",
    #     "dataset_config_path": "config/LibriTTS/dataset_sentiment_override_all_neu.yaml",
    #     "model_config_path": "config/LibriTTS/model_sentiment_input.yaml",
    #     "data_split_name": "val",
    #     "pitch_control": 1.0,
    #     "energy_control": 1.0,
    #     "duration_control": 1.0,
    #     "batch_size": 16
    # },
    # {
    #     "ckpt_path": "output/prosody_predictor_contrastive/sentiment_input-0-overrideembedding0/ckpt/40000.pth",
    #     "output_path": "output/prosody_predictor_contrastive/sentiment_input-0-overrideembedding0/pred/val_pos.csv",
    #     "dataset_config_path": "config/LibriTTS/dataset_sentiment_override_all_pos.yaml",
    #     "model_config_path": "config/LibriTTS/model_sentiment_input.yaml",
    #     "data_split_name": "val",
    #     "pitch_control": 1.0,
    #     "energy_control": 1.0,
    #     "duration_control": 1.0,
    #     "batch_size": 16
    # },

    # {
    #     "ckpt_path": "output/prosody_predictor/default/ckpt/40000.pth",
    #     "output_path": "output/prosody_predictor/default/pred/val.csv",
    #     "dataset_config_path": "config/LibriTTS/dataset.yaml",
    #     "model_config_path": "config/LibriTTS/model.yaml",
    #     "data_split_name": "val",
    #     "pitch_control": 1.0,
    #     "energy_control": 1.0,
    #     "duration_control": 1.0,
    #     "batch_size": 16
    # },

    # sentiment_input-0
    # {
    #     "ckpt_path": "output/prosody_predictor_contrastive/sentiment_input-0/ckpt/40000.pth",
    #     "output_path": "output/prosody_predictor_contrastive/sentiment_input-0/pred/val.csv",
    #     "dataset_config_path": "config/LibriTTS/dataset_sentiment.yaml",
    #     "model_config_path": "config/LibriTTS/model_sentiment_input.yaml",
    #     "data_split_name": "val",
    #     "pitch_control": 1.0,
    #     "energy_control": 1.0,
    #     "duration_control": 1.0,
    #     "batch_size": 16
    # },
    # {
    #     "ckpt_path": "output/prosody_predictor_contrastive/sentiment_input-0/ckpt/40000.pth",
    #     "output_path": "output/prosody_predictor_contrastive/sentiment_input-0/pred/val_neg.csv",
    #     "dataset_config_path": "config/LibriTTS/dataset_sentiment_override_all_neg.yaml",
    #     "model_config_path": "config/LibriTTS/model_sentiment_input.yaml",
    #     "data_split_name": "val",
    #     "pitch_control": 1.0,
    #     "energy_control": 1.0,
    #     "duration_control": 1.0,
    #     "batch_size": 16
    # },
    # {
    #     "ckpt_path": "output/prosody_predictor_contrastive/sentiment_input-0/ckpt/40000.pth",
    #     "output_path": "output/prosody_predictor_contrastive/sentiment_input-0/pred/val_neu.csv",
    #     "dataset_config_path": "config/LibriTTS/dataset_sentiment_override_all_neu.yaml",
    #     "model_config_path": "config/LibriTTS/model_sentiment_input.yaml",
    #     "data_split_name": "val",
    #     "pitch_control": 1.0,
    #     "energy_control": 1.0,
    #     "duration_control": 1.0,
    #     "batch_size": 16
    # },
    # {
    #     "ckpt_path": "output/prosody_predictor_contrastive/sentiment_input-0/ckpt/40000.pth",
    #     "output_path": "output/prosody_predictor_contrastive/sentiment_input-0/pred/val_pos.csv",
    #     "dataset_config_path": "config/LibriTTS/dataset_sentiment_override_all_pos.yaml",
    #     "model_config_path": "config/LibriTTS/model_sentiment_input.yaml",
    #     "data_split_name": "val",
    #     "pitch_control": 1.0,
    #     "energy_control": 1.0,
    #     "duration_control": 1.0,
    #     "batch_size": 16
    # },

    # # sentiment_input-0.01
    # {
    #     "ckpt_path": "output/prosody_predictor_contrastive/sentiment_input-0.01/ckpt/40000.pth",
    #     "output_path": "output/prosody_predictor_contrastive/sentiment_input-0.01/pred/val.csv",
    #     "dataset_config_path": "config/LibriTTS/dataset_sentiment.yaml",
    #     "model_config_path": "config/LibriTTS/model_sentiment_input.yaml",
    #     "data_split_name": "val",
    #     "pitch_control": 1.0,
    #     "energy_control": 1.0,
    #     "duration_control": 1.0,
    #     "batch_size": 16
    # },
    # {
    #     "ckpt_path": "output/prosody_predictor_contrastive/sentiment_input-0.01/ckpt/40000.pth",
    #     "output_path": "output/prosody_predictor_contrastive/sentiment_input-0.01/pred/val_neg.csv",
    #     "dataset_config_path": "config/LibriTTS/dataset_sentiment_override_all_neg.yaml",
    #     "model_config_path": "config/LibriTTS/model_sentiment_input.yaml",
    #     "data_split_name": "val",
    #     "pitch_control": 1.0,
    #     "energy_control": 1.0,
    #     "duration_control": 1.0,
    #     "batch_size": 16
    # },
    # {
    #     "ckpt_path": "output/prosody_predictor_contrastive/sentiment_input-0.01/ckpt/40000.pth",
    #     "output_path": "output/prosody_predictor_contrastive/sentiment_input-0.01/pred/val_neu.csv",
    #     "dataset_config_path": "config/LibriTTS/dataset_sentiment_override_all_neu.yaml",
    #     "model_config_path": "config/LibriTTS/model_sentiment_input.yaml",
    #     "data_split_name": "val",
    #     "pitch_control": 1.0,
    #     "energy_control": 1.0,
    #     "duration_control": 1.0,
    #     "batch_size": 16
    # },
    # {
    #     "ckpt_path": "output/prosody_predictor_contrastive/sentiment_input-0.01/ckpt/40000.pth",
    #     "output_path": "output/prosody_predictor_contrastive/sentiment_input-0.01/pred/val_pos.csv",
    #     "dataset_config_path": "config/LibriTTS/dataset_sentiment_override_all_pos.yaml",
    #     "model_config_path": "config/LibriTTS/model_sentiment_input.yaml",
    #     "data_split_name": "val",
    #     "pitch_control": 1.0,
    #     "energy_control": 1.0,
    #     "duration_control": 1.0,
    #     "batch_size": 16
    # },

    # # sentiment_input-0.1
    # {
    #     "ckpt_path": "output/prosody_predictor_contrastive/sentiment_input-0.1/ckpt/40000.pth",
    #     "output_path": "output/prosody_predictor_contrastive/sentiment_input-0.1/pred/val.csv",
    #     "dataset_config_path": "config/LibriTTS/dataset_sentiment.yaml",
    #     "model_config_path": "config/LibriTTS/model_sentiment_input.yaml",
    #     "data_split_name": "val",
    #     "pitch_control": 1.0,
    #     "energy_control": 1.0,
    #     "duration_control": 1.0,
    #     "batch_size": 16
    # },
    # {
    #     "ckpt_path": "output/prosody_predictor_contrastive/sentiment_input-0.1/ckpt/40000.pth",
    #     "output_path": "output/prosody_predictor_contrastive/sentiment_input-0.1/pred/val_neg.csv",
    #     "dataset_config_path": "config/LibriTTS/dataset_sentiment_override_all_neg.yaml",
    #     "model_config_path": "config/LibriTTS/model_sentiment_input.yaml",
    #     "data_split_name": "val",
    #     "pitch_control": 1.0,
    #     "energy_control": 1.0,
    #     "duration_control": 1.0,
    #     "batch_size": 16
    # },
    # {
    #     "ckpt_path": "output/prosody_predictor_contrastive/sentiment_input-0.1/ckpt/40000.pth",
    #     "output_path": "output/prosody_predictor_contrastive/sentiment_input-0.1/pred/val_neu.csv",
    #     "dataset_config_path": "config/LibriTTS/dataset_sentiment_override_all_neu.yaml",
    #     "model_config_path": "config/LibriTTS/model_sentiment_input.yaml",
    #     "data_split_name": "val",
    #     "pitch_control": 1.0,
    #     "energy_control": 1.0,
    #     "duration_control": 1.0,
    #     "batch_size": 16
    # },
    # {
    #     "ckpt_path": "output/prosody_predictor_contrastive/sentiment_input-0.1/ckpt/40000.pth",
    #     "output_path": "output/prosody_predictor_contrastive/sentiment_input-0.1/pred/val_pos.csv",
    #     "dataset_config_path": "config/LibriTTS/dataset_sentiment_override_all_pos.yaml",
    #     "model_config_path": "config/LibriTTS/model_sentiment_input.yaml",
    #     "data_split_name": "val",
    #     "pitch_control": 1.0,
    #     "energy_control": 1.0,
    #     "duration_control": 1.0,
    #     "batch_size": 16
    # },
]

if __name__ == "__main__":
    for config in tqdm.tqdm(CONFIGS, desc="Processing configurations"):
        process(**config)

    print("Processing completed for all configurations.")
