import tqdm
from fastspeech2.predict_batch_prosody_predictor import process

NONE = -1

NEG = 0
NEU = 1
POS = 2

ANGER = 0
DISGUST = 1
FEAR = 2
JOY = 3
NEUTRAL = 4
SADNESS = 5
SURPRISE = 6

CONFIGS = (
# [
#     {
#         "ckpt_path": "output/prosody_predictor_contrastive4/emotion_input_translate2-0.01/ckpt/40000.pth",
#         "output_path": f"output/prosody_predictor_contrastive4/emotion_input_translate2-0.01/pred/val{prefix}_40k.csv",
#         "dataset_config_path": "config/LibriTTS/dataset_emotion.yaml",
#         "model_config_path": "config/LibriTTS/model_emotion_input_translate2.yaml",
#         "data_split_name": "val",
#         "pitch_control": 1.0,
#         "energy_control": 1.0,
#         "duration_control": 1.0,
#         "batch_size": 16,
#         "sentiment_control": NONE,
#         "emotion_control": emotion_control,
#     } for emotion_control, prefix in [(NONE, ""), (ANGER, "_anger"), (DISGUST, "_disgust"), (FEAR, "_fear"), (JOY, "_joy"), (NEUTRAL, "_neutral"), (SADNESS, "_sadness"), (SURPRISE, "_surprise")]
# ]+

# [
#     {
#         "ckpt_path": "output/prosody_predictor_contrastive4/emotion_input_translate2-0.01/ckpt/80000.pth",
#         "output_path": f"output/prosody_predictor_contrastive4/emotion_input_translate2-0.01/pred/val{prefix}_80k.csv",
#         "dataset_config_path": "config/LibriTTS/dataset_emotion.yaml",
#         "model_config_path": "config/LibriTTS/model_emotion_input_translate2.yaml",
#         "data_split_name": "val",
#         "pitch_control": 1.0,
#         "energy_control": 1.0,
#         "duration_control": 1.0,
#         "batch_size": 16,
#         "sentiment_control": NONE,
#         "emotion_control": emotion_control,
#     } for emotion_control, prefix in [(NONE, ""), (ANGER, "_anger"), (DISGUST, "_disgust"), (FEAR, "_fear"), (JOY, "_joy"), (NEUTRAL, "_neutral"), (SADNESS, "_sadness"), (SURPRISE, "_surprise")]
# ]

# [
#     {
#         "ckpt_path": "output/prosody_predictor_contrastive3/emotion_input_translate2-0.01/ckpt/40000.pth",
#         "output_path": f"output/prosody_predictor_contrastive3/emotion_input_translate2-0.01/pred/val{prefix}_40k.csv",
#         "dataset_config_path": "config/LibriTTS/dataset_emotion.yaml",
#         "model_config_path": "config/LibriTTS/model_emotion_input_translate2.yaml",
#         "data_split_name": "val",
#         "pitch_control": 1.0,
#         "energy_control": 1.0,
#         "duration_control": 1.0,
#         "batch_size": 16,
#         "sentiment_control": NONE,
#         "emotion_control": emotion_control,
#     } for emotion_control, prefix in [(NONE, ""), (ANGER, "_anger"), (DISGUST, "_disgust"), (FEAR, "_fear"), (JOY, "_joy"), (NEUTRAL, "_neutral"), (SADNESS, "_sadness"), (SURPRISE, "_surprise")]
# ]
# [
#     {
#         "ckpt_path": "output/prosody_predictor_contrastive3/emotion_input_translate2-0.01/ckpt/80000.pth",
#         "output_path": f"output/prosody_predictor_contrastive3/emotion_input_translate2-0.01/pred/val{prefix}_80k.csv",
#         "dataset_config_path": "config/LibriTTS/dataset_emotion.yaml",
#         "model_config_path": "config/LibriTTS/model_emotion_input.yaml",
#         "data_split_name": "val",
#         "pitch_control": 1.0,
#         "energy_control": 1.0,
#         "duration_control": 1.0,
#         "batch_size": 16,
#         "sentiment_control": NONE,
#         "emotion_control": emotion_control,
#     } for emotion_control, prefix in [(NONE, ""), (ANGER, "_anger"), (DISGUST, "_disgust"), (FEAR, "_fear"), (JOY, "_joy"), (NEUTRAL, "_neutral"), (SADNESS, "_sadness"), (SURPRISE, "_surprise")]
# ]

# [
#     {
#         "ckpt_path": "output/prosody_predictor_contrastive3/emotion_input-0.01/ckpt/40000.pth",
#         "output_path": f"output/prosody_predictor_contrastive3/emotion_input-0.01/pred/val{prefix}_40k.csv",
#         "dataset_config_path": "config/LibriTTS/dataset_emotion.yaml",
#         "model_config_path": "config/LibriTTS/model_emotion_input.yaml",
#         "data_split_name": "val",
#         "pitch_control": 1.0,
#         "energy_control": 1.0,
#         "duration_control": 1.0,
#         "batch_size": 16,
#         "sentiment_control": NONE,
#         "emotion_control": emotion_control,
#     } for emotion_control, prefix in [(NONE, ""), (ANGER, "_anger"), (DISGUST, "_disgust"), (FEAR, "_fear"), (JOY, "_joy"), (NEUTRAL, "_neutral"), (SADNESS, "_sadness"), (SURPRISE, "_surprise")]
# ] + 
# [
#     {
#         "ckpt_path": "output/prosody_predictor_contrastive3/emotion_input-0.01/ckpt/80000.pth",
#         "output_path": f"output/prosody_predictor_contrastive3/emotion_input-0.01/pred/val{prefix}_80k.csv",
#         "dataset_config_path": "config/LibriTTS/dataset_emotion.yaml",
#         "model_config_path": "config/LibriTTS/model_emotion_input.yaml",
#         "data_split_name": "val",
#         "pitch_control": 1.0,
#         "energy_control": 1.0,
#         "duration_control": 1.0,
#         "batch_size": 16,
#         "sentiment_control": NONE,
#         "emotion_control": emotion_control,
#     } for emotion_control, prefix in [(NONE, ""), (ANGER, "_anger"), (DISGUST, "_disgust"), (FEAR, "_fear"), (JOY, "_joy"), (NEUTRAL, "_neutral"), (SADNESS, "_sadness"), (SURPRISE, "_surprise")]
# ]

# [
#     {
#         "ckpt_path": "output/prosody_predictor_contrastive2/emotion_input-0.1/ckpt/40000.pth",
#         "output_path": f"output/prosody_predictor_contrastive2/emotion_input-0.1/pred/val{prefix}_40k.csv",
#         "dataset_config_path": "config/LibriTTS/dataset_emotion.yaml",
#         "model_config_path": "config/LibriTTS/model_emotion_input.yaml",
#         "data_split_name": "val",
#         "pitch_control": 1.0,
#         "energy_control": 1.0,
#         "duration_control": 1.0,
#         "batch_size": 1,
#         "sentiment_control": NONE,
#         "emotion_control": emotion_control,
#     } for emotion_control, prefix in [(NONE, ""), (ANGER, "_anger"), (DISGUST, "_disgust"), (FEAR, "_fear"), (JOY, "_joy"), (NEUTRAL, "_neutral"), (SADNESS, "_sadness"), (SURPRISE, "_surprise")]
# ]
# [
#     {
#         "ckpt_path": "output/prosody_predictor_contrastive2/emotion_input_translate2-0.1/ckpt/40000.pth",
#         "output_path": f"output/prosody_predictor_contrastive2/emotion_input_translate2-0.1/pred/val{prefix}.csv",
#         "dataset_config_path": "config/LibriTTS/dataset_emotion.yaml",
#         "model_config_path": "config/LibriTTS/model_emotion_input_translate2.yaml",
#         "data_split_name": "val",
#         "pitch_control": 1.0,
#         "energy_control": 1.0,
#         "duration_control": 1.0,
#         "batch_size": 16,
#         "sentiment_control": NONE,
#         "emotion_control": emotion_control,
#     } for emotion_control, prefix in [(NONE, ""), (ANGER, "_anger"), (DISGUST, "_disgust"), (FEAR, "_fear"), (JOY, "_joy"), (NEUTRAL, "_neutral"), (SADNESS, "_sadness"), (SURPRISE, "_surprise")]
# ] +
# [
#     {
#         "ckpt_path": "output/prosody_predictor_contrastive2/emotion_input_translate2-0.1/ckpt/80000.pth",
#         "output_path": f"output/prosody_predictor_contrastive2/emotion_input_translate2-0.1/pred/val{prefix}_80k.csv",
#         "dataset_config_path": "config/LibriTTS/dataset_emotion.yaml",
#         "model_config_path": "config/LibriTTS/model_emotion_input_translate2.yaml",
#         "data_split_name": "val",
#         "pitch_control": 1.0,
#         "energy_control": 1.0,
#         "duration_control": 1.0,
#         "batch_size": 16,
#         "sentiment_control": NONE,
#         "emotion_control": emotion_control,
#     } for emotion_control, prefix in [(NONE, ""), (ANGER, "_anger"), (DISGUST, "_disgust"), (FEAR, "_fear"), (JOY, "_joy"), (NEUTRAL, "_neutral"), (SADNESS, "_sadness"), (SURPRISE, "_surprise")]
# ] +
# [
#     {
#         "ckpt_path": "output/prosody_predictor_contrastive2/emotion_input_translate2-0.1/ckpt/20000.pth",
#         "output_path": f"output/prosody_predictor_contrastive2/emotion_input_translate2-0.1/pred/val{prefix}_20k.csv",
#         "dataset_config_path": "config/LibriTTS/dataset_emotion.yaml",
#         "model_config_path": "config/LibriTTS/model_emotion_input_translate2.yaml",
#         "data_split_name": "val",
#         "pitch_control": 1.0,
#         "energy_control": 1.0,
#         "duration_control": 1.0,
#         "batch_size": 16,
#         "sentiment_control": NONE,
#         "emotion_control": emotion_control,
#     } for emotion_control, prefix in [(NONE, ""), (ANGER, "_anger"), (DISGUST, "_disgust"), (FEAR, "_fear"), (JOY, "_joy"), (NEUTRAL, "_neutral"), (SADNESS, "_sadness"), (SURPRISE, "_surprise")]
# ]
)

if __name__ == "__main__":
    for config in tqdm.tqdm(CONFIGS, desc="Processing configurations"):
        process(**config)

    print("Processing completed for all configurations.")
