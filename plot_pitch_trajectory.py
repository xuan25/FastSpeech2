# csv input format:
# data_id,phone_idx,phone,pitch,energy,duration,label

import csv

DATASET_FILE = "data/expresso.tar"

# GT_FEATRUES_FILE = "output/expresso/style/prosody_predictor_gt/gt/pred/train/0.csv"
# PREDICTED_FEATURES_FILE = "output/expresso/style/prosody_predictor/embedding_input/contrastive7_0.1_0.1/pred/train/default/361000.csv"
# BASELINE_FEATURES_FILE = "output/expresso/style/prosody_predictor/embedding_input/contrastive7_0.1_0.1/pred/train/default/361000.csv"

# FEATURE = [
#     ("Ground Truth", "output/expresso/style/prosody_predictor_gt/gt/pred/train/0.csv"),
#     ("No Embedding", "output/expresso/style/prosody_predictor/default/default/pred/train/default/361000.csv"),
#     ("Embedding", "output/expresso/style/prosody_predictor/embedding_input/default/pred/train/default/361000.csv"),
#     ("Contrastive", "output/expresso/style/prosody_predictor/embedding_input/contrastive7_0.1_0.1/pred/train/default/361000.csv"),
# ]

# FEATURE = [
#     ("Ground Truth", "output/expresso/style/prosody_predictor_gt/gt/pred/train/0.csv"),
#     ("No Embedding", "output/expresso/style/prosody_predictor/default/default/pred/train/default/41000.csv"),
#     ("Embedding", "output/expresso/style/prosody_predictor/embedding_input/default/pred/train/default/41000.csv"),
#     ("Contrastive", "output/expresso/style/prosody_predictor/embedding_input/contrastive7_0.1_0.1/pred/train/default/41000.csv"),
# ]

# FEATURE = [
#     ("Ground Truth", "output/expresso/style/prosody_predictor_gt/gt/pred/train/0.csv"),
#     ("No Embedding", "output/expresso/style/prosody_predictor/default/default/pred_teacher_forcing/train/default/11000.csv"),
#     ("Embedding", "output/expresso/style/prosody_predictor/embedding_input/default/pred_teacher_forcing/train/default/9000.csv"),
#     ("Contrastive", "output/expresso/style/prosody_predictor/embedding_input/contrastive7_0.1_0.1/pred_teacher_forcing/train/default/5000.csv"),
# ]

FEATURE = [
    ("Ground Truth", "output/expresso/style/prosody_predictor_gt/gt/pred/test/0.csv"),
    ("No Embedding", "output/expresso/style/prosody_predictor/default/default/pred_teacher_forcing/test/default/11000.csv"),
    ("Embedding", "output/expresso/style/prosody_predictor/embedding_input/default/pred_teacher_forcing/test/default/9000.csv"),
    ("Contrastive", "output/expresso/style/prosody_predictor/embedding_input/contrastive7_0.1_0.1/pred_teacher_forcing/test/default/5000.csv"),
]

# DURATION_FEATURE_OVERRIED = None

# for teacher forcing predicitons, we stick with the GT duration. the predicted duration is not used during the expansion.
DURATION_FEATURE_OVERRIED = "output/expresso/style/prosody_predictor_gt/gt/pred/test/0.csv"

# TGT_DATA = "train/0002/ex03_happy_00036"
TGT_DATA = "test/0000/ex01_happy_00363"
TGT_FEATURE = "pitch"

def load_features(data_id, feature_file):
    FEATURE_COLS = ["pitch", "energy", "duration"]

    PHONE_IDX_COL = "phone_idx"

    features = {
        feature: [] for feature in FEATURE_COLS
    }

    with open(feature_file, "r") as f:
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            if row["data_id"] != data_id:
                continue
            phone_idx = int(row[PHONE_IDX_COL])
            for feature in FEATURE_COLS:
                features[feature].append((float(row[feature]), phone_idx))

        # sort by phone_idx
        for feature in FEATURE_COLS:
            features[feature].sort(key=lambda x: x[1])
            features[feature] = [x[0] for x in features[feature]]

    return features

def extend_features_by_duration(features, durations):
    expanded_features = []
    for feature, duration in zip(features, durations):
        expanded_features.extend([feature] * int(duration))
    return expanded_features

# gt_features = load_features(TGT_DATA, GT_FEATRUES_FILE)
# predicted_features = load_features(TGT_DATA, PREDICTED_FEATURES_FILE)

# assert len(gt_features["duration"]) != 0, "No ground truth features found for the target data."
# assert len(predicted_features["duration"]) != 0, "No predicted features found for the target data."

# assert len(gt_features["pitch"]) == len(gt_features["duration"]), "Mismatch in length of pitch and duration for ground truth."
# assert len(predicted_features["pitch"]) == len(predicted_features["duration"]), "Mismatch in length of pitch and duration for predicted features."

# gt_durations = gt_features["duration"]
# gt_pitch = extend_features_by_duration(gt_features["pitch"], gt_durations)

# predicted_durations = predicted_features["duration"]
# predicted_pitch = extend_features_by_duration(predicted_features["pitch"], predicted_durations)

if DURATION_FEATURE_OVERRIED is not None:
    duration_features = load_features(TGT_DATA, DURATION_FEATURE_OVERRIED)["duration"]

feature_dict = {}
for feature_name, feature_file in FEATURE:
    features = load_features(TGT_DATA, feature_file)
    assert len(features["duration"]) != 0, f"No features found for {feature_name}."
    assert len(features["pitch"]) == len(features["duration"]), f"Mismatch in length of pitch and duration for {feature_name}."

    if DURATION_FEATURE_OVERRIED is not None:
        durations = duration_features
    else:
        durations = features["duration"]

    pitch = extend_features_by_duration(features["pitch"], durations)
    feature_dict[feature_name] = pitch





from fastspeech2.dataset.datasetfs import DatasetFS

dataset_fs = DatasetFS(DATASET_FILE)

import numpy as np
import io

with dataset_fs.open(f"mel/{TGT_DATA}.npy") as f:
    with io.BytesIO(f.read()) as buffer:
        mel: np.ndarray = np.load(buffer)
        mel = mel.astype(np.float32)
        print(f"Mel spectrogram shape: {mel.shape}")

assert len(feature_dict["Ground Truth"]) == mel.shape[0], "Mismatch in length of pitch and number of frames in mel spectrogram."

import numpy as np
import matplotlib.pyplot as plt

fig, ax_mel = plt.subplots(figsize=(12, 6))

# mel.T shape: [freq_bins, time_frames]
n_mel_bins, n_frames = mel.T.shape
n_frames_features = [len(feature_dict[feature_name]) for feature_name in feature_dict]
max_frames = max(n_frames, max(n_frames_features))


# explicitly specify the x/y range for imshow to avoid default pixel coordinate issues
im = ax_mel.imshow(
    mel.T,
    aspect="auto",
    origin="lower",
    alpha=0.5,
    cmap="viridis",
    extent=[0, n_frames - 1, 0, n_mel_bins - 1],
)

ax_mel.set_xlabel("Time (frames)")
ax_mel.set_ylabel("Mel bin")
ax_mel.set_ylim(0, n_mel_bins - 1)
ax_mel.set_xlim(0, max_frames - 1)

# the secondary y-axis for plotting normalized pitch
ax_pitch = ax_mel.twinx()

# assume the length of pitch matches the number of mel frames, we can directly plot against the same x-axis
for feature_name in feature_dict:
    x = np.arange(len(feature_dict[feature_name]))
    ax_pitch.plot(x, feature_dict[feature_name], label=feature_name)

ax_pitch.set_ylabel("Pitch (Normalized)")

plt.title(f"Pitch Trajectory for {TGT_DATA}")

# merge legends from both axes
lines1, labels1 = ax_mel.get_legend_handles_labels()
lines2, labels2 = ax_pitch.get_legend_handles_labels()
ax_pitch.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

ax_mel.grid(False)

plt.savefig(f"pitch_trajectory_{TGT_DATA.replace('/', '_')}.png", dpi=150, bbox_inches="tight")