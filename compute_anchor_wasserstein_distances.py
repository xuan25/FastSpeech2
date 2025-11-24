import csv
import os
import numpy as np
import scipy.stats
import tqdm

# DATA_LABEL_META = "data/expresso_style.csv"
# ANCHOR_FILE = "output/expresso/style/prosody_predictor_gt/gt/pred/pred_val.csv"
# # OUTPUT_DIR = "output/expresso/style/prosody_predictor_gt/gt/wasserstein_distance/val"

# EXCLUDE_LABELS = ["whisper"]
# OUTPUT_DIR = "output/expresso/style/prosody_predictor_gt/gt/wasserstein_distance_no_whisper/val"

DATA_LABEL_META = "data/expresso_style.csv"
ANCHOR_FILE = "output/expresso/style/prosody_predictor_gt/gt/pred/train/gt/0.csv"
OUTPUT_DIR = "output/expresso/style/prosody_predictor_gt/gt/wasserstein_distance/train"

EXCLUDE_LABELS = []

def get_source_label_map():
    source_label_map = {}
    with open(DATA_LABEL_META, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        assert reader.fieldnames is not None, "CSV file must have header"
        label_names = reader.fieldnames[1:]

        data_ids = []
        probs = []
        for row in reader:
            file_id = row['basename']
            prob = [float(row[label_name]) for label_name in label_names]

            data_ids.append(file_id)
            probs.append(prob)

        label_ids = np.argmax(np.array(probs), axis=1)
        for idx, file_id in enumerate(data_ids):
            label_id = label_ids[idx]
            label_name = label_names[label_id]
            source_label_map[file_id] = label_name
    return source_label_map, label_names

source_label_map, label_names = get_source_label_map()

def get_anchor_data(feature_name, label_col, label_names):
    anchor_data_map = {}

    with open(ANCHOR_FILE, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            label = int(row[label_col])
            label_name = label_names[label]
            feature_value = float(row[feature_name])
            if label_name not in anchor_data_map:
                anchor_data_map[label_name] = []
            anchor_data_map[label_name].append(feature_value)

    return anchor_data_map

pitch_anchor_data = get_anchor_data("pitch", "label", label_names)
energy_anchor_data = get_anchor_data("energy", "label", label_names)
duration_anchor_data = get_anchor_data("duration", "label", label_names)

labels = sorted(pitch_anchor_data.keys())
assert labels == sorted(energy_anchor_data.keys()) == sorted(duration_anchor_data.keys()), "Labels do not match across features."

labels = [label for label in labels if label not in EXCLUDE_LABELS]

# feature_name -> label -> data list
anchor_data_map = {
    "pitch": pitch_anchor_data,
    "energy": energy_anchor_data,
    "duration": duration_anchor_data
}


# loop over features
for feature in ["pitch", "energy", "duration"]:
    output_feature_dir = os.path.join(OUTPUT_DIR, feature)
    os.makedirs(output_feature_dir, exist_ok=True)

    output_file = os.path.join(output_feature_dir, f"0.csv")
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['source_label', 'target_label', 'wasserstein_distance']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # loop over labels
        for source_label in labels:
            pred_data = anchor_data_map[feature][source_label]

            # loop over anchor labels
            for anchor_label in labels:
                anchor_data = anchor_data_map[feature][anchor_label]

                # compute wasserstein distance
                wass_distance = scipy.stats.wasserstein_distance(pred_data, anchor_data)

                # write to csv
                writer.writerow({
                    'source_label': source_label,
                    'target_label': anchor_label,
                    'wasserstein_distance': wass_distance
                })
    print(f"Wrote Wasserstein distances to {output_file}")
