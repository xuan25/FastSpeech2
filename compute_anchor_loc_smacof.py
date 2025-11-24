
import csv
import os

import numpy as np


DATA_LABEL_META = "data/expresso_style.csv"
ANCHOR_FILE = "output/expresso/style/prosody_predictor_gt/gt/pred/pred_val.csv"
OUTPUT_DIR = "output/expresso/style/prosody_predictor_gt/gt/loc"

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

# feature_name -> label -> data list
anchor_data_map = {
    "pitch": pitch_anchor_data,
    "energy": energy_anchor_data,
    "duration": duration_anchor_data
}

for feature in ["pitch", "energy", "duration"]:
    output_path = f"{OUTPUT_DIR}/{feature}_anchor_loc_smacof.csv"

    distance_matrix_anchors = []

    for label_i in labels:
        row = []
        data_i = anchor_data_map[feature][label_i]
        for label_j in labels:
            data_j = anchor_data_map[feature][label_j]

            # Compute Wasserstein distance between data_i and data_j
            from scipy.stats import wasserstein_distance
            dist_ij = wasserstein_distance(data_i, data_j)
            row.append(dist_ij)
        distance_matrix_anchors.append(row)
    D = np.array(distance_matrix_anchors)

    from sklearn.manifold import smacof
    X, stress = smacof(dissimilarities=D, metric=False, n_components=2, random_state=0)
    print(f"Feature: {feature}, Embedding shape: {X.shape}, Stress: {stress}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['label', 'x', 'y'])
        for idx, label in enumerate(labels):
            x, y = X[idx]
            writer.writerow([label, x, y])
