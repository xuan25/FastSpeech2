import csv
import os
import pickle

import numpy as np
from sklearn.decomposition import PCA
import tqdm


ANCHOR_LOC_FILE = "output/expresso/style/prosody_predictor_gt/gt/loc/pitch_anchor_loc_pca.csv"
ANCHOR_LOC_TRANSFORM_FILE = "output/expresso/style/prosody_predictor_gt/gt/loc/pitch_anchor_pca_transform.pkl"
EXCLUDE_LABELS = []


# TARGET_DISTANCE_DIR = "output/expresso/style/prosody_predictor/embedding_input/contrastive6_0.1/wasserstein_distance/val/pitch"
# OUTPUT_DIR = "output/expresso/style/prosody_predictor/embedding_input/contrastive6_0.1/loc/val/pitch/pca"

# TARGET_DISTANCE_DIR = "output/expresso/style/prosody_predictor/embedding_input/contrastive5_0.1/wasserstein_distance/val/pitch"
# OUTPUT_DIR = "output/expresso/style/prosody_predictor/embedding_input/contrastive5_0.1/loc/val/pitch/pca"

# TARGET_DISTANCE_DIR = "output/expresso/style/prosody_predictor/embedding_input/default/wasserstein_distance/val/pitch"
# OUTPUT_DIR = "output/expresso/style/prosody_predictor/embedding_input/default/loc/val/pitch/pca"

# TARGET_DISTANCE_DIR = "output/expresso/style/prosody_predictor/embedding_input/contrastive6_0.1_0.1/wasserstein_distance/val/pitch"
# OUTPUT_DIR = "output/expresso/style/prosody_predictor/embedding_input/contrastive6_0.1_0.1/loc/val/pitch/pca"

# TARGET_DISTANCE_DIR = "output/expresso/style/prosody_predictor/embedding_input/contrastive5_0.1_0.1/wasserstein_distance/val/pitch"
# OUTPUT_DIR = "output/expresso/style/prosody_predictor/embedding_input/contrastive5_0.1_0.1/loc/val/pitch/pca"

# TARGET_DISTANCE_DIR = "output/expresso/style/prosody_predictor_gt/gt/wasserstein_distance/train/pitch"
# OUTPUT_DIR = "output/expresso/style/prosody_predictor_gt/gt/loc/train/pitch/pca"

# TARGET_DISTANCE_DIR = "output/expresso/style/prosody_predictor/embedding_input/contrastive7_0.1/wasserstein_distance/val/pitch"
# OUTPUT_DIR = "output/expresso/style/prosody_predictor/embedding_input/contrastive7_0.1/loc/val/pitch/pca"

TARGET_DISTANCE_DIR = "output/expresso/style/prosody_predictor/embedding_input/contrastive7_0.1_0.1/wasserstein_distance/val/pitch"
OUTPUT_DIR = "output/expresso/style/prosody_predictor/embedding_input/contrastive7_0.1_0.1/loc/val/pitch/pca"



# ---- no whisper ----

# ANCHOR_LOC_FILE = "output/expresso/style/prosody_predictor_gt/gt_no_whisper/loc/pitch_anchor_loc_pca.csv"
# ANCHOR_LOC_TRANSFORM_FILE = "output/expresso/style/prosody_predictor_gt/gt_no_whisper/loc/pitch_anchor_pca_transform.pkl"
# EXCLUDE_LABELS = ["whisper"]


# TARGET_DISTANCE_DIR = "output/expresso/style/prosody_predictor/embedding_input/contrastive5_0.1_0.1_no_whisper/wasserstein_distance/val/pitch"
# OUTPUT_DIR = "output/expresso/style/prosody_predictor/embedding_input/contrastive5_0.1_0.1_no_whisper/loc/val/pitch/pca"

# TARGET_DISTANCE_DIR = "output/expresso/style/prosody_predictor/embedding_input/contrastive5_0.1_no_whisper/wasserstein_distance/val/pitch"
# OUTPUT_DIR = "output/expresso/style/prosody_predictor/embedding_input/contrastive5_0.1_no_whisper/loc/val/pitch/pca"

# TARGET_DISTANCE_DIR = "output/expresso/style/prosody_predictor/embedding_input/contrastive6_0.1_0.1_no_whisper/wasserstein_distance/val/pitch"
# OUTPUT_DIR = "output/expresso/style/prosody_predictor/embedding_input/contrastive6_0.1_0.1_no_whisper/loc/val/pitch/pca"

# TARGET_DISTANCE_DIR = "output/expresso/style/prosody_predictor/embedding_input/contrastive6_0.1_no_whisper/wasserstein_distance/val/pitch"
# OUTPUT_DIR = "output/expresso/style/prosody_predictor/embedding_input/contrastive6_0.1_no_whisper/loc/val/pitch/pca"

# TARGET_DISTANCE_DIR = "output/expresso/style/prosody_predictor/embedding_input/default_no_whisper/wasserstein_distance/val/pitch"
# OUTPUT_DIR = "output/expresso/style/prosody_predictor/embedding_input/default_no_whisper/loc/val/pitch/pca"

# TARGET_DISTANCE_DIR = "output/expresso/style/prosody_predictor_gt/gt_no_whisper/wasserstein_distance/train/pitch"
# OUTPUT_DIR = "output/expresso/style/prosody_predictor_gt/gt_no_whisper/loc/train/pitch/pca"




anchor_loc_map = {}
with open(ANCHOR_LOC_FILE, 'r', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        label_name = row['label']
        x, y = float(row['x']), float(row['y'])
        anchor_loc_map[label_name] = (x, y)

# note: labels are sorted to ensure consistent order
labels = sorted(anchor_loc_map.keys())
labels = [label for label in labels if label not in EXCLUDE_LABELS]

pca: PCA = pickle.load(open(ANCHOR_LOC_TRANSFORM_FILE, 'rb'))

tgt_distance_files = os.listdir(TARGET_DISTANCE_DIR)

for tgt_distance_file in tqdm.tqdm(tgt_distance_files, desc="Computing target loc", dynamic_ncols=True, leave=False):

    target_distance_map = {}
    with open(os.path.join(TARGET_DISTANCE_DIR, tgt_distance_file), 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            source_target_label = (row['source_label'], row['target_label'])
            anchor_label = row['anchor_label']
            distance = float(row['wasserstein_distance'])
            if source_target_label not in target_distance_map:
                target_distance_map[source_target_label] = {}
            target_distance_map[source_target_label][anchor_label] = distance

        distance_matrix = []
        for (source_label, target_label), anchor_distances in target_distance_map.items():
            # PCA to compute target loc based on distances to anchors
            distance_vector = []
            for label in labels:
                distance_vector.append(anchor_distances[label])
            distance_matrix.append(distance_vector)
        distance_matrix = np.array(distance_matrix)
        X = pca.transform(distance_matrix)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, os.path.basename(tgt_distance_file)), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['source_label', 'target_label', 'x', 'y'])

        for idx, (source_label, target_label) in enumerate(target_distance_map.keys()):
            target_loc = X[idx]
        
            x, y = target_loc
            writer.writerow([source_label, target_label, x, y])
