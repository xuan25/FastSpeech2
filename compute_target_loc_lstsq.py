import csv
import os

import numpy as np
import tqdm


ANCHOR_LOC_FILE = "output/expresso/style/prosody_predictor_gt/gt/loc/pitch_anchor_loc_smacof.csv"

# TARGET_DISTANCE_DIR = "output/expresso/style/prosody_predictor/embedding_input/contrastive6_0.1/wasserstein_distance/val/pitch"
# OUTPUT_DIR = "output/expresso/style/prosody_predictor/embedding_input/contrastive6_0.1/loc/val/pitch/lstsq"

# TARGET_DISTANCE_DIR = "output/expresso/style/prosody_predictor/embedding_input/contrastive5_0.1/wasserstein_distance/val/pitch"
# OUTPUT_DIR = "output/expresso/style/prosody_predictor/embedding_input/contrastive5_0.1/loc/val/pitch/lstsq"

TARGET_DISTANCE_DIR = "output/expresso/style/prosody_predictor/embedding_input/default/wasserstein_distance/val/pitch"
OUTPUT_DIR = "output/expresso/style/prosody_predictor/embedding_input/default/loc/val/pitch/lstsq"

anchor_loc_map = {}
with open(ANCHOR_LOC_FILE, 'r', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        label_name = row['label']
        x, y = float(row['x']), float(row['y'])
        anchor_loc_map[label_name] = (x, y)

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


    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, os.path.basename(tgt_distance_file)), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['source_label', 'target_label', 'x', 'y'])

        for (source_label, target_label), anchor_distances in target_distance_map.items():
            # Multidimensional scaling (MDS) to compute target loc based on distances to anchors
            # multilateration
            anchor_labels = list(anchor_distances.keys())

            distances = np.array([anchor_distances[label] for label in anchor_labels])
            anchor_coords = np.array([anchor_loc_map[label] for label in anchor_labels])

            # Solve for target location using least squares
            A = -2 * (anchor_coords - np.mean(anchor_coords, axis=0))
            b = distances**2 - np.sum(anchor_coords**2, axis=1) + np.sum(np.mean(anchor_coords, axis=0)**2)

            target_loc, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        
            x, y = target_loc
            writer.writerow([source_label, target_label, x, y])
