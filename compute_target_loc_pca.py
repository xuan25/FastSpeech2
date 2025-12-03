import argparse
import csv
import os
import pickle

import numpy as np
from sklearn.decomposition import PCA
import tqdm

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument(
    "--anchor_loc_file", type=str, help="Path to anchor loc file. e.g. output/dataset/label/model_gt/gt/loc_anchor_split/feature_anchor_loc_pca.csv"
)
arg_parser.add_argument(
    "--anchor_loc_transform_file", type=str, help="Path to anchor loc PCA transform file. e.g. output/dataset/label/model_gt/gt/loc_anchor_split/feature_anchor_pca_transform.pkl"
)
arg_parser.add_argument(
    "--target_distance_dir", type=str, help="Directory containing target wasserstein distance files. e.g. output/dataset/label/model/variant/loss/wasserstein_distance/split/feature"
)
arg_parser.add_argument(
    "--output_dir", type=str, help="Output directory to save target loc PCA files. e.g. output/dataset/label/model/variant/loss/loc/split/feature/pca"
)
arg_parser.add_argument(
    "--exclude_labels", type=str, default=None, help="Labels to exclude. Comma separated list e.g. category1,category2"
)
args = arg_parser.parse_args()


exclude_labels = args.exclude_labels.split(',') if args.exclude_labels is not None else []


anchor_loc_map = {}
with open(args.anchor_loc_file, 'r', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        label_name = row['label']
        x, y = float(row['x']), float(row['y'])
        anchor_loc_map[label_name] = (x, y)

# note: labels are sorted to ensure consistent order
labels = sorted(anchor_loc_map.keys())
labels = [label for label in labels if label not in exclude_labels]

pca: PCA = pickle.load(open(args.anchor_loc_transform_file, 'rb'))

tgt_distance_files = os.listdir(args.target_distance_dir)

for tgt_distance_file in tqdm.tqdm(tgt_distance_files, desc="Computing target loc", dynamic_ncols=True, leave=False):

    target_distance_map = {}
    with open(os.path.join(args.target_distance_dir, tgt_distance_file), 'r', newline='') as csvfile:
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

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, os.path.basename(tgt_distance_file)), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['source_label', 'target_label', 'x', 'y'])

        for idx, (source_label, target_label) in enumerate(target_distance_map.keys()):
            target_loc = X[idx]
        
            x, y = target_loc
            writer.writerow([source_label, target_label, x, y])
