
# DISTANCE_FILE = "output/expresso/style/prosody_predictor_gt/gt/wasserstein_distance/val/pitch/0.csv"
# OUTPUT_FILE = "output/expresso/style/prosody_predictor_gt/gt/vis/val/pitch/0.png"

DISTANCE_FILE = "output/expresso/style/prosody_predictor_gt/gt/wasserstein_distance_no_whisper/val/pitch/0.csv"
OUTPUT_FILE = "output/expresso/style/prosody_predictor_gt/gt/vis_no_whisper/val/pitch/0.png"

LABLE_PREFERED_ORDER = [
    "default",
    "confused",
    "enunciated",
    "sad",
    "happy",
    "laughing",
    "whisper"
]

import os
import csv
import matplotlib.pyplot as plt
import numpy as np
import tqdm

# plot the matrix
lables_set = set()
plt.figure(figsize=(8, 8))
with open(DISTANCE_FILE, 'r', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        lables_set.add(row['source_label'])
        lables_set.add(row['target_label'])

labels = sorted(lables_set)

# reorder labels according to preferred order and append the rest to the end
ordered_labels = []
for preferred_label in LABLE_PREFERED_ORDER:
    if preferred_label in labels:
        ordered_labels.append(preferred_label)
for label in labels:
    if label not in ordered_labels:
        ordered_labels.append(label)
labels = ordered_labels

lable_idx_map = {label: idx for idx, label in enumerate(labels)}

distance_matrix = np.zeros((len(labels), len(labels)))

with open(DISTANCE_FILE, 'r', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:

        source_label = row['source_label']
        target_label = row['target_label']
        distance = float(row['wasserstein_distance'])

        source_idx = lable_idx_map[source_label]
        target_idx = lable_idx_map[target_label]

        distance_matrix[source_idx, target_idx] = distance

# plot heatmap
plt.imshow(distance_matrix, cmap='viridis', interpolation='nearest')

# plot number on each cell
for i in range(len(labels)):
    for j in range(len(labels)):
        plt.text(j, i, f"{distance_matrix[i, j]:.2f}", ha='center', va='center', color='white' if distance_matrix[i, j] < np.max(distance_matrix)/2 else 'black')

plt.colorbar(label='Wasserstein Distance')
plt.xticks(ticks=np.arange(len(labels)), labels=labels, rotation=90)
plt.yticks(ticks=np.arange(len(labels)), labels=labels)
plt.title('Wasserstein Distance Matrix')
plt.tight_layout()

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

plt.savefig(OUTPUT_FILE)
print(f"Saved distance matrix plot to {OUTPUT_FILE}")
