import csv

import numpy as np

# | model | mean_distance_to_target | mean_distance_to_opposites | num_entries |

# FILES = [
#     "output/meld/emotion/prosody_predictor/embedding_input/default/wasserstein_distance/train/pitch/361000.csv",
#     "output/meld/emotion/prosody_predictor/embedding_input/contrastive7_0.1_0.1/wasserstein_distance/train/pitch/361000.csv",
#     "output/meld/emotion/prosody_predictor/embedding_input/contrastive7_0.1_1/wasserstein_distance/train/pitch/361000.csv",
#     "output/meld/emotion/prosody_predictor/embedding_input/contrastive7_1_1/wasserstein_distance/train/pitch/361000.csv",   
# ]

# FILES = [
#     "output/expresso/style/prosody_predictor/embedding_input/default/wasserstein_distance/train/pitch/361000.csv",
#     "output/expresso/style/prosody_predictor/embedding_input/contrastive7_0.1_0.1/wasserstein_distance/train/pitch/361000.csv",
#     "output/expresso/style/prosody_predictor/embedding_input/contrastive7_0.1_1/wasserstein_distance/train/pitch/361000.csv",
#     "output/expresso/style/prosody_predictor/embedding_input/contrastive7_1_1/wasserstein_distance/train/pitch/361000.csv",
# ]

# FILES = [
#     "output/expresso/style/prosody_predictor/embedding_input/default/wasserstein_distance/train/pitch/1000.csv",
#     "output/expresso/style/prosody_predictor/embedding_input/contrastive7_0.1_0.1/wasserstein_distance/train/pitch/1000.csv",
#     "output/expresso/style/prosody_predictor/embedding_input/contrastive7_0.1_1/wasserstein_distance/train/pitch/1000.csv",
#     "output/expresso/style/prosody_predictor/embedding_input/contrastive7_1_1/wasserstein_distance/train/pitch/1000.csv",
# ]

# FILES = [
#     "output/expresso/style/prosody_predictor/embedding_input/default/wasserstein_distance/train/pitch/41000.csv",
#     "output/expresso/style/prosody_predictor/embedding_input/contrastive7_0.1_0.1/wasserstein_distance/train/pitch/41000.csv",
#     "output/expresso/style/prosody_predictor/embedding_input/contrastive7_0.1_1/wasserstein_distance/train/pitch/41000.csv",
#     "output/expresso/style/prosody_predictor/embedding_input/contrastive7_1_1/wasserstein_distance/train/pitch/41000.csv",
# ]

FILES = [
    # "output/libritts/sentiment/prosody_predictor/embedding_input/default/wasserstein_distance/train/pitch/41000.csv",
    "output/libritts/sentiment/prosody_predictor/embedding_input/archive-lr-0.01/default_old-lr-0.1/wasserstein_distance/train/pitch/41000.csv",
    # "output/libritts/sentiment/prosody_predictor/embedding_input/contrastive7_0.1_0.1/wasserstein_distance/train/pitch/41000.csv",
    "output/libritts/sentiment/prosody_predictor/embedding_input/contrastive_emb_0.1_0.1/wasserstein_distance/train/pitch/41000.csv",
    # "output/libritts/sentiment/prosody_predictor/embedding_input/contrastive7_0.1_1/wasserstein_distance/train/pitch/41000.csv",
    # "output/libritts/sentiment/prosody_predictor/embedding_input/contrastive7_1_1/wasserstein_distance/train/pitch/41000.csv",
]



# collect all the labels
all_source_labels = set()
all_target_labels = set()
all_anchor_labels = set()

for file in FILES:
    with open(file, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            source_label = row['source_label']
            target_label = row['target_label']
            anchor_label = row['anchor_label']
            all_source_labels.add(source_label)
            all_target_labels.add(target_label)
            all_anchor_labels.add(anchor_label)

all_source_labels = sorted(list(all_source_labels))
all_target_labels = sorted(list(all_target_labels))
all_anchor_labels = sorted(list(all_anchor_labels))


print("| model | mean_distance_to_target | mean_distance_to_opposites | num_entries |")
print("|-------|-------------------------|----------------------------|-------------|")

for file in FILES:
    model = file.split('/')[-5]
    distance_to_target = []
    distance_to_opposites = []
    
    with open(file, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        
        # build a map from (source_label, target_label, anchor_label) to distance
        distance_map = {}
        for row in reader:
            source_label = row['source_label']
            target_label = row['target_label']
            anchor_label = row['anchor_label']

            # only consider cases where source_label == target_label
            # where those exist in the dataset
            # no generalization beyond the dataset
            if source_label != target_label:
                continue

            distance = float(row['wasserstein_distance'])
            if target_label == anchor_label:
                distance_to_target.append(distance)
            else:
                distance_to_opposites.append(distance)

    # sum
    avg_distance_to_target = sum(distance_to_target) / len(distance_to_target) if len(distance_to_target) > 0 else 0.0
    avg_distance_to_opposites = sum(distance_to_opposites) / len(distance_to_opposites) if len(distance_to_opposites) > 0 else 0.0
    
    # standard deviation
    std_distance_to_target = np.std(distance_to_target) if len(distance_to_target) > 0 else 0.0
    std_distance_to_opposites = np.std(distance_to_opposites) if len(distance_to_opposites) > 0 else 0.0
    
    # min and max
    min_distance_to_target = min(distance_to_target) if len(distance_to_target) > 0 else 0.0
    max_distance_to_target = max(distance_to_target) if len(distance_to_target) > 0 else 0.0
    min_distance_to_opposites = min(distance_to_opposites) if len(distance_to_opposites) > 0 else 0.0
    max_distance_to_opposites = max(distance_to_opposites) if len(distance_to_opposites) > 0 else 0.0

    print(f"| {model} | {avg_distance_to_target:.3f} ± {std_distance_to_target:.3f} Range: {min_distance_to_target:.3f}-{max_distance_to_target:.3f} | {avg_distance_to_opposites:.3f} ± {std_distance_to_opposites:.3f} Range: {min_distance_to_opposites:.3f}-{max_distance_to_opposites:.3f} | {len(distance_to_target) + len(distance_to_opposites)} |")


