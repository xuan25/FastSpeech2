import csv

import numpy as np

# | model | mean_distance_to_target | mean_distance_to_opposites | num_entries |

# FILES = [
#     "output/meld/emotion/prosody_predictor/embedding_input/default/mse/train/pitch/361000.csv",
#     "output/meld/emotion/prosody_predictor/embedding_input/contrastive7_0.1_0.1/mse/train/pitch/361000.csv",
#     "output/meld/emotion/prosody_predictor/embedding_input/contrastive7_0.1_1/mse/train/pitch/361000.csv",
#     "output/meld/emotion/prosody_predictor/embedding_input/contrastive7_1_1/mse/train/pitch/361000.csv",   
# ]

# FILES = [
#     "output/expresso/style/prosody_predictor/embedding_input/default/mse/train/pitch/361000.csv",
#     "output/expresso/style/prosody_predictor/embedding_input/contrastive7_0.1_0.1/mse/train/pitch/361000.csv",
#     "output/expresso/style/prosody_predictor/embedding_input/contrastive7_0.1_1/mse/train/pitch/361000.csv",
#     "output/expresso/style/prosody_predictor/embedding_input/contrastive7_1_1/mse/train/pitch/361000.csv",   
# ]

# FILES = [
#     "output/expresso/style/prosody_predictor/embedding_input/default/mse/train/pitch/1000.csv",
#     "output/expresso/style/prosody_predictor/embedding_input/contrastive7_0.1_0.1/mse/train/pitch/1000.csv",
#     "output/expresso/style/prosody_predictor/embedding_input/contrastive7_0.1_1/mse/train/pitch/1000.csv",
#     "output/expresso/style/prosody_predictor/embedding_input/contrastive7_1_1/mse/train/pitch/1000.csv",
# ]

FILES = [
    "output/expresso/style/prosody_predictor/embedding_input/default/mse/train/pitch/41000.csv",
    "output/expresso/style/prosody_predictor/embedding_input/contrastive7_0.1_0.1/mse/train/pitch/41000.csv",
    "output/expresso/style/prosody_predictor/embedding_input/contrastive7_0.1_1/mse/train/pitch/41000.csv",
    "output/expresso/style/prosody_predictor/embedding_input/contrastive7_1_1/mse/train/pitch/41000.csv",
]


# FILES = [
#     # "output/libritts/sentiment/prosody_predictor/embedding_input/default/mse/train/pitch/361000.csv",
#     "output/libritts/sentiment/prosody_predictor/embedding_input/archive-lr-0.01/default_old-lr-0.1/mse/train/pitch/361000.csv",
#     # "output/libritts/sentiment/prosody_predictor/embedding_input/contrastive7_0.1_0.1/mse/train/pitch/361000.csv", 
#     "output/libritts/sentiment/prosody_predictor/embedding_input/contrastive_emb_0.1_0.1/mse/train/pitch/361000.csv", 
#     # "output/libritts/sentiment/prosody_predictor/embedding_input/contrastive7_0.1_1/mse/train/pitch/361000.csv", 
#     # "output/libritts/sentiment/prosody_predictor/embedding_input/contrastive7_1_1/mse/train/pitch/361000.csv", 
# ]



# collect all the labels
all_source_labels = set()
all_target_labels = set()

for file in FILES:
    with open(file, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            source_label = row['source_label']
            target_label = row['target_label']
            all_source_labels.add(source_label)
            all_target_labels.add(target_label)

all_source_labels = sorted(list(all_source_labels))
all_target_labels = sorted(list(all_target_labels))


print("| model | mean_mse | mean_mae | num_entries |")
print("|-------|----------|----------|-------------|")

for file in FILES:
    model = file.split('/')[-5]
    mse = []
    mae = []
    
    with open(file, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        
        # build a map from (source_label, target_label, anchor_label) to distance
        distance_map = {}
        for row in reader:
            source_label = row['source_label']
            target_label = row['target_label']

            # only consider cases where source_label == target_label
            # where those exist in the dataset
            # no generalization beyond the dataset
            if source_label != target_label:
                continue

            mse.append(float(row['mse']))
            mae.append(float(row['mae']))

    # sum
    avg_mse = sum(mse) / len(mse) if len(mse) > 0 else 0.0
    avg_mae = sum(mae) / len(mae) if len(mae) > 0 else 0.0
    
    # standard deviation
    std_mse = np.std(mse) if len(mse) > 0 else 0.0
    std_mae = np.std(mae) if len(mae) > 0 else 0.0
    
    # min and max\
    min_mse = min(mse) if len(mse) > 0 else 0.0
    max_mse = max(mse) if len(mse) > 0 else 0.0
    min_mae = min(mae) if len(mae) > 0 else 0.0
    max_mae = max(mae) if len(mae) > 0 else 0.0

    print(f"| {model} | {avg_mse:.3f} ± {std_mse:.3f} Range: {min_mse:.3f}-{max_mse:.3f} | {avg_mae:.3f} ± {std_mae:.3f} Range: {min_mae:.3f}-{max_mae:.3f} | {len(mse)} |")


