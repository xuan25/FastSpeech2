import argparse
import csv
import os
import numpy as np
import scipy.stats
import tqdm

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument(
    "--data_label_meta", type=str, help="CSV file containing data label meta information. e.g. data/dataset_label.csv"
)
arg_parser.add_argument(
    "--anchor_file", type=str, help="CSV file containing anchor data. e.g. output/dataset/label/model_gt/gt/pred/split/gt/0.csv"
)
arg_parser.add_argument(
    "--pred_dir", type=str, help="Directory containing prediction CSV files. e.g. output/dataset/label/model/variant/loss/pred/split"
)
arg_parser.add_argument(
    "--output_dir", type=str, help="Directory to save Wasserstein distance CSV files. e.g. output/dataset/label/model/variant/loss/wasserstein_distance/split"
)
args = arg_parser.parse_args()



def get_source_label_map():
    source_label_map = {}
    with open(args.data_label_meta, 'r', newline='') as csvfile:
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

    with open(args.anchor_file, 'r', newline='') as csvfile:
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

    

def get_ckpt_file_map():
    ckpt_file_map = {}
    for target_label in os.listdir(args.pred_dir):
        for pred_file in os.listdir(os.path.join(args.pred_dir, target_label)):
            ckpt_id = pred_file.replace(".csv", "")
            if ckpt_id not in ckpt_file_map:
                ckpt_file_map[ckpt_id] = {}
            ckpt_file_map[ckpt_id][target_label] = pred_file
    return ckpt_file_map

# [(ckpt_id, {target_label: pred_file, ...}), ...]
ckpt_file_map = get_ckpt_file_map()

# loop over features
for feature in ["pitch", "energy", "duration"]:
    output_feature_dir = os.path.join(args.output_dir, feature)
    os.makedirs(output_feature_dir, exist_ok=True)

    # loop over ckpts
    for ckpt_id, target_label_file_map in tqdm.tqdm(ckpt_file_map.items(), desc=f"Processing {feature}", dynamic_ncols=True, leave=False):
        output_file = os.path.join(output_feature_dir, f"{ckpt_id}.csv")
        # output a csv file for each ckpt
        with open(output_file, 'w', newline='') as csvfile:
            fieldnames = ['source_label', 'target_label', 'anchor_label', 'wasserstein_distance']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            # loop over target labels
            for target_label_str, pred_file in target_label_file_map.items():
                target_label = target_label_str
                pred_data_source_map = {}
        
                pred_file_path = os.path.join(args.pred_dir, target_label_str, pred_file)

                with open(pred_file_path, 'r', newline='') as pred_csvfile:
                    reader = csv.DictReader(pred_csvfile)
                    for row in reader:
                        file_id = row['data_id']
                        source_label = source_label_map[file_id]
                        feature_value = float(row[feature])
                        if source_label not in pred_data_source_map:
                            pred_data_source_map[source_label] = []
                        pred_data_source_map[source_label].append(feature_value)

                # loop over source labels
                for source_label in labels:
                    if source_label not in pred_data_source_map:
                        continue
                    pred_data = pred_data_source_map[source_label]

                    # loop over anchor labels
                    for anchor_label in labels:
                        anchor_data = anchor_data_map[feature][anchor_label]

                        # compute wasserstein distance
                        wass_distance = scipy.stats.wasserstein_distance(pred_data, anchor_data)

                        # write to csv
                        writer.writerow({
                            'source_label': source_label,
                            'target_label': target_label,
                            'anchor_label': anchor_label,
                            'wasserstein_distance': wass_distance
                        })
        print(f"Wrote Wasserstein distances to {output_file}")

