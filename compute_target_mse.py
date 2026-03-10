import argparse
import csv
import os
import numpy as np
import scipy.stats
import tqdm

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument(
    "--data_label_meta", type=str, help="CSV file containing data label meta information. e.g. data/dataset_label.csv",
    default="data/expresso_style.csv"
)
arg_parser.add_argument(
    "--gt_file", type=str, help="CSV file containing ground truth data. e.g. output/dataset/label/model_gt/gt/pred/split/gt/0.csv",
    default="output/expresso/style/prosody_predictor_gt/gt/pred/train/gt/0.csv"
)
arg_parser.add_argument(
    "--pred_dir", type=str, help="Directory containing prediction CSV files. e.g. output/dataset/label/model/variant/loss/pred/split",
    default="output/expresso/style/prosody_predictor/embedding_input/contrastive7_0.1_0.1/pred/train"
)
arg_parser.add_argument(
    "--output_dir", type=str, help="Directory to save Wasserstein distance CSV files. e.g. output/dataset/label/model/variant/loss/mse/split",
    default="output/debug/style/prosody_predictor/embedding_input/contrastive7_0.1_0.1/mse/train"
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



# def get_anchor_data(feature_name, label_col, label_names):
#     anchor_data_map = {}

#     with open(args.anchor_file, 'r', newline='') as csvfile:
#         reader = csv.DictReader(csvfile)
#         for row in reader:
#             label = int(row[label_col])
#             label_name = label_names[label]
#             feature_value = float(row[feature_name])
#             if label_name not in anchor_data_map:
#                 anchor_data_map[label_name] = []
#             anchor_data_map[label_name].append(feature_value)

#     return anchor_data_map

# pitch_anchor_data = get_anchor_data("pitch", "label", label_names)
# energy_anchor_data = get_anchor_data("energy", "label", label_names)
# duration_anchor_data = get_anchor_data("duration", "label", label_names)





# labels = sorted(pitch_anchor_data.keys())
# assert labels == sorted(energy_anchor_data.keys()) == sorted(duration_anchor_data.keys()), "Labels do not match across features."

# feature_name -> label -> data list
# anchor_data_map = {
#     "pitch": pitch_anchor_data,
#     "energy": energy_anchor_data,
#     "duration": duration_anchor_data
# }




def get_gt_data_map(feature_name, data_id_col, pos_col):
    gt_data_map = {}

    with open(args.gt_file, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data_id = row[data_id_col]
            feature_value = float(row[feature_name])
            pos = int(row[pos_col])
            if pos == 0:
                assert data_id not in gt_data_map, "Duplicate data_id in GT file or ordering issue."
                gt_data_map[data_id] = [feature_value]
            else:
                assert data_id in gt_data_map, "Data_id not found in GT file or ordering issue."
                assert len(gt_data_map[data_id]) == pos, "Data_id positions are not sequential."
                gt_data_map[data_id].append(feature_value)

    return gt_data_map


# feature_name -> data_id -> feature value sequence

gt_data_map = {
    "pitch": get_gt_data_map("pitch", "data_id", "phone_idx"),
    "energy": get_gt_data_map("energy", "data_id", "phone_idx"),
    "duration": get_gt_data_map("duration", "data_id", "phone_idx")
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
            fieldnames = ['source_label', 'target_label', 'mse']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            # loop over target labels
            for target_label_str, pred_file in target_label_file_map.items():
                target_label = target_label_str

                pred_source_map = {}
                gt_source_map = {}
                
        
                pred_file_path = os.path.join(args.pred_dir, target_label_str, pred_file)

                with open(pred_file_path, 'r', newline='') as pred_csvfile:
                    reader = csv.DictReader(pred_csvfile)
                    for row in reader:
                        file_id = row['data_id']
                        source_label = source_label_map[file_id]
                        feature_value = float(row[feature])
                        phone_idx = int(row['phone_idx'])

                        gt_feature_value = gt_data_map[feature][file_id][phone_idx]

                        if source_label not in gt_source_map:
                            gt_source_map[source_label] = []
                        gt_source_map[source_label].append(gt_feature_value)

                        if source_label not in pred_source_map:
                            pred_source_map[source_label] = []
                        pred_source_map[source_label].append(feature_value)

                # loop over source labels
                for source_label in label_names:
                    # if source_label not in pred_data_source_map:
                    #     continue
                    # pred_data = pred_data_source_map[source_label]

                    pred_data = pred_source_map[source_label]
                    gt_data = gt_source_map[source_label]

                    # compute mse and stats

                    se = (np.array(pred_data) - np.array(gt_data)) ** 2

                    import numpy as np
                    mse = np.mean(se)

                    writer.writerow({
                        'source_label': source_label,
                        'target_label': target_label,
                        'mse': mse
                    })

        tqdm.tqdm.write(f"Wrote MSE values to {output_file}")

