import argparse
import csv
import tqdm
from fastspeech2.predict_batch_prosody_predictor import process

def get_labels_from_index_file(label_index_file):
    with open(label_index_file, 'r') as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames is not None, "CSV file must have header"
        labels = reader.fieldnames[1:]  # Skip the first column which is assumed to be 'basename' or similar
    return labels

if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(
        description="Predict prosody features using a trained prosody predictor model."
    )
    arg_parser.add_argument(
        "--model_ckpt_dir", type=str, required=True, help="Model checkpoint directory. e.g. output/dataset/label/model/variant/loss"
    )
    arg_parser.add_argument(
        "--dataset_config_path", type=str, required=True, help="Dataset configuration file path. e.g. config/dataset.yaml"
    )
    arg_parser.add_argument(
        "--model_config_path", type=str, help="Model configuration file path. e.g. config/model.yaml"
    )
    arg_parser.add_argument(
        "--label_index_file", type=str, required=True, help="Label index CSV file. e.g. data/label.csv"
    )
    arg_parser.add_argument(
        "--data_split_name", type=str, default="train", help="Data split name to process. e.g. train or val"
    )
    arg_parser.add_argument(
        "--ckpt_begin", type=int, default=1000, help="Start checkpoint number to process. e.g. 1000"
    )
    arg_parser.add_argument(
        "--ckpt_end", type=int, default=400000, help="End checkpoint number to process. e.g. 400000"
    )
    arg_parser.add_argument(
        "--ckpt_step", type=int, default=40000, help="Step size between checkpoints to process. e.g. 40000"
    )
    args = arg_parser.parse_args()



    label_names = get_labels_from_index_file(args.label_index_file)

    configs = [
        {
            "ckpt_path": f"{args.model_ckpt_dir}/ckpt/{model_ckpt_num}.pth",
            "output_path": f"{args.model_ckpt_dir}/pred/{args.data_split_name}/{label_name}/{model_ckpt_num}.csv",
            "dataset_config_path":args.dataset_config_path,
            "model_config_path": args.model_config_path,
            "data_split_name": args.data_split_name,
            "pitch_control": 1.0,
            "energy_control": 1.0,
            "duration_control": 1.0,
            "batch_size": 16,
            "label_control": label_control,
        }
        for model_ckpt_num in range(args.ckpt_begin, args.ckpt_end + 1, args.ckpt_step)
        for label_control, label_name in [
            (-1, "default"),
        ] + [(i, label_names[i]) for i in range(len(label_names))]
    ]



    success_count = 0
    for config in tqdm.tqdm(configs, desc="Processing configurations", dynamic_ncols=True, leave=False):
        try:
            process(**config)
            success_count += 1
        except Exception as e:
            tqdm.tqdm.write(f"Failed to process: {config}")
            tqdm.tqdm.write(str(e))
        
    if success_count == len(configs):
        print("Processing completed for all configurations.")
    else:
        print(f"Processing completed for {success_count} of {len(configs)} configurations.")
