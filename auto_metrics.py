import csv
import os
import tqdm
from metrics.log_f0 import LogF0BatchProcesser
from metrics.mcd import MCDBatchProcesser

def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ref_dir", 
        # required=True, 
        type=str, 
        default="data/original/LibriTTS/wav"
    )

    parser.add_argument(
        "--synth_dir", 
        # required=True, 
        type=str, 
        default="output/default/synth_val"
    )

    parser.add_argument(
        "--output_file", 
        # required=True, 
        type=str, 
        default="output/default/metrics.csv"
    )

    args = parser.parse_args()

    ref_dir = args.ref_dir
    synth_dir = args.synth_dir
    output_file = args.output_file

    # collecting tgt paths

    tgt_paths = []

    for root, dirs, files in os.walk(synth_dir):
        for file in files:
            if file.endswith(".wav"):
                tgt_paths.append(os.path.join(root, file))

    file_ids = [os.path.splitext(os.path.basename(path))[0] for path in tgt_paths]

    # collecting ref paths

    ref_paths_raw = []

    for root, dirs, files in os.walk(ref_dir):
        for file in files:
            if file.endswith(".wav"):
                ref_paths_raw.append(os.path.join(root, file))


    ref_paths_map = {os.path.splitext(os.path.basename(path))[0]: path for path in ref_paths_raw}

    ref_paths = [ref_paths_map[file_id] for file_id in file_ids]

    # batch computing MCD (default hyperparameters)

    mcd_processer = MCDBatchProcesser(ref_paths, tgt_paths, num_threads=8)
    log_f0_processer = LogF0BatchProcesser(ref_paths, tgt_paths, num_threads=8)

    mcds = []
    log_f0s = []

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["audio_name", "mcd", "log_f0"])

        for tgt_path, mcd in tqdm.tqdm(
            zip(
                tgt_paths,
                mcd_processer
            ),
            desc="[computing MCD]", dynamic_ncols=True, total=len(tgt_paths)):

            mcds.append(mcd)

            tqdm.tqdm.write(f"{tgt_path} MCD: {mcd:.3f}")

            # tgt_audio_filename = os.path.splitext(os.path.basename(tgt_path))[0]
            # writer.writerow([tgt_audio_filename, mcd])

            # f.flush()

        for tgt_path, log_f0 in tqdm.tqdm(
            zip(
                tgt_paths,
                log_f0_processer
            ),
            desc="[computing Log-F0]", dynamic_ncols=True, total=len(tgt_paths)):

            log_f0s.append(log_f0)

            tqdm.tqdm.write(f"{tgt_path} Log-F0: {log_f0:.3f}")

        for tgt_path, mcd, log_f0 in zip(tgt_paths, mcds, log_f0s):
            tgt_audio_filename = os.path.splitext(os.path.basename(tgt_path))[0]
            writer.writerow([tgt_audio_filename, mcd, log_f0])

        mean_mcd = sum(mcds) / len(mcds)
        tqdm.tqdm.write(f"Mean MCD: {mean_mcd:.3f}")
        # writer.writerow(["Mean", mean_mcd])

        mean_log_f0 = sum(log_f0s) / len(log_f0s)
        tqdm.tqdm.write(f"Mean Log-F0: {mean_log_f0:.3f}")
        writer.writerow(["Mean", mean_mcd, mean_log_f0])

if __name__ == "__main__":
    main()
