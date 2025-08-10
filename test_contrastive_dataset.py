# from fastspeech2.dataset.dataset import main

# if __name__ == "__main__":
#     main()


import time
import tqdm
from fastspeech2.config import DatasetPathConfig, DatasetPreprocessingConfig
from fastspeech2.dataset.dataset import DatasetSplit, DatasetWithSentimentContrastive, OriginalDatasetWithSentiment

# MAX_SAMPLES = 5000  # Limit the number of batches for benchmarking
MAX_SAMPLES = -1

def run_benchmark(dataset, num_workers, batch_size):
    from torch.utils.data import DataLoader

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn
    )

    num_batches = 0
    num_samples = 0
    time_begin = time.time()
    time_first_batch = None

    for batch in tqdm.tqdm(dataloader, desc=f"Loading dataset (num_workers={num_workers}, batch_size={batch_size})"):
        if time_first_batch is None:
            time_first_batch = time.time()
        num_batches += 1
        num_samples += batch_size
        if MAX_SAMPLES > 0 and num_samples >= MAX_SAMPLES:
            break

    time_end = time.time()

    assert time_first_batch is not None, "No batches were processed."

    print(
        f"Processed {num_samples} samples with num_workers={num_workers}, batch_size={batch_size}")
    print(f"Total time: {time_end - time_begin:.2f} seconds")
    print(f"Warmup duration: {time_first_batch - time_begin:.2f} seconds")
    print(
        f"Average time per batch: {(time_end - time_begin) / num_batches:.2f} seconds")
    print(
        f"Number of batches per second: {num_batches / (time_end - time_begin):.2f} batches/s")


def main():

    # from torch.utils.data import DataLoader

    dataset_path_config = DatasetPathConfig.from_dict(
        base_dir="config/LibriTTS",
        config_dict={
            "base_dir": "../../data/LibriTTS.tar",
            "meta_file_train": "train.txt",
            "meta_file_val": "val.txt",
            "speaker_map_file": "speakers.json",
            "feature_dir": "",
            "stats_file": "stats.json",
            # "sentiment_file": "sentiment_scores.csv",
            "sentiment_file": "../overrides/sentiment_scores_override_all_neu.csv",
        })
    dataset_preprocessing_config = DatasetPreprocessingConfig(
        lexicon_path="../../lexicon/librispeech-lexicon.txt",
        text_cleaners=["english_cleaners"],
    )

    dataset = DatasetWithSentimentContrastive(
        dataset_path_config=dataset_path_config,
        dataset_preprocessing_config=dataset_preprocessing_config,
        split=DatasetSplit.TRAIN,
        sort=False
    )

    run_benchmark(dataset, 0, 16)

    # dataloader = DataLoader(dataset, batch_size=4, num_workers=2, collate_fn=dataset.collate_fn)

    # max_batches = 1000  # Limit the number of batches for benchmarking

    # num_batches = 0

    # time_begin = time.time()
    # time_first_batch = None
    # for batch in tqdm.tqdm(dataloader, desc="Loading dataset"):
    #     if time_first_batch is None:
    #         time_first_batch = time.time()
    #     num_batches += 1
    #     if num_batches >= max_batches:
    #         break

    # time_end = time.time()

    # assert time_first_batch is not None, "No batches were processed."

    # print(f"Total time: {time_end - time_begin:.2f} seconds")
    # print(f"Warmup duration: {time_first_batch - time_begin:.2f} seconds")
    # print(f"Average time per batch: {(time_end - time_begin) / num_batches:.2f} seconds")
    # print(f"Number of batches per second: {num_batches / (time_end - time_begin):.2f} batches/s")


if __name__ == "__main__":
    print("dataset benchmark")
    main()
