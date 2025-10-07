from fastspeech2.config import DatasetPathConfig, DatasetPreprocessingConfig
from fastspeech2.dataset.data_models import DataBatch
from fastspeech2.dataset.dataset import DatasetSplit, DatasetWithEmotionContrastive, OriginalDatasetWithSentiment


# DatasetWithEmotionContrastive()



def main():
    
    from torch.utils.data import DataLoader

    dataset_path_config = DatasetPathConfig.from_dict(
        base_dir="./",
        config_dict={
            "base_dir": "F:/data_temp/MELD.Preprocessed/data.tar",
            "meta_file_train": "train.csv",
            "meta_file_val": "dev.csv",
            "speaker_map_file": "speakers.json",
            "feature_dir": "",
            "stats_file": "stats.json",
            # "sentiment_file": "sentiment_scores.csv",
            # "sentiment_file": "../overrides/sentiment_scores_override_all_neu.csv",
            "emotion_file": "F:/data_temp/MELD.Preprocessed/emotion.csv",
        })
    dataset_preprocessing_config=DatasetPreprocessingConfig(
        lexicon_path="../../lexicon/librispeech-lexicon.txt",
        text_cleaners=["english_cleaners"],
    )

    dataset = DatasetWithEmotionContrastive(
        dataset_path_config=dataset_path_config,
        dataset_preprocessing_config=dataset_preprocessing_config,
        split=DatasetSplit.VAL,
        sort=True
    )

    sample = dataset[0]
    print(sample)
    
    # for i in range(3):
    #     sample = dataset[i]
    #     # print(sample)
    #     print(f"Speaker: {sample.speaker}, Text: {sample.text.shape}, Emotion: {sample.emotion}")
    #     print(f"Raw Text: {sample.raw_text}")
    #     print(f"Mel shape: {sample.mel.shape}, Pitch shape: {sample.pitch.shape}, Energy shape: {sample.energy.shape}, Duration shape: {sample.duration.shape}") # type: ignore
    #     print("-" * 50)


    # dataloader = DataLoader(dataset, batch_size=4, num_workers=2, collate_fn=dataset.collate_fn)
    # for batch in dataloader:
    #     batch: DataBatch = batch
    #     # print(batch)
    #     print(f"Batch size: {batch.batch_size}, Text lens: {batch.text_lens}, Mel lens: {batch.mel_lens}")
    #     print(f"Speakers: {batch.speakers}, Texts: {batch.texts.shape}")
    #     print("-" * 50)
    #     for sample in batch.data_samples:
    #         # print(sample)
    #         print(f"Speaker: {sample.speaker}, Text: {sample.text.shape}, Emotion: {sample.emotion}")
    #         print(f"Raw Text: {sample.raw_text}")
    #         print(f"Mel shape: {sample.mel.shape}, Pitch shape: {sample.pitch.shape}, Energy shape: {sample.energy.shape}, Duration shape: {sample.duration.shape}") # type: ignore
    #         print("-" * 50)

    #     break

if __name__ == "__main__":
    main()
