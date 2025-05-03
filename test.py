from fastspeech2.config import *

def get_test_configs():
    dataset_config = DatasetConfig(
        path_config=DatasetPathConfig(
            meta_file_train="data/augmented_data/LibriTTS-original/train.txt",
            meta_file_val="data/augmented_data/LibriTTS-original/val.txt",
            speaker_map_file="data/augmented_data/LibriTTS-original/speakers.json",
            feature_dir="data/augmented_data/LibriTTS-original",
            stats_file="data/augmented_data/LibriTTS-original/stats.json",
            sentiment_file="data/original/LibriTTS/sentiment_scores_libri-tts.csv",
        ),
        feature_properties_config=DatasetFeaturePropertiesConfig(
            pitch_feature_level="phoneme_level",
            energy_feature_level="phoneme_level",
            n_mel_channels=80,
            max_wav_value=32768.0,
            sampling_rate=22050,
            stft_hop_length=256,
        ),
        preprocessing_config=DatasetPreprocessingConfig(
            text_cleaners=["english_cleaners"],
        ),
    )

    model_config = ModelConfig(
        transformer_config=ModelTransformerConfig(
            encoder_layer=4,
            encoder_head=2,
            encoder_hidden=256,
            decoder_layer=4,
            decoder_head=2,
            decoder_hidden=256,
            conv_filter_size=1024,
            conv_kernel_size=[9, 1],
            encoder_dropout=0.2,
            decoder_dropout=0.2,
        ),
        variance_predictor_config=ModelVariancePredictorConfig(
            filter_size=256,
            kernel_size=3,
            dropout=0.5,
        ),
        variance_embedding_config=ModelVarianceEmbeddingConfig(
            pitch_quantization="linear",
            energy_quantization="linear",
            n_bins=256,
        ),
        vocoder_config=ModelVocoderConfig(
            model="HiFi-GAN",
            speaker="universal",
        ),
        global_config=ModelGlobalConfig(
            multi_speaker=True,
            num_sentiments=3,
            max_seq_len=1000,
        ),
    )

    train_config = TrainConfig(
        path_config=TrainPathConfig(
            ckpt_path="./output/ckpt/LibriTTS",
            log_path="./output/log/LibriTTS",
            result_path="./output/result/LibriTTS",
        ),
        optimizer_config=TrainOptimizerConfig(
            betas=[0.9, 0.98],
            eps=1e-9,
            weight_decay=0.0,
            grad_clip_thresh=1.0,
            grad_acc_step=1,
            warm_up_step=4000,
            anneal_steps=[300000, 400000, 500000],
            anneal_rate=0.3,
        ),
        step_config=TrainStepConfig(
            total_step=900000,
            log_step=100,
            synth_step=100,
            val_step=100,
            save_step=100,
            batch_size=16,
        ),
    )
    return dataset_config, model_config, train_config
