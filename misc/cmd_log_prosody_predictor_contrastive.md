
# gt

python stats_gt.py --output_dir output/prosody_predictor_gt/gt/pred --dataset_config config/LibriTTS/dataset_sentiment.yaml --data_split val


# train

python train_fastspeech2.py --dataset_config config/LibriTTS/dataset.yaml --train_config config/LibriTTS/train.yaml --model_config config/LibriTTS/model.yaml --output_dir output/fastspeech2/default

python train_prosody_predictor_contrastive.py --dataset_config config/LibriTTS/dataset_sentiment.yaml --train_config config/LibriTTS/train_prosody_predictor_contrastive-0.yaml --model_config config/LibriTTS/model_sentiment_input.yaml --output_dir output/prosody_predictor_contrastive/sentiment_input-0

python train_prosody_predictor_contrastive.py --dataset_config config/LibriTTS/dataset_sentiment.yaml --train_config config/LibriTTS/train_prosody_predictor_contrastive-0.5.yaml --model_config config/LibriTTS/model_sentiment_input.yaml --output_dir output/prosody_predictor_contrastive/sentiment_input-0.5

python train_prosody_predictor_contrastive.py --dataset_config config/LibriTTS/dataset_sentiment.yaml --train_config config/LibriTTS/train_prosody_predictor_contrastive-0.3.yaml --model_config config/LibriTTS/model_sentiment_input.yaml --output_dir output/prosody_predictor_contrastive/sentiment_input-0.3

python train_prosody_predictor_contrastive.py --dataset_config config/LibriTTS/dataset_sentiment.yaml --train_config config/LibriTTS/train_prosody_predictor_contrastive-0.3-0.1.yaml --model_config config/LibriTTS/model_sentiment_input.yaml --output_dir output/prosody_predictor_contrastive/sentiment_input-0.3-0.1

python train_prosody_predictor_contrastive.py --dataset_config config/LibriTTS/dataset_sentiment.yaml --train_config config/LibriTTS/train_prosody_predictor_contrastive-0.1-0.3.yaml --model_config config/LibriTTS/model_sentiment_input.yaml --output_dir output/prosody_predictor_contrastive/sentiment_input-0.1-0.3



python train_prosody_predictor_contrastive.py --dataset_config config/LibriTTS/dataset_sentiment.yaml --train_config config/LibriTTS/train_prosody_predictor_contrastive-0.yaml --model_config config/LibriTTS/model_sentiment_input.yaml --output_dir output/prosody_predictor_contrastive/sentiment_input-0-overrideembedding0

python train_prosody_predictor_contrastive.py --dataset_config config/LibriTTS/dataset_sentiment.yaml --train_config config/LibriTTS/train_prosody_predictor_contrastive-0.yaml --model_config config/LibriTTS/model_sentiment_input.yaml --output_dir output/prosody_predictor_contrastive/sentiment_input-0-overrideembedding0-B




python train_prosody_predictor.py --dataset_config config/LibriTTS/dataset.yaml --train_config config/LibriTTS/train_prosody_predictor.yaml --model_config config/LibriTTS/model.yaml --output_dir output/prosody_predictor/default-B

python train_prosody_predictor.py --dataset_config config/LibriTTS/dataset_sentiment.yaml --train_config config/LibriTTS/train_prosody_predictor.yaml --model_config config/LibriTTS/model_sentiment_input.yaml --output_dir output/prosody_predictor/sentiment_input

python train_prosody_predictor.py --dataset_config config/LibriTTS/dataset_sentiment.yaml --train_config config/LibriTTS/train_prosody_predictor.yaml --model_config config/LibriTTS/model_sentiment_input.yaml --output_dir output/prosody_predictor/sentiment_input_overrideembedding0



# pred

python prosody_predictor_predict_batch.py


# stat

<!-- python prosody_predictor_stats.py -->
python wasserstein_distance.py




