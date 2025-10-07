
# gt

python stats_gt.py --output_dir output/prosody_predictor_gt/gt/pred --dataset_config config/LibriTTS/dataset_sentiment.yaml --data_split val


# train

## baseline

<!-- python train_fastspeech2.py --dataset_config config/LibriTTS/dataset.yaml --train_config config/LibriTTS/train.yaml --model_config config/LibriTTS/model.yaml --output_dir output/fastspeech2/default -->

## sentiment_input - contrastive

python train_prosody_predictor_contrastive.py --dataset_config config/LibriTTS/dataset_sentiment.yaml --train_config config/LibriTTS/train_prosody_predictor_contrastive-0.1.yaml --model_config config/LibriTTS/model_sentiment_input_translate.yaml --output_dir output/prosody_predictor_contrastive/sentiment_input_translate-0.1

python train_prosody_predictor_contrastive.py --dataset_config config/LibriTTS/dataset_sentiment.yaml --train_config config/LibriTTS/train_prosody_predictor_contrastive-0.01.yaml --model_config config/LibriTTS/model_sentiment_input_translate.yaml --output_dir output/prosody_predictor_contrastive/sentiment_input_translate-0.01

python train_prosody_predictor_contrastive.py --dataset_config config/LibriTTS/dataset_sentiment.yaml --train_config config/LibriTTS/train_prosody_predictor_contrastive-0.yaml --model_config config/LibriTTS/model_sentiment_input_translate.yaml --output_dir output/prosody_predictor_contrastive/sentiment_input_translate-0

python train_prosody_predictor_contrastive.py --dataset_config config/LibriTTS/dataset_sentiment.yaml --train_config config/LibriTTS/train_prosody_predictor_contrastive-0.1.yaml --model_config config/LibriTTS/model_sentiment_input_translate2.yaml --output_dir output/prosody_predictor_contrastive/sentiment_input_translate2-0.1

python train_prosody_predictor_contrastive.py --dataset_config config/LibriTTS/dataset_sentiment.yaml --train_config config/LibriTTS/train_prosody_predictor_contrastive-0.yaml --model_config config/LibriTTS/model_sentiment_input_translate2.yaml --output_dir output/prosody_predictor_contrastive/sentiment_input_translate2-0

python train_prosody_predictor_contrastive.py --dataset_config config/LibriTTS/dataset_sentiment.yaml --train_config config/LibriTTS/train_prosody_predictor_contrastive-0.001.yaml --model_config config/LibriTTS/model_sentiment_input_translate2.yaml --output_dir output/prosody_predictor_contrastive/sentiment_input_translate2-0.001

python train_prosody_predictor_contrastive.py --dataset_config config/LibriTTS/dataset_sentiment.yaml --train_config config/LibriTTS/train_prosody_predictor_contrastive-0.0001.yaml --model_config config/LibriTTS/model_sentiment_input_translate2.yaml --output_dir output/prosody_predictor_contrastive/sentiment_input_translate2-0.0001







# new loss

python train_prosody_predictor_contrastive.py --dataset_config config/LibriTTS/dataset_sentiment.yaml --train_config config/LibriTTS/train_prosody_predictor_contrastive-0.1.yaml --model_config config/LibriTTS/model_sentiment_input_translate2.yaml --output_dir output/prosody_predictor_contrastive2/sentiment_input_translate2-0.1

python train_prosody_predictor_contrastive.py --dataset_config config/LibriTTS/dataset_sentiment.yaml --train_config config/LibriTTS/train_prosody_predictor_contrastive-0.3.yaml --model_config config/LibriTTS/model_sentiment_input_translate2.yaml --output_dir output/prosody_predictor_contrastive2/sentiment_input_translate2-0.3

python train_prosody_predictor_contrastive.py --dataset_config config/LibriTTS/dataset_sentiment.yaml --train_config config/LibriTTS/train_prosody_predictor_contrastive-0.5.yaml --model_config config/LibriTTS/model_sentiment_input_translate2.yaml --output_dir output/prosody_predictor_contrastive2/sentiment_input_translate2-0.5

python train_prosody_predictor_contrastive.py --dataset_config config/LibriTTS/dataset_sentiment.yaml --train_config config/LibriTTS/train_prosody_predictor_contrastive-1.yaml --model_config config/LibriTTS/model_sentiment_input_translate2.yaml --output_dir output/prosody_predictor_contrastive2/sentiment_input_translate2-1

python train_prosody_predictor_contrastive.py --dataset_config config/LibriTTS/dataset_sentiment.yaml --train_config config/LibriTTS/train_prosody_predictor_contrastive-10.yaml --model_config config/LibriTTS/model_sentiment_input_translate2.yaml --output_dir output/prosody_predictor_contrastive2/sentiment_input_translate2-10



<!-- python train_prosody_predictor_contrastive.py --dataset_config config/LibriTTS/dataset_sentiment.yaml --train_config config/LibriTTS/train_prosody_predictor_contrastive-1.yaml --model_config config/LibriTTS/model_sentiment_input.yaml --output_dir output/prosody_predictor_contrastive2/sentiment_input-1

python train_prosody_predictor_contrastive.py --dataset_config config/LibriTTS/dataset_sentiment.yaml --train_config config/LibriTTS/train_prosody_predictor_contrastive-0.5.yaml --model_config config/LibriTTS/model_sentiment_input.yaml --output_dir output/prosody_predictor_contrastive2/sentiment_input-0.5 -->

python train_prosody_predictor_contrastive.py --dataset_config config/LibriTTS/dataset_sentiment.yaml --train_config config/LibriTTS/train_prosody_predictor_contrastive-0.1.yaml --model_config config/LibriTTS/model_sentiment_input.yaml --output_dir output/prosody_predictor_contrastive2/sentiment_input-0.1

python train_prosody_predictor_contrastive.py --dataset_config config/LibriTTS/dataset_sentiment.yaml --train_config config/LibriTTS/train_prosody_predictor_contrastive-0.01.yaml --model_config config/LibriTTS/model_sentiment_input.yaml --output_dir output/prosody_predictor_contrastive2/sentiment_input-0.01



# pred

python prosody_predictor_predict_batch.py


# stat

<!-- python prosody_predictor_stats.py -->
python wasserstein_distance.py




