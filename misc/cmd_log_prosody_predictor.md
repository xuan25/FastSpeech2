<!-- python train_prosody_predictor.py --dataset_config config/LibriTTS/dataset.yaml --train_config config/LibriTTS/train.yaml --model_config config/LibriTTS/model.yaml --output_dir output/prosody_predictor/default
python train_prosody_predictor.py --dataset_config config/LibriTTS/dataset_sentiment.yaml --train_config config/LibriTTS/train.yaml --model_config config/LibriTTS/model_sentiment.yaml --output_dir output/prosody_predictor/sentiment
python train_prosody_predictor.py --dataset_config config/LibriTTS/dataset_sentiment_override_all_neu.yaml --train_config config/LibriTTS/train.yaml --model_config config/LibriTTS/model_sentiment.yaml --output_dir output/prosody_predictor/sentiment_override_all_neu -->





<!-- 
python train_prosody_predictor.py --dataset_config config/LibriTTS/dataset.yaml --train_config config/LibriTTS/train.yaml --model_config config/LibriTTS/model.yaml --output_dir output/prosody_predictor/default

python train_prosody_predictor.py --dataset_config config/LibriTTS/dataset_sentiment.yaml --train_config config/LibriTTS/train.yaml --model_config config/LibriTTS/model_sentiment_input.yaml --output_dir output/prosody_predictor/sentiment_input
python train_prosody_predictor.py --dataset_config config/LibriTTS/dataset_sentiment_override_all_neu.yaml --train_config config/LibriTTS/train.yaml --model_config config/LibriTTS/model_sentiment_input.yaml --output_dir output/prosody_predictor/sentiment_input_override_all_neu

python train_prosody_predictor.py --dataset_config config/LibriTTS/dataset_sentiment.yaml --train_config config/LibriTTS/train.yaml --model_config config/LibriTTS/model_sentiment_after_encoder.yaml --output_dir output/prosody_predictor/sentiment_after_encoder
python train_prosody_predictor.py --dataset_config config/LibriTTS/dataset_sentiment_override_all_neu.yaml --train_config config/LibriTTS/train.yaml --model_config config/LibriTTS/model_sentiment_after_encoder.yaml --output_dir output/prosody_predictor/sentiment_after_encoder_override_all_neu

python train_prosody_predictor.py --dataset_config config/LibriTTS/dataset_sentiment.yaml --train_config config/LibriTTS/train.yaml --model_config config/LibriTTS/model_sentiment_before_prosodic_predictors.yaml --output_dir output/prosody_predictor/sentimentt_before_prosodic_predictors
python train_prosody_predictor.py --dataset_config config/LibriTTS/dataset_sentiment_override_all_neu.yaml --train_config config/LibriTTS/train.yaml --model_config config/LibriTTS/model_sentiment_before_prosodic_predictors.yaml --output_dir output/prosody_predictor/sentiment_before_prosodic_predictors_override_all_neu -->



python train_prosody_predictor.py --dataset_config config/LibriTTS/dataset.yaml --train_config config/LibriTTS/train_prosody_predictor.yaml --model_config config/LibriTTS/model.yaml --output_dir output/prosody_predictor/test




python train_prosody_predictor.py --dataset_config config/LibriTTS/dataset.yaml --train_config config/LibriTTS/train_prosody_predictor.yaml --model_config config/LibriTTS/model.yaml --output_dir output/prosody_predictor/default

python train_prosody_predictor.py --dataset_config config/LibriTTS/dataset_sentiment.yaml --train_config config/LibriTTS/train_prosody_predictor.yaml --model_config config/LibriTTS/model_sentiment_input.yaml --output_dir output/prosody_predictor/sentiment_input
python train_prosody_predictor.py --dataset_config config/LibriTTS/dataset_sentiment_override_all_neu.yaml --train_config config/LibriTTS/train_prosody_predictor.yaml --model_config config/LibriTTS/model_sentiment_input.yaml --output_dir output/prosody_predictor/sentiment_input_override_all_neu

python train_prosody_predictor.py --dataset_config config/LibriTTS/dataset_sentiment.yaml --train_config config/LibriTTS/train_prosody_predictor.yaml --model_config config/LibriTTS/model_sentiment_after_encoder.yaml --output_dir output/prosody_predictor/sentiment_after_encoder
python train_prosody_predictor.py --dataset_config config/LibriTTS/dataset_sentiment_override_all_neu.yaml --train_config config/LibriTTS/train_prosody_predictor.yaml --model_config config/LibriTTS/model_sentiment_after_encoder.yaml --output_dir output/prosody_predictor/sentiment_after_encoder_override_all_neu

python train_prosody_predictor.py --dataset_config config/LibriTTS/dataset_sentiment.yaml --train_config config/LibriTTS/train_prosody_predictor.yaml --model_config config/LibriTTS/model_sentiment_before_prosodic_predictors.yaml --output_dir output/prosody_predictor/sentiment_before_prosodic_predictors
python train_prosody_predictor.py --dataset_config config/LibriTTS/dataset_sentiment_override_all_neu.yaml --train_config config/LibriTTS/train_prosody_predictor.yaml --model_config config/LibriTTS/model_sentiment_before_prosodic_predictors.yaml --output_dir output/prosody_predictor/sentiment_before_prosodic_predictors_override_all_neu