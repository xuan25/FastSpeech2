

<!-- python train_prosody_predictor_contrastive_trace.py --dataset_config config/LibriTTS/dataset_sentiment.yaml --train_config config/LibriTTS/train_prosody_predictor.yaml --model_config config/LibriTTS/model_sentiment_input.yaml --output_dir output/prosody_predictor_contrastive/sentiment_input -->



python train_prosody_predictor_contrastive.py --dataset_config config/LibriTTS/dataset_sentiment.yaml --train_config config/LibriTTS/train_prosody_predictor.yaml --model_config config/LibriTTS/model_sentiment_input.yaml --output_dir output/prosody_predictor_contrastive/sentiment_input


python train_prosody_predictor_contrastive.py --dataset_config config/LibriTTS/dataset_sentiment.yaml --train_config config/LibriTTS/train_prosody_predictor.yaml --model_config config/LibriTTS/model_sentiment_input.yaml --output_dir output/prosody_predictor_contrastive/sentiment_input-0.00001

python train_prosody_predictor_contrastive.py --dataset_config config/LibriTTS/dataset_sentiment.yaml --train_config config/LibriTTS/train_prosody_predictor.yaml --model_config config/LibriTTS/model_sentiment_input.yaml --output_dir output/prosody_predictor_contrastive/sentiment_input-0

python train_prosody_predictor_contrastive.py --dataset_config config/LibriTTS/dataset_sentiment.yaml --train_config config/LibriTTS/train_prosody_predictor.yaml --model_config config/LibriTTS/model_sentiment_input.yaml --output_dir output/prosody_predictor_contrastive/sentiment_input-non-contrastive-sample


python train_prosody_predictor_contrastive.py --dataset_config config/LibriTTS/dataset_sentiment.yaml --train_config config/LibriTTS/train_prosody_predictor.yaml --model_config config/LibriTTS/model_sentiment_input.yaml --output_dir output/prosody_predictor_contrastive/sentiment_input-0-A

python train_prosody_predictor_contrastive.py --dataset_config config/LibriTTS/dataset_sentiment.yaml --train_config config/LibriTTS/train_prosody_predictor.yaml --model_config config/LibriTTS/model_sentiment_input.yaml --output_dir output/prosody_predictor_contrastive/sentiment_input-0-reworked

python train_prosody_predictor_contrastive.py --dataset_config config/LibriTTS/dataset_sentiment.yaml --train_config config/LibriTTS/train_prosody_predictor.yaml --model_config config/LibriTTS/model_sentiment_input.yaml --output_dir output/prosody_predictor_contrastive/sentiment_input-0-reworked-smallerbatch

python train_prosody_predictor_contrastive.py --dataset_config config/LibriTTS/dataset_sentiment.yaml --train_config config/LibriTTS/train_prosody_predictor.yaml --model_config config/LibriTTS/model_sentiment_input.yaml --output_dir output/prosody_predictor_contrastive/sentiment_input-reworked-0.001



python train_prosody_predictor_contrastive.py --dataset_config config/LibriTTS/dataset_sentiment.yaml --train_config config/LibriTTS/train_prosody_predictor.yaml --model_config config/LibriTTS/model_sentiment_input.yaml --output_dir output/prosody_predictor_contrastive/sentiment_input-reworked-hardremovecontrastive


# pred

python predict_batch_prosody_predictor.py --ckpt_path output/prosody_predictor_contrastive/sentiment_input/ckpt/40000.pth --output_path output/prosody_predictor_contrastive/sentiment_input/pred/val.csv --dataset_config config/LibriTTS/dataset_sentiment.yaml --model_config config/LibriTTS/model_sentiment_input.yaml --data_split val

python predict_batch_prosody_predictor.py --ckpt_path output/prosody_predictor_contrastive/sentiment_input/ckpt/40000.pth --output_path output/prosody_predictor_contrastive/sentiment_input/pred/val_neg.csv --dataset_config config/LibriTTS/dataset_sentiment_override_all_neg.yaml --model_config config/LibriTTS/model_sentiment_input.yaml --data_split val

python predict_batch_prosody_predictor.py --ckpt_path output/prosody_predictor_contrastive/sentiment_input/ckpt/40000.pth --output_path output/prosody_predictor_contrastive/sentiment_input/pred/val_neu.csv --dataset_config config/LibriTTS/dataset_sentiment_override_all_neu.yaml --model_config config/LibriTTS/model_sentiment_input.yaml --data_split val

python predict_batch_prosody_predictor.py --ckpt_path output/prosody_predictor_contrastive/sentiment_input/ckpt/40000.pth --output_path output/prosody_predictor_contrastive/sentiment_input/pred/val_pos.csv --dataset_config config/LibriTTS/dataset_sentiment_override_all_pos.yaml --model_config config/LibriTTS/model_sentiment_input.yaml --data_split val



python predict_batch_prosody_predictor.py --ckpt_path output/prosody_predictor_contrastive/sentiment_input-reworked-0.001/ckpt/60000.pth --output_path output/prosody_predictor_contrastive/sentiment_input-reworked-0.001/pred/val.csv --dataset_config config/LibriTTS/dataset_sentiment.yaml --model_config config/LibriTTS/model_sentiment_input.yaml --data_split val

python predict_batch_prosody_predictor.py --ckpt_path output/prosody_predictor_contrastive/sentiment_input-reworked-0.001/ckpt/60000.pth --output_path output/prosody_predictor_contrastive/sentiment_input-reworked-0.001/pred/val_neg.csv --dataset_config config/LibriTTS/dataset_sentiment_override_all_neg.yaml --model_config config/LibriTTS/model_sentiment_input.yaml --data_split val

python predict_batch_prosody_predictor.py --ckpt_path output/prosody_predictor_contrastive/sentiment_input-reworked-0.001/ckpt/60000.pth --output_path output/prosody_predictor_contrastive/sentiment_input-reworked-0.001/pred/val_neu.csv --dataset_config config/LibriTTS/dataset_sentiment_override_all_neu.yaml --model_config config/LibriTTS/model_sentiment_input.yaml --data_split val

python predict_batch_prosody_predictor.py --ckpt_path output/prosody_predictor_contrastive/sentiment_input-reworked-0.001/ckpt/60000.pth --output_path output/prosody_predictor_contrastive/sentiment_input-reworked-0.001/pred/val_pos.csv --dataset_config config/LibriTTS/dataset_sentiment_override_all_pos.yaml --model_config config/LibriTTS/model_sentiment_input.yaml --data_split val



python predict_batch_prosody_predictor.py --ckpt_path output/prosody_predictor_contrastive/sentiment_input-0-reworked/ckpt/40000.pth --output_path output/prosody_predictor_contrastive/sentiment_input-0-reworked/pred/val.csv --dataset_config config/LibriTTS/dataset_sentiment.yaml --model_config config/LibriTTS/model_sentiment_input.yaml --data_split val

python predict_batch_prosody_predictor.py --ckpt_path output/prosody_predictor_contrastive/sentiment_input-0-reworked/ckpt/40000.pth --output_path output/prosody_predictor_contrastive/sentiment_input-0-reworked/pred/val_neg.csv --dataset_config config/LibriTTS/dataset_sentiment_override_all_neg.yaml --model_config config/LibriTTS/model_sentiment_input.yaml --data_split val

python predict_batch_prosody_predictor.py --ckpt_path output/prosody_predictor_contrastive/sentiment_input-0-reworked/ckpt/40000.pth --output_path output/prosody_predictor_contrastive/sentiment_input-0-reworked/pred/val_neu.csv --dataset_config config/LibriTTS/dataset_sentiment_override_all_neu.yaml --model_config config/LibriTTS/model_sentiment_input.yaml --data_split val

python predict_batch_prosody_predictor.py --ckpt_path output/prosody_predictor_contrastive/sentiment_input-0-reworked/ckpt/40000.pth --output_path output/prosody_predictor_contrastive/sentiment_input-0-reworked/pred/val_pos.csv --dataset_config config/LibriTTS/dataset_sentiment_override_all_pos.yaml --model_config config/LibriTTS/model_sentiment_input.yaml --data_split val





python plot_violin.py