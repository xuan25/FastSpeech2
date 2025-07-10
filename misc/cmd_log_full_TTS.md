python train.py --dataset_config config/LibriTTS/dataset.yaml --train_config config/LibriTTS/train.yaml --model_config config/LibriTTS/model.yaml --output_dir output/default
python train.py --dataset_config config/LibriTTS/dataset_sentiment.yaml --train_config config/LibriTTS/train.yaml --model_config config/LibriTTS/model_sentiment.yaml --output_dir output/sentiment
python train.py --dataset_config config/LibriTTS/dataset_sentiment_override_all_neu.yaml --train_config config/LibriTTS/train.yaml --model_config config/LibriTTS/model_sentiment.yaml --output_dir output/sentiment_override_all_neu


CUDA_VISIBLE_DEVICES=2
python synthesize_batch.py --ckpt_path output/default/ckpt/200000.pth --output_dir output/default/synth_val --dataset_config config/LibriTTS/dataset.yaml --model_config config/LibriTTS/model.yaml --data_split val
python synthesize_batch.py --ckpt_path output/sentiment/ckpt/200000.pth --output_dir output/sentiment/synth_val --dataset_config config/LibriTTS/dataset_sentiment.yaml --model_config config/LibriTTS/model_sentiment.yaml --data_split val
python synthesize_batch.py --ckpt_path output/sentiment_override_all_neu/ckpt/200000.pth --output_dir output/sentiment_override_all_neu/synth_val --dataset_config config/LibriTTS/dataset_sentiment_override_all_neu.yaml --model_config config/LibriTTS/model_sentiment.yaml --data_split val

python reconstruct.py --output_dir output/reconstruct/synth_val --dataset_config config/LibriTTS/dataset.yaml --model_config config/LibriTTS/model.yaml --data_split val






python synthesize_batch.py --ckpt_path output/sentiment/ckpt/200000.pth --output_dir output/sentiment/synth_val_override_pos --dataset_config config/LibriTTS/dataset_sentiment_override_all_pos.yaml --model_config config/LibriTTS/model_sentiment.yaml --data_split val
python synthesize_batch.py --ckpt_path output/sentiment/ckpt/200000.pth --output_dir output/sentiment/synth_val_override_neg --dataset_config config/LibriTTS/dataset_sentiment_override_all_neg.yaml --model_config config/LibriTTS/model_sentiment.yaml --data_split val
python synthesize_batch.py --ckpt_path output/sentiment/ckpt/200000.pth --output_dir output/sentiment/synth_val_override_neu --dataset_config config/LibriTTS/dataset_sentiment_override_all_neu.yaml --model_config config/LibriTTS/model_sentiment.yaml --data_split val





















CUDA_VISIBLE_DEVICES=2
CUDA_VISIBLE_DEVICES=2 python synthesize_batch.py --ckpt_path output/default-nomask/ckpt/200000.pth --output_dir output/default-nomask/synth_val --dataset_config config/LibriTTS/dataset.yaml --model_config config/LibriTTS/model.yaml --data_split val
CUDA_VISIBLE_DEVICES=2 python synthesize_batch.py --ckpt_path output/sentiment-nomask/ckpt/200000.pth --output_dir output/sentiment-nomask/synth_val --dataset_config config/LibriTTS/dataset_sentiment.yaml --model_config config/LibriTTS/model_sentiment.yaml --data_split val
CUDA_VISIBLE_DEVICES=2 python synthesize_batch.py --ckpt_path output/sentiment_override_all_neu-nomask/ckpt/200000.pth --output_dir output/sentiment_override_all_neu-nomask/synth_val --dataset_config config/LibriTTS/dataset_sentiment_override_all_neu.yaml --model_config config/LibriTTS/model_sentiment.yaml --data_split val


CUDA_VISIBLE_DEVICES=2 python synthesize_batch.py --ckpt_path output/sentiment-nomask/ckpt/200000.pth --output_dir output/sentiment-nomask/synth_val_override_pos --dataset_config config/LibriTTS/dataset_sentiment_override_all_pos.yaml --model_config config/LibriTTS/model_sentiment.yaml --data_split val
CUDA_VISIBLE_DEVICES=2 python synthesize_batch.py --ckpt_path output/sentiment-nomask/ckpt/200000.pth --output_dir output/sentiment-nomask/synth_val_override_neg --dataset_config config/LibriTTS/dataset_sentiment_override_all_neg.yaml --model_config config/LibriTTS/model_sentiment.yaml --data_split val
CUDA_VISIBLE_DEVICES=2 python synthesize_batch.py --ckpt_path output/sentiment-nomask/ckpt/200000.pth --output_dir output/sentiment-nomask/synth_val_override_neu --dataset_config config/LibriTTS/dataset_sentiment_override_all_neu.yaml --model_config config/LibriTTS/model_sentiment.yaml --data_split val