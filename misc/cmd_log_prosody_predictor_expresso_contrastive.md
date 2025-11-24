
# gt

- [x] python gt_export.py --output_dir output/expresso/style/prosody_predictor_gt/gt/pred --dataset_config config/dataset_expresso_style.yaml --data_split val
- [x] python gt_export.py --output_dir output/expresso/style/prosody_predictor_gt/gt/pred_train --dataset_config config/dataset_expresso_style.yaml --data_split train


# train

## baseline / defaults

- [x] python train_fastspeech2.py --dataset_config config/dataset_expresso.yaml --train_config config/train.yaml --model_config config/model.yaml --output_dir output/expresso/style/fastspeech2/default

- [x] python train_prosody_predictor.py --dataset_config config/dataset_expresso.yaml --train_config config/train.yaml --model_config config/model.yaml --output_dir output/expresso/style/prosody_predictor/default

- [x] python train_prosody_predictor.py --dataset_config config/dataset_expresso_style.yaml --train_config config/train.yaml --model_config config/model_label_input.yaml --output_dir output/expresso/style/prosody_predictor/embedding_input/default

## sentiment_input - contrastive5

- [x] python train_prosody_predictor_contrastive.py --dataset_config config/dataset_expresso_style.yaml --train_config config/train_prosody_predictor.yaml --model_config config/model_label_input.yaml --output_dir output/expresso/style/prosody_predictor/embedding_input/contrastive5_0

- [x] python train_prosody_predictor_contrastive.py --dataset_config config/dataset_expresso_style.yaml --train_config config/train_prosody_predictor_contrastive-0.1.yaml --model_config config/model_label_input.yaml --output_dir output/expresso/style/prosody_predictor/embedding_input/contrastive5_0.1


## sentiment_input - contrastive6

- [x] python train_prosody_predictor_contrastive.py --dataset_config config/dataset_expresso_style.yaml --train_config config/train_prosody_predictor.yaml --model_config config/model_label_input.yaml --output_dir output/expresso/style/prosody_predictor/embedding_input/contrastive6_0

- [x] ython train_prosody_predictor_contrastive.py --dataset_config config/dataset_expresso_style.yaml --train_config config/train_prosody_predictor_contrastive-0.1.yaml --model_config config/model_label_input.yaml --output_dir output/expresso/style/prosody_predictor/embedding_input/contrastive6_0.1

- [x] python train_prosody_predictor_contrastive.py --dataset_config config/dataset_expresso_style.yaml --train_config config/train_prosody_predictor_contrastive-0.1_0.1.yaml --model_config config/model_label_input.yaml --output_dir output/expresso/style/prosody_predictor/embedding_input/contrastive6_0.1_0.1

- [x] python train_prosody_predictor_contrastive.py --dataset_config config/dataset_expresso_style.yaml --train_config config/train_prosody_predictor_contrastive-0.1_0.1.yaml --model_config config/model_label_input.yaml --output_dir output/expresso/style/prosody_predictor/embedding_input/contrastive5_0.1_0.1


## sentiment_input - contrastive7

- [-] python train_prosody_predictor_contrastive.py --dataset_config config/dataset_expresso_style.yaml --train_config config/train_prosody_predictor.yaml --model_config config/model_label_input.yaml --output_dir output/expresso/style/prosody_predictor/embedding_input/contrastive7_0

- [-] python train_prosody_predictor_contrastive.py --dataset_config config/dataset_expresso_style.yaml --train_config config/train_prosody_predictor_contrastive-0.1.yaml --model_config config/model_label_input.yaml --output_dir output/expresso/style/prosody_predictor/embedding_input/contrastive7_0.1

- [-] python train_prosody_predictor_contrastive.py --dataset_config config/dataset_expresso_style.yaml --train_config config/train_prosody_predictor_contrastive-0.1_0.1.yaml --model_config config/model_label_input.yaml --output_dir output/expresso/style/prosody_predictor/embedding_input/contrastive7_0.1_0.1


## baseline / defaults - no whisper

- [ ] python train_fastspeech2.py --dataset_config config/dataset_expresso.yaml --train_config config/train.yaml --model_config config/model.yaml --output_dir output/expresso/style/fastspeech2/default

- [ ] python train_prosody_predictor.py --dataset_config config/dataset_expresso.yaml --train_config config/train.yaml --model_config config/model.yaml --output_dir output/expresso/style/prosody_predictor/default

- [x] python train_prosody_predictor.py --dataset_config config/dataset_expresso_style_no_whisper.yaml --train_config config/train.yaml --model_config config/model_label_input.yaml --output_dir output/expresso/style/prosody_predictor/embedding_input/default_no_whisper



## sentiment_input - contrastive6 - no whisper


- [x] python train_prosody_predictor_contrastive.py --dataset_config config/dataset_expresso_style_no_whisper.yaml --train_config config/train_prosody_predictor_contrastive-0.1_0.1.yaml --model_config config/model_label_input.yaml --output_dir output/expresso/style/prosody_predictor/embedding_input/contrastive6_0.1_0.1_no_whisper

- [x] python train_prosody_predictor_contrastive.py --dataset_config config/dataset_expresso_style_no_whisper.yaml --train_config config/train_prosody_predictor_contrastive-0.1.yaml --model_config config/model_label_input.yaml --output_dir output/expresso/style/prosody_predictor/embedding_input/contrastive6_0.1_no_whisper

- [x] python train_prosody_predictor_contrastive.py --dataset_config config/dataset_expresso_style_no_whisper.yaml --train_config config/train_prosody_predictor_contrastive-0.1_0.1.yaml --model_config config/model_label_input.yaml --output_dir output/expresso/style/prosody_predictor/embedding_input/contrastive5_0.1_0.1_no_whisper

- [x] python train_prosody_predictor_contrastive.py --dataset_config config/dataset_expresso_style_no_whisper.yaml --train_config config/train_prosody_predictor_contrastive-0.1.yaml --model_config config/model_label_input.yaml --output_dir output/expresso/style/prosody_predictor/embedding_input/contrastive5_0.1_no_whisper



# pred

python prosody_predictor_predict_batch_label.py


# distances

python compute_target_wasserstein_distances.py

# vis

<!-- python compute_anchor_loc_smacof.py
python compute_target_loc_lstsq.py -->

python compute_anchor_loc_pca.py
python compute_target_loc_pca.py

python plot_loc.py