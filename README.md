
## E.g. setup
```sh
# export DATASET=expresso
# export ASPECT=style
export DATASET=libritts
export ASPECT=sentiment

# export TRAIN_CONFIG=train_prosody_predictor
# export MODEL_CONFIG=default

# export TRAIN_CONFIG=train_prosody_predictor_contrastive_0.1_0.1
# export MODEL_CONFIG=contrastive7_0.1_0.1

# export TRAIN_CONFIG=train_prosody_predictor_contrastive_0.1_1
# export MODEL_CONFIG=contrastive7_0.1_1

# export TRAIN_CONFIG=train_prosody_predictor_contrastive_1_1
# export MODEL_CONFIG=contrastive7_1_1

export TRAIN_CONFIG=train_prosody_predictor_contrastive_0.1_0.1
export MODEL_CONFIG=contrastive_emb_0.1_0.1

export DATA_SPLIT=train
# export DATA_SPLIT=test
export FACTOR=pitch
```

## For each dataset

1. Export the ground truth prosody features to a CSV file for the training set.
```sh
python gt_export.py \
--dataset_config config/dataset_$DATASET\_$ASPECT.yaml \
--data_split $DATA_SPLIT \
--output_dir output/$DATASET/$ASPECT/prosody_predictor_gt/gt/pred/$DATA_SPLIT
```

2. Compute the Wasserstein distance between the anchor prosody features and store the distances in a CSV file.
```sh
python compute_anchor_wasserstein_distances.py \
--data_label_meta data/$DATASET\_$ASPECT.csv \
--anchor_file output/$DATASET/$ASPECT/prosody_predictor_gt/gt/pred/train/0.csv \
--output_dir output/$DATASET/$ASPECT/prosody_predictor_gt/gt/wasserstein_distance/train_anchors
```

3. Compute the anchor prosody feature visualization locations in the PCA space defined by the anchor prosody features, strore the transforms.
```sh
python compute_anchor_loc_pca.py \
--data_label_meta data/$DATASET\_$ASPECT.csv \
--anchor_file output/$DATASET/$ASPECT/prosody_predictor_gt/gt/pred/train/0.csv \
--output_dir output/$DATASET/$ASPECT/prosody_predictor_gt/gt/loc_anchor_train
```

## For baseline

1. Train the prosody predictor without contrastive learning.
```sh
python train_prosody_predictor.py \
--dataset_config config/dataset_$DATASET\_$ASPECT.yaml \
--train_config config/$TRAIN_CONFIG.yaml \
--model_config config/model.yaml \
--output_dir output/$DATASET/$ASPECT/prosody_predictor/default/$MODEL_CONFIG
```

2. Predict the prosody features with the trained model.
<!-- ```sh
python prosody_predictor_predict_batch_label.py \
--model_ckpt_dir output/$DATASET/$ASPECT/prosody_predictor/default/$MODEL_CONFIG \
--dataset_config_path config/dataset_$DATASET\_$ASPECT.yaml \
--model_config_path config/model.yaml \
--label_index_file data/$DATASET\_$ASPECT.csv \
--data_split_name $DATA_SPLIT \
--ckpt_begin 1000 \
--ckpt_end 400000 \
--ckpt_step 40000 \
--skip_conf_steps 0
``` -->

```sh
python prosody_predictor_predict_batch_label_teacher_forcing.py \
--model_ckpt_dir output/$DATASET/$ASPECT/prosody_predictor/default/$MODEL_CONFIG \
--dataset_config_path config/dataset_$DATASET\_$ASPECT.yaml \
--model_config_path config/model.yaml \
--label_index_file data/$DATASET\_$ASPECT.csv \
--data_split_name $DATA_SPLIT \
--ckpt_begin 1000 \
--ckpt_end 400000 \
--ckpt_step 40000 \
--skip_conf_steps 0
```

## for each model (prediction) but teacher forcing

2. Predict the prosody features with the trained model, but with teacher forcing.
```sh
python prosody_predictor_predict_batch_label_teacher_forcing.py \
--model_ckpt_dir output/$DATASET/$ASPECT/prosody_predictor/embedding_input/$MODEL_CONFIG \
--dataset_config_path config/dataset_$DATASET\_$ASPECT.yaml \
--model_config_path config/model_label_input.yaml \
--label_index_file data/$DATASET\_$ASPECT.csv \
--data_split_name $DATA_SPLIT \
--ckpt_begin 1000 \
--ckpt_end 400000 \
--ckpt_step 40000 \
--skip_conf_steps 0
```

## For each model

1. Train the prosody predictor with contrastive learning.
```sh
python train_prosody_predictor_contrastive.py \
--dataset_config config/dataset_$DATASET\_$ASPECT.yaml \
--train_config config/$TRAIN_CONFIG.yaml \
--model_config config/model_label_input.yaml \
--output_dir output/$DATASET/$ASPECT/prosody_predictor/embedding_input/$MODEL_CONFIG
```

2. Predict the prosody features with the trained model.
```sh
python prosody_predictor_predict_batch_label.py \
--model_ckpt_dir output/$DATASET/$ASPECT/prosody_predictor/embedding_input/$MODEL_CONFIG \
--dataset_config_path config/dataset_$DATASET\_$ASPECT.yaml \
--model_config_path config/model_label_input.yaml \
--label_index_file data/$DATASET\_$ASPECT.csv \
--data_split_name $DATA_SPLIT \
--ckpt_begin 1000 \
--ckpt_end 400000 \
--ckpt_step 40000 \
--skip_conf_steps 0
```

3. Compute the Wasserstein distance between the predicted prosody features and the anchor prosody features.
```sh
python compute_target_wasserstein_distances.py \
--data_label_meta data/$DATASET\_$ASPECT.csv \
--anchor_file output/$DATASET/$ASPECT/prosody_predictor_gt/gt/pred/$DATA_SPLIT/0.csv \
--pred_dir output/$DATASET/$ASPECT/prosody_predictor/embedding_input/$MODEL_CONFIG/pred/$DATA_SPLIT \
--output_dir output/$DATASET/$ASPECT/prosody_predictor/embedding_input/$MODEL_CONFIG/wasserstein_distance/$DATA_SPLIT
```

4. Compute the target prosody feature locations in the PCA space defined by the anchor prosody features, and store the locations.
```sh
python compute_target_loc_pca.py \
--anchor_loc_file output/$DATASET/$ASPECT/prosody_predictor_gt/gt/loc_anchor_train/$FACTOR\_anchor_loc_pca.csv \
--anchor_loc_transform_file output/$DATASET/$ASPECT/prosody_predictor_gt/gt/loc_anchor_train/$FACTOR\_anchor_pca_transform.pkl \
--target_distance_dir output/$DATASET/$ASPECT/prosody_predictor/embedding_input/$MODEL_CONFIG/wasserstein_distance/$DATA_SPLIT/$FACTOR \
--output_dir output/$DATASET/$ASPECT/prosody_predictor/embedding_input/$MODEL_CONFIG/loc/$DATA_SPLIT/$FACTOR/pca
```

5. Visualize the target prosody feature locations in the PCA space defined by the anchor prosody features, and save the visualization.
```sh
python plot_loc.py \
--anchor_loc_file output/$DATASET/$ASPECT/prosody_predictor_gt/gt/loc_anchor_train/$FACTOR\_anchor_loc_pca.csv \
--target_loc_dir output/$DATASET/$ASPECT/prosody_predictor/embedding_input/$MODEL_CONFIG/loc/$DATA_SPLIT/$FACTOR/pca \
--output_file output/$DATASET/$ASPECT/prosody_predictor/embedding_input/$MODEL_CONFIG/vis/$DATA_SPLIT/$FACTOR/pca.png \
--plot_margin_ratio_anchor 5 \
--tag $DATASET\_$ASPECT\_$MODEL_CONFIG\_$DATA_SPLIT\_$FACTOR\_pca
```

6. Compute the MSE between the predicted prosody features and the ground truth prosody features.
```sh
python compute_target_mse.py \
--data_label_meta data/$DATASET\_$ASPECT.csv \
--gt_file output/$DATASET/$ASPECT/prosody_predictor_gt/gt/pred/$DATA_SPLIT/0.csv \
--pred_dir output/$DATASET/$ASPECT/prosody_predictor/embedding_input/$MODEL_CONFIG/pred/$DATA_SPLIT \
--output_dir output/$DATASET/$ASPECT/prosody_predictor/embedding_input/$MODEL_CONFIG/mse/$DATA_SPLIT
```
