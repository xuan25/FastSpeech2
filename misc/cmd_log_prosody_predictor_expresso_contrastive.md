
## expresso style anchors

...


## expresso style default

...

python compute_target_loc_pca.py \
    --anchor_loc_file output/expresso/style/prosody_predictor_gt/gt/loc_anchor_train/duration_anchor_loc_pca.csv \
    --anchor_loc_transform_file output/expresso/style/prosody_predictor_gt/gt/loc_anchor_train/duration_anchor_pca_transform.pkl \
    --target_distance_dir output/expresso/style/prosody_predictor/embedding_input/default/wasserstein_distance/train/duration \
    --output_dir output/expresso/style/prosody_predictor/embedding_input/default/loc/train/duration/pca

python plot_loc.py \
    --anchor_loc_file output/expresso/style/prosody_predictor_gt/gt/loc_anchor_train/duration_anchor_loc_pca.csv \
    --target_loc_dir output/expresso/style/prosody_predictor/embedding_input/default/loc/train/duration/pca \
    --output_file output/expresso/style/prosody_predictor/embedding_input/default/vis/train/duration/pca.png \
    --tag expresso_style_default_train_duration_pca


## expresso style contrastive contrastive7_0.1_0.1

...

python compute_target_loc_pca.py \
    --anchor_loc_file output/expresso/style/prosody_predictor_gt/gt/loc_anchor_train/duration_anchor_loc_pca.csv \
    --anchor_loc_transform_file output/expresso/style/prosody_predictor_gt/gt/loc_anchor_train/duration_anchor_pca_transform.pkl \
    --target_distance_dir output/expresso/style/prosody_predictor/embedding_input/contrastive7_0.1_0.1/wasserstein_distance/train/duration \
    --output_dir output/expresso/style/prosody_predictor/embedding_input/contrastive7_0.1_0.1/loc/train/duration/pca

python plot_loc.py \
    --anchor_loc_file output/expresso/style/prosody_predictor_gt/gt/loc_anchor_train/duration_anchor_loc_pca.csv \
    --target_loc_dir output/expresso/style/prosody_predictor/embedding_input/contrastive7_0.1_0.1/loc/train/duration/pca \
    --output_file output/expresso/style/prosody_predictor/embedding_input/contrastive7_0.1_0.1/vis/train/duration/pca.png \
    --tag expresso_style_contrastive7_0.1_0.1_train_duration_pca


## expresso style contrastive contrastive7_1_1

...

python prosody_predictor_predict_batch_label.py \
    --model_ckpt_dir output/expresso/style/prosody_predictor/embedding_input/contrastive7_1_1 \
    --dataset_config_path config/dataset_expresso_style.yaml \
    --model_config_path config/model_label_input.yaml \
    --label_index_file data/expresso_style.csv \
    --data_split_name train \
    --ckpt_begin 1000 \
    --ckpt_end 400000 \
    --ckpt_step 40000 \

python compute_target_wasserstein_distances.py \
    --data_label_meta data/expresso_style.csv \
    --anchor_file output/expresso/style/prosody_predictor_gt/gt/pred/train/gt/0.csv \
    --pred_dir output/expresso/style/prosody_predictor/embedding_input/contrastive7_1_1/pred/train \
    --output_dir output/expresso/style/prosody_predictor/embedding_input/contrastive7_1_1/wasserstein_distance/train

python compute_target_loc_pca.py \
    --anchor_loc_file output/expresso/style/prosody_predictor_gt/gt/loc_anchor_train_no_whisper/pitch_anchor_loc_pca.csv \
    --anchor_loc_transform_file output/expresso/style/prosody_predictor_gt/gt/loc_anchor_train_no_whisper/pitch_anchor_pca_transform.pkl \
    --target_distance_dir output/expresso/style/prosody_predictor/embedding_input/contrastive7_1_1/wasserstein_distance/train/pitch \
    --output_dir output/expresso/style/prosody_predictor/embedding_input/contrastive7_1_1/loc_no_whisper/train/pitch/pca \
    --exclude_labels whisper

python plot_loc.py \
    --anchor_loc_file output/expresso/style/prosody_predictor_gt/gt/loc_anchor_train_no_whisper/pitch_anchor_loc_pca.csv \
    --target_loc_dir output/expresso/style/prosody_predictor/embedding_input/contrastive7_1_1/loc_no_whisper/train/pitch/pca \
    --output_file output/expresso/style/prosody_predictor/embedding_input/contrastive7_1_1/vis/train/pitch/pca_no_whisper.png \
    --tag expresso_style_contrastive7_1_1_train_pitch_pca_no_whisper \



## expresso style contrastive contrastive7_1_0.1

python train_prosody_predictor_contrastive.py \
    --dataset_config config/dataset_expresso_style.yaml \
    --train_config config/train_prosody_predictor_contrastive_1_0.1.yaml \
    --model_config config/model_label_input.yaml \
    --output_dir output/expresso/style/prosody_predictor/embedding_input/contrastive7_1_0.1

python prosody_predictor_predict_batch_label.py \
    --model_ckpt_dir output/expresso/style/prosody_predictor/embedding_input/contrastive7_1_0.1 \
    --dataset_config_path config/dataset_expresso_style.yaml \
    --model_config_path config/model_label_input.yaml \
    --label_index_file data/expresso_style.csv \
    --data_split_name train \
    --ckpt_begin 1000 \
    --ckpt_end 400000 \
    --ckpt_step 40000 \

python compute_target_wasserstein_distances.py \
    --data_label_meta data/expresso_style.csv \
    --anchor_file output/expresso/style/prosody_predictor_gt/gt/pred/train/gt/0.csv \
    --pred_dir output/expresso/style/prosody_predictor/embedding_input/contrastive7_1_0.1/pred/train \
    --output_dir output/expresso/style/prosody_predictor/embedding_input/contrastive7_1_0.1/wasserstein_distance/train

python compute_target_loc_pca.py \
    --anchor_loc_file output/expresso/style/prosody_predictor_gt/gt/loc_anchor_train_no_whisper/pitch_anchor_loc_pca.csv \
    --anchor_loc_transform_file output/expresso/style/prosody_predictor_gt/gt/loc_anchor_train_no_whisper/pitch_anchor_pca_transform.pkl \
    --target_distance_dir output/expresso/style/prosody_predictor/embedding_input/contrastive7_1_0.1/wasserstein_distance/train/pitch \
    --output_dir output/expresso/style/prosody_predictor/embedding_input/contrastive7_1_0.1/loc_no_whisper/train/pitch/pca \
    --exclude_labels whisper

python plot_loc.py \
    --anchor_loc_file output/expresso/style/prosody_predictor_gt/gt/loc_anchor_train_no_whisper/pitch_anchor_loc_pca.csv \
    --target_loc_dir output/expresso/style/prosody_predictor/embedding_input/contrastive7_1_0.1/loc_no_whisper/train/pitch/pca \
    --output_file output/expresso/style/prosody_predictor/embedding_input/contrastive7_1_0.1/vis_no_whisper/train/pitch/pca.png \
    --tag expresso_style_contrastive7_1_0.1_train_pitch_pca_no_whisper \


   



## meld emotion anchor

python gt_export.py \
    --dataset_config config/dataset_meld_emotion.yaml \
    --data_split train \
    --output_dir output/meld/emotion/prosody_predictor_gt/gt/pred/train

python compute_anchor_wasserstein_distances.py \
    --data_label_meta data/meld_emotion.csv \
    --anchor_file output/meld/emotion/prosody_predictor_gt/gt/pred/train/0.csv \
    --output_dir output/meld/emotion/prosody_predictor_gt/gt/wasserstein_distance/train_anchors

python compute_anchor_loc_pca.py \
    --data_label_meta data/meld_emotion.csv \
    --anchor_file output/meld/emotion/prosody_predictor_gt/gt/pred/train/0.csv \
    --output_dir output/meld/emotion/prosody_predictor_gt/gt/loc_anchor_train


## meld emotion contrastive 0.1 1

python train_prosody_predictor_contrastive.py \
    --dataset_config config/dataset_meld_emotion.yaml \
    --train_config config/train_prosody_predictor_contrastive_0.1_1.yaml \
    --model_config config/model_label_input.yaml \
    --output_dir output/meld/emotion/prosody_predictor/embedding_input/contrastive7_0.1_1
    
python prosody_predictor_predict_batch_label.py \
    --model_ckpt_dir output/meld/emotion/prosody_predictor/embedding_input/contrastive7_0.1_1 \
    --dataset_config_path config/dataset_meld_emotion.yaml \
    --model_config_path config/model_label_input.yaml \
    --label_index_file data/meld_emotion.csv \
    --data_split_name train \
    --ckpt_begin 1000 \
    --ckpt_end 400000 \
    --ckpt_step 40000 \

python compute_target_wasserstein_distances.py \
    --data_label_meta data/meld_emotion.csv \
    --anchor_file output/meld/emotion/prosody_predictor_gt/gt/pred/train/0.csv \
    --pred_dir output/meld/emotion/prosody_predictor/embedding_input/contrastive7_0.1_1/pred/train \
    --output_dir output/meld/emotion/prosody_predictor/embedding_input/contrastive7_0.1_1/wasserstein_distance/train

python compute_target_loc_pca.py \
    --anchor_loc_file output/meld/emotion/prosody_predictor_gt/gt/loc_anchor_train/pitch_anchor_loc_pca.csv \
    --anchor_loc_transform_file output/meld/emotion/prosody_predictor_gt/gt/loc_anchor_train/pitch_anchor_pca_transform.pkl \
    --target_distance_dir output/meld/emotion/prosody_predictor/embedding_input/contrastive7_0.1_1/wasserstein_distance/train/pitch \
    --output_dir output/meld/emotion/prosody_predictor/embedding_input/contrastive7_0.1_1/loc/train/pitch/pca

python plot_loc.py \
    --anchor_loc_file output/meld/emotion/prosody_predictor_gt/gt/loc_anchor_train/pitch_anchor_loc_pca.csv \
    --target_loc_dir output/meld/emotion/prosody_predictor/embedding_input/contrastive7_0.1_1/loc/train/pitch/pca \
    --output_file output/meld/emotion/prosody_predictor/embedding_input/contrastive7_0.1_1/vis/train/pitch/pca.png \
    --tag meld_emotion_contrastive7_0.1_1_train_pitch_pca


## meld sentiment anchor

python gt_export.py \
    --dataset_config config/dataset_meld_sentiment.yaml \
    --data_split train \
    --output_dir output/meld/sentiment/prosody_predictor_gt/gt/pred/train

python compute_anchor_wasserstein_distances.py \
    --data_label_meta data/meld_sentiment.csv \
    --anchor_file output/meld/sentiment/prosody_predictor_gt/gt/pred/train/0.csv \
    --output_dir output/meld/sentiment/prosody_predictor_gt/gt/wasserstein_distance/train_anchors

python compute_anchor_loc_pca.py \
    --data_label_meta data/meld_sentiment.csv \
    --anchor_file output/meld/sentiment/prosody_predictor_gt/gt/pred/train/0.csv \
    --output_dir output/meld/sentiment/prosody_predictor_gt/gt/loc_anchor_train

## meld sentiment contrastive 0.1 1

python train_prosody_predictor_contrastive.py \
    --dataset_config config/dataset_meld_sentiment.yaml \
    --train_config config/train_prosody_predictor_contrastive_0.1_1.yaml \
    --model_config config/model_label_input.yaml \
    --output_dir output/meld/sentiment/prosody_predictor/embedding_input/contrastive7_0.1_1

python prosody_predictor_predict_batch_label.py \
    --model_ckpt_dir output/meld/sentiment/prosody_predictor/embedding_input/contrastive7_0.1_1 \
    --dataset_config_path config/dataset_meld_sentiment.yaml \
    --model_config_path config/model_label_input.yaml \
    --label_index_file data/meld_sentiment.csv \
    --data_split_name train \
    --ckpt_begin 1000 \
    --ckpt_end 400000 \
    --ckpt_step 40000 \

python compute_target_wasserstein_distances.py \
    --data_label_meta data/meld_sentiment.csv \
    --anchor_file output/meld/sentiment/prosody_predictor_gt/gt/pred/train/0.csv \
    --pred_dir output/meld/sentiment/prosody_predictor/embedding_input/contrastive7_0.1_1/pred/train \
    --output_dir output/meld/sentiment/prosody_predictor/embedding_input/contrastive7_0.1_1/wasserstein_distance/train

python compute_target_loc_pca.py \
    --anchor_loc_file output/meld/sentiment/prosody_predictor_gt/gt/loc_anchor_train/pitch_anchor_loc_pca.csv \
    --anchor_loc_transform_file output/meld/sentiment/prosody_predictor_gt/gt/loc_anchor_train/pitch_anchor_pca_transform.pkl \
    --target_distance_dir output/meld/sentiment/prosody_predictor/embedding_input/contrastive7_0.1_1/wasserstein_distance/train/pitch \
    --output_dir output/meld/sentiment/prosody_predictor/embedding_input/contrastive7_0.1_1/loc/train/pitch/pca

python plot_loc.py \
    --anchor_loc_file output/meld/sentiment/prosody_predictor_gt/gt/loc_anchor_train/pitch_anchor_loc_pca.csv \
    --target_loc_dir output/meld/sentiment/prosody_predictor/embedding_input/contrastive7_0.1_1/loc/train/pitch/pca \
    --output_file output/meld/sentiment/prosody_predictor/embedding_input/contrastive7_0.1_1/vis/train/pitch/pca.png \
    --tag meld_sentiment_contrastive7_0.1_1_train_pitch_pca