


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

