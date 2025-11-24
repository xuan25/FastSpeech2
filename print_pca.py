import csv
import os
import pickle

import numpy as np
from sklearn.decomposition import PCA
import tqdm

ANCHOR_LOC_TRANSFORM_FILE = "output/expresso/style/prosody_predictor_gt/gt/loc/pitch_anchor_pca_transform.pkl"
OUTPUT_FILE = "output/expresso/style/prosody_predictor_gt/gt/loc/pitch_anchor_pca_info.txt"

ANCHOR_LOC_TRANSFORM_FILE = "output/expresso/style/prosody_predictor_gt/gt/loc_no_whisper/pitch_anchor_pca_transform.pkl"
OUTPUT_FILE = "output/expresso/style/prosody_predictor_gt/gt/loc_no_whisper/pitch_anchor_pca_info.txt"

pca: PCA = pickle.load(open(ANCHOR_LOC_TRANSFORM_FILE, 'rb'))

print("PCA explained variance ratio:", pca.explained_variance_ratio_)
print("PCA explained variance:", pca.explained_variance_)

with open(OUTPUT_FILE, 'w') as f:
    f.write(f"PCA explained variance ratio: {pca.explained_variance_ratio_}\n")
    f.write(f"PCA explained variance: {pca.explained_variance_}\n")