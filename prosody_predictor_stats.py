import csv
import os

# compute the average pitch, energy, and duration for each sentiment group

SENTIMENT_MAPPING_FILE = "output/prosody_predictor_contrastive/sentiment_input-0.1/pred/val.csv"

DATA_FILE = [

    "output/fastspeech2/default/pred/val.csv",

    # "output/prosody_predictor/default/pred/val.csv",

    # "output/prosody_predictor_contrastive/sentiment_input-0.1/pred/val.csv",
    # "output/prosody_predictor_contrastive/sentiment_input-0.1/pred/val_neg.csv",
    # "output/prosody_predictor_contrastive/sentiment_input-0.1/pred/val_neu.csv",
    # "output/prosody_predictor_contrastive/sentiment_input-0.1/pred/val_pos.csv",

    # "output/prosody_predictor_contrastive/sentiment_input-0.01/pred/val.csv",
    # "output/prosody_predictor_contrastive/sentiment_input-0.01/pred/val_neg.csv",
    # "output/prosody_predictor_contrastive/sentiment_input-0.01/pred/val_neu.csv",
    # "output/prosody_predictor_contrastive/sentiment_input-0.01/pred/val_pos.csv",

    # "output/prosody_predictor_contrastive/sentiment_input-0/pred/val.csv",
    # "output/prosody_predictor_contrastive/sentiment_input-0/pred/val_neg.csv",
    # "output/prosody_predictor_contrastive/sentiment_input-0/pred/val_neu.csv",
    # "output/prosody_predictor_contrastive/sentiment_input-0/pred/val_pos.csv",
]

sentiment_mapping = {}
if os.path.exists(SENTIMENT_MAPPING_FILE):
    with open(SENTIMENT_MAPPING_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data_id = row["data_id"]
            sentiment_assigned = int(row["sentiment"])
            sentiment_mapping[data_id] = sentiment_assigned

for data_file in DATA_FILE:

    pitch_neg = []
    energy_neg = []
    duration_neg = []

    pitch_neu = []
    energy_neu = []
    duration_neu = []

    pitch_pos = []
    energy_pos = []
    duration_pos = []

    with open(data_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # process each row
            data_id = row["data_id"]
            phone_idx = int(row["phone_idx"])
            phone = row["phone"]
            pitch = float(row["pitch"])
            energy = float(row["energy"])
            duration = float(row["duration"])
            # sentiment_assigned = int(row["sentiment"])
            sentiment_gt = sentiment_mapping[data_id]

            if sentiment_gt == 0:
                pitch_neg.append(pitch)
                energy_neg.append(energy)
                duration_neg.append(duration)
            elif sentiment_gt == 1:
                pitch_neu.append(pitch)
                energy_neu.append(energy)
                duration_neu.append(duration)
            elif sentiment_gt == 2:
                pitch_pos.append(pitch)
                energy_pos.append(energy)
                duration_pos.append(duration)

    pitch_mean_neg = sum(pitch_neg) / len(pitch_neg) if pitch_neg else None
    energy_mean_neg = sum(energy_neg) / len(energy_neg) if energy_neg else None
    duration_mean_neg = sum(duration_neg) / len(duration_neg) if duration_neg else None

    pitch_mean_neu = sum(pitch_neu) / len(pitch_neu) if pitch_neu else None
    energy_mean_neu = sum(energy_neu) / len(energy_neu) if energy_neu else None
    duration_mean_neu = sum(duration_neu) / len(duration_neu) if duration_neu else None

    pitch_mean_pos = sum(pitch_pos) / len(pitch_pos) if pitch_pos else None
    energy_mean_pos = sum(energy_pos) / len(energy_pos) if energy_pos else None
    duration_mean_pos = sum(duration_pos) / len(duration_pos) if duration_pos else None

    # Calculate means
    pitch_all = (pitch_neg + pitch_neu + pitch_pos)
    energy_all = (energy_neg + energy_neu + energy_pos)
    duration_all = (duration_neg + duration_neu + duration_pos)

    pitch_mean_all = sum(pitch_all) / len(pitch_all) if pitch_all else None
    energy_mean_all = sum(energy_all) / len(energy_all) if energy_all else None
    duration_mean_all = sum(duration_all) / len(duration_all) if duration_all else None

    
    output_file = os.path.join(os.path.dirname(data_file), f"stats_{os.path.basename(data_file)}")

    with open(output_file, "w", encoding="utf-8", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow([
            "pitch_mean_all", "pitch_mean_neg", "pitch_mean_neu", "pitch_mean_pos",
            "energy_mean_all", "energy_mean_neg", "energy_mean_neu", "energy_mean_pos",
            "duration_mean_all", "duration_mean_neg", "duration_mean_neu", "duration_mean_pos"
        ])
        csv_writer.writerow([
            pitch_mean_all, pitch_mean_neg, pitch_mean_neu, pitch_mean_pos,
            energy_mean_all, energy_mean_neg, energy_mean_neu, energy_mean_pos,
            duration_mean_all, duration_mean_neg, duration_mean_neu, duration_mean_pos
        ])
        print(f"Statistics saved to {output_file}")
