import csv
import os

# compute the average pitch, energy, and duration for each sentiment group

# DATA_FILE = [
#     "output/prosody_predictor_contrastive/sentiment_input/pred/val.csv",
#     "output/prosody_predictor_contrastive/sentiment_input/pred/val_neg.csv",
#     "output/prosody_predictor_contrastive/sentiment_input/pred/val_neu.csv",
#     "output/prosody_predictor_contrastive/sentiment_input/pred/val_pos.csv",
# ]

# OUTPUT_DIR = "output/prosody_predictor_contrastive/sentiment_input/pred/plots"

# DATA_FILE = [
#     "output/prosody_predictor_contrastive/sentiment_input-reworked-0.001/pred/val.csv",
#     "output/prosody_predictor_contrastive/sentiment_input-reworked-0.001/pred/val_neg.csv",
#     "output/prosody_predictor_contrastive/sentiment_input-reworked-0.001/pred/val_neu.csv",
#     "output/prosody_predictor_contrastive/sentiment_input-reworked-0.001/pred/val_pos.csv",
# ]

# OUTPUT_DIR = "output/prosody_predictor_contrastive/sentiment_input-reworked-0.001/pred/plots"

DATA_FILE = [
    "output/prosody_predictor_contrastive/sentiment_input-0-reworked/pred/val.csv",
    "output/prosody_predictor_contrastive/sentiment_input-0-reworked/pred/val_neg.csv",
    "output/prosody_predictor_contrastive/sentiment_input-0-reworked/pred/val_neu.csv",
    "output/prosody_predictor_contrastive/sentiment_input-0-reworked/pred/val_pos.csv",
]

OUTPUT_DIR = "output/prosody_predictor_contrastive/sentiment_input-0-reworked/pred/plots"


DATA_TAGS = [
    "true",
    "negative",
    "neutral",
    "positive",
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

features_dict = {
    "pitch": [],
    "energy": [],
    "duration": []
}

for data_file in DATA_FILE:

    with open(data_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        pitchs = []
        energies = []
        durations = []
        for row in reader:
            # process each row
            data_id = row["data_id"]
            phone_idx = int(row["phone_idx"])
            phone = row["phone"]
            pitch = float(row["pitch"])
            energy = float(row["energy"])
            duration = float(row["duration"])

            pitchs.append(pitch)
            energies.append(energy)
            durations.append(duration)

        avg_pitch = sum(pitchs) / len(pitchs) if pitchs else 0
        avg_energy = sum(energies) / len(energies) if energies else 0
        avg_duration = sum(durations) / len(durations) if durations else 0

        print(f"File: {data_file}")
        print(f"Average Pitch: {avg_pitch:.2f}")
        print(f"Average Energy: {avg_energy:.2f}")
        print(f"Average Duration: {avg_duration:.2f}")
        print("-" * 40)

        features_dict["pitch"].append(pitchs)
        features_dict["energy"].append(energies)
        features_dict["duration"].append(durations)

#plotting the violin plots for each feature
import matplotlib.pyplot as plt
def plot_violin(data, title, ylabel, output_path):
    plt.figure(figsize=(10, 6))
    plt.violinplot(data, showmeans=True)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks([1, 2, 3, 4], DATA_TAGS)
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

plot_violin(features_dict["pitch"], "Pitch Distribution by Sentiment", "Pitch", os.path.join(OUTPUT_DIR, "pitch_distribution.png"))
plot_violin(features_dict["energy"], "Energy Distribution by Sentiment", "Energy", os.path.join(OUTPUT_DIR, "energy_distribution.png"))
plot_violin(features_dict["duration"], "Duration Distribution by Sentiment", "Duration", os.path.join(OUTPUT_DIR, "duration_distribution.png"))