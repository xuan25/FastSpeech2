import csv
import scipy.stats

SENTIMENT_REF_FILE = "output/prosody_predictor_contrastive/sentiment_input-0/pred/val.csv"

NEG = "0"
NEU = "1"
POS = "2"

sentiment_mapping = {}
with open(SENTIMENT_REF_FILE, 'r', encoding='utf-8') as sentiment_file:
    reader = csv.DictReader(sentiment_file)
    for row in reader:
        data_id = row['data_id']
        sentiment = row['sentiment']
        if sentiment not in sentiment_mapping:
            sentiment_mapping[data_id] = sentiment

class Task:
    def __init__(self, data_file_a, data_file_b, feat_a, feat_b, text_sentiment_a=None, text_sentiment_b=None):
        self.data_file_a = data_file_a
        self.feat_a = feat_a
        self.text_sentiment_a = text_sentiment_a
        self.data_file_b = data_file_b
        self.feat_b = feat_b
        self.text_sentiment_b = text_sentiment_b

tasks = [
    # Task("output/prosody_predictor_gt/gt/pred/val.csv", "output/prosody_predictor_gt/gt/pred/val.csv", "pitch", "pitch", NEG, NEG),
    # Task("output/prosody_predictor_gt/gt/pred/val.csv", "output/fastspeech2/default/pred/val.csv", "pitch", "pitch", NEG, NEG),
    # Task("output/prosody_predictor_gt/gt/pred/val.csv", "output/prosody_predictor/default/pred/val.csv", "pitch", "pitch", NEG, NEG),
    # Task("output/prosody_predictor_gt/gt/pred/val.csv", "output/prosody_predictor_contrastive/sentiment_input-0/pred/val.csv", "pitch", "pitch", NEG, NEG),
    # Task("output/prosody_predictor_gt/gt/pred/val.csv", "output/prosody_predictor_contrastive/sentiment_input-0.01/pred/val.csv", "pitch", "pitch", NEG, NEG),
    # Task("output/prosody_predictor_gt/gt/pred/val.csv", "output/prosody_predictor_contrastive/sentiment_input-0.1/pred/val.csv", "pitch", "pitch", NEG, NEG),
    # Task("output/prosody_predictor_gt/gt/pred/val.csv", "output/prosody_predictor_contrastive/sentiment_input-0-overrideembedding0/pred/val.csv", "pitch", "pitch", NEG, NEG),
    # Task("output/prosody_predictor_gt/gt/pred/val.csv", "output/prosody_predictor_contrastive/sentiment_input-0-overrideembedding0-B/pred/val.csv", "pitch", "pitch", NEG, NEG),
    # Task("output/prosody_predictor_gt/gt/pred/val.csv", "output/prosody_predictor/default-B/pred/val.csv", "pitch", "pitch", NEG, NEG),
    # Task("output/prosody_predictor_gt/gt/pred/val.csv", "output/prosody_predictor/sentiment_input/pred/val.csv", "pitch", "pitch", NEG, NEG),
    Task("output/prosody_predictor_gt/gt/pred/val.csv", "output/prosody_predictor/sentiment_input_overrideembedding0/pred/val.csv", "pitch", "pitch", NEG, NEG),

    # Task("output/prosody_predictor_gt/gt/pred/val.csv", "output/prosody_predictor_gt/gt/pred/val.csv", "pitch", "pitch", NEU, NEG),
    # Task("output/prosody_predictor_gt/gt/pred/val.csv", "output/fastspeech2/default/pred/val.csv", "pitch", "pitch", NEU, NEG),
    # Task("output/prosody_predictor_gt/gt/pred/val.csv", "output/prosody_predictor/default/pred/val.csv", "pitch", "pitch", NEU, NEG),
    # Task("output/prosody_predictor_gt/gt/pred/val.csv", "output/prosody_predictor_contrastive/sentiment_input-0/pred/val.csv", "pitch", "pitch", NEU, NEG),
    # Task("output/prosody_predictor_gt/gt/pred/val.csv", "output/prosody_predictor_contrastive/sentiment_input-0.01/pred/val.csv", "pitch", "pitch", NEU, NEG),
    # Task("output/prosody_predictor_gt/gt/pred/val.csv", "output/prosody_predictor_contrastive/sentiment_input-0.1/pred/val.csv", "pitch", "pitch", NEU, NEG),
    # Task("output/prosody_predictor_gt/gt/pred/val.csv", "output/prosody_predictor_contrastive/sentiment_input-0-overrideembedding0/pred/val.csv", "pitch", "pitch", NEU, NEG),
    # Task("output/prosody_predictor_gt/gt/pred/val.csv", "output/prosody_predictor_contrastive/sentiment_input-0-overrideembedding0-B/pred/val.csv", "pitch", "pitch", NEU, NEG),
    # Task("output/prosody_predictor_gt/gt/pred/val.csv", "output/prosody_predictor/default-B/pred/val.csv", "pitch", "pitch", NEU, NEG),
    # Task("output/prosody_predictor_gt/gt/pred/val.csv", "output/prosody_predictor/sentiment_input/pred/val.csv", "pitch", "pitch", NEU, NEG),
    Task("output/prosody_predictor_gt/gt/pred/val.csv", "output/prosody_predictor/sentiment_input_overrideembedding0/pred/val.csv", "pitch", "pitch", NEU, NEG),


    # Task("output/prosody_predictor_gt/gt/pred/val.csv", "output/prosody_predictor_gt/gt/pred/val.csv", "pitch", "pitch", NEU, NEU),
    # Task("output/prosody_predictor_gt/gt/pred/val.csv", "output/fastspeech2/default/pred/val.csv", "pitch", "pitch", NEU, NEU),
    # Task("output/prosody_predictor_gt/gt/pred/val.csv", "output/prosody_predictor/default/pred/val.csv", "pitch", "pitch", NEU, NEU),
    # Task("output/prosody_predictor_gt/gt/pred/val.csv", "output/prosody_predictor_contrastive/sentiment_input-0/pred/val.csv", "pitch", "pitch", NEU, NEU),
    # Task("output/prosody_predictor_gt/gt/pred/val.csv", "output/prosody_predictor_contrastive/sentiment_input-0.01/pred/val.csv", "pitch", "pitch", NEU, NEU),
    # Task("output/prosody_predictor_gt/gt/pred/val.csv", "output/prosody_predictor_contrastive/sentiment_input-0.1/pred/val.csv", "pitch", "pitch", NEU, NEU),
    # Task("output/prosody_predictor_gt/gt/pred/val.csv", "output/prosody_predictor_contrastive/sentiment_input-0-overrideembedding0/pred/val.csv", "pitch", "pitch", NEU, NEU),
    # Task("output/prosody_predictor_gt/gt/pred/val.csv", "output/prosody_predictor_contrastive/sentiment_input-0-overrideembedding0-B/pred/val.csv", "pitch", "pitch", NEU, NEU),
    # Task("output/prosody_predictor_gt/gt/pred/val.csv", "output/prosody_predictor/default-B/pred/val.csv", "pitch", "pitch", NEU, NEU),
    # Task("output/prosody_predictor_gt/gt/pred/val.csv", "output/prosody_predictor/sentiment_input/pred/val.csv", "pitch", "pitch", NEU, NEU),
    Task("output/prosody_predictor_gt/gt/pred/val.csv", "output/prosody_predictor/sentiment_input_overrideembedding0/pred/val.csv", "pitch", "pitch", NEU, NEU),


    # Task("output/prosody_predictor_gt/gt/pred/val.csv", "output/prosody_predictor_gt/gt/pred/val.csv", "pitch", "pitch", NEG, NEU),
    # Task("output/prosody_predictor_gt/gt/pred/val.csv", "output/fastspeech2/default/pred/val.csv", "pitch", "pitch", NEG, NEU),
    # Task("output/prosody_predictor_gt/gt/pred/val.csv", "output/prosody_predictor/default/pred/val.csv", "pitch", "pitch", NEG, NEU),
    # Task("output/prosody_predictor_gt/gt/pred/val.csv", "output/prosody_predictor_contrastive/sentiment_input-0/pred/val.csv", "pitch", "pitch", NEG, NEU),
    # Task("output/prosody_predictor_gt/gt/pred/val.csv", "output/prosody_predictor_contrastive/sentiment_input-0.01/pred/val.csv", "pitch", "pitch", NEG, NEU),
    # Task("output/prosody_predictor_gt/gt/pred/val.csv", "output/prosody_predictor_contrastive/sentiment_input-0.1/pred/val.csv", "pitch", "pitch", NEG, NEU),
    # Task("output/prosody_predictor_gt/gt/pred/val.csv", "output/prosody_predictor_contrastive/sentiment_input-0-overrideembedding0/pred/val.csv", "pitch", "pitch", NEG, NEU),
    # Task("output/prosody_predictor_gt/gt/pred/val.csv", "output/prosody_predictor_contrastive/sentiment_input-0-overrideembedding0-B/pred/val.csv", "pitch", "pitch", NEG, NEU),
    # Task("output/prosody_predictor_gt/gt/pred/val.csv", "output/prosody_predictor/default-B/pred/val.csv", "pitch", "pitch", NEG, NEU),
    # Task("output/prosody_predictor_gt/gt/pred/val.csv", "output/prosody_predictor/sentiment_input/pred/val.csv", "pitch", "pitch", NEG, NEU),
    Task("output/prosody_predictor_gt/gt/pred/val.csv", "output/prosody_predictor/sentiment_input_overrideembedding0/pred/val.csv", "pitch", "pitch", NEG, NEU),



    # Task("output/prosody_predictor_gt/gt/pred/val.csv", "output/prosody_predictor_contrastive/sentiment_input-0/pred/val_neu.csv", "pitch", "pitch", NEG, NEG),
    # Task("output/prosody_predictor_gt/gt/pred/val.csv", "output/prosody_predictor_contrastive/sentiment_input-0.01/pred/val_neu.csv", "pitch", "pitch", NEG, NEG),
    # Task("output/prosody_predictor_gt/gt/pred/val.csv", "output/prosody_predictor_contrastive/sentiment_input-0.1/pred/val_neu.csv", "pitch", "pitch", NEG, NEG),
    # Task("output/prosody_predictor_gt/gt/pred/val.csv", "output/prosody_predictor_contrastive/sentiment_input-0-overrideembedding0/pred/val_neu.csv", "pitch", "pitch", NEG, NEG),
    # Task("output/prosody_predictor_gt/gt/pred/val.csv", "output/prosody_predictor_contrastive/sentiment_input-0-overrideembedding0-B/pred/val_neu.csv", "pitch", "pitch", NEG, NEG),
    # Task("output/prosody_predictor_gt/gt/pred/val.csv", "output/prosody_predictor/sentiment_input/pred/val_neu.csv", "pitch", "pitch", NEG, NEG),
    Task("output/prosody_predictor_gt/gt/pred/val.csv", "output/prosody_predictor/sentiment_input_overrideembedding0/pred/val_neu.csv", "pitch", "pitch", NEG, NEG),

    # Task("output/prosody_predictor_gt/gt/pred/val.csv", "output/prosody_predictor_contrastive/sentiment_input-0/pred/val_neu.csv", "pitch", "pitch", NEU, NEG),
    # Task("output/prosody_predictor_gt/gt/pred/val.csv", "output/prosody_predictor_contrastive/sentiment_input-0.01/pred/val_neu.csv", "pitch", "pitch", NEU, NEG),
    # Task("output/prosody_predictor_gt/gt/pred/val.csv", "output/prosody_predictor_contrastive/sentiment_input-0.1/pred/val_neu.csv", "pitch", "pitch", NEU, NEG),
    # Task("output/prosody_predictor_gt/gt/pred/val.csv", "output/prosody_predictor_contrastive/sentiment_input-0-overrideembedding0/pred/val_neu.csv", "pitch", "pitch", NEU, NEG),
    # Task("output/prosody_predictor_gt/gt/pred/val.csv", "output/prosody_predictor_contrastive/sentiment_input-0-overrideembedding0-B/pred/val_neu.csv", "pitch", "pitch", NEU, NEG),
    # Task("output/prosody_predictor_gt/gt/pred/val.csv", "output/prosody_predictor/sentiment_input/pred/val_neu.csv", "pitch", "pitch", NEU, NEG),
    Task("output/prosody_predictor_gt/gt/pred/val.csv", "output/prosody_predictor/sentiment_input_overrideembedding0/pred/val_neu.csv", "pitch", "pitch", NEU, NEG),

    # Task("output/prosody_predictor_gt/gt/pred/val.csv", "output/prosody_predictor_contrastive/sentiment_input-0/pred/val_neg.csv", "pitch", "pitch", NEU, NEU),
    # Task("output/prosody_predictor_gt/gt/pred/val.csv", "output/prosody_predictor_contrastive/sentiment_input-0.01/pred/val_neg.csv", "pitch", "pitch", NEU, NEU),
    # Task("output/prosody_predictor_gt/gt/pred/val.csv", "output/prosody_predictor_contrastive/sentiment_input-0.1/pred/val_neg.csv", "pitch", "pitch", NEU, NEU),
    # Task("output/prosody_predictor_gt/gt/pred/val.csv", "output/prosody_predictor_contrastive/sentiment_input-0-overrideembedding0/pred/val_neg.csv", "pitch", "pitch", NEU, NEU),
    # Task("output/prosody_predictor_gt/gt/pred/val.csv", "output/prosody_predictor_contrastive/sentiment_input-0-overrideembedding0-B/pred/val_neg.csv", "pitch", "pitch", NEU, NEU),
    # Task("output/prosody_predictor_gt/gt/pred/val.csv", "output/prosody_predictor/sentiment_input/pred/val_neg.csv", "pitch", "pitch", NEU, NEU),
    Task("output/prosody_predictor_gt/gt/pred/val.csv", "output/prosody_predictor/sentiment_input_overrideembedding0/pred/val_neg.csv", "pitch", "pitch", NEU, NEU),


    # Task("output/prosody_predictor_gt/gt/pred/val.csv", "output/prosody_predictor_contrastive/sentiment_input-0/pred/val_neg.csv", "pitch", "pitch", NEG, NEU),
    # Task("output/prosody_predictor_gt/gt/pred/val.csv", "output/prosody_predictor_contrastive/sentiment_input-0.01/pred/val_neg.csv", "pitch", "pitch", NEG, NEU),
    # Task("output/prosody_predictor_gt/gt/pred/val.csv", "output/prosody_predictor_contrastive/sentiment_input-0.1/pred/val_neg.csv", "pitch", "pitch", NEG, NEU),
    # Task("output/prosody_predictor_gt/gt/pred/val.csv", "output/prosody_predictor_contrastive/sentiment_input-0-overrideembedding0/pred/val_neg.csv", "pitch", "pitch", NEG, NEU),
    # Task("output/prosody_predictor_gt/gt/pred/val.csv", "output/prosody_predictor_contrastive/sentiment_input-0-overrideembedding0-B/pred/val_neg.csv", "pitch", "pitch", NEG, NEU),
    # Task("output/prosody_predictor_gt/gt/pred/val.csv", "output/prosody_predictor/sentiment_input/pred/val_neg.csv", "pitch", "pitch", NEG, NEU),
    Task("output/prosody_predictor_gt/gt/pred/val.csv", "output/prosody_predictor/sentiment_input_overrideembedding0/pred/val_neg.csv", "pitch", "pitch", NEG, NEU),






]

OUTPUT_FILE = "output/wasserstein_distance.csv"

with open(OUTPUT_FILE, 'w', encoding='utf-8') as output_file:
    writer = csv.writer(output_file)
    writer.writerow(['data_file_a', 'feat_a', 'data_file_b', 'feat_b', 'wasserstein_distance'])

    for task in tasks:
        with open(task.data_file_a, 'r', encoding='utf-8') as file_a, open(task.data_file_b, 'r', encoding='utf-8') as file_b:
            reader_a = csv.DictReader(file_a)
            reader_b = csv.DictReader(file_b)
            data_a = [float(row[task.feat_a]) for row in reader_a if (task.text_sentiment_a is None or sentiment_mapping.get(row['data_id']) == task.text_sentiment_a)]
            data_b = [float(row[task.feat_b]) for row in reader_b if (task.text_sentiment_b is None or sentiment_mapping.get(row['data_id']) == task.text_sentiment_b)]
        wasserstein_distance = scipy.stats.wasserstein_distance(data_a, data_b)
        writer.writerow([task.data_file_a, task.feat_a, task.data_file_b, task.feat_b, wasserstein_distance])
        print(f"Processed task: {task.data_file_a}-{task.feat_a}-{task.text_sentiment_a} vs {task.data_file_b}-{task.feat_b}-{task.text_sentiment_b}, Wasserstein Distance: {wasserstein_distance}")