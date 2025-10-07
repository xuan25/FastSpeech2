
import csv
import os


NEG = 0
NEU = 1
POS = 2


SENTIMENTS_REF_FILE = "output/prosody_predictor_gt/gt/pred/val.csv"
sentiment_mapping = {}
try:
    with open(SENTIMENTS_REF_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sentiment_mapping[row["data_id"]] = int(row["sentiment"])
except FileNotFoundError:
    print(f"Warning: Sentiment reference file not found: {SENTIMENTS_REF_FILE}")
except (UnicodeDecodeError, csv.Error) as e:
    print(f"Error reading sentiment reference file: {e}")

def get_data_sentiment(data_id: str):
    return sentiment_mapping.get(data_id, -1)

class Distribution:
    def __init__(self, samples_file: str, sentiment_filter: int, feature: str):
        self.samples_file = samples_file
        self.sentiment_filter = sentiment_filter
        self.feature = feature
        self.samples = self.load_data(samples_file, sentiment_filter, feature)

    def load_data(self, samples_file: str, sentiment_filter: int, feature: str):
        data = []
        try:
            with open(samples_file, "r", encoding="utf-8") as file:
                csv_reader = csv.DictReader(file)
                for data_row in csv_reader:
                    if get_data_sentiment(data_row["data_id"]) == sentiment_filter:
                        try:
                            data.append(float(data_row[feature]))
                        except (ValueError, KeyError) as e:
                            print(f"Warning: Skipping invalid data in {samples_file}: {e}")
                            continue
        except FileNotFoundError:
            print(f"Error: File not found: {samples_file}")
            return []
        except (UnicodeDecodeError, csv.Error) as e:
            print(f"Error reading {samples_file}: {e}")
            return []
        return data

class Position:
    def __init__(self, distribution: Distribution, position_label: str = ""):
        self.distribution = distribution
        self.label = position_label

class Anchor(Position):
    pass

class Target(Position):
    pass

class Distance:
    def __init__(self, anchor: Anchor, target: Target):
        self.anchor = anchor
        self.target = target

    def compute_distance(self):
        distance = wasserstein_distance(self.anchor.distribution.samples, self.target.distribution.samples)
        return distance
    
anchors_shared = [
    Anchor(Distribution("output/prosody_predictor_gt/gt/pred/val.csv", NEG, "pitch"), "GT_NEG"),
    Anchor(Distribution("output/prosody_predictor_gt/gt/pred/val.csv", NEU, "pitch"), "GT_NEU"),
    Anchor(Distribution("output/prosody_predictor_gt/gt/pred/val.csv", POS, "pitch"), "GT_POS"),
]

class Task:
    def __init__(self, task_anchors: list[Anchor], task_targets: list[Target], task_output: str, task_label: str = ""):
        self.anchors = task_anchors
        self.targets = task_targets
        self.output = task_output
        self.label = task_label

tasks = [

    Task(
        anchors_shared,
        [
            Target(Distribution("output/prosody_predictor/sentiment_input_concat/pred/val_neu.csv", NEG, "pitch"), "NEG->SI_NEU"),
            Target(Distribution("output/prosody_predictor/sentiment_input_concat/pred/val_neu.csv", NEU, "pitch"), "NEU->SI_NEU"),
            Target(Distribution("output/prosody_predictor/sentiment_input_concat/pred/val_neu.csv", POS, "pitch"), "POS->SI_NEU"),
            Target(Distribution("output/prosody_predictor/sentiment_input_concat/pred/val_neg.csv", NEG, "pitch"), "NEG->SI_NEG"),
            Target(Distribution("output/prosody_predictor/sentiment_input_concat/pred/val_neg.csv", NEU, "pitch"), "NEU->SI_NEG"),
            Target(Distribution("output/prosody_predictor/sentiment_input_concat/pred/val_neg.csv", POS, "pitch"), "POS->SI_NEG"),
            Target(Distribution("output/prosody_predictor/sentiment_input_concat/pred/val_pos.csv", NEG, "pitch"), "NEG->SI_POS"),
            Target(Distribution("output/prosody_predictor/sentiment_input_concat/pred/val_pos.csv", NEU, "pitch"), "NEU->SI_POS"),
            Target(Distribution("output/prosody_predictor/sentiment_input_concat/pred/val_pos.csv", POS, "pitch"), "POS->SI_POS"),

        ],
        task_output="output/plots/trilateration-eqtri-ols/prosody-predictor_sentiment-input-concat.png",
        task_label="prosody-predictor_sentiment-input-concat"
    ),

    # Task(
    #     anchors_shared,
    #     [
    #         Target(Distribution("output/prosody_predictor_contrastive/sentiment_input_concat-0/pred/val_neu.csv", NEG, "pitch"), "NEG->SI_NEU"),
    #         Target(Distribution("output/prosody_predictor_contrastive/sentiment_input_concat-0/pred/val_neu.csv", NEU, "pitch"), "NEU->SI_NEU"),
    #         Target(Distribution("output/prosody_predictor_contrastive/sentiment_input_concat-0/pred/val_neu.csv", POS, "pitch"), "POS->SI_NEU"),
    #         Target(Distribution("output/prosody_predictor_contrastive/sentiment_input_concat-0/pred/val_neg.csv", NEG, "pitch"), "NEG->SI_NEG"),
    #         Target(Distribution("output/prosody_predictor_contrastive/sentiment_input_concat-0/pred/val_neg.csv", NEU, "pitch"), "NEU->SI_NEG"),
    #         Target(Distribution("output/prosody_predictor_contrastive/sentiment_input_concat-0/pred/val_neg.csv", POS, "pitch"), "POS->SI_NEG"),
    #         Target(Distribution("output/prosody_predictor_contrastive/sentiment_input_concat-0/pred/val_pos.csv", NEG, "pitch"), "NEG->SI_POS"),
    #         Target(Distribution("output/prosody_predictor_contrastive/sentiment_input_concat-0/pred/val_pos.csv", NEU, "pitch"), "NEU->SI_POS"),
    #         Target(Distribution("output/prosody_predictor_contrastive/sentiment_input_concat-0/pred/val_pos.csv", POS, "pitch"), "POS->SI_POS"),

    #     ],
    #     task_output="output/plots/trilateration-eqtri-ols/prosody-predictor-contrastive_sentiment-input-concat-0.png",
    #     task_label="prosody-predictor-contrastive_sentiment-input-concat-0"
    # ),
    # Task(
    #     anchors_shared,
    #     [
    #         Target(Distribution("output/prosody_predictor_contrastive/sentiment_input_concat-0.1/pred/val_neu.csv", NEG, "pitch"), "NEG->SI_NEU"),
    #         Target(Distribution("output/prosody_predictor_contrastive/sentiment_input_concat-0.1/pred/val_neu.csv", NEU, "pitch"), "NEU->SI_NEU"),
    #         Target(Distribution("output/prosody_predictor_contrastive/sentiment_input_concat-0.1/pred/val_neu.csv", POS, "pitch"), "POS->SI_NEU"),
    #         Target(Distribution("output/prosody_predictor_contrastive/sentiment_input_concat-0.1/pred/val_neg.csv", NEG, "pitch"), "NEG->SI_NEG"),
    #         Target(Distribution("output/prosody_predictor_contrastive/sentiment_input_concat-0.1/pred/val_neg.csv", NEU, "pitch"), "NEU->SI_NEG"),
    #         Target(Distribution("output/prosody_predictor_contrastive/sentiment_input_concat-0.1/pred/val_neg.csv", POS, "pitch"), "POS->SI_NEG"),
    #         Target(Distribution("output/prosody_predictor_contrastive/sentiment_input_concat-0.1/pred/val_pos.csv", NEG, "pitch"), "NEG->SI_POS"),
    #         Target(Distribution("output/prosody_predictor_contrastive/sentiment_input_concat-0.1/pred/val_pos.csv", NEU, "pitch"), "NEU->SI_POS"),
    #         Target(Distribution("output/prosody_predictor_contrastive/sentiment_input_concat-0.1/pred/val_pos.csv", POS, "pitch"), "POS->SI_POS"),
    #     ],
    #     task_output="output/plots/trilateration-eqtri-ols/prosody-predictor-contrastive_sentiment-input-concat-0.1.png",
    #     task_label="prosody-predictor-contrastive_sentiment-input-concat-0.1"
    # ),


    # Task(
    #     anchors_shared,
    #     [
    #         # Target(Distribution("output/prosody_predictor/sentiment_input/pred/val.csv", NEG, "pitch"), "SI_NEG"),
    #         # Target(Distribution("output/prosody_predictor/sentiment_input/pred/val.csv", NEU, "pitch"), "SI_NEU"),
    #         # Target(Distribution("output/prosody_predictor/sentiment_input/pred/val.csv", POS, "pitch"), "SI_POS"),
    #         Target(Distribution("output/prosody_predictor/sentiment_input/pred/val_neu.csv", NEG, "pitch"), "NEG->SI_NEU"),
    #         Target(Distribution("output/prosody_predictor/sentiment_input/pred/val_neu.csv", NEU, "pitch"), "NEU->SI_NEU"),
    #         Target(Distribution("output/prosody_predictor/sentiment_input/pred/val_neu.csv", POS, "pitch"), "POS->SI_NEU"),
    #         Target(Distribution("output/prosody_predictor/sentiment_input/pred/val_neg.csv", NEG, "pitch"), "NEG->SI_NEG"),
    #         Target(Distribution("output/prosody_predictor/sentiment_input/pred/val_neg.csv", NEU, "pitch"), "NEU->SI_NEG"),
    #         Target(Distribution("output/prosody_predictor/sentiment_input/pred/val_neg.csv", POS, "pitch"), "POS->SI_NEG"),
    #         Target(Distribution("output/prosody_predictor/sentiment_input/pred/val_pos.csv", NEG, "pitch"), "NEG->SI_POS"),
    #         Target(Distribution("output/prosody_predictor/sentiment_input/pred/val_pos.csv", NEU, "pitch"), "NEU->SI_POS"),
    #         Target(Distribution("output/prosody_predictor/sentiment_input/pred/val_pos.csv", POS, "pitch"), "POS->SI_POS"),

    #     ],
    #     # task_output="output/prosody_predictor/sentiment_input/plots/trilateration_pitch.png",
    #     task_output="output/plots/trilateration-eqtri-ols/prosody-predictor_sentiment-input.png",
    #     task_label="prosody-predictor_sentiment-input"
    # ),
    # Task(
    #     anchors_shared,
    #     [
    #         # Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0/pred/val.csv", NEG, "pitch"), "SI_NEG"),
    #         # Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0/pred/val.csv", NEU, "pitch"), "SI_NEU"),
    #         # Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0/pred/val.csv", POS, "pitch"), "SI_POS"),
    #         Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0/pred/val_neu.csv", NEG, "pitch"), "NEG->SI_NEU"),
    #         Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0/pred/val_neu.csv", NEU, "pitch"), "NEU->SI_NEU"),
    #         Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0/pred/val_neu.csv", POS, "pitch"), "POS->SI_NEU"),
    #         Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0/pred/val_neg.csv", NEG, "pitch"), "NEG->SI_NEG"),
    #         Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0/pred/val_neg.csv", NEU, "pitch"), "NEU->SI_NEG"),
    #         Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0/pred/val_neg.csv", POS, "pitch"), "POS->SI_NEG"),
    #         Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0/pred/val_pos.csv", NEG, "pitch"), "NEG->SI_POS"),
    #         Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0/pred/val_pos.csv", NEU, "pitch"), "NEU->SI_POS"),
    #         Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0/pred/val_pos.csv", POS, "pitch"), "POS->SI_POS"),
    #     ],
    #     # task_output="output/prosody_predictor_contrastive/sentiment_input-0/plots/trilateration_pitch.png",
    #     task_output="output/plots/trilateration-eqtri-ols/prosody-predictor_sentiment-input_contrastive-0.png",
    #     task_label="prosody-predictor_sentiment-input_contrastive-0"
    # ),
    # Task(
    #     anchors_shared,
    #     [
    #         # Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0.01/pred/val.csv", NEG, "pitch"), "SI_NEG"),
    #         # Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0.01/pred/val.csv", NEU, "pitch"), "SI_NEU"),
    #         # Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0.01/pred/val.csv", POS, "pitch"), "SI_POS"),
    #         Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0.01/pred/val_neu.csv", NEG, "pitch"), "NEG->SI_NEU"),
    #         Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0.01/pred/val_neu.csv", NEU, "pitch"), "NEU->SI_NEU"),
    #         Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0.01/pred/val_neu.csv", POS, "pitch"), "POS->SI_NEU"),
    #         Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0.01/pred/val_neg.csv", NEG, "pitch"), "NEG->SI_NEG"),
    #         Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0.01/pred/val_neg.csv", NEU, "pitch"), "NEU->SI_NEG"),
    #         Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0.01/pred/val_neg.csv", POS, "pitch"), "POS->SI_NEG"),
    #         Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0.01/pred/val_pos.csv", NEG, "pitch"), "NEG->SI_POS"),
    #         Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0.01/pred/val_pos.csv", NEU, "pitch"), "NEU->SI_POS"),
    #         Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0.01/pred/val_pos.csv", POS, "pitch"), "POS->SI_POS"),
    #     ],
    #     # task_output="output/prosody_predictor_contrastive/sentiment_input-0.01/plots/trilateration_pitch.png",
    #     task_output="output/plots/trilateration-eqtri-ols/prosody-predictor_sentiment-input_contrastive-0.01.png",
    #     task_label="prosody-predictor_sentiment-input_contrastive-0.01"
    # ),
    # Task(
    #     anchors_shared,
    #     [
    #         # Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0.1/pred/val.csv", NEG, "pitch"), "SI_NEG"),
    #         # Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0.1/pred/val.csv", NEU, "pitch"), "SI_NEU"),
    #         # Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0.1/pred/val.csv", POS, "pitch"), "SI_POS"),
    #         Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0.1/pred/val_neu.csv", NEG, "pitch"), "NEG->SI_NEU"),
    #         Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0.1/pred/val_neu.csv", NEU, "pitch"), "NEU->SI_NEU"),
    #         Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0.1/pred/val_neu.csv", POS, "pitch"), "POS->SI_NEU"),
    #         Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0.1/pred/val_neg.csv", NEG, "pitch"), "NEG->SI_NEG"),
    #         Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0.1/pred/val_neg.csv", NEU, "pitch"), "NEU->SI_NEG"),
    #         Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0.1/pred/val_neg.csv", POS, "pitch"), "POS->SI_NEG"),
    #         Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0.1/pred/val_pos.csv", NEG, "pitch"), "NEG->SI_POS"),
    #         Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0.1/pred/val_pos.csv", NEU, "pitch"), "NEU->SI_POS"),
    #         Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0.1/pred/val_pos.csv", POS, "pitch"), "POS->SI_POS"),
    #     ],
    #     # task_output="output/prosody_predictor_contrastive/sentiment_input-0.1/plots/trilateration_pitch.png",
    #     task_output="output/plots/trilateration-eqtri-ols/prosody-predictor_sentiment-input_contrastive-0.1.png",
    #     task_label="prosody-predictor_sentiment-input_contrastive-0.1"
    # ),
]

from scipy.stats import wasserstein_distance
import numpy as np
import matplotlib.pyplot as plt


# # === 用三边测量把每个 target 映射到 2D ===
# # 固定三角形（等边，边长=1）
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, np.sqrt(3)/2])

A *= 0.4
B *= 0.4
C *= 0.4

BA = B - A
CA = C - A
M = np.vstack([BA, CA])  # 2x2
ATA = A @ A
BTB = B @ B
CTC = C @ C


for task in tasks:
    anchors = task.anchors
    targets = task.targets
    output = task.output
    label = task.label

    anchor_labels = [a.label for a in anchors]
    target_labels = [t.label for t in targets]

    # 预取 anchor 的样本数组
    anchor_samples = {a.label: np.asarray(a.distribution.samples, dtype=float) for a in anchors}

    # 距离表：dist[target_label][anchor_label] = scalar
    dist = {}
    for t in targets:
        t_samples = np.asarray(t.distribution.samples, dtype=float)
        dist[t.label] = {}
        for a in anchors:
            d = wasserstein_distance(anchor_samples[a.label], t_samples)
            dist[t.label][a.label] = float(d)

    # 最小二乘解
    # ordinary least squares
    def trilaterate(dist_a, dist_b, dist_c):
        # dist_a, dist_b, dist_c 分别是到 A,B,C 的距离（这里用 Wasserstein）
        b1 = (BTB - ATA + dist_a**2 - dist_b**2) / 2.0
        b2 = (CTC - ATA + dist_a**2 - dist_c**2) / 2.0
        p, *_ = np.linalg.lstsq(M, np.array([b1, b2]), rcond=None)
        return p


    # 保证 anchor 的顺序与 d_a, d_b, d_c 的匹配
    order = ["GT_NEG", "GT_NEU", "GT_POS"]  # 你也可以换成别的顺序，但三者要与 A,B,C 一一对应
    coords = {}
    for tl in target_labels:
        d_a = dist[tl][order[0]]
        d_b = dist[tl][order[1]]
        d_c = dist[tl][order[2]]
        coords[tl] = trilaterate(d_a, d_b, d_c)



    # === 画图：三角形 + targets ===
    # plt.figure(figsize=(6,6))
    # # 锚点
    # anchors2d = np.vstack([A, B, C])
    # plt.scatter(anchors2d[:,0], anchors2d[:,1], marker='^', s=120, c='k', label='Anchors')
    # for lab, xy in zip(order, anchors2d):
    #     plt.text(xy[0], xy[1], f" {lab}", va='bottom', fontsize=10)

    # # 目标点
    # for tl in target_labels:
    #     x, y = coords[tl]
    #     plt.scatter([x], [y], s=80, label=tl)
    #     plt.text(x, y, f" {tl}", va='bottom', fontsize=9)

    from adjustText import adjust_text

    plt.figure(figsize=(6,6))
    # 画锚点
    anchors2d = np.vstack([A, B, C])
    plt.scatter(anchors2d[:,0], anchors2d[:,1], marker='^', s=120, c='k', label='Anchors')

    texts = []
    # 锚点标签
    for lab, xy in zip(order, anchors2d):
        texts.append(plt.text(xy[0], xy[1], f" {lab}", fontsize=10, va='bottom', ha='left'))

    # 目标点
    for tl in target_labels:
        x, y = coords[tl]
        plt.scatter([x], [y], s=80)
        texts.append(plt.text(x, y, f" {tl}", fontsize=9, va='bottom', ha='left'))

    # 关键：自动避让（可加箭头提高可读性）
    adjust_text(
        texts,
        only_move={'texts':'xy'},   # 常用设置：点尽量不动，文字可动
        force_points=0.2, force_text=0.5,         # 力度可调
        expand_points=(1.1, 1.2), expand_text=(1.1, 1.2),
        arrowprops=dict(arrowstyle='->', lw=0.6, alpha=0.7)
    )

    plt.axis('equal')
    # plt.title("Trilateration Visualization under Wasserstein Distance (pitch distributions)")
    if label:
        plt.suptitle(label, fontsize=10)
    plt.tight_layout()
    plt.legend(loc='best', fontsize=8)
    # plt.show()
    os.makedirs(os.path.dirname(output), exist_ok=True)
    plt.savefig(output, dpi=300)
    plt.close()

