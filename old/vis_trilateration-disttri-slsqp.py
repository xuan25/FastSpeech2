
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
            # Target(Distribution("output/prosody_predictor/sentiment_input/pred/val.csv", NEG, "pitch"), "SI_NEG"),
            # Target(Distribution("output/prosody_predictor/sentiment_input/pred/val.csv", NEU, "pitch"), "SI_NEU"),
            # Target(Distribution("output/prosody_predictor/sentiment_input/pred/val.csv", POS, "pitch"), "SI_POS"),
            Target(Distribution("output/prosody_predictor/sentiment_input/pred/val_neu.csv", NEG, "pitch"), "NEG->SI_NEU"),
            Target(Distribution("output/prosody_predictor/sentiment_input/pred/val_neu.csv", NEU, "pitch"), "NEU->SI_NEU"),
            Target(Distribution("output/prosody_predictor/sentiment_input/pred/val_neu.csv", POS, "pitch"), "POS->SI_NEU"),
            Target(Distribution("output/prosody_predictor/sentiment_input/pred/val_neg.csv", NEG, "pitch"), "NEG->SI_NEG"),
            Target(Distribution("output/prosody_predictor/sentiment_input/pred/val_neg.csv", NEU, "pitch"), "NEU->SI_NEG"),
            Target(Distribution("output/prosody_predictor/sentiment_input/pred/val_neg.csv", POS, "pitch"), "POS->SI_NEG"),
            Target(Distribution("output/prosody_predictor/sentiment_input/pred/val_pos.csv", NEG, "pitch"), "NEG->SI_POS"),
            Target(Distribution("output/prosody_predictor/sentiment_input/pred/val_pos.csv", NEU, "pitch"), "NEU->SI_POS"),
            Target(Distribution("output/prosody_predictor/sentiment_input/pred/val_pos.csv", POS, "pitch"), "POS->SI_POS"),

        ],
        # task_output="output/prosody_predictor/sentiment_input/plots/trilateration_pitch.png",
        task_output="output/plots/trilateration-disttri-slsqp/prosody-predictor_sentiment-input.png",
        task_label="prosody-predictor_sentiment-input"
    ),
    Task(
        anchors_shared,
        [
            # Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0/pred/val.csv", NEG, "pitch"), "SI_NEG"),
            # Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0/pred/val.csv", NEU, "pitch"), "SI_NEU"),
            # Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0/pred/val.csv", POS, "pitch"), "SI_POS"),
            Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0/pred/val_neu.csv", NEG, "pitch"), "NEG->SI_NEU"),
            Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0/pred/val_neu.csv", NEU, "pitch"), "NEU->SI_NEU"),
            Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0/pred/val_neu.csv", POS, "pitch"), "POS->SI_NEU"),
            Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0/pred/val_neg.csv", NEG, "pitch"), "NEG->SI_NEG"),
            Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0/pred/val_neg.csv", NEU, "pitch"), "NEU->SI_NEG"),
            Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0/pred/val_neg.csv", POS, "pitch"), "POS->SI_NEG"),
            Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0/pred/val_pos.csv", NEG, "pitch"), "NEG->SI_POS"),
            Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0/pred/val_pos.csv", NEU, "pitch"), "NEU->SI_POS"),
            Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0/pred/val_pos.csv", POS, "pitch"), "POS->SI_POS"),
        ],
        # task_output="output/prosody_predictor_contrastive/sentiment_input-0/plots/trilateration_pitch.png",
        task_output="output/plots/trilateration-disttri-slsqp/prosody-predictor_sentiment-input_contrastive-0.png",
        task_label="prosody-predictor_sentiment-input_contrastive-0"
    ),
    Task(
        anchors_shared,
        [
            # Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0.01/pred/val.csv", NEG, "pitch"), "SI_NEG"),
            # Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0.01/pred/val.csv", NEU, "pitch"), "SI_NEU"),
            # Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0.01/pred/val.csv", POS, "pitch"), "SI_POS"),
            Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0.01/pred/val_neu.csv", NEG, "pitch"), "NEG->SI_NEU"),
            Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0.01/pred/val_neu.csv", NEU, "pitch"), "NEU->SI_NEU"),
            Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0.01/pred/val_neu.csv", POS, "pitch"), "POS->SI_NEU"),
            Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0.01/pred/val_neg.csv", NEG, "pitch"), "NEG->SI_NEG"),
            Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0.01/pred/val_neg.csv", NEU, "pitch"), "NEU->SI_NEG"),
            Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0.01/pred/val_neg.csv", POS, "pitch"), "POS->SI_NEG"),
            Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0.01/pred/val_pos.csv", NEG, "pitch"), "NEG->SI_POS"),
            Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0.01/pred/val_pos.csv", NEU, "pitch"), "NEU->SI_POS"),
            Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0.01/pred/val_pos.csv", POS, "pitch"), "POS->SI_POS"),
        ],
        # task_output="output/prosody_predictor_contrastive/sentiment_input-0.01/plots/trilateration_pitch.png",
        task_output="output/plots/trilateration-disttri-slsqp/prosody-predictor_sentiment-input_contrastive-0.01.png",
        task_label="prosody-predictor_sentiment-input_contrastive-0.01"
    ),
    Task(
        anchors_shared,
        [
            # Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0.1/pred/val.csv", NEG, "pitch"), "SI_NEG"),
            # Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0.1/pred/val.csv", NEU, "pitch"), "SI_NEU"),
            # Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0.1/pred/val.csv", POS, "pitch"), "SI_POS"),
            Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0.1/pred/val_neu.csv", NEG, "pitch"), "NEG->SI_NEU"),
            Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0.1/pred/val_neu.csv", NEU, "pitch"), "NEU->SI_NEU"),
            Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0.1/pred/val_neu.csv", POS, "pitch"), "POS->SI_NEU"),
            Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0.1/pred/val_neg.csv", NEG, "pitch"), "NEG->SI_NEG"),
            Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0.1/pred/val_neg.csv", NEU, "pitch"), "NEU->SI_NEG"),
            Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0.1/pred/val_neg.csv", POS, "pitch"), "POS->SI_NEG"),
            Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0.1/pred/val_pos.csv", NEG, "pitch"), "NEG->SI_POS"),
            Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0.1/pred/val_pos.csv", NEU, "pitch"), "NEU->SI_POS"),
            Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0.1/pred/val_pos.csv", POS, "pitch"), "POS->SI_POS"),
        ],
        # task_output="output/prosody_predictor_contrastive/sentiment_input-0.1/plots/trilateration_pitch.png",
        task_output="output/plots/trilateration-disttri-slsqp/prosody-predictor_sentiment-input_contrastive-0.1.png",
        task_label="prosody-predictor_sentiment-input_contrastive-0.1"
    ),
]

from scipy.stats import wasserstein_distance
import numpy as np
import matplotlib.pyplot as plt

# === 用三锚点之间的真实 Wasserstein 距离在 2D 放置锚点 ===
def place_three_anchors_by_dist(s0, s1, s2, scale=1.0, flip=False):
    """
    给定三个样本数组 s0,s1,s2，计算它们两两的 Wasserstein 距离，
    并在 2D 中放置 A=(0,0), B=(d01,0), C=(xC, yC)。scale 用于缩放显示，不改变相对/绝对比例。
    """
    from scipy.stats import wasserstein_distance
    d01 = wasserstein_distance(s0, s1)  # |A-B|
    d02 = wasserstein_distance(s0, s2)  # |A-C|
    d12 = wasserstein_distance(s1, s2)  # |B-C|

    eps = 1e-12
    A = np.array([0.0, 0.0], float)
    B = np.array([max(d01, eps), 0.0], float)  # 放在 x 轴
    # 由三边长解 C
    xC = (d02**2 - d12**2 + d01**2) / (2.0 * max(d01, eps))
    y_sq = max(d02**2 - xC**2, 0.0)           # 数值保护
    yC = np.sqrt(y_sq)
    if flip:
        yC = -yC
    C = np.array([xC, yC], float)

    # 可选缩放（仅用于绘图尺寸控制；留 scale=1 保持绝对尺度）
    A *= scale; B *= scale; C *= scale
    return A, B, C

# 你用于匹配 A/B/C 的锚点顺序（和之前一致即可）
anchor_order = ["GT_NEG", "GT_NEU", "GT_POS"]  # A,B,C 分别对应的锚点标签

for task in tasks:
    anchors = task.anchors
    targets = task.targets
    output = task.output
    label = task.label

    anchor_labels = [a.label for a in anchors]
    target_labels = [t.label for t in targets]

    # 预取 anchor 的样本数组
    anchor_samples = {a.label: np.asarray(a.distribution.samples, dtype=float) for a in anchors}

    # === 关键：用真实距离放置三锚点（保持形状与绝对尺度） ===
    # 取出与 A/B/C 对应的三个样本数组
    sA = anchor_samples[anchor_order[0]]
    sB = anchor_samples[anchor_order[1]]
    sC = anchor_samples[anchor_order[2]]

    # 如需统一图幅可设 scale=0.4（不建议；保持 1.0 就是绝对尺度）
    A, B, C = place_three_anchors_by_dist(sA, sB, sC, scale=1.0, flip=False)

    # 基于真实 A/B/C 的三边测量线性化矩阵
    BA = B - A
    CA = C - A
    M = np.vstack([BA, CA])  # 2x2
    ATA = A @ A
    BTB = B @ B
    CTC = C @ C

    # 距离表：dist[target_label][anchor_label] = scalar
    dist = {}
    for t in targets:
        t_samples = np.asarray(t.distribution.samples, dtype=float)
        dist[t.label] = {}
        for a in anchors:
            d = wasserstein_distance(anchor_samples[a.label], t_samples)
            dist[t.label][a.label] = float(d)

    # # 最小二乘解（到真实 A/B/C 的距离）
    # def trilaterate(dA, dB, dC):
    #     b1 = (BTB - ATA + dA**2 - dB**2) / 2.0
    #     b2 = (CTC - ATA + dA**2 - dC**2) / 2.0
    #     p, *_ = np.linalg.lstsq(M, np.array([b1, b2]), rcond=None)
    #     return p

    from scipy.optimize import minimize

    def fit_inside_triangle(dA, dB, dC, A, B, C, u0v0=(1/3, 1/3)):
        # 变量 x = (u, v), w = 1 - u - v
        def loss(x):
            u, v = x
            w = 1.0 - u - v
            # 约束外时给大惩罚，避免数值漂出
            if (u < -1e-9) or (v < -1e-9) or (w < -1e-9):
                return 1e6 + (min(u, v, w))**2
            P = u*A + v*B + w*C
            rA = np.linalg.norm(P - A) - dA
            rB = np.linalg.norm(P - B) - dB
            rC = np.linalg.norm(P - C) - dC
            return rA*rA + rB*rB + rC*rC

        cons = [
            {'type': 'ineq', 'fun': lambda x: x[0]},                 # u >= 0
            {'type': 'ineq', 'fun': lambda x: x[1]},                 # v >= 0
            {'type': 'ineq', 'fun': lambda x: 1.0 - x[0] - x[1]},    # w >= 0
        ]
        res = minimize(loss, x0=np.array(u0v0, float), constraints=cons, method='SLSQP',
                    options={'maxiter': 200, 'ftol': 1e-12, 'disp': False})
        u, v = res.x
        w = 1.0 - u - v
        P = u*A + v*B + w*C
        return P

    # 生成坐标
    coords = {}
    for tl in target_labels:
        dA = dist[tl][anchor_order[0]]
        dB = dist[tl][anchor_order[1]]
        dC = dist[tl][anchor_order[2]]
        p = fit_inside_triangle(dA, dB, dC, A, B, C)  # 直接在三角形内拟合
        coords[tl] = p

    # === 下面的绘图代码保持不变（只是 anchors2d 改用新的 A/B/C） ===
    from adjustText import adjust_text

    plt.figure(figsize=(6,6))
    anchors2d = np.vstack([A, B, C])
    plt.scatter(anchors2d[:,0], anchors2d[:,1], marker='^', s=120, c='k', label='Anchors')

    texts = []
    for lab, xy in zip(anchor_order, anchors2d):
        texts.append(plt.text(xy[0], xy[1], f" {lab}", fontsize=10, va='bottom', ha='left'))

    for tl in target_labels:
        x, y = coords[tl]
        plt.scatter([x], [y], s=80)
        texts.append(plt.text(x, y, f" {tl}", fontsize=9, va='bottom', ha='left'))

    adjust_text(
        texts,
        only_move={'texts':'xy'},
        force_points=0.2, force_text=0.5,
        expand_points=(1.1, 1.2), expand_text=(1.1, 1.2),
        arrowprops=dict(arrowstyle='->', lw=0.6, alpha=0.7)
    )

    plt.axis('equal')
    if label:
        plt.suptitle(label, fontsize=10)
    plt.tight_layout()
    plt.legend(loc='best', fontsize=8)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    plt.savefig(output, dpi=300)
    plt.close()