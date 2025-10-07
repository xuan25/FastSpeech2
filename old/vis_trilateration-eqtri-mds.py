
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
        task_output="output/plots/trilateration-eqtri-mds/prosody-predictor_sentiment-input.png",
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
        task_output="output/plots/trilateration-eqtri-mds/prosody-predictor_sentiment-input_contrastive-0.png",
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
        task_output="output/plots/trilateration-eqtri-mds/prosody-predictor_sentiment-input_contrastive-0.01.png",
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
        task_output="output/plots/trilateration-eqtri-mds/prosody-predictor_sentiment-input_contrastive-0.1.png",
        task_label="prosody-predictor_sentiment-input_contrastive-0.1"
    ),
]

from scipy.stats import wasserstein_distance
import numpy as np
import matplotlib.pyplot as plt
from adjustText import adjust_text
import os
from sklearn.manifold import MDS
from scipy.spatial.distance import squareform
from scipy.linalg import lstsq


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
    # ========= Metric MDS 替换版 =========

    # 1) 组织样本：锚点 + target
    labels_all = [a.label for a in anchors] + [t.label for t in targets]
    samples_by_label = {}
    for a in anchors:
        samples_by_label[a.label] = np.asarray(a.distribution.samples, dtype=float)
    for t in targets:
        samples_by_label[t.label] = np.asarray(t.distribution.samples, dtype=float)

    # 简单的空数据检查
    empty_labs = [lab for lab in labels_all if samples_by_label[lab].size == 0]
    if empty_labs:
        print(f"Skipping task '{label}' due to empty distributions: {empty_labs}")
        continue

    # 2) 构建成对 Wasserstein 距离矩阵 D (n x n)
    n = len(labels_all)
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        si = samples_by_label[labels_all[i]]
        for j in range(i+1, n):
            sj = samples_by_label[labels_all[j]]
            dij = wasserstein_distance(si, sj)
            D[i, j] = D[j, i] = dij

    # 3) 运行 Metric MDS（SMACOF）
    mds = MDS(
        n_components=2,
        metric=True,
        dissimilarity='precomputed',
        n_init=4,
        max_iter=600,
        random_state=0,
        normalized_stress='auto'  # sklearn >=1.1 才有；老版本可去掉
    )
    X2d = mds.fit_transform(D)         # (n, 2)
    stress = getattr(mds, 'stress_', None)

    # 4) （可选）将三锚点对齐到你想要的等边三角形 A,B,C（相似变换：旋转+缩放+平移）
    ALIGN_TO_TRIANGLE = True
    order = ["GT_NEG", "GT_NEU", "GT_POS"]  # 与你的三角形 A,B,C 一一对应

    def _solve_similarity_transform(src_pts, dst_pts):
        """
        给定 src_pts(kx2) 与 dst_pts(kx2)，求最小二乘相似变换 x -> s*R*x + t
        返回 s, R(2x2), t(2,)
        """
        # 中心化
        src_mean = src_pts.mean(axis=0)
        dst_mean = dst_pts.mean(axis=0)
        X = src_pts - src_mean
        Y = dst_pts - dst_mean

        # 允许有缩放的 Procrustes：用最小二乘解 R 与 s
        # 解 R,s:  min || s*X*R - Y ||_F
        # 先解 R（旋转+反射），用 SVD 的 Kabsch；再解 s
        U, S, Vt = np.linalg.svd(X.T @ Y)
        R = U @ Vt
        # 防止反射翻转：保证 det(R)=+1（可按需允许镜像）
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = U @ Vt
        # 缩放
        s = np.trace((X @ R).T @ Y) / np.trace(X.T @ X) if np.trace(X.T @ X) > 0 else 1.0
        t = dst_mean - s * (src_mean @ R)
        return s, R, t

    if ALIGN_TO_TRIANGLE:
        # 当前三锚在 MDS 空间中的坐标
        idx_map = {lab: i for i, lab in enumerate(labels_all)}
        try:
            src = np.vstack([X2d[idx_map[order[0]]],
                            X2d[idx_map[order[1]]],
                            X2d[idx_map[order[2]]]])  # (3,2)
            dst = np.vstack([A, B, C])  # 你前面定义并缩放过的等边三角形坐标

            s, R, t = _solve_similarity_transform(src, dst)
            X2d_aligned = (X2d @ R) * s + t
        except KeyError:
            print(f"Warning: missing some anchors in order={order}; skip alignment.")
            X2d_aligned = X2d
    else:
        X2d_aligned = X2d

    # 5) 画图
    plt.figure(figsize=(6, 6))

    # 画锚点
    anchors_idx = [labels_all.index(lab) for lab in order if lab in labels_all]
    anchors_xy = X2d_aligned[anchors_idx]
    plt.scatter(anchors_xy[:, 0], anchors_xy[:, 1], marker='^', s=120, c='k', label='Anchors')

    texts = []
    # 锚点标签
    for lab in order:
        if lab in labels_all:
            i = labels_all.index(lab)
            x, y = X2d_aligned[i]
            texts.append(plt.text(x, y, f" {lab}", fontsize=10, va='bottom', ha='left'))

    # 画 targets
    for tl in [t.label for t in targets]:
        i = labels_all.index(tl)
        x, y = X2d_aligned[i]
        plt.scatter([x], [y], s=80)
        texts.append(plt.text(x, y, f" {tl}", fontsize=9, va='bottom', ha='left'))

    # 文本避让
    adjust_text(
        texts, only_move={'texts': 'xy'},
        force_points=0.2, force_text=0.5,
        expand_points=(1.1, 1.2), expand_text=(1.1, 1.2),
        arrowprops=dict(arrowstyle='->', lw=0.6, alpha=0.7)
    )

    plt.axis('equal')
    title = label if label else "Metric MDS (Wasserstein)"
    if stress is not None:
        title += f" | stress={stress:.3g}"
    plt.suptitle(title, fontsize=10)
    plt.tight_layout(rect=(0, 0, 1, 0.96))

    os.makedirs(os.path.dirname(output), exist_ok=True)
    plt.savefig(output, dpi=300)
    plt.close()

    # （可选）评估：Shepard 相关性（W₁ vs. 嵌入欧氏）
    # 你也可以把这段改成打印或记录到文件
    Euclid = np.sqrt(((X2d_aligned[:, None, :] - X2d_aligned[None, :, :])**2).sum(-1))
    upper = np.triu_indices(n, 1)
    pearson = np.corrcoef(D[upper], Euclid[upper])[0, 1]
    print(f"[{label}] Shepard correlation (W1 vs. 2D Euclid): r={pearson:.3f}")