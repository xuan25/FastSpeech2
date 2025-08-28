
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
        task_output="output/plots/trilateration/prosody-predictor_sentiment-input.png",
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
        task_output="output/plots/trilateration/prosody-predictor_sentiment-input_contrastive-0.png",
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
        task_output="output/plots/trilateration/prosody-predictor_sentiment-input_contrastive-0.01.png",
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
        task_output="output/plots/trilateration/prosody-predictor_sentiment-input_contrastive-0.1.png",
        task_label="prosody-predictor_sentiment-input_contrastive-0.1"
    ),
]
from scipy.stats import wasserstein_distance
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
from adjustText import adjust_text
import os

# ----------------------- 工具函数 -----------------------
def place_three_anchors_by_dist(s0, s1, s2, scale=1.0, flip=False):
    """用三个分布的 Wasserstein 距离在 2D 放置锚点 A,B,C。"""
    d01 = wasserstein_distance(s0, s1)  # |A-B|
    d02 = wasserstein_distance(s0, s2)  # |A-C|
    d12 = wasserstein_distance(s1, s2)  # |B-C|
    eps = 1e-12
    A = np.array([0.0, 0.0], float)
    B = np.array([max(d01, eps), 0.0], float)
    xC = (d02**2 - d12**2 + d01**2) / (2.0 * max(d01, eps))
    y_sq = max(d02**2 - xC**2, 0.0)
    yC = np.sqrt(y_sq)
    if flip:
        yC = -yC
    C = np.array([xC, yC], float)
    A *= scale; B *= scale; C *= scale
    return A, B, C

def build_trilateration_mats(A, B, C):
    BA = B - A
    CA = C - A
    M = np.vstack([BA, CA])  # 2x2
    ATA = float(A @ A)
    BTB = float(B @ B)
    CTC = float(C @ C)
    return M, ATA, BTB, CTC

def trilaterate_ols(dA, dB, dC, M, ATA, BTB, CTC):
    """线性化最小二乘（可能落在三角形外）"""
    b1 = (BTB - ATA + dA**2 - dB**2) / 2.0
    b2 = (CTC - ATA + dA**2 - dC**2) / 2.0
    p, *_ = np.linalg.lstsq(M, np.array([b1, b2], float), rcond=None)
    return p

def _barycentric_coords(P, A, B, C):
    v0 = B - A; v1 = C - A; v2 = P - A
    d00 = float(v0 @ v0); d01 = float(v0 @ v1); d11 = float(v1 @ v1)
    d20 = float(v2 @ v0); d21 = float(v2 @ v1)
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-20:
        return (1.0, 0.0, 0.0)
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return (u, v, w)

def _project_point_to_segment(P, X, Y):
    XY = Y - X
    denom = float(XY @ XY)
    if denom <= 0:
        return X.copy()
    t = float((P - X) @ XY) / denom
    t = max(0.0, min(1.0, t))
    return X + t * XY

def project_to_triangle(P, A, B, C):
    u, v, w = _barycentric_coords(P, A, B, C)
    if (u >= 0) and (v >= 0) and (w >= 0):
        return P
    Pab = _project_point_to_segment(P, A, B)
    Pbc = _project_point_to_segment(P, B, C)
    Pca = _project_point_to_segment(P, C, A)
    def d2(Q): return float(np.sum((P - Q) ** 2))
    return min([Pab, Pbc, Pca], key=d2)

def fit_inside_triangle(dA, dB, dC, A, B, C, u0v0=(1/3, 1/3)):
    """在重心坐标约束（u,v,w>=0, u+v+w=1）下最小化距离残差平方和。"""
    def loss(x):
        u, v = x
        w = 1.0 - u - v
        if (u < -1e-9) or (v < -1e-9) or (w < -1e-9):
            return 1e6 + min(u, v, w)**2
        P = u*A + v*B + w*C
        rA = np.linalg.norm(P - A) - dA
        rB = np.linalg.norm(P - B) - dB
        rC = np.linalg.norm(P - C) - dC
        return rA*rA + rB*rB + rC*rC

    cons = [
        {'type': 'ineq', 'fun': lambda x: x[0]},               # u >= 0
        {'type': 'ineq', 'fun': lambda x: x[1]},               # v >= 0
        {'type': 'ineq', 'fun': lambda x: 1.0 - x[0] - x[1]},  # w >= 0
    ]
    res = minimize(loss, x0=np.array(u0v0, float), constraints=cons, method='SLSQP',
                   options={'maxiter': 200, 'ftol': 1e-12, 'disp': False})
    u, v = res.x
    w = 1.0 - u - v
    return u*A + v*B + w*C

def draw_plot(A, B, C, coords_dict, anchor_order, target_labels, title, out_path):
    """统一绘图：anchors+targets+标签避让，并保存到 out_path"""
    plt.figure(figsize=(6,6))
    anchors2d = np.vstack([A, B, C])
    plt.scatter(anchors2d[:,0], anchors2d[:,1], marker='^', s=120, c='k', label='Anchors')

    texts = []
    for lab, xy in zip(anchor_order, anchors2d):
        texts.append(plt.text(xy[0], xy[1], f" {lab}", fontsize=10, va='bottom', ha='left'))

    for tl in target_labels:
        x, y = coords_dict[tl]
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
    if title:
        plt.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.legend(loc='best', fontsize=8)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()

# ----------------------- 主循环：为每个 task 生成三张图 -----------------------
anchor_order = ["GT_NEG", "GT_NEU", "GT_POS"]  # A, B, C 的锚点顺序

for task in tasks:
    anchors = task.anchors
    targets = task.targets
    base_output = task.output  # 原来的输出路径
    task_title = task.label

    target_labels = [t.label for t in targets]

    # 预取 anchor 的样本数组
    anchor_samples = {a.label: np.asarray(a.distribution.samples, dtype=float) for a in anchors}

    # === 1) 用真实距离放置三锚点（保持形状和尺度） ===
    sA = anchor_samples[anchor_order[0]]
    sB = anchor_samples[anchor_order[1]]
    sC = anchor_samples[anchor_order[2]]
    A, B, C = place_three_anchors_by_dist(sA, sB, sC, scale=1.0, flip=False)

    # === 2) 预计算最小二乘矩阵 ===
    M, ATA, BTB, CTC = build_trilateration_mats(A, B, C)

    # === 3) target -> anchor 的 Wasserstein 距离表 ===
    dist = {}
    for t in targets:
        ts = np.asarray(t.distribution.samples, dtype=float)
        dist[t.label] = {
            anchor_order[0]: float(wasserstein_distance(sA, ts)),
            anchor_order[1]: float(wasserstein_distance(sB, ts)),
            anchor_order[2]: float(wasserstein_distance(sC, ts)),
        }

    # === 4) 三种坐标解法 ===
    coords_unconstrained = {}
    coords_projected = {}
    coords_constrained = {}

    for tl in target_labels:
        dA = dist[tl][anchor_order[0]]
        dB = dist[tl][anchor_order[1]]
        dC = dist[tl][anchor_order[2]]

        # 4.1 普通最小二乘
        p_ols = trilaterate_ols(dA, dB, dC, M, ATA, BTB, CTC)
        coords_unconstrained[tl] = p_ols

        # 4.2 OLS 后投影到三角形
        coords_projected[tl] = project_to_triangle(p_ols, A, B, C)

        # 4.3 约束优化（在三角形内拟合）
        coords_constrained[tl] = fit_inside_triangle(dA, dB, dC, A, B, C)

    # === 5) 保存三张图 ===
    root, ext = os.path.splitext(base_output)
    out_ols  = f"{root}_unconstrained{ext}"
    out_proj = f"{root}_projected{ext}"
    out_cons = f"{root}_constrained{ext}"

    draw_plot(A, B, C, coords_unconstrained, anchor_order, target_labels,
              f"{task_title} | OLS (unconstrained)", out_ols)
    draw_plot(A, B, C, coords_projected, anchor_order, target_labels,
              f"{task_title} | OLS → projected to triangle", out_proj)
    draw_plot(A, B, C, coords_constrained, anchor_order, target_labels,
              f"{task_title} | Constrained inside triangle (SLSQP)", out_cons)