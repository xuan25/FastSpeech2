import csv
import os
from enum import IntEnum
from typing import Literal, cast

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
from matplotlib.lines import Line2D


# =========================
# Sentiment Enum & Aliases
# =========================
class Sentiment(IntEnum):
    NEG = 0
    NEU = 1
    POS = 2

# 兼容旧常量名（可删）
NEG, NEU, POS = Sentiment.NEG, Sentiment.NEU, Sentiment.POS
SENT_NAMES = {Sentiment.NEG: "NEG", Sentiment.NEU: "NEU", Sentiment.POS: "POS"}


# =========================
# Ground-truth mapping
# =========================
# SENTIMENTS_REF_FILE = "output/prosody_predictor_gt/gt/pred/val.csv"
SENTIMENTS_REF_FILE = "output/prosody_predictor/sentiment_input_concat/pred/train.csv"
sentiment_mapping: dict[str, int] = {}

try:
    with open(SENTIMENTS_REF_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 行为：data_id -> 0/1/2
            sentiment_mapping[row["data_id"]] = int(row["sentiment"])
except FileNotFoundError:
    print(f"Warning: Sentiment reference file not found: {SENTIMENTS_REF_FILE}")
except (UnicodeDecodeError, csv.Error) as e:
    print(f"Error reading sentiment reference file: {e}")


def get_data_sentiment(data_id: str) -> int:
    """返回 0/1/2；未知返回 -1。"""
    return sentiment_mapping.get(data_id, -1)


# =========================
# Data containers
# =========================
class Distribution:
    def __init__(self, samples_file: str, sentiment_filter: Sentiment, feature: str):
        self.samples_file = samples_file
        self.sentiment_filter = Sentiment(sentiment_filter)
        self.feature = feature
        self.samples = self.load_data(samples_file, self.sentiment_filter, feature)

    def load_data(self, samples_file: str, sentiment_filter: Sentiment, feature: str):
        data = []
        try:
            with open(samples_file, "r", encoding="utf-8") as file:
                csv_reader = csv.DictReader(file)
                kept = 0
                total = 0
                for data_row in csv_reader:
                    total += 1
                    if get_data_sentiment(data_row.get("data_id", "")) == int(sentiment_filter):
                        try:
                            data.append(float(data_row[feature]))
                            kept += 1
                        except (ValueError, KeyError) as e:
                            print(f"Warning: Skipping invalid data in {samples_file}: {e}")
                if kept == 0:
                    print(
                        f"Warning: No rows kept from {samples_file} "
                        f"for sentiment={SENT_NAMES[sentiment_filter]} and feature='{feature}'. (total rows={total})"
                    )
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
    """锚点带显式情绪（用于填充样式编码）"""
    def __init__(self, distribution: Distribution, sentiment: Sentiment, position_label: str):
        label = position_label or f"GT_{SENT_NAMES[sentiment]}"
        super().__init__(distribution, position_label)
        self.sentiment = Sentiment(sentiment)


class Target(Position):
    """
    orig_sentiment: 样本的原始情绪 (ground truth)
    target_sentiment: 模型条件/目标情绪（你之前的 SI 情绪，现在命名更清晰）
    """
    def __init__(
        self,
        distribution: Distribution,
        orig_sentiment: Sentiment,
        target_sentiment: Sentiment,
        position_label: str | None = None
    ):
        label = position_label or f"{SENT_NAMES[orig_sentiment]}->TGT_{SENT_NAMES[target_sentiment]}"
        super().__init__(distribution, label)
        self.orig_sentiment = Sentiment(orig_sentiment)
        self.target_sentiment = Sentiment(target_sentiment)


class Task:
    def __init__(self, task_anchors: list[Anchor], task_targets: list[Target], task_output: str, task_label: str = ""):
        self.anchors = task_anchors
        self.targets = task_targets
        self.output = task_output
        self.label = task_label

# =========================
# Anchors (GT distributions)
# =========================
anchors_shared = [
    Anchor(Distribution("output/prosody_predictor_gt/gt/pred/val.csv", NEG, "pitch"), NEG, "GT_NEG"),
    Anchor(Distribution("output/prosody_predictor_gt/gt/pred/val.csv", NEU, "pitch"), NEU, "GT_NEU"),
    Anchor(Distribution("output/prosody_predictor_gt/gt/pred/val.csv", POS, "pitch"), POS, "GT_POS"),
]

anchors_shared_train = [
    Anchor(Distribution("output/prosody_predictor/sentiment_input_concat/pred/train.csv", NEG, "pitch"), NEG, "GT_NEG"),
    Anchor(Distribution("output/prosody_predictor/sentiment_input_concat/pred/train.csv", NEU, "pitch"), NEU, "GT_NEU"),
    Anchor(Distribution("output/prosody_predictor/sentiment_input_concat/pred/train.csv", POS, "pitch"), POS, "GT_POS"),
]

# =========================
# Tasks
# =========================
tasks = [

    Task(
        anchors_shared_train,
        [
            Target(Distribution("output/prosody_predictor/sentiment_input_concat/pred/train_neu.csv", NEG, "pitch"), NEG, NEU),
            Target(Distribution("output/prosody_predictor/sentiment_input_concat/pred/train_neu.csv", NEU, "pitch"), NEU, NEU),
            Target(Distribution("output/prosody_predictor/sentiment_input_concat/pred/train_neu.csv", POS, "pitch"), POS, NEU),
            Target(Distribution("output/prosody_predictor/sentiment_input_concat/pred/train_neg.csv", NEG, "pitch"), NEG, NEG),
            Target(Distribution("output/prosody_predictor/sentiment_input_concat/pred/train_neg.csv", NEU, "pitch"), NEU, NEG),
            Target(Distribution("output/prosody_predictor/sentiment_input_concat/pred/train_neg.csv", POS, "pitch"), POS, NEG),
            Target(Distribution("output/prosody_predictor/sentiment_input_concat/pred/train_pos.csv", NEG, "pitch"), NEG, POS),
            Target(Distribution("output/prosody_predictor/sentiment_input_concat/pred/train_pos.csv", NEU, "pitch"), NEU, POS),
            Target(Distribution("output/prosody_predictor/sentiment_input_concat/pred/train_pos.csv", POS, "pitch"), POS, POS),
        ],
        task_output="output/plots/trilateration-eqtri-ols/prosody-predictor_sentiment-input-concat_train.png",
        task_label="prosody-predictor_sentiment-input-concat_train"
    ),


    # Task(
    #     anchors_shared,
    #     [
    #         Target(Distribution("output/prosody_predictor/sentiment_input_concat/pred/val_neu.csv", NEG, "pitch"), NEG, NEU),
    #         Target(Distribution("output/prosody_predictor/sentiment_input_concat/pred/val_neu.csv", NEU, "pitch"), NEU, NEU),
    #         Target(Distribution("output/prosody_predictor/sentiment_input_concat/pred/val_neu.csv", POS, "pitch"), POS, NEU),
    #         Target(Distribution("output/prosody_predictor/sentiment_input_concat/pred/val_neg.csv", NEG, "pitch"), NEG, NEG),
    #         Target(Distribution("output/prosody_predictor/sentiment_input_concat/pred/val_neg.csv", NEU, "pitch"), NEU, NEG),
    #         Target(Distribution("output/prosody_predictor/sentiment_input_concat/pred/val_neg.csv", POS, "pitch"), POS, NEG),
    #         Target(Distribution("output/prosody_predictor/sentiment_input_concat/pred/val_pos.csv", NEG, "pitch"), NEG, POS),
    #         Target(Distribution("output/prosody_predictor/sentiment_input_concat/pred/val_pos.csv", NEU, "pitch"), NEU, POS),
    #         Target(Distribution("output/prosody_predictor/sentiment_input_concat/pred/val_pos.csv", POS, "pitch"), POS, POS),

    #     ],
    #     task_output="output/plots/trilateration-eqtri-ols/prosody-predictor_sentiment-input-concat.png",
    #     task_label="prosody-predictor_sentiment-input-concat"
    # ),

    # Task(
    #     anchors_shared,
    #     [
    #         Target(Distribution("output/prosody_predictor_contrastive/sentiment_input_concat-0/pred/val_neu.csv", NEG, "pitch"), NEG, NEU),
    #         Target(Distribution("output/prosody_predictor_contrastive/sentiment_input_concat-0/pred/val_neu.csv", NEU, "pitch"), NEU, NEU),
    #         Target(Distribution("output/prosody_predictor_contrastive/sentiment_input_concat-0/pred/val_neu.csv", POS, "pitch"), POS, NEU),
    #         Target(Distribution("output/prosody_predictor_contrastive/sentiment_input_concat-0/pred/val_neg.csv", NEG, "pitch"), NEG, NEG),
    #         Target(Distribution("output/prosody_predictor_contrastive/sentiment_input_concat-0/pred/val_neg.csv", NEU, "pitch"), NEU, NEG),
    #         Target(Distribution("output/prosody_predictor_contrastive/sentiment_input_concat-0/pred/val_neg.csv", POS, "pitch"), POS, NEG),
    #         Target(Distribution("output/prosody_predictor_contrastive/sentiment_input_concat-0/pred/val_pos.csv", NEG, "pitch"), NEG, POS),
    #         Target(Distribution("output/prosody_predictor_contrastive/sentiment_input_concat-0/pred/val_pos.csv", NEU, "pitch"), NEU, POS),
    #         Target(Distribution("output/prosody_predictor_contrastive/sentiment_input_concat-0/pred/val_pos.csv", POS, "pitch"), POS, POS),

    #     ],
    #     task_output="output/plots/trilateration-eqtri-ols/prosody-predictor_sentiment-input-concat_contrastive-0.png",
    #     task_label="prosody-predictor_sentiment-input-concat_contrastive-0"
    # ),
    # Task(
    #     anchors_shared,
    #     [
    #         Target(Distribution("output/prosody_predictor_contrastive/sentiment_input_concat-0.1/pred/val_neu.csv", NEG, "pitch"), NEG, NEU),
    #         Target(Distribution("output/prosody_predictor_contrastive/sentiment_input_concat-0.1/pred/val_neu.csv", NEU, "pitch"), NEU, NEU),
    #         Target(Distribution("output/prosody_predictor_contrastive/sentiment_input_concat-0.1/pred/val_neu.csv", POS, "pitch"), POS, NEU),
    #         Target(Distribution("output/prosody_predictor_contrastive/sentiment_input_concat-0.1/pred/val_neg.csv", NEG, "pitch"), NEG, NEG),
    #         Target(Distribution("output/prosody_predictor_contrastive/sentiment_input_concat-0.1/pred/val_neg.csv", NEU, "pitch"), NEU, NEG),
    #         Target(Distribution("output/prosody_predictor_contrastive/sentiment_input_concat-0.1/pred/val_neg.csv", POS, "pitch"), POS, NEG),
    #         Target(Distribution("output/prosody_predictor_contrastive/sentiment_input_concat-0.1/pred/val_pos.csv", NEG, "pitch"), NEG, POS),
    #         Target(Distribution("output/prosody_predictor_contrastive/sentiment_input_concat-0.1/pred/val_pos.csv", NEU, "pitch"), NEU, POS),
    #         Target(Distribution("output/prosody_predictor_contrastive/sentiment_input_concat-0.1/pred/val_pos.csv", POS, "pitch"), POS, POS),
    #     ],
    #     task_output="output/plots/trilateration-eqtri-ols/prosody-predictor_sentiment-input-concat_contrastive-0.1.png",
    #     task_label="prosody-predictor_sentiment-input-concat_contrastive-0.1"
    # ),


    # Task(
    #     anchors_shared,
    #     [
    #         Target(Distribution("output/prosody_predictor/sentiment_input/pred/val_neu.csv", NEG, "pitch"), NEG, NEU),
    #         Target(Distribution("output/prosody_predictor/sentiment_input/pred/val_neu.csv", NEU, "pitch"), NEU, NEU),
    #         Target(Distribution("output/prosody_predictor/sentiment_input/pred/val_neu.csv", POS, "pitch"), POS, NEU),
    #         Target(Distribution("output/prosody_predictor/sentiment_input/pred/val_neg.csv", NEG, "pitch"), NEG, NEG),
    #         Target(Distribution("output/prosody_predictor/sentiment_input/pred/val_neg.csv", NEU, "pitch"), NEU, NEG),
    #         Target(Distribution("output/prosody_predictor/sentiment_input/pred/val_neg.csv", POS, "pitch"), POS, NEG),
    #         Target(Distribution("output/prosody_predictor/sentiment_input/pred/val_pos.csv", NEG, "pitch"), NEG, POS),
    #         Target(Distribution("output/prosody_predictor/sentiment_input/pred/val_pos.csv", NEU, "pitch"), NEU, POS),
    #         Target(Distribution("output/prosody_predictor/sentiment_input/pred/val_pos.csv", POS, "pitch"), POS, POS),

    #     ],
    #     task_output="output/plots/trilateration-eqtri-ols/prosody-predictor_sentiment-input.png",
    #     task_label="prosody-predictor_sentiment-input"
    # ),
    # Task(
    #     anchors_shared,
    #     [
    #         Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0/pred/val_neu.csv", NEG, "pitch"), NEG, NEU),
    #         Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0/pred/val_neu.csv", NEU, "pitch"), NEU, NEU),
    #         Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0/pred/val_neu.csv", POS, "pitch"), POS, NEU),
    #         Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0/pred/val_neg.csv", NEG, "pitch"), NEG, NEG),
    #         Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0/pred/val_neg.csv", NEU, "pitch"), NEU, NEG),
    #         Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0/pred/val_neg.csv", POS, "pitch"), POS, NEG),
    #         Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0/pred/val_pos.csv", NEG, "pitch"), NEG, POS),
    #         Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0/pred/val_pos.csv", NEU, "pitch"), NEU, POS),
    #         Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0/pred/val_pos.csv", POS, "pitch"), POS, POS),
    #     ],
    #     task_output="output/plots/trilateration-eqtri-ols/prosody-predictor_sentiment-input_contrastive-0.png",
    #     task_label="prosody-predictor_sentiment-input_contrastive-0"
    # ),
    # Task(
    #     anchors_shared,
    #     [
    #         Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0.01/pred/val_neu.csv", NEG, "pitch"), NEG, NEU),
    #         Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0.01/pred/val_neu.csv", NEU, "pitch"), NEU, NEU),
    #         Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0.01/pred/val_neu.csv", POS, "pitch"), POS, NEU),
    #         Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0.01/pred/val_neg.csv", NEG, "pitch"), NEG, NEG),
    #         Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0.01/pred/val_neg.csv", NEU, "pitch"), NEU, NEG),
    #         Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0.01/pred/val_neg.csv", POS, "pitch"), POS, NEG),
    #         Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0.01/pred/val_pos.csv", NEG, "pitch"), NEG, POS),
    #         Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0.01/pred/val_pos.csv", NEU, "pitch"), NEU, POS),
    #         Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0.01/pred/val_pos.csv", POS, "pitch"), POS, POS),
    #     ],
    #     task_output="output/plots/trilateration-eqtri-ols/prosody-predictor_sentiment-input_contrastive-0.01.png",
    #     task_label="prosody-predictor_sentiment-input_contrastive-0.01"
    # ),
    # Task(
    #     anchors_shared,
    #     [
    #         Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0.1/pred/val_neu.csv", NEG, "pitch"), NEG, NEU),
    #         Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0.1/pred/val_neu.csv", NEU, "pitch"), NEU, NEU),
    #         Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0.1/pred/val_neu.csv", POS, "pitch"), POS, NEU),
    #         Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0.1/pred/val_neg.csv", NEG, "pitch"), NEG, NEG),
    #         Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0.1/pred/val_neg.csv", NEU, "pitch"), NEU, NEG),
    #         Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0.1/pred/val_neg.csv", POS, "pitch"), POS, NEG),
    #         Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0.1/pred/val_pos.csv", NEG, "pitch"), NEG, POS),
    #         Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0.1/pred/val_pos.csv", NEU, "pitch"), NEU, POS),
    #         Target(Distribution("output/prosody_predictor_contrastive/sentiment_input-0.1/pred/val_pos.csv", POS, "pitch"), POS, POS),
    #     ],
    #     task_output="output/plots/trilateration-eqtri-ols/prosody-predictor_sentiment-input_contrastive-0.1.png",
    #     task_label="prosody-predictor_sentiment-input_contrastive-0.1"
    # ),
]
# =========================
# Geometry: fixed equilateral triangle
# =========================
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, np.sqrt(3)/2])

# 缩放到更紧凑的范围
scale = 0.4
A *= scale
B *= scale
C *= scale

BA = B - A
CA = C - A
M = np.vstack([BA, CA])  # 2x2
ATA = A @ A
BTB = B @ B
CTC = C @ C

# Anchors 顺序需要与 order 对应
order = ["GT_NEG", "GT_NEU", "GT_POS"]


# =========================
# Plot style mappings
# =========================

FillStyle = Literal["full", "left", "right", "bottom", "top", "none"]

# 形状 = 原始情绪
orig_to_marker = {
    Sentiment.NEG: "v",   # 下三角
    Sentiment.NEU: "o",   # 圆
    Sentiment.POS: "^",   # 上三角
}
# 填充 = 目标情绪
target_to_fill: dict[Sentiment, FillStyle] = {
    Sentiment.NEG: "full",    # 实心
    Sentiment.NEU: "none",    # 空心
    Sentiment.POS: "bottom",  # 半填（底部）
}
anchor_to_fill: dict[Sentiment, FillStyle] = {
    Sentiment.NEG: "full",
    Sentiment.NEU: "none",
    Sentiment.POS: "bottom",
}

def get_target_fill(s: Sentiment) -> FillStyle:
    return target_to_fill.get(s, cast(FillStyle, "none"))

def get_anchor_fill(s: Sentiment) -> FillStyle:
    return anchor_to_fill.get(s, cast(FillStyle, "none"))

# 尺寸参数：scatter 的 s 是面积；plot 的 markersize 是半径像素
MARKER_AREA = 120.0
MARKER_SIZE = float(np.sqrt(MARKER_AREA))  # 用于 plot(Line2D)
EDGE_WIDTH  = 1.2
ALPHA       = 0.9

# 锚点样式（与 target 形状完全脱钩）
ANCHOR_MARKER = 'D'   # 菱形；也可改为 's' 方块
ANCHOR_SIZE   = MARKER_SIZE * 1.1
ANCHOR_EDGEW  = 1.4


# =========================
# Main loop
# =========================
for task in tasks:
    anchors = task.anchors
    targets = task.targets
    output = task.output
    label = task.label

    # 1) 基础校验
    if any(len(a.distribution.samples) == 0 for a in anchors):
        print(f"Error: One or more anchor distributions are empty for task '{label}'. Skip plotting.")
        continue
    anchor_labels = [a.label for a in anchors]
    if anchor_labels != order:
        print(f"Note: anchor label order {anchor_labels} != expected {order}. Make sure order matches A,B,C.")

    # 2) 预取锚点样本并计算 Wasserstein 距离
    anchor_samples = {a.label: np.asarray(a.distribution.samples, dtype=float) for a in anchors}

    dist: dict[str, dict[str, float]] = {}
    for t in targets:
        t_samples = np.asarray(t.distribution.samples, dtype=float)
        if t_samples.size == 0:
            print(f"Warning: Target '{t.label}' has empty distribution. It will be skipped.")
        dist[t.label] = {}
        for a in anchors:
            d = wasserstein_distance(anchor_samples[a.label], t_samples) if t_samples.size else np.nan
            dist[t.label][a.label] = float(d)

    # 3) OLS trilateration
    def trilaterate(dist_a: float, dist_b: float, dist_c: float) -> np.ndarray:
        if not (np.isfinite(dist_a) and np.isfinite(dist_b) and np.isfinite(dist_c)):
            return np.array([np.nan, np.nan], dtype=float)
        b1 = (BTB - ATA + dist_a**2 - dist_b**2) / 2.0
        b2 = (CTC - ATA + dist_a**2 - dist_c**2) / 2.0
        p, *_ = np.linalg.lstsq(M, np.array([b1, b2]), rcond=None)
        return p

    coords: dict[str, np.ndarray] = {}
    for t in targets:
        d_a = dist[t.label][order[0]]
        d_b = dist[t.label][order[1]]
        d_c = dist[t.label][order[2]]
        coords[t.label] = trilaterate(d_a, d_b, d_c)

    # 4) 绘图
    fig, ax = plt.subplots(figsize=(6, 6))

    # 参考等边三角形边（灰色虚线）
    tri_x = [A[0], B[0], C[0], A[0]]
    tri_y = [A[1], B[1], C[1], A[1]]
    ax.plot(tri_x, tri_y, linestyle='--', linewidth=1.0, color='0.6')

    # 锚点（固定菱形，填充样式=锚点情绪），并给三点注释做像素级偏移
    for (lab, xy, anc) in zip(order, np.vstack([A, B, C]), anchors):
        afill = get_anchor_fill(anc.sentiment)
        mfc   = 'black' if afill != 'none' else 'none'
        mfc2  = 'white'
        ax.plot([xy[0]], [xy[1]],
                linestyle='None',
                marker=ANCHOR_MARKER, markersize=ANCHOR_SIZE,
                markerfacecolor=mfc, markerfacecoloralt=mfc2,
                markeredgecolor='black', markeredgewidth=ANCHOR_EDGEW,
                fillstyle=afill)

    # 锚点文本偏移（避免与角落/图例冲突）
    ax.annotate(" GT_NEG", xy=(A[0], A[1]), xytext=(6, 6),   textcoords='offset points',
                ha='left',  va='bottom', fontsize=10)
    ax.annotate(" GT_NEU", xy=(B[0], B[1]), xytext=(-6, 6),  textcoords='offset points',
                ha='right', va='bottom', fontsize=10)
    ax.annotate(" GT_POS", xy=(C[0], C[1]), xytext=(6, -6),  textcoords='offset points',
                ha='left',  va='top',    fontsize=10)

    # 目标点（形状=原始；填充=目标），不加点旁文字
    plotted_any = False
    for t in targets:
        xy = coords[t.label]
        if not np.all(np.isfinite(xy)):
            continue
        marker = orig_to_marker.get(t.orig_sentiment, "s")
        fill   = get_target_fill(t.target_sentiment)
        mfc  = 'black' if fill != 'none' else 'none'
        mfc2 = 'white'
        ax.plot([xy[0]], [xy[1]],
                linestyle='None',
                marker=marker,
                markersize=MARKER_SIZE,
                markerfacecolor=mfc,
                markerfacecoloralt=mfc2,
                markeredgecolor='black',
                markeredgewidth=EDGE_WIDTH,
                fillstyle=fill,
                alpha=ALPHA)
        plotted_any = True

    if not plotted_any:
        print(f"Warning: no valid target points for task '{label}'. Skipping save to {output}.")
        plt.close(fig)
        continue

    ax.set_aspect('equal', adjustable='box')
    if label:
        fig.suptitle(label, fontsize=10)
    plt.tight_layout(rect=(0, 0, 1, 0.93))  # 顶部多留白，容纳上方图例

    # 5) 三组图例：左上两块 + 右上一块，避开上方中间的 GT_POS
    # Anchors legend（左上第二块）
    anchor_fill_handles = [
        Line2D([0], [0], marker=ANCHOR_MARKER, linestyle='None',
               markersize=ANCHOR_SIZE * 0.85,
               markerfacecolor='black', markerfacecoloralt='white',
               markeredgecolor='black', markeredgewidth=ANCHOR_EDGEW,
               fillstyle=anchor_to_fill[Sentiment.NEG], label='GT_NEG'),
        Line2D([0], [0], marker=ANCHOR_MARKER, linestyle='None',
               markersize=ANCHOR_SIZE * 0.85,
               markerfacecolor='none',  markerfacecoloralt='white',
               markeredgecolor='black', markeredgewidth=ANCHOR_EDGEW,
               fillstyle=anchor_to_fill[Sentiment.NEU], label='GT_NEU'),
        Line2D([0], [0], marker=ANCHOR_MARKER, linestyle='None',
               markersize=ANCHOR_SIZE * 0.85,
               markerfacecolor='black', markerfacecoloralt='white',
               markeredgecolor='black', markeredgewidth=ANCHOR_EDGEW,
               fillstyle=anchor_to_fill[Sentiment.POS], label='GT_POS'),
    ]

    # Shape legend（左上第一块）
    shape_handles = [
        Line2D([0], [0], marker=orig_to_marker[Sentiment.NEG], linestyle='None',
               markersize=MARKER_SIZE * 0.85,
               markerfacecolor='none', markeredgecolor='black', markeredgewidth=EDGE_WIDTH, label='NEG'),
        Line2D([0], [0], marker=orig_to_marker[Sentiment.NEU], linestyle='None',
               markersize=MARKER_SIZE * 0.85,
               markerfacecolor='none', markeredgecolor='black', markeredgewidth=EDGE_WIDTH, label='NEU'),
        Line2D([0], [0], marker=orig_to_marker[Sentiment.POS], linestyle='None',
               markersize=MARKER_SIZE * 0.85,
               markerfacecolor='none', markeredgecolor='black', markeredgewidth=EDGE_WIDTH, label='POS'),
    ]

    # Target fill legend（右上）
    fill_handles = [
        Line2D([0], [0], marker='o', linestyle='None',
               markersize=MARKER_SIZE * 0.85,
               markerfacecolor='black', markerfacecoloralt='white',
               markeredgecolor='black', markeredgewidth=EDGE_WIDTH,
               fillstyle=target_to_fill[Sentiment.NEG], label='TGT_NEG'),
        Line2D([0], [0], marker='o', linestyle='None',
               markersize=MARKER_SIZE * 0.85,
               markerfacecolor='none',  markerfacecoloralt='white',
               markeredgecolor='black', markeredgewidth=EDGE_WIDTH,
               fillstyle=target_to_fill[Sentiment.NEU], label='TGT_NEU'),
        Line2D([0], [0], marker='o', linestyle='None',
               markersize=MARKER_SIZE * 0.85,
               markerfacecolor='black', markerfacecoloralt='white',
               markeredgecolor='black', markeredgewidth=EDGE_WIDTH,
               fillstyle=target_to_fill[Sentiment.POS], label='TGT_POS'),
    ]

    # 左上列：1) Original（顶）；2) Anchor（其下）
    legend_shape = ax.legend(handles=shape_handles, title="Original sentiment (shape)",
                             loc='upper left', bbox_to_anchor=(0.02, 0.98),
                             borderaxespad=0.0, fontsize=8, frameon=True)
    ax.add_artist(legend_shape)

    legend_anchor = ax.legend(handles=anchor_fill_handles, title="Anchor sentiment (fill)",
                              loc='upper left', bbox_to_anchor=(0.02, 0.80),
                              borderaxespad=0.0, fontsize=8, frameon=True)
    ax.add_artist(legend_anchor)

    # 右上列：3) Target（顶）
    legend_fill = ax.legend(handles=fill_handles, title="Target sentiment (fill)",
                            loc='upper right', bbox_to_anchor=(0.98, 0.98),
                            borderaxespad=0.0, fontsize=8, frameon=True)
    ax.add_artist(legend_fill)

    # 6) 保存
    os.makedirs(os.path.dirname(output), exist_ok=True)
    fig.savefig(output, dpi=300)
    plt.close(fig)

print("Done.")