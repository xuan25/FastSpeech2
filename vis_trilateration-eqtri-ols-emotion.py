from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Callable, Dict, List, Literal, Mapping, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import wasserstein_distance

# =========================
# Sentiment Enum & Aliases
# =========================
class Sentiment(IntEnum):
    ANGER = 0
    DISGUST = 1
    FEAR = 2
    JOY = 3
    NEU = 4
    SAD = 5
    SUP = 6

SAD, NEU, JOY = Sentiment.SAD, Sentiment.NEU, Sentiment.JOY
SENT_NAMES: Dict[Sentiment, str] = {SAD: "SAD", NEU: "NEU", JOY: "JOY"}


# =========================
# Ground-truth mapping
# =========================

def load_emotion_mapping(ref_csv: Path) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    try:
        with ref_csv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    mapping[row["basename"]] = np.argmax(
                        [float(row["anger"]), float(row["disgust"]), float(row["fear"]), float(row["joy"]), float(row["neutral"]), float(row["sadness"]), float(row["surprise"])]
                    ).item()
                except (ValueError, KeyError):
                    continue
    except FileNotFoundError:
        print(f"Warning: Emotion reference file not found: {ref_csv}")
    except (UnicodeDecodeError, csv.Error) as e:
        print(f"Error reading emotion reference file: {e}")
    return mapping


# =========================
# Data containers
# =========================
@dataclass
class Distribution:
    samples_file: Path
    sentiment_filter: Sentiment
    feature: str
    mapping: Mapping[str, int]

    _cached: Optional[np.ndarray] = None  # 内部缓存

    def load_samples(self) -> np.ndarray:
        """懒加载 + 缓存"""
        if self._cached is not None:
            return self._cached

        data: list[float] = []
        sentiment_filter_int = int(self.sentiment_filter)
        try:
            with self.samples_file.open("r", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                kept = 0
                total = 0
                for row in reader:
                    total += 1
                    sid = self.mapping.get(row.get("data_id", ""), -1)
                    if sid == sentiment_filter_int:
                        try:
                            v = float(row[self.feature])
                            if math.isfinite(v):
                                data.append(v)
                                kept += 1
                        except (ValueError, KeyError):
                            continue
                if kept == 0:
                    print(
                        f"Warning: No rows kept from {self.samples_file} "
                        f"for sentiment={SENT_NAMES[self.sentiment_filter]} "
                        f"and feature='{self.feature}' (total rows={total})"
                    )
        except FileNotFoundError:
            print(f"Error: File not found: {self.samples_file}")
            self._cached = np.array([], dtype=float)
            return self._cached
        except (UnicodeDecodeError, csv.Error) as e:
            print(f"Error reading {self.samples_file}: {e}")
            self._cached = np.array([], dtype=float)
            return self._cached

        self._cached = np.asarray(data, dtype=float)
        return self._cached


@dataclass
class Position:
    distribution: Distribution
    label: str


@dataclass
class Anchor(Position):
    sentiment: Sentiment

    @classmethod
    def make(
        cls,
        samples_file: Path,
        sentiment: Sentiment,
        feature: str,
        mapping: Mapping[str, int],
        position_label: Optional[str] = None,
    ) -> "Anchor":
        label = position_label or f"GT_{SENT_NAMES[sentiment]}"
        dist = Distribution(samples_file, sentiment, feature, mapping)
        return cls(dist, label, sentiment)


@dataclass
class Target(Position):
    orig_sentiment: Sentiment
    target_sentiment: Sentiment

    @classmethod
    def make(
        cls,
        samples_file: Path,
        data_sentiment_filter: Sentiment,
        feature: str,
        mapping: Mapping[str, int],
        orig_sentiment: Sentiment,
        target_sentiment: Sentiment,
        position_label: Optional[str] = None,
    ) -> "Target":
        auto = position_label or f"{SENT_NAMES[orig_sentiment]}->{SENT_NAMES[target_sentiment]}"
        dist = Distribution(samples_file, data_sentiment_filter, feature, mapping)
        return cls(dist, auto, orig_sentiment, target_sentiment)


@dataclass
class Task:
    anchors: List[Anchor]
    targets: List[Target]
    output: Path
    label: str = ""
    scale: float = 1.0


emotion_mapping = load_emotion_mapping(Path("data/emotion_meld.csv"))

# =========================
# Anchors (GT distributions)
# =========================
anchors_shared = [
    Anchor.make(Path("output/prosody_predictor_gt/gt_MELD/pred/val.csv"), Sentiment.SAD, "pitch", emotion_mapping),
    Anchor.make(Path("output/prosody_predictor_gt/gt_MELD/pred/val.csv"), Sentiment.NEU, "pitch", emotion_mapping),
    Anchor.make(Path("output/prosody_predictor_gt/gt_MELD/pred/val.csv"), Sentiment.JOY, "pitch", emotion_mapping),
]

# =========================
# Tasks
# =========================
tasks = [

    Task(
        anchors_shared,
        [
            Target.make(Path("output/prosody_predictor_contrastive5/emotion_input-0.01-B/pred/val_neutral_80k.csv"), Sentiment.SAD, "pitch", emotion_mapping, Sentiment.SAD, Sentiment.NEU),
            Target.make(Path("output/prosody_predictor_contrastive5/emotion_input-0.01-B/pred/val_neutral_80k.csv"), Sentiment.NEU, "pitch", emotion_mapping, Sentiment.NEU, Sentiment.NEU),
            Target.make(Path("output/prosody_predictor_contrastive5/emotion_input-0.01-B/pred/val_neutral_80k.csv"), Sentiment.JOY, "pitch", emotion_mapping, Sentiment.JOY, Sentiment.NEU),
            Target.make(Path("output/prosody_predictor_contrastive5/emotion_input-0.01-B/pred/val_sadness_80k.csv"), Sentiment.SAD, "pitch", emotion_mapping, Sentiment.SAD, Sentiment.SAD),
            Target.make(Path("output/prosody_predictor_contrastive5/emotion_input-0.01-B/pred/val_sadness_80k.csv"), Sentiment.NEU, "pitch", emotion_mapping, Sentiment.NEU, Sentiment.SAD),
            Target.make(Path("output/prosody_predictor_contrastive5/emotion_input-0.01-B/pred/val_sadness_80k.csv"), Sentiment.JOY, "pitch", emotion_mapping, Sentiment.JOY, Sentiment.SAD),
            Target.make(Path("output/prosody_predictor_contrastive5/emotion_input-0.01-B/pred/val_joy_80k.csv"), Sentiment.SAD, "pitch", emotion_mapping, Sentiment.SAD, Sentiment.JOY),
            Target.make(Path("output/prosody_predictor_contrastive5/emotion_input-0.01-B/pred/val_joy_80k.csv"), Sentiment.NEU, "pitch", emotion_mapping, Sentiment.NEU, Sentiment.JOY),
            Target.make(Path("output/prosody_predictor_contrastive5/emotion_input-0.01-B/pred/val_joy_80k.csv"), Sentiment.JOY, "pitch", emotion_mapping, Sentiment.JOY, Sentiment.JOY),

        ],
        output=Path("output/plots/trilateration-eqtri-ols/prosody-predictor_emotion-input_contrastive5-0.01-B_80k.png"),
        label="prosody-predictor_emotion-input_contrastive5-0.01-B_80k",
        scale=0.4
    ),

    Task(
        anchors_shared,
        [
            Target.make(Path("output/prosody_predictor_contrastive5/emotion_input-0.01-B/pred/val_neutral_40k.csv"), Sentiment.SAD, "pitch", emotion_mapping, Sentiment.SAD, Sentiment.NEU),
            Target.make(Path("output/prosody_predictor_contrastive5/emotion_input-0.01-B/pred/val_neutral_40k.csv"), Sentiment.NEU, "pitch", emotion_mapping, Sentiment.NEU, Sentiment.NEU),
            Target.make(Path("output/prosody_predictor_contrastive5/emotion_input-0.01-B/pred/val_neutral_40k.csv"), Sentiment.JOY, "pitch", emotion_mapping, Sentiment.JOY, Sentiment.NEU),
            Target.make(Path("output/prosody_predictor_contrastive5/emotion_input-0.01-B/pred/val_sadness_40k.csv"), Sentiment.SAD, "pitch", emotion_mapping, Sentiment.SAD, Sentiment.SAD),
            Target.make(Path("output/prosody_predictor_contrastive5/emotion_input-0.01-B/pred/val_sadness_40k.csv"), Sentiment.NEU, "pitch", emotion_mapping, Sentiment.NEU, Sentiment.SAD),
            Target.make(Path("output/prosody_predictor_contrastive5/emotion_input-0.01-B/pred/val_sadness_40k.csv"), Sentiment.JOY, "pitch", emotion_mapping, Sentiment.JOY, Sentiment.SAD),
            Target.make(Path("output/prosody_predictor_contrastive5/emotion_input-0.01-B/pred/val_joy_40k.csv"), Sentiment.SAD, "pitch", emotion_mapping, Sentiment.SAD, Sentiment.JOY),
            Target.make(Path("output/prosody_predictor_contrastive5/emotion_input-0.01-B/pred/val_joy_40k.csv"), Sentiment.NEU, "pitch", emotion_mapping, Sentiment.NEU, Sentiment.JOY),
            Target.make(Path("output/prosody_predictor_contrastive5/emotion_input-0.01-B/pred/val_joy_40k.csv"), Sentiment.JOY, "pitch", emotion_mapping, Sentiment.JOY, Sentiment.JOY),

        ],
        output=Path("output/plots/trilateration-eqtri-ols/prosody-predictor_emotion-input_contrastive5-0.01-B_40k.png"),
        label="prosody-predictor_emotion-input_contrastive5-0.01-B_40k",
        scale=0.4
    ),

    # Task(
    #     anchors_shared,
    #     [
    #         Target.make(Path("output/prosody_predictor_contrastive5/emotion_input-0.1/pred/val_neutral_40k.csv"), Sentiment.SAD, "pitch", emotion_mapping, Sentiment.SAD, Sentiment.NEU),
    #         Target.make(Path("output/prosody_predictor_contrastive5/emotion_input-0.1/pred/val_neutral_40k.csv"), Sentiment.NEU, "pitch", emotion_mapping, Sentiment.NEU, Sentiment.NEU),
    #         Target.make(Path("output/prosody_predictor_contrastive5/emotion_input-0.1/pred/val_neutral_40k.csv"), Sentiment.JOY, "pitch", emotion_mapping, Sentiment.JOY, Sentiment.NEU),
    #         Target.make(Path("output/prosody_predictor_contrastive5/emotion_input-0.1/pred/val_sadness_40k.csv"), Sentiment.SAD, "pitch", emotion_mapping, Sentiment.SAD, Sentiment.SAD),
    #         Target.make(Path("output/prosody_predictor_contrastive5/emotion_input-0.1/pred/val_sadness_40k.csv"), Sentiment.NEU, "pitch", emotion_mapping, Sentiment.NEU, Sentiment.SAD),
    #         Target.make(Path("output/prosody_predictor_contrastive5/emotion_input-0.1/pred/val_sadness_40k.csv"), Sentiment.JOY, "pitch", emotion_mapping, Sentiment.JOY, Sentiment.SAD),
    #         Target.make(Path("output/prosody_predictor_contrastive5/emotion_input-0.1/pred/val_joy_40k.csv"), Sentiment.SAD, "pitch", emotion_mapping, Sentiment.SAD, Sentiment.JOY),
    #         Target.make(Path("output/prosody_predictor_contrastive5/emotion_input-0.1/pred/val_joy_40k.csv"), Sentiment.NEU, "pitch", emotion_mapping, Sentiment.NEU, Sentiment.JOY),
    #         Target.make(Path("output/prosody_predictor_contrastive5/emotion_input-0.1/pred/val_joy_40k.csv"), Sentiment.JOY, "pitch", emotion_mapping, Sentiment.JOY, Sentiment.JOY),

    #     ],
    #     output=Path("output/plots/trilateration-eqtri-ols/prosody-predictor_emotion-input_contrastive5-0.1_40k.png"),
    #     label="prosody-predictor_emotion-input_contrastive5-0.1_40k",
    #     scale=0.4
    # ),


    

    # Task(
    #     anchors_shared,
    #     [
    #         Target.make(Path("output/prosody_predictor_contrastive5/emotion_input-0/pred/val_neutral_40k.csv"), Sentiment.SAD, "pitch", emotion_mapping, Sentiment.SAD, Sentiment.NEU),
    #         Target.make(Path("output/prosody_predictor_contrastive5/emotion_input-0/pred/val_neutral_40k.csv"), Sentiment.NEU, "pitch", emotion_mapping, Sentiment.NEU, Sentiment.NEU),
    #         Target.make(Path("output/prosody_predictor_contrastive5/emotion_input-0/pred/val_neutral_40k.csv"), Sentiment.JOY, "pitch", emotion_mapping, Sentiment.JOY, Sentiment.NEU),
    #         Target.make(Path("output/prosody_predictor_contrastive5/emotion_input-0/pred/val_sadness_40k.csv"), Sentiment.SAD, "pitch", emotion_mapping, Sentiment.SAD, Sentiment.SAD),
    #         Target.make(Path("output/prosody_predictor_contrastive5/emotion_input-0/pred/val_sadness_40k.csv"), Sentiment.NEU, "pitch", emotion_mapping, Sentiment.NEU, Sentiment.SAD),
    #         Target.make(Path("output/prosody_predictor_contrastive5/emotion_input-0/pred/val_sadness_40k.csv"), Sentiment.JOY, "pitch", emotion_mapping, Sentiment.JOY, Sentiment.SAD),
    #         Target.make(Path("output/prosody_predictor_contrastive5/emotion_input-0/pred/val_joy_40k.csv"), Sentiment.SAD, "pitch", emotion_mapping, Sentiment.SAD, Sentiment.JOY),
    #         Target.make(Path("output/prosody_predictor_contrastive5/emotion_input-0/pred/val_joy_40k.csv"), Sentiment.NEU, "pitch", emotion_mapping, Sentiment.NEU, Sentiment.JOY),
    #         Target.make(Path("output/prosody_predictor_contrastive5/emotion_input-0/pred/val_joy_40k.csv"), Sentiment.JOY, "pitch", emotion_mapping, Sentiment.JOY, Sentiment.JOY),

    #     ],
    #     output=Path("output/plots/trilateration-eqtri-ols/prosody-predictor_emotion-input_contrastive5-0_40k.png"),
    #     label="prosody-predictor_emotion-input_contrastive5-0_40k",
    #     scale=0.4
    # ),

    # Task(
    #     anchors_shared,
    #     [
    #         Target.make(Path("output/prosody_predictor_contrastive5/emotion_input-0/pred/val_neutral_80k.csv"), Sentiment.SAD, "pitch", emotion_mapping, Sentiment.SAD, Sentiment.NEU),
    #         Target.make(Path("output/prosody_predictor_contrastive5/emotion_input-0/pred/val_neutral_80k.csv"), Sentiment.NEU, "pitch", emotion_mapping, Sentiment.NEU, Sentiment.NEU),
    #         Target.make(Path("output/prosody_predictor_contrastive5/emotion_input-0/pred/val_neutral_80k.csv"), Sentiment.JOY, "pitch", emotion_mapping, Sentiment.JOY, Sentiment.NEU),
    #         Target.make(Path("output/prosody_predictor_contrastive5/emotion_input-0/pred/val_sadness_80k.csv"), Sentiment.SAD, "pitch", emotion_mapping, Sentiment.SAD, Sentiment.SAD),
    #         Target.make(Path("output/prosody_predictor_contrastive5/emotion_input-0/pred/val_sadness_80k.csv"), Sentiment.NEU, "pitch", emotion_mapping, Sentiment.NEU, Sentiment.SAD),
    #         Target.make(Path("output/prosody_predictor_contrastive5/emotion_input-0/pred/val_sadness_80k.csv"), Sentiment.JOY, "pitch", emotion_mapping, Sentiment.JOY, Sentiment.SAD),
    #         Target.make(Path("output/prosody_predictor_contrastive5/emotion_input-0/pred/val_joy_80k.csv"), Sentiment.SAD, "pitch", emotion_mapping, Sentiment.SAD, Sentiment.JOY),
    #         Target.make(Path("output/prosody_predictor_contrastive5/emotion_input-0/pred/val_joy_80k.csv"), Sentiment.NEU, "pitch", emotion_mapping, Sentiment.NEU, Sentiment.JOY),
    #         Target.make(Path("output/prosody_predictor_contrastive5/emotion_input-0/pred/val_joy_80k.csv"), Sentiment.JOY, "pitch", emotion_mapping, Sentiment.JOY, Sentiment.JOY),

    #     ],
    #     output=Path("output/plots/trilateration-eqtri-ols/prosody-predictor_emotion-input_contrastive5-0_80k.png"),
    #     label="prosody-predictor_emotion-input_contrastive5-0_80k",
    #     scale=0.4,
    # ),    

    
    # Task(
    #     anchors_shared,
    #     [
    #         Target.make(Path("output/prosody_predictor_contrastive5/emotion_input_translate2-0/pred/val_neutral_40k.csv"), Sentiment.SAD, "pitch", emotion_mapping, Sentiment.SAD, Sentiment.NEU),
    #         Target.make(Path("output/prosody_predictor_contrastive5/emotion_input_translate2-0/pred/val_neutral_40k.csv"), Sentiment.NEU, "pitch", emotion_mapping, Sentiment.NEU, Sentiment.NEU),
    #         Target.make(Path("output/prosody_predictor_contrastive5/emotion_input_translate2-0/pred/val_neutral_40k.csv"), Sentiment.JOY, "pitch", emotion_mapping, Sentiment.JOY, Sentiment.NEU),
    #         Target.make(Path("output/prosody_predictor_contrastive5/emotion_input_translate2-0/pred/val_sadness_40k.csv"), Sentiment.SAD, "pitch", emotion_mapping, Sentiment.SAD, Sentiment.SAD),
    #         Target.make(Path("output/prosody_predictor_contrastive5/emotion_input_translate2-0/pred/val_sadness_40k.csv"), Sentiment.NEU, "pitch", emotion_mapping, Sentiment.NEU, Sentiment.SAD),
    #         Target.make(Path("output/prosody_predictor_contrastive5/emotion_input_translate2-0/pred/val_sadness_40k.csv"), Sentiment.JOY, "pitch", emotion_mapping, Sentiment.JOY, Sentiment.SAD),
    #         Target.make(Path("output/prosody_predictor_contrastive5/emotion_input_translate2-0/pred/val_joy_40k.csv"), Sentiment.SAD, "pitch", emotion_mapping, Sentiment.SAD, Sentiment.JOY),
    #         Target.make(Path("output/prosody_predictor_contrastive5/emotion_input_translate2-0/pred/val_joy_40k.csv"), Sentiment.NEU, "pitch", emotion_mapping, Sentiment.NEU, Sentiment.JOY),
    #         Target.make(Path("output/prosody_predictor_contrastive5/emotion_input_translate2-0/pred/val_joy_40k.csv"), Sentiment.JOY, "pitch", emotion_mapping, Sentiment.JOY, Sentiment.JOY),

    #     ],
    #     output=Path("output/plots/trilateration-eqtri-ols/prosody-predictor_emotion-input-translate2_contrastive5-0_40k.png"),
    #     label="prosody-predictor_emotion-input-translate2_contrastive5-0_40k",
    #     scale=1
    # ),

    # Task(
    #     anchors_shared,
    #     [
    #         Target.make(Path("output/prosody_predictor_contrastive5/emotion_input_translate2-0/pred/val_neutral_80k.csv"), Sentiment.SAD, "pitch", emotion_mapping, Sentiment.SAD, Sentiment.NEU),
    #         Target.make(Path("output/prosody_predictor_contrastive5/emotion_input_translate2-0/pred/val_neutral_80k.csv"), Sentiment.NEU, "pitch", emotion_mapping, Sentiment.NEU, Sentiment.NEU),
    #         Target.make(Path("output/prosody_predictor_contrastive5/emotion_input_translate2-0/pred/val_neutral_80k.csv"), Sentiment.JOY, "pitch", emotion_mapping, Sentiment.JOY, Sentiment.NEU),
    #         Target.make(Path("output/prosody_predictor_contrastive5/emotion_input_translate2-0/pred/val_sadness_80k.csv"), Sentiment.SAD, "pitch", emotion_mapping, Sentiment.SAD, Sentiment.SAD),
    #         Target.make(Path("output/prosody_predictor_contrastive5/emotion_input_translate2-0/pred/val_sadness_80k.csv"), Sentiment.NEU, "pitch", emotion_mapping, Sentiment.NEU, Sentiment.SAD),
    #         Target.make(Path("output/prosody_predictor_contrastive5/emotion_input_translate2-0/pred/val_sadness_80k.csv"), Sentiment.JOY, "pitch", emotion_mapping, Sentiment.JOY, Sentiment.SAD),
    #         Target.make(Path("output/prosody_predictor_contrastive5/emotion_input_translate2-0/pred/val_joy_80k.csv"), Sentiment.SAD, "pitch", emotion_mapping, Sentiment.SAD, Sentiment.JOY),
    #         Target.make(Path("output/prosody_predictor_contrastive5/emotion_input_translate2-0/pred/val_joy_80k.csv"), Sentiment.NEU, "pitch", emotion_mapping, Sentiment.NEU, Sentiment.JOY),
    #         Target.make(Path("output/prosody_predictor_contrastive5/emotion_input_translate2-0/pred/val_joy_80k.csv"), Sentiment.JOY, "pitch", emotion_mapping, Sentiment.JOY, Sentiment.JOY),

    #     ],
    #     output=Path("output/plots/trilateration-eqtri-ols/prosody-predictor_emotion-input-translate2_contrastive5-0_80k.png"),
    #     label="prosody-predictor_emotion-input-translate2_contrastive5-0_80k",
    #     scale=1
    # ),  

        
    # Task(
    #     anchors_shared,
    #     [
    #         Target.make(Path("output/prosody_predictor_contrastive5/emotion_input-0.01/pred/val_neutral_40k.csv"), Sentiment.SAD, "pitch", emotion_mapping, Sentiment.SAD, Sentiment.NEU),
    #         Target.make(Path("output/prosody_predictor_contrastive5/emotion_input-0.01/pred/val_neutral_40k.csv"), Sentiment.NEU, "pitch", emotion_mapping, Sentiment.NEU, Sentiment.NEU),
    #         Target.make(Path("output/prosody_predictor_contrastive5/emotion_input-0.01/pred/val_neutral_40k.csv"), Sentiment.JOY, "pitch", emotion_mapping, Sentiment.JOY, Sentiment.NEU),
    #         Target.make(Path("output/prosody_predictor_contrastive5/emotion_input-0.01/pred/val_sadness_40k.csv"), Sentiment.SAD, "pitch", emotion_mapping, Sentiment.SAD, Sentiment.SAD),
    #         Target.make(Path("output/prosody_predictor_contrastive5/emotion_input-0.01/pred/val_sadness_40k.csv"), Sentiment.NEU, "pitch", emotion_mapping, Sentiment.NEU, Sentiment.SAD),
    #         Target.make(Path("output/prosody_predictor_contrastive5/emotion_input-0.01/pred/val_sadness_40k.csv"), Sentiment.JOY, "pitch", emotion_mapping, Sentiment.JOY, Sentiment.SAD),
    #         Target.make(Path("output/prosody_predictor_contrastive5/emotion_input-0.01/pred/val_joy_40k.csv"), Sentiment.SAD, "pitch", emotion_mapping, Sentiment.SAD, Sentiment.JOY),
    #         Target.make(Path("output/prosody_predictor_contrastive5/emotion_input-0.01/pred/val_joy_40k.csv"), Sentiment.NEU, "pitch", emotion_mapping, Sentiment.NEU, Sentiment.JOY),
    #         Target.make(Path("output/prosody_predictor_contrastive5/emotion_input-0.01/pred/val_joy_40k.csv"), Sentiment.JOY, "pitch", emotion_mapping, Sentiment.JOY, Sentiment.JOY),

    #     ],
    #     output=Path("output/plots/trilateration-eqtri-ols/prosody-predictor_emotion-input_contrastive5-0.01_40k.png"),
    #     label="prosody-predictor_emotion-input_contrastive5-0.01_40k",
    #     scale=0.4
    # ),

    # Task(
    #     anchors_shared,
    #     [
    #         Target.make(Path("output/prosody_predictor_contrastive5/emotion_input-0.01/pred/val_neutral_80k.csv"), Sentiment.SAD, "pitch", emotion_mapping, Sentiment.SAD, Sentiment.NEU),
    #         Target.make(Path("output/prosody_predictor_contrastive5/emotion_input-0.01/pred/val_neutral_80k.csv"), Sentiment.NEU, "pitch", emotion_mapping, Sentiment.NEU, Sentiment.NEU),
    #         Target.make(Path("output/prosody_predictor_contrastive5/emotion_input-0.01/pred/val_neutral_80k.csv"), Sentiment.JOY, "pitch", emotion_mapping, Sentiment.JOY, Sentiment.NEU),
    #         Target.make(Path("output/prosody_predictor_contrastive5/emotion_input-0.01/pred/val_sadness_80k.csv"), Sentiment.SAD, "pitch", emotion_mapping, Sentiment.SAD, Sentiment.SAD),
    #         Target.make(Path("output/prosody_predictor_contrastive5/emotion_input-0.01/pred/val_sadness_80k.csv"), Sentiment.NEU, "pitch", emotion_mapping, Sentiment.NEU, Sentiment.SAD),
    #         Target.make(Path("output/prosody_predictor_contrastive5/emotion_input-0.01/pred/val_sadness_80k.csv"), Sentiment.JOY, "pitch", emotion_mapping, Sentiment.JOY, Sentiment.SAD),
    #         Target.make(Path("output/prosody_predictor_contrastive5/emotion_input-0.01/pred/val_joy_80k.csv"), Sentiment.SAD, "pitch", emotion_mapping, Sentiment.SAD, Sentiment.JOY),
    #         Target.make(Path("output/prosody_predictor_contrastive5/emotion_input-0.01/pred/val_joy_80k.csv"), Sentiment.NEU, "pitch", emotion_mapping, Sentiment.NEU, Sentiment.JOY),
    #         Target.make(Path("output/prosody_predictor_contrastive5/emotion_input-0.01/pred/val_joy_80k.csv"), Sentiment.JOY, "pitch", emotion_mapping, Sentiment.JOY, Sentiment.JOY),

    #     ],
    #     output=Path("output/plots/trilateration-eqtri-ols/prosody-predictor_emotion-input_contrastive5-0.01_80k.png"),
    #     label="prosody-predictor_emotion-input_contrastive5-0.01_80k",
    #     scale=0.4,
    # ),    

    
    # Task(
    #     anchors_shared,
    #     [
    #         Target.make(Path("output/prosody_predictor_contrastive5/emotion_input_translate2-0.01/pred/val_neutral_40k.csv"), Sentiment.SAD, "pitch", emotion_mapping, Sentiment.SAD, Sentiment.NEU),
    #         Target.make(Path("output/prosody_predictor_contrastive5/emotion_input_translate2-0.01/pred/val_neutral_40k.csv"), Sentiment.NEU, "pitch", emotion_mapping, Sentiment.NEU, Sentiment.NEU),
    #         Target.make(Path("output/prosody_predictor_contrastive5/emotion_input_translate2-0.01/pred/val_neutral_40k.csv"), Sentiment.JOY, "pitch", emotion_mapping, Sentiment.JOY, Sentiment.NEU),
    #         Target.make(Path("output/prosody_predictor_contrastive5/emotion_input_translate2-0.01/pred/val_sadness_40k.csv"), Sentiment.SAD, "pitch", emotion_mapping, Sentiment.SAD, Sentiment.SAD),
    #         Target.make(Path("output/prosody_predictor_contrastive5/emotion_input_translate2-0.01/pred/val_sadness_40k.csv"), Sentiment.NEU, "pitch", emotion_mapping, Sentiment.NEU, Sentiment.SAD),
    #         Target.make(Path("output/prosody_predictor_contrastive5/emotion_input_translate2-0.01/pred/val_sadness_40k.csv"), Sentiment.JOY, "pitch", emotion_mapping, Sentiment.JOY, Sentiment.SAD),
    #         Target.make(Path("output/prosody_predictor_contrastive5/emotion_input_translate2-0.01/pred/val_joy_40k.csv"), Sentiment.SAD, "pitch", emotion_mapping, Sentiment.SAD, Sentiment.JOY),
    #         Target.make(Path("output/prosody_predictor_contrastive5/emotion_input_translate2-0.01/pred/val_joy_40k.csv"), Sentiment.NEU, "pitch", emotion_mapping, Sentiment.NEU, Sentiment.JOY),
    #         Target.make(Path("output/prosody_predictor_contrastive5/emotion_input_translate2-0.01/pred/val_joy_40k.csv"), Sentiment.JOY, "pitch", emotion_mapping, Sentiment.JOY, Sentiment.JOY),

    #     ],
    #     output=Path("output/plots/trilateration-eqtri-ols/prosody-predictor_emotion-input-translate2_contrastive5-0.01_40k.png"),
    #     label="prosody-predictor_emotion-input-translate2_contrastive5-0.01_40k",
    #     scale=1
    # ),

    # Task(
    #     anchors_shared,
    #     [
    #         Target.make(Path("output/prosody_predictor_contrastive5/emotion_input_translate2-0.01/pred/val_neutral_80k.csv"), Sentiment.SAD, "pitch", emotion_mapping, Sentiment.SAD, Sentiment.NEU),
    #         Target.make(Path("output/prosody_predictor_contrastive5/emotion_input_translate2-0.01/pred/val_neutral_80k.csv"), Sentiment.NEU, "pitch", emotion_mapping, Sentiment.NEU, Sentiment.NEU),
    #         Target.make(Path("output/prosody_predictor_contrastive5/emotion_input_translate2-0.01/pred/val_neutral_80k.csv"), Sentiment.JOY, "pitch", emotion_mapping, Sentiment.JOY, Sentiment.NEU),
    #         Target.make(Path("output/prosody_predictor_contrastive5/emotion_input_translate2-0.01/pred/val_sadness_80k.csv"), Sentiment.SAD, "pitch", emotion_mapping, Sentiment.SAD, Sentiment.SAD),
    #         Target.make(Path("output/prosody_predictor_contrastive5/emotion_input_translate2-0.01/pred/val_sadness_80k.csv"), Sentiment.NEU, "pitch", emotion_mapping, Sentiment.NEU, Sentiment.SAD),
    #         Target.make(Path("output/prosody_predictor_contrastive5/emotion_input_translate2-0.01/pred/val_sadness_80k.csv"), Sentiment.JOY, "pitch", emotion_mapping, Sentiment.JOY, Sentiment.SAD),
    #         Target.make(Path("output/prosody_predictor_contrastive5/emotion_input_translate2-0.01/pred/val_joy_80k.csv"), Sentiment.SAD, "pitch", emotion_mapping, Sentiment.SAD, Sentiment.JOY),
    #         Target.make(Path("output/prosody_predictor_contrastive5/emotion_input_translate2-0.01/pred/val_joy_80k.csv"), Sentiment.NEU, "pitch", emotion_mapping, Sentiment.NEU, Sentiment.JOY),
    #         Target.make(Path("output/prosody_predictor_contrastive5/emotion_input_translate2-0.01/pred/val_joy_80k.csv"), Sentiment.JOY, "pitch", emotion_mapping, Sentiment.JOY, Sentiment.JOY),

    #     ],
    #     output=Path("output/plots/trilateration-eqtri-ols/prosody-predictor_emotion-input-translate2_contrastive5-0.01_80k.png"),
    #     label="prosody-predictor_emotion-input-translate2_contrastive5-0.01_80k",
    #     scale=1
    # ),    


    # Task(
    #     anchors_shared,
    #     [
    #         Target.make(Path("output/prosody_predictor_contrastive4/emotion_input_translate2-0.01/pred/val_neutral_40k.csv"), Sentiment.SAD, "pitch", emotion_mapping, Sentiment.SAD, Sentiment.NEU),
    #         Target.make(Path("output/prosody_predictor_contrastive4/emotion_input_translate2-0.01/pred/val_neutral_40k.csv"), Sentiment.NEU, "pitch", emotion_mapping, Sentiment.NEU, Sentiment.NEU),
    #         Target.make(Path("output/prosody_predictor_contrastive4/emotion_input_translate2-0.01/pred/val_neutral_40k.csv"), Sentiment.JOY, "pitch", emotion_mapping, Sentiment.JOY, Sentiment.NEU),
    #         Target.make(Path("output/prosody_predictor_contrastive4/emotion_input_translate2-0.01/pred/val_sadness_40k.csv"), Sentiment.SAD, "pitch", emotion_mapping, Sentiment.SAD, Sentiment.SAD),
    #         Target.make(Path("output/prosody_predictor_contrastive4/emotion_input_translate2-0.01/pred/val_sadness_40k.csv"), Sentiment.NEU, "pitch", emotion_mapping, Sentiment.NEU, Sentiment.SAD),
    #         Target.make(Path("output/prosody_predictor_contrastive4/emotion_input_translate2-0.01/pred/val_sadness_40k.csv"), Sentiment.JOY, "pitch", emotion_mapping, Sentiment.JOY, Sentiment.SAD),
    #         Target.make(Path("output/prosody_predictor_contrastive4/emotion_input_translate2-0.01/pred/val_joy_40k.csv"), Sentiment.SAD, "pitch", emotion_mapping, Sentiment.SAD, Sentiment.JOY),
    #         Target.make(Path("output/prosody_predictor_contrastive4/emotion_input_translate2-0.01/pred/val_joy_40k.csv"), Sentiment.NEU, "pitch", emotion_mapping, Sentiment.NEU, Sentiment.JOY),
    #         Target.make(Path("output/prosody_predictor_contrastive4/emotion_input_translate2-0.01/pred/val_joy_40k.csv"), Sentiment.JOY, "pitch", emotion_mapping, Sentiment.JOY, Sentiment.JOY),

    #     ],
    #     output=Path("output/plots/trilateration-eqtri-ols/prosody-predictor_emotion-input-translate2_contrastive4-0.01_40k.png"),
    #     label="prosody-predictor_emotion-input-translate4_contrastive42-0.01_40k",
    # ),

    # Task(
    #     anchors_shared,
    #     [
    #         Target.make(Path("output/prosody_predictor_contrastive4/emotion_input_translate2-0.01/pred/val_neutral_80k.csv"), Sentiment.SAD, "pitch", emotion_mapping, Sentiment.SAD, Sentiment.NEU),
    #         Target.make(Path("output/prosody_predictor_contrastive4/emotion_input_translate2-0.01/pred/val_neutral_80k.csv"), Sentiment.NEU, "pitch", emotion_mapping, Sentiment.NEU, Sentiment.NEU),
    #         Target.make(Path("output/prosody_predictor_contrastive4/emotion_input_translate2-0.01/pred/val_neutral_80k.csv"), Sentiment.JOY, "pitch", emotion_mapping, Sentiment.JOY, Sentiment.NEU),
    #         Target.make(Path("output/prosody_predictor_contrastive4/emotion_input_translate2-0.01/pred/val_sadness_80k.csv"), Sentiment.SAD, "pitch", emotion_mapping, Sentiment.SAD, Sentiment.SAD),
    #         Target.make(Path("output/prosody_predictor_contrastive4/emotion_input_translate2-0.01/pred/val_sadness_80k.csv"), Sentiment.NEU, "pitch", emotion_mapping, Sentiment.NEU, Sentiment.SAD),
    #         Target.make(Path("output/prosody_predictor_contrastive4/emotion_input_translate2-0.01/pred/val_sadness_80k.csv"), Sentiment.JOY, "pitch", emotion_mapping, Sentiment.JOY, Sentiment.SAD),
    #         Target.make(Path("output/prosody_predictor_contrastive4/emotion_input_translate2-0.01/pred/val_joy_80k.csv"), Sentiment.SAD, "pitch", emotion_mapping, Sentiment.SAD, Sentiment.JOY),
    #         Target.make(Path("output/prosody_predictor_contrastive4/emotion_input_translate2-0.01/pred/val_joy_80k.csv"), Sentiment.NEU, "pitch", emotion_mapping, Sentiment.NEU, Sentiment.JOY),
    #         Target.make(Path("output/prosody_predictor_contrastive4/emotion_input_translate2-0.01/pred/val_joy_80k.csv"), Sentiment.JOY, "pitch", emotion_mapping, Sentiment.JOY, Sentiment.JOY),

    #     ],
    #     output=Path("output/plots/trilateration-eqtri-ols/prosody-predictor_emotion-input-translate2_contrastive4-0.01_80k.png"),
    #     label="prosody-predictor_emotion-input0translate4_contrastive2-0.01_80k",
    # ),    

    # Task(
    #     anchors_shared,
    #     [
    #         Target.make(Path("output/prosody_predictor_contrastive3/emotion_input-0.01/pred/val_neutral_40k.csv"), Sentiment.SAD, "pitch", emotion_mapping, Sentiment.SAD, Sentiment.NEU),
    #         Target.make(Path("output/prosody_predictor_contrastive3/emotion_input-0.01/pred/val_neutral_40k.csv"), Sentiment.NEU, "pitch", emotion_mapping, Sentiment.NEU, Sentiment.NEU),
    #         Target.make(Path("output/prosody_predictor_contrastive3/emotion_input-0.01/pred/val_neutral_40k.csv"), Sentiment.JOY, "pitch", emotion_mapping, Sentiment.JOY, Sentiment.NEU),
    #         Target.make(Path("output/prosody_predictor_contrastive3/emotion_input-0.01/pred/val_sadness_40k.csv"), Sentiment.SAD, "pitch", emotion_mapping, Sentiment.SAD, Sentiment.SAD),
    #         Target.make(Path("output/prosody_predictor_contrastive3/emotion_input-0.01/pred/val_sadness_40k.csv"), Sentiment.NEU, "pitch", emotion_mapping, Sentiment.NEU, Sentiment.SAD),
    #         Target.make(Path("output/prosody_predictor_contrastive3/emotion_input-0.01/pred/val_sadness_40k.csv"), Sentiment.JOY, "pitch", emotion_mapping, Sentiment.JOY, Sentiment.SAD),
    #         Target.make(Path("output/prosody_predictor_contrastive3/emotion_input-0.01/pred/val_joy_40k.csv"), Sentiment.SAD, "pitch", emotion_mapping, Sentiment.SAD, Sentiment.JOY),
    #         Target.make(Path("output/prosody_predictor_contrastive3/emotion_input-0.01/pred/val_joy_40k.csv"), Sentiment.NEU, "pitch", emotion_mapping, Sentiment.NEU, Sentiment.JOY),
    #         Target.make(Path("output/prosody_predictor_contrastive3/emotion_input-0.01/pred/val_joy_40k.csv"), Sentiment.JOY, "pitch", emotion_mapping, Sentiment.JOY, Sentiment.JOY),

    #     ],
    #     output=Path("output/plots/trilateration-eqtri-ols/prosody-predictor_emotion-input_contrastive3-0.01_40k.png"),
    #     label="prosody-predictor_emotion-input_contrastive3-0.01_40k",
    # ),

    # Task(
    #     anchors_shared,
    #     [
    #         Target.make(Path("output/prosody_predictor_contrastive3/emotion_input-0.01/pred/val_neutral_80k.csv"), Sentiment.SAD, "pitch", emotion_mapping, Sentiment.SAD, Sentiment.NEU),
    #         Target.make(Path("output/prosody_predictor_contrastive3/emotion_input-0.01/pred/val_neutral_80k.csv"), Sentiment.NEU, "pitch", emotion_mapping, Sentiment.NEU, Sentiment.NEU),
    #         Target.make(Path("output/prosody_predictor_contrastive3/emotion_input-0.01/pred/val_neutral_80k.csv"), Sentiment.JOY, "pitch", emotion_mapping, Sentiment.JOY, Sentiment.NEU),
    #         Target.make(Path("output/prosody_predictor_contrastive3/emotion_input-0.01/pred/val_sadness_80k.csv"), Sentiment.SAD, "pitch", emotion_mapping, Sentiment.SAD, Sentiment.SAD),
    #         Target.make(Path("output/prosody_predictor_contrastive3/emotion_input-0.01/pred/val_sadness_80k.csv"), Sentiment.NEU, "pitch", emotion_mapping, Sentiment.NEU, Sentiment.SAD),
    #         Target.make(Path("output/prosody_predictor_contrastive3/emotion_input-0.01/pred/val_sadness_80k.csv"), Sentiment.JOY, "pitch", emotion_mapping, Sentiment.JOY, Sentiment.SAD),
    #         Target.make(Path("output/prosody_predictor_contrastive3/emotion_input-0.01/pred/val_joy_80k.csv"), Sentiment.SAD, "pitch", emotion_mapping, Sentiment.SAD, Sentiment.JOY),
    #         Target.make(Path("output/prosody_predictor_contrastive3/emotion_input-0.01/pred/val_joy_80k.csv"), Sentiment.NEU, "pitch", emotion_mapping, Sentiment.NEU, Sentiment.JOY),
    #         Target.make(Path("output/prosody_predictor_contrastive3/emotion_input-0.01/pred/val_joy_80k.csv"), Sentiment.JOY, "pitch", emotion_mapping, Sentiment.JOY, Sentiment.JOY),

    #     ],
    #     output=Path("output/plots/trilateration-eqtri-ols/prosody-predictor_emotion-input_contrastive3-0.01_80k.png"),
    #     label="prosody-predictor_emotion-input_contrastive3-0.01_80k",
    # ),    



    # Task(
    #     anchors_shared,
    #     [
    #         Target.make(Path("output/prosody_predictor_contrastive2/emotion_input_translate2-0.1/pred/val_neutral.csv"), Sentiment.SAD, "pitch", emotion_mapping, Sentiment.SAD, Sentiment.NEU),
    #         Target.make(Path("output/prosody_predictor_contrastive2/emotion_input_translate2-0.1/pred/val_neutral.csv"), Sentiment.NEU, "pitch", emotion_mapping, Sentiment.NEU, Sentiment.NEU),
    #         Target.make(Path("output/prosody_predictor_contrastive2/emotion_input_translate2-0.1/pred/val_neutral.csv"), Sentiment.JOY, "pitch", emotion_mapping, Sentiment.JOY, Sentiment.NEU),
    #         Target.make(Path("output/prosody_predictor_contrastive2/emotion_input_translate2-0.1/pred/val_sadness.csv"), Sentiment.SAD, "pitch", emotion_mapping, Sentiment.SAD, Sentiment.SAD),
    #         Target.make(Path("output/prosody_predictor_contrastive2/emotion_input_translate2-0.1/pred/val_sadness.csv"), Sentiment.NEU, "pitch", emotion_mapping, Sentiment.NEU, Sentiment.SAD),
    #         Target.make(Path("output/prosody_predictor_contrastive2/emotion_input_translate2-0.1/pred/val_sadness.csv"), Sentiment.JOY, "pitch", emotion_mapping, Sentiment.JOY, Sentiment.SAD),
    #         Target.make(Path("output/prosody_predictor_contrastive2/emotion_input_translate2-0.1/pred/val_joy.csv"), Sentiment.SAD, "pitch", emotion_mapping, Sentiment.SAD, Sentiment.JOY),
    #         Target.make(Path("output/prosody_predictor_contrastive2/emotion_input_translate2-0.1/pred/val_joy.csv"), Sentiment.NEU, "pitch", emotion_mapping, Sentiment.NEU, Sentiment.JOY),
    #         Target.make(Path("output/prosody_predictor_contrastive2/emotion_input_translate2-0.1/pred/val_joy.csv"), Sentiment.JOY, "pitch", emotion_mapping, Sentiment.JOY, Sentiment.JOY),

    #     ],
    #     output=Path("output/plots/trilateration-eqtri-ols/prosody-predictor_emotion-input-translate2_contrastive2-0.1.png"),
    #     label="prosody-predictor_emotion-input-translate2_contrastive2-0.1",
    # ),
    # Task(
    #     anchors_shared,
    #     [
    #         Target.make(Path("output/prosody_predictor_contrastive2/emotion_input_translate2-0.1/pred/val_neutral_20k.csv"), Sentiment.SAD, "pitch", emotion_mapping, Sentiment.SAD, Sentiment.NEU),
    #         Target.make(Path("output/prosody_predictor_contrastive2/emotion_input_translate2-0.1/pred/val_neutral_20k.csv"), Sentiment.NEU, "pitch", emotion_mapping, Sentiment.NEU, Sentiment.NEU),
    #         Target.make(Path("output/prosody_predictor_contrastive2/emotion_input_translate2-0.1/pred/val_neutral_20k.csv"), Sentiment.JOY, "pitch", emotion_mapping, Sentiment.JOY, Sentiment.NEU),
    #         Target.make(Path("output/prosody_predictor_contrastive2/emotion_input_translate2-0.1/pred/val_sadness_20k.csv"), Sentiment.SAD, "pitch", emotion_mapping, Sentiment.SAD, Sentiment.SAD),
    #         Target.make(Path("output/prosody_predictor_contrastive2/emotion_input_translate2-0.1/pred/val_sadness_20k.csv"), Sentiment.NEU, "pitch", emotion_mapping, Sentiment.NEU, Sentiment.SAD),
    #         Target.make(Path("output/prosody_predictor_contrastive2/emotion_input_translate2-0.1/pred/val_sadness_20k.csv"), Sentiment.JOY, "pitch", emotion_mapping, Sentiment.JOY, Sentiment.SAD),
    #         Target.make(Path("output/prosody_predictor_contrastive2/emotion_input_translate2-0.1/pred/val_joy_20k.csv"), Sentiment.SAD, "pitch", emotion_mapping, Sentiment.SAD, Sentiment.JOY),
    #         Target.make(Path("output/prosody_predictor_contrastive2/emotion_input_translate2-0.1/pred/val_joy_20k.csv"), Sentiment.NEU, "pitch", emotion_mapping, Sentiment.NEU, Sentiment.JOY),
    #         Target.make(Path("output/prosody_predictor_contrastive2/emotion_input_translate2-0.1/pred/val_joy_20k.csv"), Sentiment.JOY, "pitch", emotion_mapping, Sentiment.JOY, Sentiment.JOY),

    #     ],
    #     output=Path("output/plots/trilateration-eqtri-ols/prosody-predictor_emotion-input-translate2-20k_contrastive2-0.1.png"),
    #     label="prosody-predictor_emotion-input-translate2-20k_contrastive2-0.1",
    # ),
    # Task(
    #     anchors_shared,
    #     [
    #         Target.make(Path("output/prosody_predictor_contrastive2/emotion_input_translate2-0.1/pred/val_neutral_80k.csv"), Sentiment.SAD, "pitch", emotion_mapping, Sentiment.SAD, Sentiment.NEU),
    #         Target.make(Path("output/prosody_predictor_contrastive2/emotion_input_translate2-0.1/pred/val_neutral_80k.csv"), Sentiment.NEU, "pitch", emotion_mapping, Sentiment.NEU, Sentiment.NEU),
    #         Target.make(Path("output/prosody_predictor_contrastive2/emotion_input_translate2-0.1/pred/val_neutral_80k.csv"), Sentiment.JOY, "pitch", emotion_mapping, Sentiment.JOY, Sentiment.NEU),
    #         Target.make(Path("output/prosody_predictor_contrastive2/emotion_input_translate2-0.1/pred/val_sadness_80k.csv"), Sentiment.SAD, "pitch", emotion_mapping, Sentiment.SAD, Sentiment.SAD),
    #         Target.make(Path("output/prosody_predictor_contrastive2/emotion_input_translate2-0.1/pred/val_sadness_80k.csv"), Sentiment.NEU, "pitch", emotion_mapping, Sentiment.NEU, Sentiment.SAD),
    #         Target.make(Path("output/prosody_predictor_contrastive2/emotion_input_translate2-0.1/pred/val_sadness_80k.csv"), Sentiment.JOY, "pitch", emotion_mapping, Sentiment.JOY, Sentiment.SAD),
    #         Target.make(Path("output/prosody_predictor_contrastive2/emotion_input_translate2-0.1/pred/val_joy_80k.csv"), Sentiment.SAD, "pitch", emotion_mapping, Sentiment.SAD, Sentiment.JOY),
    #         Target.make(Path("output/prosody_predictor_contrastive2/emotion_input_translate2-0.1/pred/val_joy_80k.csv"), Sentiment.NEU, "pitch", emotion_mapping, Sentiment.NEU, Sentiment.JOY),
    #         Target.make(Path("output/prosody_predictor_contrastive2/emotion_input_translate2-0.1/pred/val_joy_80k.csv"), Sentiment.JOY, "pitch", emotion_mapping, Sentiment.JOY, Sentiment.JOY),

    #     ],
    #     output=Path("output/plots/trilateration-eqtri-ols/prosody-predictor_emotion-input-translate2-80k_contrastive2-0.1.png"),
    #     label="prosody-predictor_emotion-input-translate2-80k_contrastive2-0.1",
    # ),

]

# =========================
# Geometry with sentiment mapping
# =========================

def triangle_vertices_by_sentiment(scale: float = 0.4) -> dict[Sentiment, np.ndarray]:
    """Return vertex coordinates keyed by Sentiment:
       NEG -> bottom-left, NEU -> bottom-right, POS -> top."""
    verts = {
        # Sentiment.NEG: np.array([-0.5, 0.0], dtype=float),               # bottom-left
        # Sentiment.NEU: np.array([ 0.5, 0.0], dtype=float),               # bottom-right
        # Sentiment.POS: np.array([ 0.0, np.sqrt(3) / 2.0], dtype=float),  # top
        Sentiment.JOY: np.array([ 0.0, 1.0 ], dtype=float),  # top
        Sentiment.NEU: np.array([ np.sqrt(3) / 2.0, 0.5 ], dtype=float),               # bottom-right
        Sentiment.SAD: np.array([ 0.0, 0.0 ], dtype=float),               # bottom-left
    }
    for s in verts:
        verts[s] *= scale
    return verts

def make_trilaterator_from_task(
    anchors: Sequence[Anchor],
    vertices_by_emot: Mapping[Sentiment, np.ndarray],
) -> Tuple[Callable[[float, float, float], np.ndarray], List[Sentiment], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Use Task.anchors order to define (A,B,C). Returns (trilaterate, order_sents, (A,B,C)).
    """
    if len(anchors) != 3:
        raise ValueError(f"Expected exactly 3 anchors, got {len(anchors)}")

    order_emots: List[Sentiment] = [a.sentiment for a in anchors]

    # Optional safety checks
    if len(set(order_emots)) < 3:
        print(f"Warning: duplicate sentiments in anchors order: {[SENT_NAMES[s] for s in order_emots]}")

    A = vertices_by_emot[order_emots[0]]
    B = vertices_by_emot[order_emots[1]]
    C = vertices_by_emot[order_emots[2]]

    # Build trilaterator for these exact A,B,C
    BA = B - A
    CA = C - A
    M = np.vstack([BA, CA])  # 2x2
    ATA = float(A @ A)
    BTB = float(B @ B)
    CTC = float(C @ C)

    def trilaterate(dist_a: float, dist_b: float, dist_c: float) -> np.ndarray:
        if not (np.isfinite(dist_a) and np.isfinite(dist_b) and np.isfinite(dist_c)):
            return np.array([np.nan, np.nan], dtype=float)
        b1 = (BTB - ATA + dist_a**2 - dist_b**2) / 2.0
        b2 = (CTC - ATA + dist_a**2 - dist_c**2) / 2.0
        p, *_ = np.linalg.lstsq(M, np.array([b1, b2]), rcond=None)
        return p.astype(float)

    return trilaterate, order_emots, (A, B, C)


# =========================
# Plot style mappings
# =========================
FillStyle = Literal["full", "left", "right", "bottom", "top", "none"]

orig_to_marker: Dict[Sentiment, str] = {SAD: "v", NEU: "o", JOY: "^"}
target_to_fill: Dict[Sentiment, FillStyle] = {SAD: "bottom", NEU: "full", JOY: "top"}

MARKER_AREA = 120.0
MARKER_SIZE = float(math.sqrt(MARKER_AREA))
EDGE_WIDTH = 1.2
ALPHA = 0.9

# ANCHOR_MARKER = "D"
# ANCHOR_SIZE = MARKER_SIZE * 1.1
ANCHOR_EDGEW = 1.4


# Wash-out style for anchors (distinct from targets)
ANCHOR_ALPHA       = 1.0          # fainter than targets
ANCHOR_EDGE_COLOR  = '0.45'       # gray edge
ANCHOR_FACE_COLOR  = '0.70'       # gray fill (used when fillstyle != 'none')
ANCHOR_FACE_ALT    = '1.0'        # white for the other half in half-fill styles

BORDER_ZORDER      = 0            # draw behind everything
ANCHOR_ZORDER      = 1            # draw behind targets
TARGET_ZORDER      = 2            # draw targets above anchors

# Let anchors follow original sentiment shape instead of diamond
# (keep orig_to_marker as-is; we’ll use it for anchors too)
# Example:
# orig_to_marker = {NEG: 'v', NEU: 'o', POS: '^'}

# Optional: size tweak to subtly de-emphasize anchors
ANCHOR_SIZE = MARKER_SIZE * 0.95

def compute_wasserstein_distances_in_order(
    anchors: Sequence[Anchor],
    targets: Sequence[Target],
) -> Dict[str, List[float]]:
    """
    Returns:
      dist_ordered[target_label] = [d_to_anchor0, d_to_anchor1, d_to_anchor2]
    where the order matches anchors as given in the task.
    """
    # preload anchor samples in task order
    anchor_samps_ordered: List[np.ndarray] = []
    for a in anchors:
        samps = a.distribution.load_samples()
        if samps.size == 0:
            print(f"Error: Anchor distribution for {SENT_NAMES[a.sentiment]} is empty.")
        anchor_samps_ordered.append(samps)

    dist_ordered: Dict[str, List[float]] = {}

    for t in targets:
        t_samps = t.distribution.load_samples()
        if t_samps.size == 0:
            print(f"Warning: Target '{t.label}' has empty distribution. It will be skipped.")
        d_list: List[float] = []
        for a_samps in anchor_samps_ordered:
            d = wasserstein_distance(a_samps, t_samps) if t_samps.size else float("nan")
            d_list.append(float(d))
        dist_ordered[t.label] = d_list

    return dist_ordered

def embedd_targets_from_order(
    dist_ordered: Mapping[str, Sequence[float]],
    trilaterate: Callable[[float, float, float], np.ndarray],
) -> Dict[str, np.ndarray]:
    coords: Dict[str, np.ndarray] = {}
    for t_label, dvec in dist_ordered.items():
        if len(dvec) != 3:
            print(f"Error: expected 3 distances for '{t_label}', got {len(dvec)}")
            coords[t_label] = np.array([np.nan, np.nan], dtype=float)
            continue
        coords[t_label] = trilaterate(dvec[0], dvec[1], dvec[2])  # exact same order as anchors
    return coords

def plot_embedding(
    *,
    vertices_by_sent: Mapping[Sentiment, np.ndarray],  # from triangle_vertices_by_sentiment
    anchors: Sequence[Anchor],
    targets: Sequence[Target],
    target_coords: Mapping[str, np.ndarray],
    out_path: Path,
    title: str = "",
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))

    A = vertices_by_sent[Sentiment.SAD]
    B = vertices_by_sent[Sentiment.NEU]
    C = vertices_by_sent[Sentiment.JOY]

    # triangle border
    ax.plot([A[0], B[0], C[0], A[0]], [A[1], B[1], C[1], A[1]],
            linestyle="--", linewidth=1.0, color="0.9", zorder=BORDER_ZORDER)

    # anchors at their sentiment-defined vertex
    # anchors (by their sentiment)
    # anchors (washed out; shape follows original sentiment; fill matches target labeling)
    for anc in anchors:
        xy = vertices_by_sent[anc.sentiment]   # or coords_map[...] if you use that API
        marker = orig_to_marker[anc.sentiment] # shape = original sentiment
        fill   = target_to_fill[anc.sentiment] # fill = target labeling style (NEG full, NEU none, POS bottom)

        # washed-out colors
        if fill == 'none':
            mfc  = 'none'
            mfc2 = ANCHOR_FACE_ALT
        else:
            mfc  = ANCHOR_FACE_COLOR
            mfc2 = ANCHOR_FACE_ALT

        ax.plot([xy[0]], [xy[1]],
                linestyle='None',
                marker=marker,
                markersize=ANCHOR_SIZE,
                markerfacecolor=mfc,
                markerfacecoloralt=mfc2,
                markeredgecolor=ANCHOR_EDGE_COLOR,
                markeredgewidth=ANCHOR_EDGEW,
                fillstyle=fill,
                alpha=ANCHOR_ALPHA,
                zorder=ANCHOR_ZORDER)

    # targets
    plotted_any = False
    for t in targets:
        xy = target_coords.get(t.label, np.array([np.nan, np.nan]))
        if not np.all(np.isfinite(xy)):
            continue
        marker = orig_to_marker.get(t.orig_sentiment, "s")
        fill   = target_to_fill.get(t.target_sentiment, "none")
        mfc = "black" if fill != "none" else "none"
        ax.plot([xy[0]], [xy[1]],
            linestyle='None',
            marker=marker,
            markersize=MARKER_SIZE,
            markerfacecolor=mfc,
            markerfacecoloralt='white',
            markeredgecolor='black',
            markeredgewidth=EDGE_WIDTH,
            fillstyle=fill,
            alpha=ALPHA,
            zorder=TARGET_ZORDER)
        plotted_any = True

    if not plotted_any:
        print(f"Warning: no valid target points for plot '{title}'. Skipping save to {out_path}.")
        plt.close(fig); return

    # legends (unchanged)
    shape_handles = [
        Line2D([0],[0], marker=orig_to_marker[Sentiment.JOY], linestyle="None",
               markersize=MARKER_SIZE*0.85, markerfacecolor="none",
               markeredgecolor="black", markeredgewidth=EDGE_WIDTH, label="JOY"),
        Line2D([0],[0], marker=orig_to_marker[Sentiment.NEU], linestyle="None",
               markersize=MARKER_SIZE*0.85, markerfacecolor="none",
               markeredgecolor="black", markeredgewidth=EDGE_WIDTH, label="NEU"),
        Line2D([0],[0], marker=orig_to_marker[Sentiment.SAD], linestyle="None",
               markersize=MARKER_SIZE*0.85, markerfacecolor="none",
               markeredgecolor="black", markeredgewidth=EDGE_WIDTH, label="SAD"),
    ]
    anchor_fill_handles = [
        Line2D([0], [0],
            marker=orig_to_marker[Sentiment.JOY], linestyle='None',
            markersize=ANCHOR_SIZE * 0.85,
            markerfacecolor=ANCHOR_FACE_COLOR,
            markerfacecoloralt=ANCHOR_FACE_ALT,
            markeredgecolor=ANCHOR_EDGE_COLOR, markeredgewidth=ANCHOR_EDGEW,
            fillstyle=target_to_fill[Sentiment.JOY],
            alpha=ANCHOR_ALPHA,
            label='GT_JOY'),
        Line2D([0], [0],
            marker=orig_to_marker[Sentiment.NEU], linestyle='None',
            markersize=ANCHOR_SIZE * 0.85,
            markerfacecolor='none',
            markerfacecoloralt=ANCHOR_FACE_ALT,
            markeredgecolor=ANCHOR_EDGE_COLOR, markeredgewidth=ANCHOR_EDGEW,
            fillstyle=target_to_fill[Sentiment.NEU],
            alpha=ANCHOR_ALPHA,
            label='GT_NEU'),
        Line2D([0], [0],
            marker=orig_to_marker[Sentiment.SAD], linestyle='None',
            markersize=ANCHOR_SIZE * 0.85,
            markerfacecolor=ANCHOR_FACE_COLOR,
            markerfacecoloralt=ANCHOR_FACE_ALT,
            markeredgecolor=ANCHOR_EDGE_COLOR, markeredgewidth=ANCHOR_EDGEW,
            fillstyle=target_to_fill[Sentiment.SAD],
            alpha=ANCHOR_ALPHA,
            label='GT_SAD'),
    ]
    fill_handles = [
        Line2D([0],[0], marker="o", linestyle="None", markersize=MARKER_SIZE*0.85,
               markerfacecolor="black", markerfacecoloralt="white",
               markeredgecolor="black", markeredgewidth=EDGE_WIDTH,
               fillstyle=target_to_fill[Sentiment.JOY], label="TGT_JOY"),
        Line2D([0],[0], marker="o", linestyle="None", markersize=MARKER_SIZE*0.85,
               markerfacecolor="none", markerfacecoloralt="white",
               markeredgecolor="black", markeredgewidth=EDGE_WIDTH,
               fillstyle=target_to_fill[Sentiment.NEU], label="TGT_NEU"),
        Line2D([0],[0], marker="o", linestyle="None", markersize=MARKER_SIZE*0.85,
               markerfacecolor="black", markerfacecoloralt="white",
               markeredgecolor="black", markeredgewidth=EDGE_WIDTH,
               fillstyle=target_to_fill[Sentiment.SAD], label="TGT_SAD"),
    ]

    legend_shape = ax.legend(handles=shape_handles, title="Original sentiment (shape)",
                             loc="upper right", bbox_to_anchor=(0.98, 0.98),
                             borderaxespad=0.0, fontsize=8, frameon=True)
    ax.add_artist(legend_shape)

    legend_fill = ax.legend(handles=fill_handles, title="Target sentiment (fill)",
                            loc="upper right", bbox_to_anchor=(0.98, 0.80),
                            borderaxespad=0.0, fontsize=8, frameon=True)
    ax.add_artist(legend_fill)

    legend_anchor = ax.legend(handles=anchor_fill_handles, title="Anchor sentiment",
                              loc="lower right", bbox_to_anchor=(0.98, 0.02),
                              borderaxespad=0.0, fontsize=8, frameon=True)
    ax.add_artist(legend_anchor)

    ax.set_aspect("equal", adjustable="box")
    if title:
        fig.suptitle(title, fontsize=10)
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# =========================
# Main
# =========================
def main() -> None:


    for task in tasks:
        # vertices keyed by sentiment
        verts = triangle_vertices_by_sentiment(scale=task.scale)

        trilaterate, order_emots, (A, B, C) = make_trilaterator_from_task(task.anchors, verts)

        dist_ordered = compute_wasserstein_distances_in_order(task.anchors, task.targets)

        coords = embedd_targets_from_order(dist_ordered, trilaterate)

        plot_embedding(
            vertices_by_sent=verts,
            anchors=task.anchors,
            targets=task.targets,
            target_coords=coords,
            out_path=task.output,
            title=task.label,
        )

    print("Done.")


if __name__ == "__main__":
    main()
