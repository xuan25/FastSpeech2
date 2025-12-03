import argparse

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument(
    "--anchor_loc_file", type=str, help="CSV file containing anchor locations. e.g. output/dataset/label/model_gt/gt/loc/feature_anchor_loc_method.csv"
)
arg_parser.add_argument(
    "--target_loc_dir", type=str, help="Directory containing target location CSV files. e.g. output/dataset/label/model/variant/loss/loc/split/feature/method"
)
arg_parser.add_argument(
    "--output_file", type=str, help="Path to save the location plot e.g. output/dataset/label/model/variant/loss/vis/split/feature/method.png"
)
arg_parser.add_argument(
    "--tag", type=str, help="Tag for the plot title e.g. dataset_label_model_variant_loss_split_feature_method"
)
arg_parser.add_argument(
    "--plot_margin_ratio_anchor", type=float, default=0.2, help="Margin ratio around anchor points for plot axes"
)
arg_parser.add_argument(
    "--val_fit_step", type=int, default=None, help="Validation fit step to mark on the plot"
)
args = arg_parser.parse_args()



# SOURCE_TARGET = ("happy", "happy")
# SOURCE_TARGET = ("sad", "sad")
# SOURCE_TARGET = ("confused", "confused")
SOURCE_TARGETS = [
    # (("whisper", "whisper"), "r"),
    # (("confused", "confused"), "g"),
    # (("sad", "sad"), "b"),
    # (("happy", "happy"), "c"),

    (("confused", "confused"), "#E74C3C"),      # bright red
    (("default", "default"), "#3498DB"),        # bright blue
    (("enunciated", "enunciated"), "#2ECC71"),  # bright green
    (("happy", "happy"), "#F39C12"),            # bright orange
    (("laughing", "laughing"), "#9B59B6"),      # bright purple
    (("sad", "sad"), "#9EBC1A"),                # bright teal
    (("whisper", "whisper"), "#E91E63"),        # bright pink

    (("confused", "gt"), "#C0392B"),            # dark red
    (("default", "gt"), "#2980B9"),             # dark blue
    (("enunciated", "gt"), "#27AE60"),          # dark green
    (("happy", "gt"), "#D68910"),               # dark orange
    (("laughing", "gt"), "#8E44AD"),            # dark purple
    (("sad", "gt"), "#92A016"),                 # dark teal
    (("whisper", "gt"), "#AD1457"),             # dark pink
]

ANCHOR_ZOOM = 1

import os
import csv
import matplotlib.pyplot as plt
import numpy as np
import tqdm

plt.figure(figsize=(8, 8))

anchor_xs = []
anchor_ys = []
anchor_labels = []

# plot anchor loc
with open(args.anchor_loc_file, 'r', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        label_name = row['label']
        x, y = float(row['x']), float(row['y'])
        anchor_xs.append(x * ANCHOR_ZOOM)
        anchor_ys.append(y * ANCHOR_ZOOM)
        anchor_labels.append(label_name)

plt.scatter(anchor_xs, anchor_ys, c='red', label='Anchors')

# we constaint the axis of the plot around these anchors
x_range = (np.min(anchor_xs), np.max(anchor_xs))
x_margin = (x_range[1] - x_range[0]) * args.plot_margin_ratio_anchor
x_range = (x_range[0] - x_margin, x_range[1] + x_margin)
y_range = (np.min(anchor_ys), np.max(anchor_ys))
y_margin = (y_range[1] - y_range[0]) * args.plot_margin_ratio_anchor
y_range = (y_range[0] - y_margin, y_range[1] + y_margin)

plt.xlim(x_range)
plt.ylim(y_range)


for i, label in enumerate(anchor_labels):
    plt.annotate(label, (anchor_xs[i], anchor_ys[i]), textcoords="offset points", xytext=(0,10), ha='center', color='red')

# plot target locs as a line
tgt_loc_files = os.listdir(args.target_loc_dir)
# sort by number in filename
tgt_loc_files = sorted(tgt_loc_files, key=lambda x: int(x.split('.')[0].split('_')[-1]))

for SOURCE_TARGET, color in SOURCE_TARGETS:

    tgt_xs = []
    tgt_ys = []
    for tgt_loc_file in tgt_loc_files:
        with open(os.path.join(args.target_loc_dir, tgt_loc_file), 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if (row['source_label'], row['target_label']) != SOURCE_TARGET:
                    continue
                x, y = float(row['x']), float(row['y'])
                tgt_xs.append(x)
                tgt_ys.append(y)
                break  # only one entry per file for the given source-target pair
    # plot line, no markers
    plt.plot(tgt_xs, tgt_ys, c=color, alpha=0.2)

    # annotate last point with a x mark
    if tgt_xs and tgt_ys:
        plt.scatter(tgt_xs[-1], tgt_ys[-1], c=color, marker='x', label=f'{"-".join(SOURCE_TARGET)}')

        # annotate validation low point with a o mark
        if args.val_fit_step:
            plt.scatter(tgt_xs[args.val_fit_step - 1], tgt_ys[args.val_fit_step - 1], c=color, marker='o', label=f'{"-".join(SOURCE_TARGET)} val low')
    




plt.legend()


# plt.title('Location Plot for Source-Target: ' + ' to '.join(SOURCE_TARGET))

plt.title(f"Location Plot - {args.tag}")
plt.xlabel('X')
plt.ylabel('Y')
plt.grid()
os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

# plt.savefig(os.path.join(OUTPUT_DIR, f'loc_plot_{"_".join(SOURCE_TARGET)}.png'))
plt.savefig(args.output_file)
plt.close()