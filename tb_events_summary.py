import csv
import os
from tensorboard.backend.event_processing import event_accumulator

CONFIG_NAMES = [
    "default",
    "sentiment_input",
    "sentiment_input_override_all_neu",
    "sentiment_after_encoder",
    "sentiment_after_encoder_override_all_neu",
    "sentiment_before_prosodic_predictors",
    "sentiment_before_prosodic_predictors_override_all_neu",
]

INPUT_PATHS = [
    f"output/prosody_predictor_averaged/{config_name}/log/val" for config_name in CONFIG_NAMES
]

METRIC_LABELS = [
    "Loss/duration_loss",
    "Loss/energy_loss",
    "Loss/pitch_loss",
    "Loss/total_loss",
]

OUTPUT_PATH = "output/prosody_predictor_averaged/val_loss_summary.csv"


# def scan_for_summary_dirs(path: str) -> list:
#     dirs = []
#     for root, _, files in os.walk(path):
#         for file in files:
#             if file.startswith("events.out.tfevents"):
#                 rel_path = os.path.relpath(root, path)
#                 dirs.append(rel_path)
#                 break
#     return dirs

# def average_scalar_events_across_runs(paths: list[str]) -> dict[str, list[tuple[int, float]]]:
#     eas = []
#     for path in paths:
#         ea = event_accumulator.EventAccumulator(path)
#         ea.Reload()
#         eas.append(ea)
    
#     tags = set()
#     for ea in eas:
#         tags.update(ea.Tags()["scalars"])

#     outout_dict = {}
#     for tag in tags:
#         steps: list[int] = []
#         values: list[float] = []
#         for ea in eas:
#             if tag in ea.Tags()["scalars"]:
#                 events = ea.Scalars(tag)
#                 steps.extend(event.step for event in events)
#                 values.extend(event.value for event in events)

#         if not steps:
#             continue

#         # Average values at each step
#         step_value_map: dict[int, list[float]] = {}
#         for step, value in zip(steps, values):
#             if step not in step_value_map:
#                 step_value_map[step] = []
#             step_value_map[step].append(value)

#         averaged_value_map = {step: sum(values) / len(values) for step, values in step_value_map.items()}

#         steps_sorted = sorted(averaged_value_map.keys())

#         step_averaged_value_grouped = [(step, averaged_value_map[step]) for step in steps_sorted]

#         outout_dict[tag] = step_averaged_value_grouped

#     return outout_dict


        
def main():

    with open(OUTPUT_PATH, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Config Name"] + METRIC_LABELS)
    
        for path, config_name in zip(INPUT_PATHS, CONFIG_NAMES):
            ea = event_accumulator.EventAccumulator(path).Reload()

            tags = ea.Tags()["scalars"]

            print(f"Processing {path} for config '{config_name}'...")
            
            row = []

            for tag in METRIC_LABELS:
                if tag not in tags:
                    print(f"Tag '{tag}' not found in {path}.")
                    row.append(None)
                    continue
                
                events = ea.Scalars(tag)
                if not events:
                    print(f"No events found for tag '{tag}' in {path}.")
                    row.append(None)
                    continue
                
                min_top_values = sorted(event.value for event in events)[:3]

                best_value_smoothed = sum(min_top_values) / len(min_top_values)
                print(f"Minimum value for '{tag}' in {config_name}: {best_value_smoothed:.6f}")
                row.append(f"{best_value_smoothed:.4f}")
            
            writer.writerow([config_name] + row)

    print("All summaries processed.")

if __name__ == "__main__":
    main()
