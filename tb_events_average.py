import os
from tensorboard.backend.event_processing import event_accumulator

INPUT_PATHS = [
    "output/prosody_predictor-1",
    "output/prosody_predictor-2",
    "output/prosody_predictor-3",
]

OUTPUT_PATH = "output/prosody_predictor_averaged"

def scan_for_summary_dirs(path: str) -> list:
    dirs = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.startswith("events.out.tfevents"):
                rel_path = os.path.relpath(root, path)
                dirs.append(rel_path)
                break
    return dirs

def average_scalar_events_across_runs(paths: list[str]) -> dict[str, list[tuple[int, float]]]:
    eas = []
    for path in paths:
        ea = event_accumulator.EventAccumulator(path)
        ea.Reload()
        eas.append(ea)
    
    tags = set()
    for ea in eas:
        tags.update(ea.Tags()["scalars"])

    outout_dict = {}
    for tag in tags:
        steps: list[int] = []
        values: list[float] = []
        for ea in eas:
            if tag in ea.Tags()["scalars"]:
                events = ea.Scalars(tag)
                steps.extend(event.step for event in events)
                values.extend(event.value for event in events)

        if not steps:
            continue

        # Average values at each step
        step_value_map: dict[int, list[float]] = {}
        for step, value in zip(steps, values):
            if step not in step_value_map:
                step_value_map[step] = []
            step_value_map[step].append(value)

        averaged_value_map = {step: sum(values) / len(values) for step, values in step_value_map.items()}

        steps_sorted = sorted(averaged_value_map.keys())

        step_averaged_value_grouped = [(step, averaged_value_map[step]) for step in steps_sorted]

        outout_dict[tag] = step_averaged_value_grouped

    return outout_dict
        
def main():
    
    summary_rel_dirs = set()
    for input_path in INPUT_PATHS:
        summary_rel_dirs.update(scan_for_summary_dirs(input_path))

    summary_rel_dirs = sorted(summary_rel_dirs)

    print(f"Found {len(summary_rel_dirs)} summary directories:")
    print("   " + "\n   ".join(summary_rel_dirs))

    for summary_rel_dir in summary_rel_dirs:
        print(f"Processing: {summary_rel_dir}")
        averaged_scalar_stats = average_scalar_events_across_runs([os.path.join(input_path, summary_rel_dir) for input_path in INPUT_PATHS])

        print(f"    Found {len(averaged_scalar_stats)} tags with averaged values across runs.")
        for tag, values in averaged_scalar_stats.items():
            print(f"    Tag: {tag}, Values: {values[:5]}... Total steps: {len(values)}")

        output_path = os.path.join(OUTPUT_PATH, summary_rel_dir)
        os.makedirs(output_path, exist_ok=True)
        print(f"    Writing averaged scalar stats to: {output_path}")

        from torch.utils.tensorboard.writer import SummaryWriter
        summary_writer = SummaryWriter(output_path)
        for tag, value_pairs in averaged_scalar_stats.items():
            for step, value in value_pairs:
                summary_writer.add_scalar(tag, value, step)
        summary_writer.close()

        print(f"    Finished processing: {summary_rel_dir}")
    
    print("All summaries processed.")

if __name__ == "__main__":
    main()
