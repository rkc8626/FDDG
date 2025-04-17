from tensorboard.backend.event_processing import event_accumulator
import os

# Create event accumulator
ea = event_accumulator.EventAccumulator("/home/chenz1/toorange/FDDG/fddg/train_output/logs/tensorboard/events.out.tfevents.1744737378.c0309a-s25.ufhpc")
ea.Reload()

# Open file for writing
output_path = os.path.join(os.path.dirname(__file__), "tensorboard_results.txt")
with open(output_path, "w") as f:
    # Iterate through all scalar tags
    for tag in ea.Tags()['scalars']:
        f.write(f"== {tag} ==\n")
        for event in ea.Scalars(tag):
            f.write(f"Step: {event.step}, Value: {event.value}\n")
        f.write("\n")  # Add blank line between different metrics

print(f"Results have been written to: {output_path}")