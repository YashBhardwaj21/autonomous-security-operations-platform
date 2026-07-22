from collections import Counter, defaultdict
from pathlib import Path
from src.graph.dataset import ActivityGraphDataset

dataset_path = Path("data/processed/activities")
raw_dataset = ActivityGraphDataset(root=str(dataset_path))
dataset = [raw_dataset[i] for i in range(len(raw_dataset)) if raw_dataset[i].num_nodes > 0]

scenarios = [g.scenario_id if hasattr(g, "scenario_id") else "unknown" for g in dataset]
labels = [g.y.item() for g in dataset]

unique_scenarios = set(scenarios)
print(f"Total Graphs           : {len(dataset)}")
print(f"Unique Scenarios       : {len(unique_scenarios)}")

scenario_to_labels = defaultdict(Counter)
for g in dataset:
    scen = g.scenario_id if hasattr(g, "scenario_id") else "unknown"
    scenario_to_labels[scen][g.y.item()] += 1

multi_label_scenarios = 0
for scen, counts in scenario_to_labels.items():
    if len(counts) > 1:
        multi_label_scenarios += 1
        print(f"  Multi-label Scenario {scen}: {dict(counts)}")

print(f"\nMulti-label Scenarios Count: {multi_label_scenarios} / {len(unique_scenarios)}")
print("\nScenario -> Class Counts (First 20 Scenarios):")
for scen in sorted(list(unique_scenarios))[:20]:
    print(f"  {scen:<30} : {dict(scenario_to_labels[scen])}")
