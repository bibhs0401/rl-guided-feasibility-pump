# split_instances.py
import random, shutil
from pathlib import Path

random.seed(42)
instances = sorted(Path("instances_3").glob("*.npz"))
random.shuffle(instances)

split = int(0.8 * len(instances))
train, test = instances[:split], instances[split:]

Path("instances_3/train").mkdir(exist_ok=True)
Path("instances_3/test").mkdir(exist_ok=True)

for p in train:
    shutil.copy(p, Path("instances_3/train") / p.name)
for p in test:
    shutil.copy(p, Path("instances_3/test") / p.name)

print(f"Train: {len(train)}, Test: {len(test)}")