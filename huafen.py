import os, json, random

config_path = r"E:\Photoacoustic\Others\unified_train_predict_bundle\unified_config.json"

with open(config_path, 'r') as f:
    config = json.load(f)

patients = [p for p in os.listdir(config["features_dir"]) if p.startswith("s")]
patients = [p for p in patients if os.path.isdir(os.path.join(config["features_dir"], p))]
random.seed(42)
random.shuffle(patients)

split_idx = int(len(patients) * 0.7)
config["train_patients"] = patients[:split_idx]
config["test_patients"] = patients[split_idx:]

with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)

print("Train patients:", len(config["train_patients"]))
print("Test patients:", len(config["test_patients"]))
