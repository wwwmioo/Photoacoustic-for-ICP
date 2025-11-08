import os
import pandas as pd
import json

# 载入配置
with open("unified_config.json", "r") as f:
    config = json.load(f)

print("Features dir:", config["features_dir"])
print("Train patients:", len(config["train_patients"]), "Test patients:", len(config["test_patients"]))

# 检查第一个训练患者
p = config["train_patients"][0]
patient_path = os.path.join(config["features_dir"], p)
print("\nCheck patient folder:", patient_path)

for wl in config["wavelengths"]:
    csv_path = os.path.join(patient_path, f"{wl}nm.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"{wl}nm.csv ✅ shape={df.shape}")
    else:
        print(f"{wl}nm.csv ❌ missing")
