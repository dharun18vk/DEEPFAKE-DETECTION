import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm

# ----------- CONFIG -------------
DATASETS = [
    r"D:\deepfake_train\DATASET IMAGES\real_and_fake_face",
    r"D:\deepfake_train\DATASET IMAGES\real_vs_fake\real-vs-fake",
    r"D:\deepfake_train\DATASET IMAGES\Dataset"
]

OUTPUT_ROOT = "final_image_dataset"
SPLIT_RATIO = (0.70, 0.15, 0.15)  # train, val, test
CLASSES = ["real", "fake"]        # Folder names to create
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp"]
# --------------------------------


def collect_images(dataset_path):
    collected = {"real": [], "fake": []}

    for root, _, files in os.walk(dataset_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in IMAGE_EXTENSIONS):
                file_path = os.path.join(root, file)

                if "real" in root.lower():
                    collected["real"].append(file_path)
                elif "fake" in root.lower():
                    collected["fake"].append(file_path)
    return collected


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


print("🔍 Collecting images from datasets...")
all_images = {"real": [], "fake": []}

for dataset in DATASETS:
    if os.path.exists(dataset):
        print(f"📁 Scanning: {dataset}")
        imgs = collect_images(dataset)
        all_images["real"].extend(imgs["real"])
        all_images["fake"].extend(imgs["fake"])
    else:
        print(f"⚠️ Skipped missing folder: {dataset}")

total_real = len(all_images["real"])
total_fake = len(all_images["fake"])
print(f"\n✅ Total collected → Real: {total_real}, Fake: {total_fake}\n")


# ------------ SHUFFLE + SPLIT ------------
def split_data(data_list):
    random.shuffle(data_list)
    n = len(data_list)

    train_end = int(n * SPLIT_RATIO[0])
    val_end = train_end + int(n * SPLIT_RATIO[1])

    return data_list[:train_end], data_list[train_end:val_end], data_list[val_end:]


train_set = {"real": [], "fake": []}
val_set = {"real": [], "fake": []}
test_set = {"real": [], "fake": []}

for cls in CLASSES:
    train_set[cls], val_set[cls], test_set[cls] = split_data(all_images[cls])

# ------------ CREATE OUTPUT DIRS & COPY ------------
print("📦 Creating combined dataset...")

for split_name, split_dict in zip(["train", "val", "test"], [train_set, val_set, test_set]):
    for cls in CLASSES:
        folder = os.path.join(OUTPUT_ROOT, split_name, cls)
        ensure_dir(folder)
        print(f"📁 Copying {cls} → {split_name} ({len(split_dict[cls])} files)")

        for src in tqdm(split_dict[cls]):
            dst = os.path.join(folder, os.path.basename(src))
            shutil.copy2(src, dst)  # keeps metadata


print("\n🎉 DONE! Combined dataset created at:", OUTPUT_ROOT)
