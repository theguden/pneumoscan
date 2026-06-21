import os
import shutil
import random

BASE_DIR = "dataset_brain/train"
VAL_DIR = "dataset_brain/val"

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

VAL_SPLIT = 0.15


def split_class(class_name):
    source_folder = os.path.join(BASE_DIR, class_name)
    val_folder = os.path.join(VAL_DIR, class_name)

    files = os.listdir(source_folder)
    random.shuffle(files)

    split_index = int(len(files) * VAL_SPLIT)
    val_files = files[:split_index]

    for file in val_files:
        src = os.path.join(source_folder, file)
        dst = os.path.join(val_folder, file)
        shutil.move(src, dst)

    print(f"{class_name}: {len(val_files)} файлов перенесено в val")


def main():
    print("Разделение датасета...")

    for cls in CLASSES:
        os.makedirs(os.path.join(VAL_DIR, cls), exist_ok=True)
        split_class(cls)

    print("Готово!")


if __name__ == "__main__":
    main()