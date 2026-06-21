import os
import random
import shutil

SOURCE_DIR = "dataset/source"
BASE_DIR = "dataset"

CLASSES = ["normal", "viral", "bacterial"]
ALLOWED_EXTENSIONS = (".png", ".jpg", ".jpeg")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42


def get_image_files(folder_path):
    if not os.path.exists(folder_path):
        return []

    return [
        file_name for file_name in os.listdir(folder_path)
        if file_name.lower().endswith(ALLOWED_EXTENSIONS)
    ]


def recreate_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path, exist_ok=True)


def copy_files(file_list, source_folder, target_folder):
    os.makedirs(target_folder, exist_ok=True)

    for file_name in file_list:
        source_path = os.path.join(source_folder, file_name)
        target_path = os.path.join(target_folder, file_name)
        shutil.copy2(source_path, target_path)


def split_class_files(class_name):
    source_folder = os.path.join(SOURCE_DIR, class_name)
    files = get_image_files(source_folder)

    if not files:
        print(f"[!] В папке {source_folder} нет изображений.")
        return

    random.shuffle(files)

    total_count = len(files)
    train_count = int(total_count * TRAIN_RATIO)
    val_count = int(total_count * VAL_RATIO)
    test_count = total_count - train_count - val_count

    train_files = files[:train_count]
    val_files = files[train_count:train_count + val_count]
    test_files = files[train_count + val_count:]

    train_target = os.path.join(BASE_DIR, "train", class_name)
    val_target = os.path.join(BASE_DIR, "val", class_name)
    test_target = os.path.join(BASE_DIR, "test", class_name)

    recreate_folder(train_target)
    recreate_folder(val_target)
    recreate_folder(test_target)

    copy_files(train_files, source_folder, train_target)
    copy_files(val_files, source_folder, val_target)
    copy_files(test_files, source_folder, test_target)

    print(f"\nКласс: {class_name}")
    print(f"Всего: {total_count}")
    print(f"Train: {len(train_files)}")
    print(f"Val:   {len(val_files)}")
    print(f"Test:  {len(test_files)}")


def main():
    random.seed(RANDOM_SEED)

    print("=" * 50)
    print("АВТО-РАЗДЕЛЕНИЕ ДАТАСЕТА")
    print("=" * 50)

    if not os.path.exists(SOURCE_DIR):
        print(f"Ошибка: папка {SOURCE_DIR} не найдена.")
        return

    for class_name in CLASSES:
        split_class_files(class_name)

    print("\nГотово. Датасет разделен на train / val / test.")
    print("=" * 50)


if __name__ == "__main__":
    main()