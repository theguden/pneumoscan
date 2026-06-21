import os
import random
import shutil

SOURCE_DIR = "dataset/covid_source"

TRAIN_DIR = "dataset/train/covid"
VAL_DIR = "dataset/val/covid"
TEST_DIR = "dataset/test/covid"

ALLOWED_EXTENSIONS = (".png", ".jpg", ".jpeg")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42


def get_images(folder):
    return [
        f for f in os.listdir(folder)
        if f.lower().endswith(ALLOWED_EXTENSIONS)
    ]


def recreate_folder(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder, exist_ok=True)


def copy_files(files, src, dst):
    os.makedirs(dst, exist_ok=True)
    for f in files:
        shutil.copy2(os.path.join(src, f), os.path.join(dst, f))


def main():
    random.seed(RANDOM_SEED)

    if not os.path.exists(SOURCE_DIR):
        print("❌ Папка covid_source не найдена")
        return

    files = get_images(SOURCE_DIR)

    if not files:
        print("❌ В covid_source нет изображений")
        return

    random.shuffle(files)

    total = len(files)
    train_count = int(total * TRAIN_RATIO)
    val_count = int(total * VAL_RATIO)

    train_files = files[:train_count]
    val_files = files[train_count:train_count + val_count]
    test_files = files[train_count + val_count:]

    recreate_folder(TRAIN_DIR)
    recreate_folder(VAL_DIR)
    recreate_folder(TEST_DIR)

    copy_files(train_files, SOURCE_DIR, TRAIN_DIR)
    copy_files(val_files, SOURCE_DIR, VAL_DIR)
    copy_files(test_files, SOURCE_DIR, TEST_DIR)

    print("\nРазделение COVID завершено:")
    print(f"Всего: {total}")
    print(f"Train: {len(train_files)}")
    print(f"Val:   {len(val_files)}")
    print(f"Test:  {len(test_files)}")


if __name__ == "__main__":
    main()