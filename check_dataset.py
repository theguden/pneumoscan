import os

DATASET_DIR = "dataset"

SPLITS = ["train", "val", "test"]
CLASSES = ["normal", "viral", "bacterial", "covid"]
ALLOWED_EXTENSIONS = (".png", ".jpg", ".jpeg")


def count_images(folder_path):
    if not os.path.exists(folder_path):
        return None

    files = [
        file_name for file_name in os.listdir(folder_path)
        if file_name.lower().endswith(ALLOWED_EXTENSIONS)
    ]
    return len(files)


def check_dataset():
    print("=" * 50)
    print("ПРОВЕРКА СТРУКТУРЫ ДАТАСЕТА")
    print("=" * 50)

    if not os.path.exists(DATASET_DIR):
        print(f"Ошибка: папка '{DATASET_DIR}' не найдена.")
        return

    total_images = 0
    has_errors = False

    for split in SPLITS:
        print(f"\n[{split.upper()}]")

        split_path = os.path.join(DATASET_DIR, split)

        if not os.path.exists(split_path):
            print(f"  Ошибка: папка '{split_path}' отсутствует.")
            has_errors = True
            continue

        for class_name in CLASSES:
            class_path = os.path.join(split_path, class_name)
            image_count = count_images(class_path)

            if image_count is None:
                print(f"  Ошибка: папка '{class_path}' отсутствует.")
                has_errors = True
            elif image_count == 0:
                print(f"  Внимание: папка '{class_path}' пуста.")
                has_errors = True
            else:
                print(f"  {class_name:<12} -> {image_count} изображений")
                total_images += image_count

    print("\n" + "=" * 50)
    print(f"Всего найдено изображений: {total_images}")

    if has_errors:
        print("Проверка завершена: есть проблемы, которые нужно исправить.")
    else:
        print("Проверка завершена: структура датасета в порядке.")
    print("=" * 50)


if __name__ == "__main__":
    check_dataset()