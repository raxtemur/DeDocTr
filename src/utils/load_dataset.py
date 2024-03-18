import json
import os
from typing import List, Tuple

def load_your_dataset(dataset_dir: str) -> List[Tuple[str, str]]:
    """
    Загружает датасет изображений и их текстовых аннотаций.

    Args:
        dataset_dir (str): Путь к директории датасета, содержащей изображения и файл gt.json.

    Returns:
        List[Tuple[str, str]]: Список кортежей, где первый элемент — путь к изображению, а второй — текст.
    """
    # Путь к файлу с аннотациями
    gt_path = os.path.join(dataset_dir, "gt.json")

    # Загружаем аннотации из файла
    with open(gt_path, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    # Создаём список кортежей (путь к изображению, текст)
    dataset = []
    for img_name, text in annotations.items():
        img_path = os.path.join(dataset_dir, img_name)
        if os.path.exists(img_path):  # Проверяем, что файл изображения существует
            dataset.append((img_path, text))
        else:
            raise FileNotFoundError(f"Image file {img_path} not found.")

    return dataset
