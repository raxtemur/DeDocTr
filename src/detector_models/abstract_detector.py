from abc import ABC, abstractmethod
import cv2
from typing import List, Tuple, Any

class Detector(ABC):
    def __init__(self):
        self.transforms = None
        self.binarizer = None
        self.skew_corrector = None
        self.bboxes = []  # Хранение обнаруженных боксов
        self.sorted_bboxes = [] # Хранение кластеризованных по строкам боксов

    @abstractmethod
    def preprocess(self, image: Any) -> Any:
        """
        Предварительная обработка изображения перед детекцией текста.

        :param image: Исходное изображение.
        :return: Обработанное изображение.
        """
        pass

    @abstractmethod
    def detect(self, image: Any) -> List[Tuple[int, int, int, int]]:
        """
        Детектирует текстовые блоки на изображении.

        :param image: Изображение для детекции.
        :return: Список кортежей с координатами текстовых блоков (x_min, y_min, x_max, y_max).
        """
        pass

    @abstractmethod
    def cluster_lines(self) -> None:
        """
        Кластеризация обнаруженных текстовых блоков по строкам.
        """
        pass

    def process_image(self, image: Any) -> None:
        """
        Выполняет полный цикл обработки изображения: предварительная обработка, детекция и кластеризация.

        :param image: Исходное изображение.
        """
        preprocessed_image = self.preprocess(image)
        self.bboxes = self.detect(preprocessed_image)
        self.cluster_lines()
