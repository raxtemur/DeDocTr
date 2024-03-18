from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np


class Recognizer(ABC):
    def __init__(self, model_name_or_path: str):
        pass
    @abstractmethod
    def process_images(self, images: Union[np.ndarray, List[np.ndarray]]) -> List[str]:
        """
        Распознает текст в выделенных блоках на изображении.
        
        :param images: Изображение или список изображений для распознавания. Изображение должно содержать слово или словосочетание.
        """
        pass