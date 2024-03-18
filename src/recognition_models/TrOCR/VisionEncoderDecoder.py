import torch
import numpy as np
from transformers import VisionEncoderDecoderModel, AutoTokenizer, AutoImageProcessor
from typing import List, Union

from src.recognition_models.abstract_recognizer import Recognizer

class RuTrOCR(Recognizer):
    def __init__(self, model_name_or_path: str = "raxtemur/trocr-base-ru", **kwargs) -> None:
        """
        Initializes the model either from a Hugging Face model or a local path.

        Args:
            model_name_or_path (str): A string that specifies the model name on Hugging Face or the path to the local model directory.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.feature_extractor = AutoImageProcessor.from_pretrained(model_name_or_path)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name_or_path)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        self.model.eval()
        self.model.to(self.device)
        
    def process_images(self, images: Union[np.ndarray, List[np.ndarray]]):
        """
        Converts images to text.

        Args:
            images (Union[np.ndarray, List[np.ndarray]]): An image or a list of images in np.array format.

        Returns:
            List[str]: A list of recognized text strings.
        """
        if isinstance(images, np.ndarray):
            images = [images]
        
        inputs = self.feature_extractor(images=images, return_tensors="pt", padding=True)
        pixel_values = inputs["pixel_values"]

        pixel_values = pixel_values.to(self.device)
        
        # Generate text without using labels
        with torch.no_grad():
            output_ids = self.model.generate(pixel_values)
        
        output_texts = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        return output_texts
