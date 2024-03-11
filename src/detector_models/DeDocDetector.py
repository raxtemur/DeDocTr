from typing import List, Tuple
import albumentations as A
from dedocutils.data_structures.bbox import BBox
from dedocutils.line_segmentation import ClusteringLineSegmenter
from dedocutils.preprocessing import AdaptiveBinarizer, SkewCorrector
from dedocutils.text_detection import DoctrTextDetector
import numpy as np

from src.detector_models.abstract_detector import Detector

class DeDocDetector(Detector):
    def __init__(self):
        super().__init__()
        self.transforms = A.Compose([
            A.CLAHE(clip_limit=4.0, tile_grid_size=(6, 8), always_apply=True),
        ])
        self.binarizer = AdaptiveBinarizer()
        self.skew_corrector = SkewCorrector()
        self.text_detector = DoctrTextDetector()
        self.line_segmenter = ClusteringLineSegmenter()

    def preprocess(self, image, use_clahe=True):
        if use_clahe and self.transforms is not None:
            img_clahed = self.transforms(image=image)["image"]
        else:
            img_clahed = image
        binarized_image, _ = self.binarizer.preprocess(img_clahed)
        self.preprocessed_image, _ = self.skew_corrector.preprocess(binarized_image)
        return self.preprocessed_image

    def detect(self, image) -> List[BBox]:
        self.bboxes = self.text_detector.detect(image)
        return self.bboxes
    
    def cluster_lines(self, bboxes):
        return self.line_segmenter.segment(bboxes)

    def process_image(self, image, use_clahe=True, multiple_lines=True):
        self.preprocess(image, use_clahe=use_clahe)
        bboxes = self.detect(self.preprocessed_image)
        if multiple_lines:
            self.sorted_bboxes = self.cluster_lines(bboxes)
        else:
            # Если предполагается только одна строка, то все боксы считаются частью одной строки
            self.sorted_bboxes = [bboxes]
        return self.sorted_bboxes

    def get_by_bbox(self, image, bbox: BBox) -> np.ndarray:
        return image[bbox.y_top_left:bbox.y_bottom_right, bbox.x_top_left:bbox.x_bottom_right]