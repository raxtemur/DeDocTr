import logging
import torch
import numpy as np
from typing import List, Union
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image

from src.recognition_models.attention_model.resize_normalization import AlignCollate
from src.recognition_models.abstract_recognizer import Recognizer
from src.recognition_models.attention_model.label_converting import AttnLabelConverter
from src.recognition_models.attention_model.model import Model as AttnModel

class Options:
    def __init__(self, characters : str, 
                 batch_max_length : int = 40, img_h : int = 32, img_w : int = 100, 
                 rgb : bool = False, pad : bool = False, num_class : int = None,
                 data_dir : str = "./datasets", label_files : List[str] = ["label_file.csv"],
                 eval_stage : str = "test", batch_size : int = 128, write_errors : bool = False,
                 saved_model : str = "path_to_saved_model", log_dir : str = "logs", log_name : str = "test_log.txt",
                 device : torch.device = None, logger : logging.Logger = None, **kwargs) -> None:
        """
        Initialize the class with default values for parameters:
        - batch_max_length: int = 40 -- Maximum label length
        - img_h: int = 32 -- The height of the input image
        - img_w: int = 100 -- The width of the input image
        - rgb: bool = False -- Use rgb input
        - pad: bool = False -- Whether to keep ratio then pad for image resize
        - num_class: int = None
        - data_dir: str = "./datasets"
        - label_files: List[str] = ["label_file.csv"]
        - eval_stage: str = "test"
        - batch_size: int = 128
        - write_errors: bool = False
        - saved_model: str = "path_to_saved_model"
        - log_dir: str = "logs"
        - log_name: str = "test_log.txt"
        - device: torch.device = None
        - logger: logging.Logger = None
        """
        self.batch_max_length = batch_max_length
        self.img_h = img_h
        self.img_w = img_w
        self.rgb = rgb
        self.pad = pad
        self.num_class = num_class
        self.data_dir = data_dir
        self.label_files = label_files
        self.eval_stage = eval_stage
        self.batch_size = batch_size
        self.write_errors = write_errors
        self.saved_model = saved_model
        self.log_dir = log_dir
        self.log_name = log_name
        self.character = characters
        
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        
        if logger is None:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)

class AttentionModel(Recognizer):
    def __init__(self, model_name_or_path: str, opt: Union[Options, None] = None, **kwargs) -> None:
        if opt is None:
            opt = Options(**kwargs)
        self.opt = opt
        
        self.device = opt.device
        self.converter = AttnLabelConverter(opt.character)
        self.align_collate = AlignCollate(img_h=opt.img_h, img_w=opt.img_w, keep_ratio_with_pad=opt.pad)
        opt.num_class = len(self.converter.character)
        
        self.model = AttnModel(opt, opt.logger).to(self.device)
        self.model = torch.nn.DataParallel(self.model, device_ids=[0])
        self.model.load_state_dict(torch.load(model_name_or_path, map_location=self.device))
        # alternative way to load properly is simplier:
        # new_state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
        # self.model.load_state_dict(new_state_dict)

        self.model.eval()
        
        # Настройка преобразований изображения
        input_channel = 3 if opt.rgb else 1
        '''
        self.transform = Compose([
            Resize((opt.img_h, opt.img_w)),
            ToTensor(),
            Normalize(mean=[0.5] * input_channel, std=[0.5] * input_channel),
        ])
        '''

    def process_images(self, images: Union[np.ndarray, List[np.ndarray]]) -> List[str]:
        """
        processed_image = self.transform(images)
        if self.opt.rgb:
            processed_image = Image.fromarray(processed_image).convert('RGB')
        else:
            processed_image = Image.fromarray(processed_image).convert('L')
        """
        if isinstance(images, np.ndarray):  # Если на вход подан один массив, преобразуем его в список
            images = [images]
            
        processed_images = [Image.fromarray(img).convert('RGB') if self.opt.rgb else Image.fromarray(img).convert('L') 
                            for img in images]
        pseudo_text = '[GO]' * len(processed_images)
        length_for_pred = torch.IntTensor([self.opt.batch_max_length] * len(processed_images)).to(self.device)
        
        batch_images, _ = self.align_collate(zip(processed_images, pseudo_text))

        with torch.no_grad():
            # text_for_pred = torch.LongTensor([self.converter.dict[char] for char in '[GO]' * len(processed_images)]).to(self.device)
            preds = self.model(batch_images.to(self.device), text=None, is_train=False)

        _, preds_index = preds.max(2)
        preds_str = self.converter.decode(preds_index, length=length_for_pred)
        # prune after "end of sentence" token ([s])
        preds_str = [pred[:pred.find('[s]')] for pred in preds_str]

        return preds_str

