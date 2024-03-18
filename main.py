import argparse
import logging
import os
import cv2
from tqdm import tqdm
from src.detector_models.DeDocDetector import DeDocDetector
from src.recognition_models.TrOCR.VisionEncoderDecoder import RuTrOCR
from src.recognition_models.AttentionModel import AttentionModel
from src.utils.metrics import cer, wer, string_accuracy
from src.utils.load_dataset import load_your_dataset

DETECTORS = { 
    'DeDocDetector': DeDocDetector,
    
    }

RECOGNIZERS = {
    'RuTrOCR': RuTrOCR,
    'AttnentionModel': AttentionModel
}


def main(args):
    # Setting up logger
    dataset_path = args.dataset_path
    log_file = args.log_file
    log_dir = os.path.dirname(log_file)
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("OCRLogger")
    logger.setLevel(logging.INFO)
    # Remove all handlers associated with the logger object
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    # Create a new handler for the file logger
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    # Define the format of the log messages
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    # Add the handler to the logger object
    logger.addHandler(file_handler)
    
    logger.info(args)
    logger.info('Loading models...')
    detector = DETECTORS[args.detector]()
    recognizer = RECOGNIZERS[args.recognition](**vars(args))

    logger.info('Loading dataset...')
    dataset = load_your_dataset(dataset_path)

    total_cer, total_wer, total_accuracy = [], [], []

    pred_texts = []
    annotations = []
    for image_path, annotation in tqdm(dataset):
        logger.info(f"Filepath:{image_path}")
        image = cv2.imread(image_path)
        detected_lines = detector.process_image(image)
        pred_text = ""
        for line in detected_lines:
            for i, bbox in enumerate(line):
                cropped_image = detector.get_by_bbox(detector.preprocessed_image, bbox)
                text = recognizer.process_images(cropped_image)[0]
                pred_text+=text.strip()
                if i != len(line)-1:
                    pred_text+=" "
            pred_text+="\n"

        pred_texts.append(pred_text)
        annotations.append(annotation)

        logger.info(f"Predicted text: \n{pred_text}")

        total_cer.append(cer([pred_text], [annotation]))
        total_wer.append(wer([pred_text], [annotation]))
        total_accuracy.append(string_accuracy([pred_text], [annotation]))
        logger.info(f"Is text correct: {total_accuracy[-1]}")
        logger.info("-----------------------------------------\n")
        
    mean_cer = cer(pred_texts, annotations)
    mean_wer = wer(pred_texts, annotations)
    mean_accuracy = string_accuracy(pred_texts, annotations)
        
    logger.info(f"Mean CER: {mean_cer}")
    logger.info(f"Mean WER: {mean_wer}")
    logger.info(f"Mean Accuracy: {mean_accuracy}")
    
    logger.info(f"Mean CER: {sum(total_cer) / len(total_cer)}")
    logger.info(f"Mean WER: {sum(total_wer) / len(total_wer)}")
    logger.info(f"Mean Accuracy: {sum(total_accuracy) / len(total_accuracy)}")
    logger.info(f"-----------------------------------------\n")

if __name__ == '__main__':
    """
    
    """
    parser = argparse.ArgumentParser(description="Script to evaluate OCR performance on a dataset.")
    parser.add_argument('--dataset_path', type=str, default="./datasets/htr_lising_testing_data/good_data", help="Path to the dataset directory.")
    parser.add_argument('--log_file', type=str, default="log.txt", help="File to log the output.")
    parser.add_argument('--detector', type=str, default="DeDocDetector", 
                        help="Name of the detector model to use. Possible values: DeDocDetector")
    parser.add_argument('--recognition', type=str, default="RuTrOCR", 
                        help="Name of the recognition model to use. Possible values: RuTrOCR, AttnentionModel")
    parser.add_argument('--model_name_or_path', type=str, default="raxtemur/trocr-base-ru", help="Name of the recognition model to use.")
    parser.add_argument('--characters_file', type=str, default="./source/attention_cyrillic_hkr_synthetic_stackmix/character.txt", help="Name of the recognition model to use.")
    args = parser.parse_args()
    
    with open(args.characters_file, 'r') as f:
            args.characters = f.read()
    
    main(args)
