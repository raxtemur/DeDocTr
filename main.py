import argparse
import cv2
from tqdm import tqdm
from src.detector_models.DeDocDetector import DeDocDetector
from src.recognition_models.TrOCR.VisionEncoderDecoder import SimpleTrOCR
from src.utils.metrics import cer, wer, string_accuracy
from src.utils.load_dataset import load_your_dataset

def main(dataset_path, log_file):
    detector = DeDocDetector()
    recognizer = SimpleTrOCR(model_name_or_path='raxtemur/trocr-base-ru')

    dataset = load_your_dataset(dataset_path)

    total_cer, total_wer, total_accuracy = [], [], []

    with open(log_file, 'w') as log:
        for image_path, annotation in tqdm(dataset):
            print(f"Filepath:{image_path}", file=log)
            image = cv2.imread(image_path)
            detected_lines = detector.process_image(image)
            pred_texts = []
            pred_text = ""
            for line in detected_lines:
                for i, bbox in enumerate(line):
                    cropped_image = detector.get_by_bbox(detector.preprocessed_image, bbox)
                    text = recognizer.process_images(cropped_image)[0]
                    pred_texts.append(text.strip())
                    if i != len(line)-1:
                        pred_texts.append(" ")
                pred_texts.append("\n")

            pred_text = ''.join(pred_texts)
            print(pred_text, file=log)

            total_cer.append(cer([pred_text], [annotation]))
            total_wer.append(wer([pred_text], [annotation]))
            total_accuracy.append(string_accuracy([pred_text], [annotation]))
        
        print(f"Mean CER: {sum(total_cer) / len(total_cer)}", file=log)
        print(f"Mean WER: {sum(total_wer) / len(total_wer)}", file=log)
        print(f"Mean Accuracy: {sum(total_accuracy) / len(total_accuracy)}", file=log)
        print(f"-----------------------------------------\n\n\n", file=log)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script to evaluate OCR performance on a dataset.")
    parser.add_argument('--dataset_path', type=str, default="./datasets/htr_lising_testing_data/good_data", help="Path to the dataset directory.")
    parser.add_argument('--log_file', type=str, default="log.txt", help="File to log the output.")
    
    args = parser.parse_args()
    
    main(args.dataset_path, args.log_file)
