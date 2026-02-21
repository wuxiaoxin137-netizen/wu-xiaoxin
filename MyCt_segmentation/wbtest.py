import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from mypath import Path

current_dir = os.path.dirname(os.path.abspath(__file__))

class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def Precision(self):
        """Calculate precision for the target class (class 1)."""
        TP = np.diag(self.confusion_matrix)[1]
        FP = np.sum(self.confusion_matrix, axis=0)[1] - TP
        precision = TP / (TP + FP) if TP + FP > 0 else 0
        return precision

    def Recall(self):
        """Calculate recall for the target class (class 1)."""
        TP = np.diag(self.confusion_matrix)[1]
        FN = np.sum(self.confusion_matrix, axis=1)[1] - TP
        recall = TP / (TP + FN) if TP + FN > 0 else 0
        return recall

    def Pixel_Accuracy(self):
        """Calculate pixel accuracy for all pixels."""
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def IoU(self):
        """
        Calculate Intersection over Union (IoU) for the target class (class 1).
        """
        TP = np.diag(self.confusion_matrix)[1]
        FP = np.sum(self.confusion_matrix, axis=0)[1] - TP
        FN = np.sum(self.confusion_matrix, axis=1)[1] - TP
        iou = TP / (TP + FP + FN) if TP + FP + FN > 0 else 0
        return iou

    def Dice(self):
        """Calculate Dice Coefficient for the target class (class 1)."""
        TP = np.diag(self.confusion_matrix)[1]
        FP = np.sum(self.confusion_matrix, axis=0)[1] - TP
        FN = np.sum(self.confusion_matrix, axis=1)[1] - TP
        dice = 2 * TP / (2 * TP + FP + FN) if 2 * TP + FP + FN > 0 else 0
        return dice

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)


def modify_and_evaluate(predicted_folder, ground_truth_folder):
    evaluator = Evaluator(num_class=2)  # 2 class: background (0) and vessel (1)

    predicted_files = os.listdir(predicted_folder)
    ground_truth_files = os.listdir(ground_truth_folder)

    # Sort files to make sure they match
    predicted_files.sort()
    ground_truth_files.sort()

    if len(predicted_files) != len(ground_truth_files):
        raise ValueError("Number of predicted images does not match the number of ground truth images.")

    for predicted_file, ground_truth_file in tqdm(zip(predicted_files, ground_truth_files), total=len(predicted_files),
                                                  desc="Processing images"):
        # Load Predicted Image
        predicted_image_path = os.path.join(predicted_folder, predicted_file)
        predicted_image = Image.open(predicted_image_path).convert('RGB')
        predicted_image = np.array(predicted_image)

        # Modify Predicted Image
        red_pixels = np.all(predicted_image == [255, 0, 0], axis=2)
        predicted_image[red_pixels] = [255, 255, 255]

        # Load Ground Truth Image
        ground_truth_image_path = os.path.join(ground_truth_folder, ground_truth_file)
        ground_truth_image = Image.open(ground_truth_image_path).convert('RGB')
        ground_truth_image = np.array(ground_truth_image)

        # Convert RGB to single channel for evaluation
        predicted_mask = np.all(predicted_image == [255, 255, 255], axis=2).astype(int)
        ground_truth_mask = np.all(ground_truth_image == [255, 255, 255], axis=2).astype(int)

        evaluator.add_batch(ground_truth_mask, predicted_mask)

    acc = evaluator.Pixel_Accuracy()
    precision = evaluator.Precision()
    recall = evaluator.Recall()
    iou = evaluator.IoU()
    dice = evaluator.Dice()

    print("--------------------------------")
    print(f"Overall Accuracy: {acc:.4f}")
    print(f"Vessel Precision: {precision:.4f}")
    print(f"Vessel Recall: {recall:.4f}")
    print(f"Vessel IoU: {iou:.4f}")
    print(f"Vessel Dice: {dice:.4f}")


if __name__ == '__main__':
    predicted_folder = Path.db_root_dir('custom') + "/images/saved/"  # Replace with your predicted labels folder path
    ground_truth_folder = Path.db_root_dir('custom') + "/masks/resizemask/"  # Replace with your ground truth labels folder path
    modify_and_evaluate(predicted_folder, ground_truth_folder)