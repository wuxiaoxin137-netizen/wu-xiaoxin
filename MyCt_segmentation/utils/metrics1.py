import numpy as np


class Evaluator(object):
    """
    一个用于计算图像分割指标的类，支持多分类任务。
    它通过累积混淆矩阵来高效地计算所有指标。
    """

    def __init__(self, num_class):
        """
        初始化评估器。
        参数:
            num_class (int): 数据集中的类别总数。
        """
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def Pixel_Accuracy(self):
        """
        计算整体像素准确率 (Overall Accuracy)。
        公式: (正确分类的像素总数) / (总像素数)
        返回:
            float: 整体像素准确率。
        """
        # np.diag(self.confusion_matrix).sum() 是所有类别TP的总和
        # self.confusion_matrix.sum() 是总像素数
        with np.errstate(divide='ignore', invalid='ignore'):
            acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return acc if not np.isnan(acc) else 0.0

    def Pixel_Accuracy_Class(self):
        """
        计算每个类别的像素准确率 (per-class accuracy) 和它们的平均值 (mPA)。
        注意：每个类别的准确率在数学上与每个类别的召回率是相同的。
        返回:
            tuple: (per_class_acc, mPA)
                   - per_class_acc (np.ndarray): 包含每个类别准确率的数组。
                   - mPA (float): 平均像素准确率。
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            per_class_acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)

        mPA = np.nanmean(per_class_acc)
        return per_class_acc, mPA

    def Precision(self):
        """
        计算每个类别的精确率 (Precision) 和它们的平均值。
        公式: TP / (TP + FP)
        返回:
            tuple: (precision, mean_precision)
                   - precision (np.ndarray): 包含每个类别精确率的数组。
                   - mean_precision (float): 平均精确率。
        """
        TP = np.diag(self.confusion_matrix)
        FP = np.sum(self.confusion_matrix, axis=0) - TP
        with np.errstate(divide='ignore', invalid='ignore'):
            precision = TP / (TP + FP)

        mean_precision = np.nanmean(precision)
        return precision, mean_precision

    def Recall(self):
        """
        计算每个类别的召回率 (Recall) 和它们的平均值。
        返回:
            tuple: (recall, mean_recall)
                   - recall (np.ndarray): 包含每个类别召回率的数组。
                   - mean_recall (float): 平均召回率。
        """
        TP = np.diag(self.confusion_matrix)
        FN = np.sum(self.confusion_matrix, axis=1) - TP
        with np.errstate(divide='ignore', invalid='ignore'):
            recall = TP / (TP + FN)

        mean_recall = np.nanmean(recall)
        return recall, mean_recall

    def IoU(self):
        """
        计算每个类别的交并比 (Intersection over Union) 和它们的平均值 (mIoU)。
        返回:
            tuple: (iou, mIoU)
                   - iou (np.ndarray): 包含每个类别IoU的数组。
                   - mIoU (float): 平均交并比。
        """
        TP = np.diag(self.confusion_matrix)
        FP = np.sum(self.confusion_matrix, axis=0) - TP
        FN = np.sum(self.confusion_matrix, axis=1) - TP
        with np.errstate(divide='ignore', invalid='ignore'):
            iou = TP / (TP + FP + FN)

        mIoU = np.nanmean(iou)
        return iou, mIoU

    def Mean_Intersection_over_Union(self):
        """
        直接计算平均交并比 (mIoU) 的便捷方法。
        返回:
            float: mIoU 的值。
        """
        _, mIoU = self.IoU()  # 直接复用IoU方法的计算结果
        return mIoU

    def Dice(self):
        """
        计算每个类别的 Dice 系数和它们的平均值。
        返回:
            tuple: (dice, mean_dice)
                   - dice (np.ndarray): 包含每个类别Dice系数的数组。
                   - mean_dice (float): 平均Dice系数。
        """
        TP = np.diag(self.confusion_matrix)
        FP = np.sum(self.confusion_matrix, axis=0) - TP
        FN = np.sum(self.confusion_matrix, axis=1) - TP
        with np.errstate(divide='ignore', invalid='ignore'):
            dice = 2 * TP / (2 * TP + FP + FN)

        mean_dice = np.nanmean(dice)
        return dice, mean_dice

    def _generate_matrix(self, gt_image, pre_image):
        """为单对真实图和预测图生成混淆矩阵。"""
        # 掩码，用于选择有效的类别标签（例如，忽略值为255的像素）
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        # 高效计算混淆矩阵的技巧
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        """将一个批次的预测结果和真实标签添加到总的混淆矩阵中。"""
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        """重置混淆矩阵，为新一轮评估做准备。"""
        self.confusion_matrix = np.zeros((self.num_class,) * 2)