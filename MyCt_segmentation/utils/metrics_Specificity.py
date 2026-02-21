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
        # 初始化混淆矩阵，大小为 num_class x num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def Pixel_Accuracy(self):
        """
        计算整体像素准确率 (Overall Accuracy)。
        公式: (正确分类的像素总数) / (总像素数)
        返回:
            float: 整体像素准确率。
        """
        # 如果总像素数为0（例如，在评估开始前调用），直接返回0
        if self.confusion_matrix.sum() == 0:
            return 0.0

        # np.diag(self.confusion_matrix).sum() 是所有类别TP的总和
        # self.confusion_matrix.sum() 是总像素数
        acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return acc

    def Pixel_Accuracy_Class(self):
        """
        计算每个类别的像素准确率 (Per-class Accuracy) 和它们的平均值 (mPA)。
        公式: TP_i / (TP_i + FN_i) for each class i
        返回:
            float: 平均像素准确率。
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            # Acc_per_class 是每个类别的准确率数组
            Acc_per_class = np.diag(self.confusion_matrix) / (self.confusion_matrix.sum(axis=1) + 1e-8)

        # mPA 是忽略了 NaN 值的平均准确率
        mPA = np.nanmean(Acc_per_class)
        return mPA

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
        # sum(axis=0) 是每个预测类别的像素总数 (TP + FP)
        # FP = (TP + FP) - TP
        FP = np.sum(self.confusion_matrix, axis=0) - TP
        with np.errstate(divide='ignore', invalid='ignore'):
            precision = TP / (TP + FP + 1e-8)

        mean_precision = np.nanmean(precision)
        return precision, mean_precision

    def Recall(self):
        """
        计算每个类别的召回率 (Recall) 和它们的平均值。
        公式: TP / (TP + FN)
        返回:
            tuple: (recall, mean_recall)
                   - recall (np.ndarray): 包含每个类别召回率的数组。
                   - mean_recall (float): 平均召回率。
        """
        TP = np.diag(self.confusion_matrix)
        # sum(axis=1) 是每个真实类别的像素总数 (TP + FN)
        # FN = (TP + FN) - TP
        FN = np.sum(self.confusion_matrix, axis=1) - TP
        with np.errstate(divide='ignore', invalid='ignore'):
            recall = TP / (TP + FN + 1e-8)

        mean_recall = np.nanmean(recall)
        return recall, mean_recall

    def Specificity(self):
        """
        计算每个类别的特异性 (Specificity) 和它们的平均值。
        公式: TN / (TN + FP)
        返回:
            tuple: (specificity, mean_specificity)
                   - specificity (np.ndarray): 包含每个类别特异性的数组。
                   - mean_specificity (float): 平均特异性。
        """
        # 总像素数
        total_pixels = self.confusion_matrix.sum()
        
        # TP: 真阳性 - 混淆矩阵对角线元素
        TP = np.diag(self.confusion_matrix)
        
        # FP: 假阳性 - 每列的和减去TP
        FP = np.sum(self.confusion_matrix, axis=0) - TP
        
        # TN: 真阴性 - 对于多分类任务的正确计算公式
        # TN_i = 总像素数 - (预测为i的所有像素 + 真实为i的所有像素 - TP_i)
        TN = total_pixels - (np.sum(self.confusion_matrix, axis=0) +
                             np.sum(self.confusion_matrix, axis=1) - TP)
        
        # 计算特异性，使用 np.errstate 避免除零警告
        with np.errstate(divide='ignore', invalid='ignore'):
            specificity = TN / (TN + FP)
        
        # 计算平均特异性，自动忽略 NaN 值
        mean_specificity = np.nanmean(specificity)
        
        return specificity, mean_specificity

    def IoU(self):
        """
        计算每个类别的交并比 (Intersection over Union) 和它们的平均值 (mIoU)。
        公式: TP / (TP + FP + FN)
        返回:
            tuple: (iou, mIoU)
                   - iou (np.ndarray): 包含每个类别IoU的数组。
                   - mIoU (float): 平均交并比。
        """
        TP = np.diag(self.confusion_matrix)
        FP = np.sum(self.confusion_matrix, axis=0) - TP
        FN = np.sum(self.confusion_matrix, axis=1) - TP
        with np.errstate(divide='ignore', invalid='ignore'):
            iou = TP / (TP + FP + FN + 1e-8)

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
        公式: 2 * TP / (2 * TP + FP + FN)
        返回:
            tuple: (dice, mean_dice)
                   - dice (np.ndarray): 包含每个类别Dice系数的数组。
                   - mean_dice (float): 平均Dice系数。
        """
        TP = np.diag(self.confusion_matrix)
        FP = np.sum(self.confusion_matrix, axis=0) - TP
        FN = np.sum(self.confusion_matrix, axis=1) - TP
        with np.errstate(divide='ignore', invalid='ignore'):
            dice = 2 * TP / (2 * TP + FP + FN + 1e-8)

        mean_dice = np.nanmean(dice)
        return dice, mean_dice

    def _generate_matrix(self, gt_image, pre_image):
        """
        为单对真实图和预测图生成混淆矩阵。
        参数:
            gt_image (np.ndarray): 真实标签图 (H, W)。
            pre_image (np.ndarray): 预测标签图 (H, W)。
        返回:
            np.ndarray: 当前图像对的混淆矩阵 (num_class, num_class)。
        """
        # 更严格的掩码，确保两个图像都在有效范围内
        mask = (gt_image >= 0) & (gt_image < self.num_class) & (pre_image >= 0) & (pre_image < self.num_class)
        
        # 使用bincount方法高效计算混淆矩阵
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        """
        将一个批次的预测结果和真实标签添加到总的混淆矩阵中。
        参数:
            gt_image (np.ndarray): 批次的真实标签图 (Batch_size, H, W)。
            pre_image (np.ndarray): 批次的预测标签图 (Batch_size, H, W)。
        """
        assert gt_image.shape == pre_image.shape, f"形状不匹配: gt_image {gt_image.shape}, pre_image {pre_image.shape}"
        
        # 如果是批量数据，逐个处理
        if len(gt_image.shape) == 3:  # (Batch_size, H, W)
            for i in range(gt_image.shape[0]):
                self.confusion_matrix += self._generate_matrix(gt_image[i], pre_image[i])
        else:  # 单张图像 (H, W)
            self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        """重置混淆矩阵，为新一轮评估做准备。"""
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def get_confusion_matrix(self):
        """
        获取当前的混淆矩阵。
        返回:
            np.ndarray: 混淆矩阵。
        """
        return self.confusion_matrix