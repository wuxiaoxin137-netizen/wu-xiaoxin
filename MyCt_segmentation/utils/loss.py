import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt  # 导入距离变换函数


class SegmentationLosses(object):
    def __init__(self, weight=True, size_average=True, batch_average=True, ignore_index=254, cuda=False):
        # 可以忽略计算背景的标签索引
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average  # 注意: size_average 在新版 PyTorch 中已弃用，建议使用 reduction
        self.batch_average = batch_average
        self.cuda = cuda

    def build_loss(self, mode='ce'):
        """Choices: ['ce', 'focal', or 'dpce']"""  # 【修改】增加了 'dpce' 选项
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        elif mode == 'dpce':
            # 【新增】返回新的 DPCE 损失函数
            return self.DistancePenalizedCrossEntropyLoss
        else:
            raise NotImplementedError

    def _create_distance_penalty_map(self, gt_batch, n_classes):
        """
        【新增】辅助函数：根据论文描述，为多类别任务生成距离图惩罚权重 Φ。

        Args:
            gt_batch (torch.Tensor): 真实标签掩码批次，形状 (B, H, W)，值为类别索引。
            n_classes (int): 数据集中的总类别数。

        Returns:
            torch.Tensor: 惩罚权重图 Φ，形状为 (B, H, W)。
        """
        batch_penalty_maps = []
        for i in range(gt_batch.shape[0]):  # 遍历 batch 中的每个样本
            gt = gt_batch[i].squeeze().cpu().numpy().astype(np.uint8)

            # 为每个类别独立计算距离图并组合
            # 背景类（通常是0）不计算惩罚
            overall_penalty_map = np.zeros_like(gt, dtype=np.float32)

            for c in range(1, n_classes):  # 从类别1开始，跳过背景
                # 提取当前类别的二值掩码
                class_mask = (gt == c).astype(np.uint8)

                if np.sum(class_mask) == 0:  # 如果图像中没有这个类别的像素，跳过
                    continue

                # 计算标准距离变换 (边界为0, 中心最亮)
                dist_map = distance_transform_edt(class_mask)

                # 反转距离图 (用最大值减去当前值)
                max_dist = np.max(dist_map)
                if max_dist > 0:
                    penalty_map = max_dist - dist_map
                else:
                    penalty_map = np.zeros_like(dist_map)

                # 将当前类别的惩罚图合并到总图中
                # 如果一个像素同时属于多个类别的边界区域（理论上不可能），这里会取最大值
                overall_penalty_map = np.maximum(overall_penalty_map, penalty_map)

            batch_penalty_maps.append(torch.tensor(overall_penalty_map, dtype=torch.float32))

        # 重新组合成一个batch, 并移动到正确的设备
        phi = torch.stack(batch_penalty_maps).to(gt_batch.device)
        return phi

    def DistancePenalizedCrossEntropyLoss(self, logit, target):
        """
        【新增】实现论文中描述的、带有距离图惩罚项的多类别交叉熵损失。
        """
        n, c, h, w = logit.size()

        # 1. 生成惩罚权重图 Φ
        phi = self._create_distance_penalty_map(target, n_classes=c)

        # 2. 计算逐像素的交叉熵损失
        # 使用 reduction='none' 来获取损失图
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index, reduction='none')
        if self.cuda:
            criterion = criterion.cuda()

        loss_map = criterion(logit, target.long())

        # 3. 应用惩罚项 (根据论文公式2)
        # L = (1 + Φ) * L_ce
        penalized_loss_map = (1 + phi) * loss_map

        # 4. 根据 size_average (reduction) 决定如何聚合损失
        if self.size_average:
            loss = penalized_loss_map.mean()
        else:
            loss = penalized_loss_map.sum()

        if self.batch_average:
            loss /= n

        return loss

    def CrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()
        loss = criterion(logit, target.long())
        if self.batch_average:
            loss /= n
        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        n, c, h, w = logit.size()
        # 注意：Focal Loss 的标准实现通常不直接使用 reduction='mean' 的 CE
        # 这里为了保持与您原始代码一致，暂时保留。
        # 一个更标准的实现会使用 reduction='none' 的 CE 来计算 logpt。
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        reduction='none' if self.size_average else 'sum')
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            # Alpha weighting can be done per-class or as a single value
            # A common way is to use alpha for positive class and (1-alpha) for negative
            # For multi-class, it's more complex. Here we apply it uniformly.
            logpt *= alpha

        loss = -((1 - pt) ** gamma) * logpt

        if self.size_average:
            loss = loss.mean()

        if self.batch_average:
            loss /= n

        return loss

#
# if __name__ == "__main__":
#     # --- 示例用法 ---
#     B, C, H, W = 2, 4, 64, 64  # 假设 batch=2, 4个类别, 64x64图像
#
#     # 实例化损失函数类
#     # weight=None 表示不使用类别权重
#     loss_builder = SegmentationLosses(cuda=torch.cuda.is_available(), weight=None)
#
#     # 准备模拟数据
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#
#     # logit: 模型的原始预测输出, (B, C, H, W)
#     logit = torch.randn(B, C, H, W, requires_grad=True).to(device)
#
#     # target: 地面真相掩码, (B, H, W), 像素值为 0, 1, 2, 3
#     target = torch.randint(0, C, (B, H, W)).to(device)
#
#     # --- 测试 DPCE Loss ---
#     print("--- 测试 Distance Penalized Cross-Entropy Loss ---")
#     dpce_criterion = loss_builder.build_loss(mode='dpce')
#     dpce_loss_value = dpce_criterion(logit, target)
#     print(f"DPCE Loss: {dpce_loss_value.item()}")
#     dpce_loss_value.backward()  # 检查反向传播
#     print("DPCE Loss 反向传播成功。")
#
#     # --- 测试标准 CE Loss ---
#     print("\n--- 测试 Standard Cross-Entropy Loss ---")
#     logit.grad.zero_()  # 清空梯度
#     ce_criterion = loss_builder.build_loss(mode='ce')
#     ce_loss_value = ce_criterion(logit, target)
#     print(f"Standard CE Loss: {ce_loss_value.item()}")
#
#     # --- 测试 Focal Loss ---
#     print("\n--- 测试 Focal Loss ---")
#     focal_criterion = loss_builder.build_loss(mode='focal')
#     focal_loss_value = focal_criterion(logit, target)
#     print(f"Focal Loss: {focal_loss_value.item()}")