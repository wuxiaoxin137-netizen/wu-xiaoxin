import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt


class SegmentationLosses(object):
    def __init__(self, weight=False, size_average=True, batch_average=True, ignore_index=254, cuda=False):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda

    def build_loss(self, mode='dice'):
        """Choices: ['ce', 'focal', 'dpce', or 'dice']"""  # 【修改】增加了 'dice' 选项
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        elif mode == 'dpce':
            return self.DistancePenalizedCrossEntropyLoss
        elif mode == 'dice':
            # 【新增】返回新的 Dice 损失函数
            return self.DiceLoss
        else:
            raise NotImplementedError

    # --- Dice Loss 实现 ---
    def DiceLoss(self, logit, target, smooth=1e-6, ignore_background=False):
        """
        【新增】实现多类别 Dice Loss。
        采用"一对全"策略，为每个类别计算二值Dice Loss，然后加权平均。

        Args:
            logit (torch.Tensor): 模型的原始 logits 输出，形状 (B, C, H, W)。
            target (torch.Tensor): 真实标签掩码，形状 (B, H, W)。
            smooth (float): 平滑项，防止分母为0。
            ignore_background (bool): 是否忽略背景类别（通常是类别0）的损失。
        """
        n, c, h, w = logit.size()

        # 1. 将输出转换为概率
        probs = F.softmax(logit, dim=1)

        # 2. 将标签转换为 one-hot 编码
        # 创建一个 (B, C, H, W) 的全0张量
        target_one_hot = torch.zeros_like(logit)
        # 使用 scatter_ 将 target 中的类别索引填充到对应的 channel 上
        target_one_hot.scatter_(1, target.long().unsqueeze(1), 1)

        # 3. 准备类别权重
        if self.weight is not None:
            # 确保权重张量在正确的设备上
            class_weights = self.weight.to(logit.device)
            if class_weights.shape[0] != c:
                raise ValueError("权重张量的长度与类别数不匹配")
        else:
            # 如果没有提供权重，则所有类别权重为1
            class_weights = torch.ones(c).to(logit.device)

        total_loss = 0
        total_weight = 0

        # 4. 逐类别计算 Dice Loss
        start_index = 1 if ignore_background else 0
        for i in range(start_index, c):
            # 提取当前类别的预测概率和真实标签
            prob = probs[:, i, :, :]
            ref = target_one_hot[:, i, :, :]

            # 计算交集和并集
            intersection = torch.sum(prob * ref)
            union = torch.sum(prob) + torch.sum(ref)

            # 计算当前类别的 Dice score
            dice_score = (2. * intersection + smooth) / (union + smooth)

            # Dice Loss = 1 - Dice Score
            dice_loss = 1 - dice_score

            # 应用类别权重
            total_loss += dice_loss * class_weights[i]
            total_weight += class_weights[i]

        # 5. 返回加权平均损失
        if total_weight == 0:  # 如果所有前景类都不存在，返回0损失
            return torch.tensor(0.0).to(logit.device)

        return total_loss / total_weight

    # --- 其他损失函数 (与之前基本相同) ---
    def _create_distance_penalty_map(self, gt_batch, n_classes):
        batch_penalty_maps = []
        for i in range(gt_batch.shape[0]):
            gt = gt_batch[i].squeeze().cpu().numpy().astype(np.uint8)
            overall_penalty_map = np.zeros_like(gt, dtype=np.float32)
            for c in range(1, n_classes):
                class_mask = (gt == c).astype(np.uint8)
                if np.sum(class_mask) == 0: continue
                dist_map = distance_transform_edt(class_mask)
                max_dist = np.max(dist_map)
                penalty_map = max_dist - dist_map if max_dist > 0 else np.zeros_like(dist_map)
                overall_penalty_map = np.maximum(overall_penalty_map, penalty_map)
            batch_penalty_maps.append(torch.tensor(overall_penalty_map, dtype=torch.float32))
        return torch.stack(batch_penalty_maps).to(gt_batch.device)

    def DistancePenalizedCrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()
        phi = self._create_distance_penalty_map(target, n_classes=c)
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index, reduction='none')
        if self.cuda: criterion = criterion.cuda()
        loss_map = criterion(logit, target.long())
        penalized_loss_map = (1 + phi) * loss_map
        loss = penalized_loss_map.mean() if self.size_average else penalized_loss_map.sum()
        if self.batch_average: loss /= n
        return loss

    def CrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda: criterion = criterion.cuda()
        loss = criterion(logit, target.long())
        if self.batch_average: loss /= n
        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index, reduction='none')
        if self.cuda: criterion = criterion.cuda()
        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None: logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt
        loss = loss.mean() if self.size_average else loss.sum()
        if self.batch_average: loss /= n
        return loss


# # --- 示例用法 ---
# if __name__ == "__main__":
#     # 类别定义：0:背景, 1:血管(白), 2:导管(红)
#     # 假设类别权重：背景权重低，血管和导管权重高
#     class_weights = torch.tensor([0.5, 2.0, 3.0])
#
#     # 实例化损失函数类，并传入类别权重
#     loss_builder = SegmentationLosses(cuda=torch.cuda.is_available(), weight=class_weights)
#
#     # 准备模拟数据
#     B, C, H, W = 2, 3, 64, 64  # batch=2, 3个类别, 64x64图像
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#
#     logit = torch.randn(B, C, H, W, requires_grad=True).to(device)
#     target = torch.randint(0, C, (B, H, W)).to(device)
#
#     # --- 测试 Dice Loss ---
#     print("--- 测试 Dice Loss ---")
#     dice_criterion = loss_builder.build_loss(mode='dice')
#     dice_loss_value = dice_criterion(logit, target)
#     print(f"Dice Loss (忽略背景): {dice_loss_value.item()}")
#
#     # 测试包含背景的 Dice Loss
#     dice_loss_with_bg = dice_criterion(logit, target, ignore_background=False)
#     print(f"Dice Loss (包含背景): {dice_loss_with_bg.item()}")
#
#     dice_loss_value.backward()
#     print("Dice Loss 反向传播成功。")
#
#     # --- 测试其他损失 ---
#     print("\n--- 测试 DPCE Loss ---")
#     logit.grad.zero_()
#     dpce_criterion = loss_builder.build_loss(mode='dpce')
#     dpce_loss_value = dpce_criterion(logit, target)
#     print(f"DPCE Loss: {dpce_loss_value.item()}")