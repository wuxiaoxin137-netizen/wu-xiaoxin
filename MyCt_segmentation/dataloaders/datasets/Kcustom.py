from __future__ import print_function, division
import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import KFold
import json
from typing import List, Tuple, Dict, Any

# 假设这些模块路径正确
from mypath import Path
from dataloaders import custom_transforms as tr
from dataloaders.utils1 import encode_segmap


class CUSTOMSegmentation(Dataset):
    """
    一个灵活的分割数据集类，通过索引列表加载指定的数据子集。
    此版本不进行随机数据增强，仅进行固定的尺寸调整和标准化。
    """
    NUM_CLASSES = 3  # 背景0, 血管1, 导管2

    def __init__(self, args: Any, base_dir: str, indices: List[int], split_type: str = 'train'):
        super().__init__()
        self._base_dir = base_dir
        self._image_dir = os.path.join(self._base_dir, 'images', 'train')
        self._mask_dir = os.path.join(self._base_dir, 'masks', 'train')
        self.args = args
        self.split_type = split_type  # 保留以备未来可能需要区分

        all_images = sorted([f for f in os.listdir(self._image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.images = [os.path.join(self._image_dir, all_images[i]) for i in indices]
        self.masks = [os.path.join(self._mask_dir, all_images[i]) for i in indices]

        print(f"初始化一个 '{split_type}' 数据集，包含 {len(self.images)} 张图像。")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        _img = Image.open(self.images[index]).convert('RGB')
        _mask_np = np.array(Image.open(self.masks[index]))

        # 假设 encode_segmap 返回一个单通道的类别索引 NumPy 数组
        _mask_encoded = encode_segmap(_mask_np)

        # 确保数据类型为 uint8 以便转换为 PIL Image
        _mask = Image.fromarray(_mask_encoded.astype(np.uint8))

        sample = {'image': _img, 'label': _mask}

        # 因为训练集和验证集处理方式相同，可以直接调用同一个 transform 方法
        return self.transform(sample)

    def transform(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        统一的数据变换流程，适用于训练集和验证集。
        """
        composed_transforms = transforms.Compose([
            # 步骤1: 将图像和标签都统一缩放到 crop_size
            tr.FixedResize(size=self.args.crop_size),

            # 步骤2: 对图像进行标准化
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),

            # 步骤3: 转换为 PyTorch Tensor
            tr.ToTensor()
        ])
        return composed_transforms(sample)

def create_kfold_datasets(args: Any, base_dir: str, n_splits: int = 5, random_state: int = 42) -> List[
    Tuple[Dataset, Dataset]]:
    """
    为K折交叉验证创建训练和验证数据集对。
    它会生成或加载一个索引文件，以保证实验的可复现性。

    Args:
        args: 命令行参数对象。
        base_dir: 数据集根目录。
        n_splits: 折数 (K值)。
        random_state: 随机种子，用于保证每次划分一致。

    Returns:
        一个列表，其中每个元素都是一个 (train_dataset, val_dataset) 的元组。
    """
    # 假设所有数据都在 'train' 子目录中
    train_images_dir = os.path.join(base_dir, 'images', 'train')

    # 获取所有图像文件名，这将决定总样本数
    all_image_files = sorted(
        [f for f in os.listdir(train_images_dir) if os.path.isfile(os.path.join(train_images_dir, f))])
    num_samples = len(all_image_files)

    if num_samples == 0:
        raise FileNotFoundError(f"在目录 '{train_images_dir}' 中没有找到任何图像文件。")

    all_indices = np.arange(num_samples)
    kfold_indices_file = os.path.join(base_dir, f'kfold_indices_{n_splits}_splits_seed_{random_state}.json')

    kfold_datasets = []

    if os.path.exists(kfold_indices_file):
        print(f"信息: 从 '{kfold_indices_file}' 加载已存在的K折划分索引。")
        with open(kfold_indices_file, 'r') as f:
            kfold_indices = json.load(f)
    else:
        print(f"信息: 正在为 {num_samples} 个样本生成新的 {n_splits} 折交叉验证索引...")
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        kfold_indices = []
        for train_index, val_index in kf.split(all_indices):
            kfold_indices.append((train_index.tolist(), val_index.tolist()))

        with open(kfold_indices_file, 'w') as f:
            json.dump(kfold_indices, f, indent=4)
        print(f"信息: 新的K折划分索引已保存到 '{kfold_indices_file}'。")

    print("-" * 30)
    for i, (train_indices, val_indices) in enumerate(kfold_indices):
        print(f"创建第 {i + 1}/{n_splits} 折的数据集...")
        train_dataset = CUSTOMSegmentation(args, base_dir, indices=train_indices, split_type='train')
        val_dataset = CUSTOMSegmentation(args, base_dir, indices=val_indices, split_type='val')
        kfold_datasets.append((train_dataset, val_dataset))
        print("-" * 30)

    return kfold_datasets