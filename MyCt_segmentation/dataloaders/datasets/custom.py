from __future__ import print_function, division
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

from mypath import Path
# from dataloaders import custom_transforms as tr
# from .. import custom_transforms as tr
# import sys
# import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataloaders import custom_transforms as tr
from dataloaders.utils1 import encode_segmap

class CUSTOMSegmentation(Dataset):
    """
    Custom segmentation dataset without ImageSets directory.
    Assumes images and masks folders have corresponding files with the same names.
    """
    NUM_CLASSES = 3  # 根据你的数据集类别数修改

    def __init__(self, args, base_dir, split='train'):
        """
        :param base_dir: 数据集的根目录
        :param split: train/val
        """
        super().__init__()
        self._base_dir = base_dir
        self._image_dir = os.path.join(self._base_dir, 'images', split)
        self._mask_dir = os.path.join(self._base_dir, 'masks', split)
        self.split = split
        self.args = args

        # 获取所有图像和标注文件路径
        self.images = sorted([os.path.join(self._image_dir, f) for f in os.listdir(self._image_dir)
                              if os.path.isfile(os.path.join(self._image_dir, f))])
        self.masks = sorted([os.path.join(self._mask_dir, f) for f in os.listdir(self._mask_dir)
                             if os.path.isfile(os.path.join(self._mask_dir, f))])

        # 检查数量是否一致
        assert len(self.images) == len(self.masks), \
            "Images and masks folder do not contain the same number of files."

        # 检查文件名是否一一对应
        for img_path, mask_path in zip(self.images, self.masks):
            assert os.path.basename(img_path) == os.path.basename(mask_path), \
                f"Mismatch in filenames: {os.path.basename(img_path)} and {os.path.basename(mask_path)}"

        # 显示统计信息
        print(f"Found {len(self.images)} {split} images.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # 加载图像和对应的标注
        img_path = self.images[index]
        mask_path = self.masks[index]

        img = Image.open(img_path).convert('RGB')
        # mask = Image.open(mask_path)
        mask = np.array(Image.open(mask_path))  # 转换为 NumPy 数组
        # 将标签图像编码为单通道类别索引
        mask = encode_segmap(mask)

        sample = {'image': img, 'label': mask}

        if self.split == "train":
            return self.transform_tr(sample)
        elif self.split == 'val':
            return self.transform_val(sample)

    def transform_tr(self, sample):
        # 训练集数据增强
        composed_transforms = transforms.Compose([
            # tr.RandomHorizontalFlip(),
            # tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
            # tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()
        ])
        return composed_transforms(sample)

    def transform_val(self, sample):
        # 验证集数据增强
        composed_transforms = transforms.Compose([
            # tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()
        ])
        return composed_transforms(sample)


# if __name__ == '__main__':
#     from dataloaders.utils1 import decode_segmap
#
#     # from ..utils1 import decode_segmap
#     # import sys
#     # import os
#     # sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#     # from dataloaders.utils1 import decode_segmap
#
#     from torch.utils.data import DataLoader
#     import matplotlib.pyplot as plt
#     import argparse
#
#     # 模拟命令行参数
#     parser = argparse.ArgumentParser()
#     args = parser.parse_args()
#     args.base_size = 512
#     args.crop_size = 512
#
#     base_dir = Path.db_root_dir('custom')
#     # print(f"Custom dataset path: {base_dir}")
#
#     custom_train = CUSTOMSegmentation(args, base_dir=base_dir, split='train')
#     dataloader = DataLoader(custom_train, batch_size=5, shuffle=True, num_workers=0)
#
#     for ii, sample in enumerate(dataloader):
#         for jj in range(sample["image"].size()[0]):
#             img = sample['image'].numpy()
#             gt = sample['label'].numpy()
#             tmp = np.array(gt[jj]).astype(np.uint8)
#             segmap = decode_segmap(tmp, dataset='custom')
#             img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
#             img_tmp *= (0.229, 0.224, 0.225)
#             img_tmp += (0.485, 0.456, 0.406)
#             img_tmp *= 255.0
#             img_tmp = img_tmp.astype(np.uint8)
#             plt.figure()
#             plt.title('display')
#             plt.subplot(211)
#             plt.imshow(img_tmp)
#             plt.subplot(212)
#             plt.imshow(segmap)
#
#         if ii == 1:
#             break
#
#     plt.show(block=True)
