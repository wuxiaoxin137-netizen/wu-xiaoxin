# demo.py
import argparse
import os
import numpy as np
import time
import torch
# from modeling.unet import *

from dataloaders import custom_transforms as tr
from PIL import Image
from torchvision import transforms
from dataloaders.utils import *
from dataloaders.utils1 import decode_seg_map_sequence
from mypath import Path
from utils.metrics_Specificity import Evaluator  # 确保 Evaluator 类已导入
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


def encode_segmap(mask):
    """将 RGB 标签图像编码为单通道类别索引
    Args:
        mask (np.ndarray): 原始 RGB 标签图像 (H, W, 3)
    Returns:
        np.ndarray: 单通道类别索引图像 (H, W)
    """
    mask = mask.astype(int)
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)

    # 根据 RGB 值将像素映射为类别索引
    for ii, label in enumerate(get_custom_labels()):
        label_mask[np.all(mask == label, axis=-1)] = ii

    return label_mask.astype(int)


def decode_segmap(label_mask, dataset, plot=False):
    """解码单通道标签图像为 RGB 图像
    Args:
        label_mask (np.ndarray): 单通道类别索引图像 (H, W)
        dataset (str): 数据集名称（这里是 custom）
        plot (bool): 是否展示解码后的图像
    Returns:
        np.ndarray: RGB 图像 (H, W, 3)
    """
    if dataset == 'pascal' or dataset == 'coco':
        n_classes = 21
        label_colours = get_pascal_labels()
    elif dataset == 'cityscapes':
        n_classes = 19
        label_colours = get_cityscapes_labels()
    elif dataset == 'custom':
        n_classes = 3
        label_colours = get_custom_labels()
    else:
        raise NotImplementedError

    # 初始化 RGB 通道
    r = np.zeros_like(label_mask, dtype=np.uint8)
    g = np.zeros_like(label_mask, dtype=np.uint8)
    b = np.zeros_like(label_mask, dtype=np.uint8)

    # 根据类别索引填充颜色
    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]

    # 合成 RGB 图像
    rgb = np.stack([r, g, b], axis=2)

    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb


def decode_seg_map_sequence(label_masks, dataset='custom'):
    """将一批标签图像解码为 RGB 图像序列"""
    rgb_masks = []
    for label_mask in label_masks:
        rgb_mask = decode_segmap(label_mask, dataset)
        rgb_masks.append(rgb_mask)
    rgb_masks = torch.from_numpy(np.array(rgb_masks).transpose([0, 3, 1, 2]))
    return rgb_masks


def get_cityscapes_labels():
    """Cityscapes 数据集的颜色映射"""
    return np.array([
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32]])


def get_pascal_labels():
    """Pascal VOC 数据集的颜色映射"""
    return np.asarray([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                       [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                       [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                       [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                       [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                       [0, 64, 128]])


def get_custom_labels():
    """自定义数据集的颜色映射"""
    return np.array([
        [0, 0, 0],  # 背景
        [255, 255, 255],  # 血管
        [255, 0, 0]  # 导管
    ])


class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_names = os.listdir(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image_path = os.path.join(self.image_dir, image_name)
        mask_path = os.path.join(self.mask_dir, image_name)

        image = Image.open(image_path).convert('RGB')
        target = Image.open(mask_path).convert('RGB')

        sample = {'image': image, 'label': target}

        if self.transform:
            sample = self.transform(sample)
        target = np.array(sample['label']).astype(np.int32)  # shape:(h,w,3)
        target = encode_segmap(target)  # shape(h,w)
        target = np.array(target)  # shape(h,w)

        sample['label'] = target
        return sample


def compute_mean_std(image_dir):
    images = []
    for name in os.listdir(image_dir):
        image = np.array(Image.open(os.path.join(image_dir, name)).convert('RGB')) / 255.0
        images.append(image)

    images = np.stack(images, axis=0)
    mean = np.mean(images, axis=(0, 1, 2))
    std = np.std(images, axis=(0, 1, 2))
    return mean, std


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="PyTorch Model Test")
    # 测试时需要替换以下两个的路径
    parser.add_argument('--in-path', type=str, required=False, default=Path.db_root_dir('custom') + "/images/test",
                        help='image to test2')
    parser.add_argument('--mask-path', type=str, required=False, default=Path.db_root_dir('custom') + "/masks/test",
                        help='mask to test2')
    parser.add_argument('--ckpt', type=str,
                        default=os.path.join(current_dir, 'run/custom/'),
                        help='saved model')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--dataset', type=str, default='custom',
                        choices=['pascal', 'coco', 'cityscapes', 'invoice', 'custom'],
                        help='dataset name (default: custom)')
    parser.add_argument('--crop-size', type=int, default=512,
                        help='crop image size')
    # 代码优化，numclass自适应识别类别数

    parser.add_argument('--num_classes', type=int, default=3,
                        help='class')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--test-batch-size', type=int, default=2,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    model_s_time = time.time()
    model = UNet(n_channels=3, n_classes=3)
    ckpt = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'])
    model = model.cuda() if args.cuda else model
    model_u_time = time.time()
    model_load_time = model_u_time - model_s_time
    print("model load time is {}".format(model_load_time))

    mean, std = compute_mean_std(args.in_path)
    composed_transforms = transforms.Compose([
        tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        # tr.Normalize(mean=mean, std=std),
        tr.ToTensor()])

    test_dataset = CustomDataset(image_dir=args.in_path, mask_dir=args.mask_path, transform=composed_transforms)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)

    evaluator = Evaluator(args.num_classes)
    evaluator.reset()

    model.eval()  # Set the model to evaluation mode

    for i, sample in enumerate(test_loader):
        image = sample['image']
        target = sample['label']

        if args.cuda:
            image, target = image.cuda(), target.cuda()

        with torch.no_grad():
            output = model(image)
        pred = output.data.cpu().numpy()
        pred = np.argmax(pred, axis=1)

        # 保存预测图像
        for idx_in_batch in range(len(target)):
            # 获取当前脚本的目录
            current_dir = os.path.dirname(os.path.abspath(__file__))
            save_dir = os.path.join(current_dir, 'saved')
            os.makedirs(save_dir, exist_ok=True)
            name = test_dataset.image_names[i * args.test_batch_size + idx_in_batch]
            # 将 grid_image 转换为 numpy 数组
            ndarr = decode_seg_map_sequence(pred[idx_in_batch].reshape(1, pred.shape[1], pred.shape[2]),
                                            dataset='custom')
            ndarr = ndarr.detach().cpu().numpy()  # Add this line to convert tensor to numpy array
            ndarr = np.clip(ndarr, 0, 255).astype(np.uint8)
            ndarr = ndarr.squeeze(0)  # Add this line to remove the first dimension
            if len(ndarr.shape) == 3:
                ndarr = ndarr.transpose(1, 2, 0)
            # ndarr = ndarr[:, :, 0] # If it's still RGB after squeeze, just take one channel
            image_path = os.path.join(save_dir, '{}.png'.format(name.split('.')[0]))
            im = Image.fromarray(ndarr)
            im.save(image_path)

        # Add batch sample into evaluator
        for t, p in zip(target, pred):
            evaluator.add_batch(t.cpu().numpy(), p)
    # 计算指标
    Acc = evaluator.Pixel_Accuracy() * 100
    Acc_class = evaluator.Pixel_Accuracy_Class() * 100
    mIoU = evaluator.Mean_Intersection_over_Union() * 100

    # Precision, Recall, IoU, Dice, Specificity for each class
    precision, _ = evaluator.Precision()
    recall, _ = evaluator.Recall()
    iou, _ = evaluator.IoU()
    dice, mean_dice = evaluator.Dice()  # 获取所有类别的Dice系数和平均Dice系数
    specificity, mean_specificity = evaluator.Specificity()  # **添加：获取所有类别的特异性系数和平均特异性系数**

    # Extract specific class metrics
    background_precision = round(precision[0] * 100, 2)
    background_recall = round(recall[0] * 100, 2)
    background_iou = round(iou[0] * 100, 2)
    background_dice = round(dice[0] * 100, 2)
    background_specificity = round(specificity[0] * 100, 2)  # **添加：背景特异性**

    blood_vessels_precision = round(precision[1] * 100, 2)
    blood_vessels_recall = round(recall[1] * 100, 2)
    blood_vessels_iou = round(iou[1] * 100, 2)
    blood_vessels_dice = round(dice[1] * 100, 2)
    blood_vessels_specificity = round(specificity[1] * 100, 2)  # **添加：血管特异性**

    catheter_precision = round(precision[2] * 100, 2)
    catheter_recall = round(recall[2] * 100, 2)
    catheter_iou = round(iou[2] * 100, 2)
    catheter_dice = round(dice[2] * 100, 2)
    catheter_specificity = round(specificity[2] * 100, 2)  # **添加：导管特异性**

    mean_dice = round(mean_dice * 100, 2)  # 格式化平均dice系数
    mean_specificity = round(mean_specificity * 100, 2)  # **添加：格式化平均特异性系数**

    # Print metrics
    print('Test Results:')
    # **修改：在打印中加入平均特异性**
    print("Acc: {:.2f}%, Acc_class: {:.2f}%, mIoU: {:.2f}%, mean Dice:{:.2f}%, mean Specificity:{:.2f}%".format(Acc,
                                                                                                                Acc_class,
                                                                                                                mIoU,
                                                                                                                mean_dice,
                                                                                                                mean_specificity))

    # **修改：在每个类别的打印中加入特异性**
    print(
        "Background - Precision: {:.2f}%, Recall: {:.2f}%, IoU: {:.2f}%, Dice: {:.2f}%, Specificity: {:.2f}%".format(
            background_precision,
            background_recall,
            background_iou,
            background_dice,
            background_specificity))
    print(
        "Blood Vessels - Precision: {:.2f}%, Recall: {:.2f}%, IoU: {:.2f}%, Dice: {:.2f}%, Specificity: {:.2f}%".format(
            blood_vessels_precision,
            blood_vessels_recall,
            blood_vessels_iou,
            blood_vessels_dice,
            blood_vessels_specificity))
    print("Catheter - Precision: {:.2f}%, Recall: {:.2f}%, IoU: {:.2f}%, Dice: {:.2f}%, Specificity: {:.2f}%".format(
        catheter_precision,
        catheter_recall,
        catheter_iou,
        catheter_dice,
        catheter_specificity))


if __name__ == "__main__":
    main()