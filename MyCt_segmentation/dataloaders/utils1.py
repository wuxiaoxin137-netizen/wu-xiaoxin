import matplotlib.pyplot as plt
import numpy as np
import torch


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
        # n_classes = 2
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
