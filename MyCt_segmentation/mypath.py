import os
class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        # 获取当前文件所在的目录（即 mypath.py 的目录）
        base_dir = os.path.dirname(os.path.abspath(__file__))
        datasets_dir = os.path.join(base_dir, 'datasets')  # 假定数据集都存放在 datasets 文件夹中

        if dataset == 'pascal':
            return os.path.join(datasets_dir, 'pascal')  # pascal 数据集目录
        elif dataset == 'cityscapes':
            return os.path.join(datasets_dir, 'cityscapes')  # cityscapes 数据集目录
        elif dataset == 'coco':
            return os.path.join(datasets_dir, 'coco')  # coco 数据集目录
        elif dataset == 'custom':
            return os.path.join(datasets_dir, 'custom')  # custom 数据集目录
        else:
            raise NotImplementedError(f"Dataset {dataset} is not available.")
