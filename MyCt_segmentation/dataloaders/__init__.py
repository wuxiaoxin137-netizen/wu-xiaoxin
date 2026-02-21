# from dataloaders.datasets import cityscapes, coco, combine_dbs, pascal, custom
# from .datasets import cityscapes, coco, combine_dbs, pascal, custom

from dataloaders.datasets import cityscapes, combine_dbs, pascal, custom
from torch.utils.data import DataLoader
from mypath import Path
import os

def make_data_loader(args, val_batch_size=None, **kwargs):

    # default to args.batch_size if no test_batch_size is passed
    if val_batch_size is None:
        val_batch_size = args.batch_size

    if args.dataset == 'pascal':
        train_set = pascal.VOCSegmentation(args, split='train')
        val_set = pascal.VOCSegmentation(args, split='val')
        test_set = pascal.VOCSegmentation(args, split='test2')

        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = None

        return train_loader, val_loader, test_loader, num_class


    elif args.dataset == 'cityscapes':
        train_set = cityscapes.CityscapesSegmentation(args, split='train')
        val_set = cityscapes.CityscapesSegmentation(args, split='val')
        test_set = cityscapes.CityscapesSegmentation(args, split='test2')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader, num_class

    # elif args.dataset == 'coco':
    #     train_set = coco.COCOSegmentation(args, split='train')
    #     val_set = coco.COCOSegmentation(args, split='val')
    #     num_class = train_set.NUM_CLASSES
    #     train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
    #     val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
    #     test_loader = None
    #     return train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'custom':
        base_dir = Path.db_root_dir('custom')  # 获取数据集根目录
        train_set = custom.CUSTOMSegmentation(args, base_dir=base_dir, split='train')
        val_set = custom.CUSTOMSegmentation(args, base_dir=base_dir, split='val')
        test_set = custom.CUSTOMSegmentation(args, base_dir=base_dir, split='test')

        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        # val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        val_loader = DataLoader(val_set, batch_size=val_batch_size, shuffle=False,
                                **kwargs)  # Use test_batch_size for validation

        # test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = None

        return train_loader, val_loader, test_loader, num_class

    else:
        raise NotImplementedError

