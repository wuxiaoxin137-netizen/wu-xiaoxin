import argparse
import os
import numpy as np
from tqdm import tqdm
import json
from typing import List, Tuple, Dict, Any
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import KFold

# --- 假设以下模块和文件路径都存在且正确 ---
from mypath import Path

# --- 模型、损失函数、工具等导入 ---
# 请根据您的项目结构和模型名称，修改这里的导入语句
# from modeling.unet import *
from dataloaders import custom_transforms as tr
from dataloaders.utils1 import encode_segmap
from utils.dice_loss import SegmentationLosses  # 假设使用dice_bce_loss
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
# 确保您导入的Evaluator类包含了Specificity方法
from utils.metrics_Specificity import Evaluator
from torch.cuda.amp import GradScaler, autocast


# ==============================================================================
# 1. K折交叉验证的数据集定义
# ==============================================================================

class CUSTOMSegmentation(Dataset):
    """
    一个灵活的分割数据集类，通过索引列表加载指定的数据子集。
    此版本不进行随机数据增强，仅进行固定的尺寸调整和标准化。
    """
    NUM_CLASSES = 2  # 背景0, 血管1

    def __init__(self, args: Any, base_dir: str, indices: List[int], split_type: str = 'train'):
        super().__init__()
        self._base_dir = base_dir
        self._image_dir = os.path.join(self._base_dir, 'images', 'train')
        self._mask_dir = os.path.join(self._base_dir, 'masks', 'train')
        self.args = args
        self.split_type = split_type

        all_images = sorted([f for f in os.listdir(self._image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.images = [os.path.join(self._image_dir, all_images[i]) for i in indices]
        self.masks = [os.path.join(self._mask_dir, all_images[i]) for i in indices]

        # print(f"初始化一个 '{split_type}' 数据集，包含 {len(self.images)} 张图像。") # 注释掉以减少冗余输出

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        _img = Image.open(self.images[index]).convert('RGB')
        _mask_np = np.array(Image.open(self.masks[index]))
        _mask_encoded = encode_segmap(_mask_np)
        _mask = Image.fromarray(_mask_encoded.astype(np.uint8))
        sample = {'image': _img, 'label': _mask}
        return self.transform(sample)

    def transform(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        composed_transforms = transforms.Compose([
            tr.FixedResize(size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()
        ])
        return composed_transforms(sample)


def create_kfold_datasets(args: Any, base_dir: str, n_splits: int, random_state: int) -> List[Tuple[Dataset, Dataset]]:
    """为K折交叉验证创建训练和验证数据集对。"""
    train_images_dir = os.path.join(base_dir, 'images', 'train')
    all_image_files = sorted([f for f in os.listdir(train_images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    num_samples = len(all_image_files)
    if num_samples == 0:
        raise FileNotFoundError(f"目录 '{train_images_dir}' 中无图像。")

    all_indices = np.arange(num_samples)
    kfold_indices_file = os.path.join(base_dir, f'kfold_indices_{n_splits}_splits_seed_{random_state}.json')
    if os.path.exists(kfold_indices_file):
        print(f"从文件加载已有的K折划分: {kfold_indices_file}")
        with open(kfold_indices_file, 'r') as f:
            kfold_indices = json.load(f)
    else:
        print(f"创建新的K折划分 (n={n_splits}, seed={random_state}) 并保存...")
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        kfold_indices = [(train_idx.tolist(), val_idx.tolist()) for train_idx, val_idx in kf.split(all_indices)]
        with open(kfold_indices_file, 'w') as f:
            json.dump(kfold_indices, f, indent=4)
        print(f"K折划分已保存到: {kfold_indices_file}")

    kfold_datasets = []
    for i, (train_indices, val_indices) in enumerate(kfold_indices):
        train_dataset = CUSTOMSegmentation(args, base_dir, indices=train_indices, split_type='train')
        val_dataset = CUSTOMSegmentation(args, base_dir, indices=val_indices, split_type='val')
        print(f"Fold {i + 1}: {len(train_indices)} train images, {len(val_indices)} validation images.")
        kfold_datasets.append((train_dataset, val_dataset))
    return kfold_datasets


# ==============================================================================
# 2. Trainer 类的定义
# ==============================================================================

class Trainer(object):
    def __init__(self, args: argparse.Namespace, train_loader: DataLoader, val_loader: DataLoader, fold_num: int):
        self.args = args
        self.fold_num = fold_num
        # 为每一折创建独立的保存目录
        args.checkname = f"{args.checkname_prefix}/fold_{fold_num}"
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()
        self.train_loader, self.val_loader = train_loader, val_loader
        self.nclass = self.train_loader.dataset.NUM_CLASSES
        self.scaler = GradScaler() if args.mixed_precision else None

        # 模型定义
        # model = unet(n_channels=3, n_classes=self.nclass)
        # 优化器定义
        train_params = [{'params': model.parameters(), 'lr': args.lr}]
        optimizer = torch.optim.SGD(train_params, momentum=args.momentum, weight_decay=args.weight_decay,
                                    nesterov=args.nesterov)

        # 损失函数定义
        weight = None
        if args.use_balanced_weights:
            classes_weights_path = os.path.join(Path.db_root_dir(args.dataset), args.dataset + '_classes_weights.npy')
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight = calculate_weigths_labels(args.dataset, self.train_loader, self.nclass)
            weight = torch.from_numpy(weight.astype(np.float32))
        self.criterion = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode=args.loss_type)

        self.model, self.optimizer = model, optimizer
        self.evaluator = Evaluator(self.nclass)
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr, args.epochs, len(self.train_loader))

        # 使用GPU
        if args.cuda:
            # self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids).cuda()
            # 只有当 GPU 数量大于 1 时才使用 DataParallel
            if len(self.args.gpu_ids) > 1:
                self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids).cuda()
            else:
                self.model = self.model.cuda() # 单 GPU 直接转 cuda 即可，跳过检测

        self.best_pred = 0.0

    def training(self, epoch: int):
        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.train_loader, desc=f'Fold {self.fold_num} | Train Epoch {epoch + 1}/{self.args.epochs}')
        num_img_tr = len(self.train_loader)
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()

            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()

            with autocast(enabled=self.args.mixed_precision):
                output = self.model(image)
                loss = self.criterion(output, target)

            if self.args.mixed_precision:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            train_loss += loss.item()
            tbar.set_description(f'Fold {self.fold_num} | Train Epoch {epoch + 1} | Loss: {train_loss / (i + 1):.3f}')

            global_step = i + num_img_tr * epoch
            self.writer.add_scalar(f'Fold_{self.fold_num}/train/total_loss_iter', loss.item(), global_step)

            if i % (num_img_tr // 10) == 0:
                self.summary.visualize_image(self.writer, self.args.dataset, image, target, output, global_step)

        self.writer.add_scalar(f'Fold_{self.fold_num}/train/total_loss_epoch', train_loss / num_img_tr, epoch)
        print(f"  [Fold {self.fold_num}, Epoch: {epoch + 1}] Training Loss: {train_loss / num_img_tr:.3f}")

        loss_save_path = os.path.join(self.saver.experiment_dir, 'loss.txt')
        with open(loss_save_path, mode='a') as file_save:
            file_save.write(f'epoch:{epoch + 1} loss:{train_loss:.4f}\n')

    def validation(self, epoch: int) -> Tuple[float, Dict[str, float]]:
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc=f'Fold {self.fold_num} | Validation Epoch {epoch + 1}')
        test_loss = 0.0

        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()

            with torch.no_grad():
                with autocast(enabled=self.args.mixed_precision):
                    output = self.model(image)
                    loss = self.criterion(output, target)

            test_loss += loss.item()
            tbar.set_description(f'Fold {self.fold_num} | Val Epoch {epoch + 1} | Loss: {test_loss / (i + 1):.3f}')
            pred = torch.argmax(output, dim=1).cpu().numpy()
            target = target.cpu().numpy()
            self.evaluator.add_batch(target, pred)

        # 1. 计算所有指标
        Acc = self.evaluator.Pixel_Accuracy()
        _, mPA = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        precision, mean_precision = self.evaluator.Precision()
        recall, mean_recall = self.evaluator.Recall()
        specificity, mean_specificity = self.evaluator.Specificity()
        iou, _ = self.evaluator.IoU()
        dice, mean_dice = self.evaluator.Dice()
        class_names = ['Background', 'Blood_Vessel']

        # 2. 将所有指标打包到字典中
        metrics_dict = {
            'Overall_Acc': Acc,
            'mPA': mPA,
            'mIoU': mIoU,
            'Mean_Dice': mean_dice,
            'Mean_Precision': mean_precision,
            'Mean_Recall': mean_recall,
            'Mean_Specificity': mean_specificity
        }
        for i, name in enumerate(class_names):
            metrics_dict[f'Class_{i}_{name}/Dice'] = dice[i]
            metrics_dict[f'Class_{i}_{name}/IoU'] = iou[i]
            metrics_dict[f'Class_{i}_{name}/Precision'] = precision[i]
            metrics_dict[f'Class_{i}_{name}/Recall'] = recall[i]
            metrics_dict[f'Class_{i}_{name}/Specificity'] = specificity[i]

        # 3. TensorBoard 日志记录
        self.writer.add_scalar(f'Fold_{self.fold_num}/val/total_loss_epoch', test_loss / len(self.val_loader), epoch)
        for key, value in metrics_dict.items():
            self.writer.add_scalar(f'Fold_{self.fold_num}/val/{key}', value, epoch)

        # 4. 控制台打印
        print(f'\n  [Fold {self.fold_num}, Epoch: {epoch + 1}] Validation Results:')
        print(f"    Overall Acc: {Acc:.4f}, mPA: {mPA:.4f}, mIoU: {mIoU:.4f}, Mean Dice: {mean_dice:.4f}")
        print(
            f"    Mean Precision: {mean_precision:.4f}, Mean Recall: {mean_recall:.4f}, Mean Specificity: {mean_specificity:.4f}")
        print("\n    --- Per-Class Metrics ---")
        for i, name in enumerate(class_names):
            print(f"    Class {i} ({name}):")
            print(f"      - Precision: {precision[i]:.4f}, Recall: {recall[i]:.4f}, Specificity: {specificity[i]:.4f}")
            print(f"      - IoU: {iou[i]:.4f}, Dice: {dice[i]:.4f}")
        print("    -------------------------\n")

        # 5. 保存最佳模型 (以Mean Dice为标准)
        current_metric = mean_dice
        if current_metric > self.best_pred:
            self.best_pred = current_metric
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict() if hasattr(self.model,
                                                                        'module') else self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best=True)

        # 6. 修改返回值
        return self.best_pred, metrics_dict


# ==============================================================================
# 3. 主函数 (K折交叉验证控制器)
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="PyTorch Model K-Fold Cross-Validation")
    parser.add_argument('--dataset', type=str, default='custom', help='dataset name')
    parser.add_argument('--workers', type=int, default=0, help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=512, help='base image size')
    parser.add_argument('--crop-size', type=int, default=512, help='crop image size')
    parser.add_argument('--loss-type', type=str, default='dice', help='loss func type')
    parser.add_argument('--epochs', type=int, default=80, help='number of epochs')
    parser.add_argument('--start_epoch', type=int, default=0, help='start epoch')
    parser.add_argument('--batch-size', type=int, default=4, help='train batch size')
    parser.add_argument('--test-batch-size', type=int, default=1, help='val batch size')
    parser.add_argument('--use-balanced-weights', action='store_true', default=True, help='use balanced weights')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--lr-scheduler', type=str, default='poly', help='lr scheduler')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='w-decay')
    parser.add_argument('--nesterov', action='store_true', default=False, help='use nesterov')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='no cuda')
    parser.add_argument('--gpu-ids', type=str, default='0', help='gpu ids')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--resume', type=str, default=None, help='resume path')
    parser.add_argument('--checkname', type=str, default='my_kfold_resnet34_experiment130', help='experiment name prefix')
    parser.add_argument('--ft', action='store_true', default=False, help='finetuning')
    parser.add_argument('--eval-interval', type=int, default=1, help='evaluation interval')
    parser.add_argument('--no-val', action='store_true', default=False, help='skip validation')
    parser.add_argument('--mixed-precision', action='store_true', default=True, help='use mixed precision')
    parser.add_argument('--k-splits', type=int, default=5, help='K-Fold splits')
    parser.add_argument('--k-seed', type=int, default=42, help='K-Fold random seed')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError("Argument --gpu-ids must be a comma-separated list of integers, e.g. '0,1'")

    # 固定随机种子以保证可复现性
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # 保存checkname前缀，因为后面会修改它
    args.checkname_prefix = args.checkname
    print("命令行参数:\n", args)

    # K折主循环
    base_dir = Path.db_root_dir(args.dataset)
    kfold_datasets = create_kfold_datasets(args, base_dir=base_dir, n_splits=args.k_splits, random_state=args.k_seed)

    # 用于存储所有折的最佳指标的字典
    all_folds_best_metrics = {}

    for fold, (train_dataset, val_dataset) in enumerate(kfold_datasets):
        fold_num = fold + 1
        print("\n" + "=" * 25, f"开始第 {fold_num}/{args.k_splits} 折训练", "=" * 25)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                                  pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers,
                                pin_memory=True)

        trainer = Trainer(args, train_loader, val_loader, fold_num)

        best_fold_metric = 0.0
        best_fold_metrics_dict = None

        for epoch in range(args.start_epoch, args.epochs):
            trainer.training(epoch)
            if not args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
                # 接收 validation 返回的两个值
                current_best_mean_dice, current_metrics_dict = trainer.validation(epoch)

                # 如果当前 epoch 的 Mean Dice 更好，则更新本折的最佳记录
                if current_best_mean_dice > best_fold_metric:
                    best_fold_metric = current_best_mean_dice
                    best_fold_metrics_dict = current_metrics_dict

        print(f"--- 第 {fold_num} 折训练完成，本折最佳 Mean Dice: {best_fold_metric:.4f} ---")

        # 将本折的最佳指标字典存入总的字典中
        if best_fold_metrics_dict:
            for key, value in best_fold_metrics_dict.items():
                if key not in all_folds_best_metrics:
                    all_folds_best_metrics[key] = []
                all_folds_best_metrics[key].append(value)

        trainer.writer.close()

    # 全新的、更全面的总结部分
    print("\n" + "=" * 30, "K折交叉验证最终总结", "=" * 30)

    if not all_folds_best_metrics:
        print("没有收集到任何验证结果，无法进行总结。")
        return

    final_summary = {}
    metric_keys = sorted(all_folds_best_metrics.keys())

    for key in metric_keys:
        values = all_folds_best_metrics[key]
        if len(values) < args.k_splits:
            print(f"警告: 指标 '{key}' 只收集到 {len(values)}/{args.k_splits} 折的结果。")
        mean_val = np.mean(values)
        std_val = np.std(values)
        final_summary[key] = {'mean': mean_val, 'std': std_val, 'values': values}

    # 打印和保存总结
    results_summary_path = os.path.join('run', args.checkname_prefix, 'kfold_full_summary1.txt')
    os.makedirs(os.path.dirname(results_summary_path), exist_ok=True)

    with open(results_summary_path, 'w') as f:
        f.write(f"K-Fold Cross-Validation Full Summary ({args.k_splits} splits)\n")
        f.write("=" * 60 + "\n")
        f.write(f"{'Metric':<35} | {'Mean':>10} | {'Std Dev':>10}\n")
        f.write("-" * 60 + "\n")

        print(f"\n{'Metric':<35} | {'Mean':>10} | {'Std Dev':>10}")
        print("-" * 60)

        for key in metric_keys:
            summary = final_summary[key]
            mean_str = f"{summary['mean']:.4f}"
            std_str = f"{summary['std']:.4f}"

            line = f"{key:<35} | {mean_str:>10} | {std_str:>10}"
            print(line)
            f.write(line + "\n")

        f.write("\n\n" + "=" * 60 + "\n")
        f.write("Per-Fold Best Metrics Data (based on best Mean Dice epoch)\n")
        f.write("=" * 60 + "\n")
        for key in metric_keys:
            values_str = ", ".join([f"{v:.4f}" for v in final_summary[key]['values']])
            f.write(f"{key:<35}: [{values_str}]\n")

    print("-" * 60)
    print(f"\n交叉验证详细总结已保存到: {results_summary_path}")


if __name__ == "__main__":
    main()