import argparse
import os
import numpy as np
from tqdm import tqdm
import torch  # 确保 torch 已导入

from mypath import Path
from dataloaders import make_data_loader
# from modeling.unet import *
from utils.loss1 import SegmentationLosses
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator
from torch.cuda.amp import GradScaler, autocast
class Trainer(object):
    def __init__(self, args):
        self.args = args
        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()
        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(args,
                                                                                             val_batch_size=args.test_batch_size,
                                                                                             **kwargs)
        # 初始化混合精度训练的 GradScaler
        self.scaler = GradScaler() if args.mixed_precision else None

        # Define network
        # model = Unet(n_channels=3, n_classes=3)
        train_params = [{'params': model.parameters(), 'lr': args.lr}]

        # Define Optimizer
        optimizer = torch.optim.SGD(train_params, momentum=args.momentum,
                                    weight_decay=args.weight_decay, nesterov=args.nesterov)

        # Define Criterion
        if args.use_balanced_weights:
            classes_weights_path = os.path.join(Path.db_root_dir(args.dataset), args.dataset + '_classes_weights.npy')
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight = calculate_weigths_labels(args.dataset, self.train_loader, self.nclass)
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None
        self.criterion = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode=args.loss_type)
        self.model, self.optimizer = model, optimizer

        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)
        # Define lr scheduler
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                      args.epochs, len(self.train_loader))

        # Using cuda
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            self.model = self.model.cuda()

        # Resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

        # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0

    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.train_loader)
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
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

            if i % (num_img_tr // 10) == 0:
                global_step = i + num_img_tr * epoch
                self.summary.visualize_image(self.writer, self.args.dataset, image, target, output, global_step)

        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, (i + 1) * self.args.batch_size))
        print('Loss: %.3f' % train_loss)

        current_dir = os.path.dirname(os.path.abspath(__file__))
        loss_save_dir = os.path.join(current_dir, 'curve')
        os.makedirs(loss_save_dir, exist_ok=True)
        loss_save_path = os.path.join(loss_save_dir, 'loss.txt')
        with open(loss_save_path, mode='a') as file_save:
            file_save.write(f'\nepoch:{epoch} {self.args.loss_type} loss:{train_loss}')

    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)
            loss = self.criterion(output, target)
            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            self.evaluator.add_batch(target, pred)

        # 1. 计算所有指标
        Acc = self.evaluator.Pixel_Accuracy() * 100
        class_accuracies, mPA = self.evaluator.Pixel_Accuracy_Class()
        iou_per_class, mIoU = self.evaluator.IoU()
        precision_per_class, mPrecision = self.evaluator.Precision()
        recall_per_class, mRecall = self.evaluator.Recall()
        dice_per_class, mDice = self.evaluator.Dice()

        # 2. 将指标提取并格式化
        # Overall metrics
        mPA *= 100
        mIoU *= 100
        mPrecision *= 100
        mRecall *= 100
        mDice *= 100

        # Per-class metrics
        class_names = ['Background', 'BloodVessel', 'Catheter']
        metrics_per_class = {}
        for i, name in enumerate(class_names):
            metrics_per_class[name] = {
                'ACC': round(class_accuracies[i] * 100, 2) if not np.isnan(class_accuracies[i]) else 0.0,
                'Precision': round(precision_per_class[i] * 100, 2) if not np.isnan(precision_per_class[i]) else 0.0,
                'Recall': round(recall_per_class[i] * 100, 2) if not np.isnan(recall_per_class[i]) else 0.0,
                'IoU': round(iou_per_class[i] * 100, 2) if not np.isnan(iou_per_class[i]) else 0.0,
                'Dice': round(dice_per_class[i] * 100, 2) if not np.isnan(dice_per_class[i]) else 0.0,
            }

        # 3. 将指标写入TensorBoard日志
        self.writer.add_scalar('val/loss', test_loss, epoch)
        self.writer.add_scalar('val/overall_acc', Acc, epoch)
        self.writer.add_scalar('val/mPA', mPA, epoch)
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        self.writer.add_scalar('val/mPrecision', mPrecision, epoch)
        self.writer.add_scalar('val/mRecall', mRecall, epoch)
        self.writer.add_scalar('val/mDice', mDice, epoch)

        for name, metrics in metrics_per_class.items():
            for metric_name, value in metrics.items():
                self.writer.add_scalar(f'val/{name}_{metric_name}', value, epoch)

        # 4. 在控制台打印指标
        print("\n" + "=" * 80)
        print(f"Validation Results for Epoch: {epoch}")
        print("=" * 80)
        print("Overall Metrics:")
        print(f"  - Pixel Accuracy (Acc): {Acc:.2f}%")
        print(f"  - Mean Pixel Accuracy (mPA): {mPA:.2f}%")
        print(f"  - Mean IoU (mIoU): {mIoU:.2f}%")
        print(f"  - Mean Precision: {mPrecision:.2f}%")
        print(f"  - Mean Recall: {mRecall:.2f}%")
        print(f"  - Mean Dice: {mDice:.2f}%")
        print(f"  - Validation Loss: {test_loss:.4f}")
        print("-" * 80)
        print("Per-Class Metrics:")
        header = f"{'Class':<15} | {'ACC':<10} | {'Precision':<10} | {'Recall':<10} | {'IoU':<10} | {'Dice':<10}"
        print(header)
        print("-" * len(header))
        for i, name in enumerate(class_names):
            metrics = metrics_per_class[name]
            print(
                f"{f'{i}: {name}':<15} | {metrics['ACC']:<10.2f} | {metrics['Precision']:<10.2f} | {metrics['Recall']:<10.2f} | {metrics['IoU']:<10.2f} | {metrics['Dice']:<10.2f}")
        print("=" * 80 + "\n")

        # 5. 保存最佳模型
        new_pred = mIoU
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict() if hasattr(self.model,
                                                                        'module') else self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)


def main():
    parser = argparse.ArgumentParser(description="PyTorch UNet Training")

    # Dataset and paths
    parser.add_argument('--dataset', type=str, default='custom', choices=['pascal', 'coco', 'cityscapes', 'custom'],
                        help='dataset name')
    parser.add_argument('--checkname', type=str, default=None, help='set the checkpoint name')

    # Dataloader arguments
    parser.add_argument('--workers', type=int, default=0, metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=512, help='base image size')
    parser.add_argument('--crop-size', type=int, default=512, help='crop image size')

    # Training hyper-parameters
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0, metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                        help='input batch size for testing (default: auto)')

    # Loss and weights
    parser.add_argument('--loss-type', type=str, default='focal', choices=['ce', 'focal'],
                        help='loss func type (default: ce)')
    parser.add_argument('--use-balanced-weights', action='store_true', default=False,
                        help='whether to use balanced weights')

    # Optimizer parameters
    parser.add_argument('--lr', type=float, default=None, metavar='LR', help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='poly', choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4, metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False, help='whether use nesterov (default: False)')

    # CUDA, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

    # Finetuning and resuming
    parser.add_argument('--resume', type=str, default=None, help='put the path to resuming file if needed')
    parser.add_argument('--ft', action='store_true', default=False, help='finetuning on a different dataset')

    # Evaluation options
    parser.add_argument('--eval-interval', type=int, default=1, help='evaluation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False, help='skip validation during training')

    # Mixed precision
    parser.add_argument('--mixed-precision', action='store_true', default=True, help='use mixed precision training')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    # Set up default hyper-parameters if not provided
    if args.epochs is None:
        epoches = {'coco': 30, 'cityscapes': 200, 'pascal': 50, 'custom': 100}
        args.epochs = epoches.get(args.dataset.lower(), 100)

    if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids)

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    if args.lr is None:
        lrs = {'coco': 0.1, 'cityscapes': 0.01, 'pascal': 0.007, 'custom': 0.01}
        # A common practice is to scale LR with batch size
        args.lr = lrs.get(args.dataset.lower(), 0.01) / (4 * len(args.gpu_ids)) * args.batch_size

    if args.checkname is None:
        args.checkname = f'modle2-{args.dataset}'

    print("-------------------- Configuration --------------------")
    for arg in vars(args):
        print(f"{str(arg):<20}: {str(getattr(args, arg))}")
    print("-----------------------------------------------------")

    torch.manual_seed(args.seed)
    trainer = Trainer(args)

    print(f"\nStarting Training for {trainer.args.epochs} epochs on {'CUDA' if args.cuda else 'CPU'}")
    print(f"Start Epoch: {trainer.args.start_epoch}")
    print(f"Total Epochs: {trainer.args.epochs}")

    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        if not trainer.args.no_val and (epoch + 1) % args.eval_interval == 0:
            trainer.validation(epoch)

    trainer.writer.close()
    print("\nTraining finished.")


if __name__ == "__main__":
    main()