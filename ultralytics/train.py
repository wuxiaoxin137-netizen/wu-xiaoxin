import warnings, os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"    # 代表用cpu训练 不推荐！没意义！ 而且有些模块不能在cpu上跑
# os.environ["CUDA_VISIBLE_DEVICES"]="0"     # 代表用第一张卡进行训练  0：第一张卡 1：第二张卡
# 多卡训练参考<YOLOV8V10配置文件.md>下方常见错误和解决方案
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('/home/ultralytics-main/ultralytics/cfg/models/v8/yolov8.yaml')
    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(data='/home/ultralytics-main/dataset/data.yaml',
                cache=True,
                imgsz=512,
                epochs=200,
                batch=16,
                close_mosaic=10,
                workers=0, # Windows下出现莫名其妙卡主的情况可以尝试把workers设置为0
                optimizer='SGD', # using SGD
                split='val',
                # close_mosaic=1,  # 关闭 Mosaic 数据增强
                # fliplr=0,  # 关闭左右翻转
                # flipud=0,  # 关闭上下翻转
                # degrees=0,  # 关闭旋转
                # translate=0,  # 关闭平移
                # scale=1.0,  # 关闭缩放
                # shear=0,  # 关闭剪切
                # perspective=0,  # 关闭透视变换
                # mixup=0,  # 关闭 MixUp
                # copy_paste=0,  # 关闭 CopyPaste
                # hsv_h=0,  # 关闭 HSV 色调调整
                # hsv_s=0,  # 关闭 HSV 饱和度调整
                # hsv_v=0,  # 关闭 HSV 明度调整
                device='0', # 指定显卡和多卡训练参考<YOLOV8V10配置文件.md>下方常见错误和解决方案
                # patience=0, # set 0 to close earlystop.
                # resume=True, # 断点续训,YOLO初始化时选择last.pt,例如YOLO('last.pt')
                # amp=False, # close amp
                # fraction=0.2,
                iou=0.7,  # 设置 IOU 阈值为 0.5
                project='runs/train',
                name='exp',
                )