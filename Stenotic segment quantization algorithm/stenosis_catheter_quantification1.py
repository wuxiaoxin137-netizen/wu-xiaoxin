# 文件名: step1_preprocess_rois.py
# 作用: ROI提取、超分辨率放大、YOLO坐标变换模块。

import cv2
import numpy as np
import os


# ==============================================================================
# 1. 核心处理函数 (低层级)
# ==============================================================================

def _process_and_save_single_roi(
        image_full,
        bbox_to_crop,
        name_suffix,
        output_dir,
        original_img_name,
        params,
        original_yolo_bbox_with_id=None  # 可选，包含class_id和bbox
):
    """
    一个底层的辅助函数，负责处理单个ROI的扩展、裁剪、放大、填充、保存和标签转换。
    """
    # 从参数字典中解包
    mag_factor = params.get('magnification_factor', 8)
    expand_pixels = params.get('expand_pixels', 10)
    padding_pixels = params.get('padding_pixels', 0)

    if bbox_to_crop is None:
        print(f"  - 提示: 跳过 '{name_suffix}' 的处理，因为未提供边界框。")
        return

    x1_orig, y1_orig, x2_orig, y2_orig = bbox_to_crop
    H_full, W_full = image_full.shape[:2]

    # 扩展边界
    x1_exp = max(0, x1_orig - expand_pixels)
    y1_exp = max(0, y1_orig - expand_pixels)
    x2_exp = min(W_full, x2_orig + expand_pixels)
    y2_exp = min(H_full, y2_orig + expand_pixels)

    # 裁剪
    img_roi_cropped = image_full[y1_exp:y2_exp, x1_exp:x2_exp]
    if img_roi_cropped.size == 0:
        print(f"  - 警告: 裁剪后的 '{name_suffix}' ROI为空，已跳过。")
        return

    # 放大
    H_crop, W_crop = img_roi_cropped.shape[:2]
    target_W = max(1, int(W_crop * mag_factor))
    target_H = max(1, int(H_crop * mag_factor))
    img_roi_mag = cv2.resize(img_roi_cropped, (target_W, target_H), interpolation=cv2.INTER_NEAREST)

    # 填充
    if padding_pixels > 0:
        img_roi_final = cv2.copyMakeBorder(img_roi_mag, padding_pixels, padding_pixels, padding_pixels, padding_pixels,
                                           cv2.BORDER_CONSTANT, value=[0, 0, 0])
    else:
        img_roi_final = img_roi_mag

    H_final, W_final = img_roi_final.shape[:2]

    # 保存图像
    base_name = os.path.splitext(os.path.basename(original_img_name))[0]
    output_img_filename = f"{base_name}_{name_suffix}_mag{mag_factor}x_exp{expand_pixels}p_pad{padding_pixels}p.png"
    output_img_path = os.path.join(output_dir, output_img_filename)
    cv2.imwrite(output_img_path, img_roi_final)
    print(f"  - ROI图像已保存: '{output_img_filename}'")

    # 如果是狭窄ROI，计算并保存新标签
    if original_yolo_bbox_with_id:
        class_id = original_yolo_bbox_with_id['class_id']
        x1_yolo, y1_yolo, x2_yolo, y2_yolo = original_yolo_bbox_with_id['bbox']

        # 计算YOLO框在扩展后、放大前、带填充的最终图像中的新坐标
        yolo_x1_in_exp = x1_yolo - x1_exp
        yolo_y1_in_exp = y1_yolo - y1_exp
        w_yolo_orig = x2_yolo - x1_yolo
        h_yolo_orig = y2_yolo - y1_yolo

        x_center_final = (yolo_x1_in_exp + w_yolo_orig / 2) * mag_factor + padding_pixels
        y_center_final = (yolo_y1_in_exp + h_yolo_orig / 2) * mag_factor + padding_pixels
        w_final = w_yolo_orig * mag_factor
        h_final = h_yolo_orig * mag_factor

        # 归一化
        cx_norm = x_center_final / W_final
        cy_norm = y_center_final / H_final
        w_norm = w_final / W_final
        h_norm = h_final / H_final

        output_label_filename = f"{os.path.splitext(output_img_filename)[0]}.txt"
        output_label_path = os.path.join(output_dir, output_label_filename)

        with open(output_label_path, 'w') as f:
            f.write(f"{class_id} {cx_norm:.6f} {cy_norm:.6f} {w_norm:.6f} {h_norm:.6f}\n")
        print(f"  - YOLO标签已保存: '{output_label_filename}'")


# ==============================================================================
# 2. 主调用函数 (高层级) - 这是未来主程序要调用的函数
# ==============================================================================

def run_step1_preprocess_rois(
        original_image_path,
        yolo_label_path,
        output_folder,
        params
):
    """
    执行ROI预处理流程：从一张原始图和YOLO标签中，提取、放大并保存导管和所有狭窄ROI。
    所有处理参数都从 params 字典中获取。
    """
    img_orig = cv2.imread(original_image_path)
    if img_orig is None:
        print(f"错误: 无法读取图像 '{original_image_path}'。")
        return

    H_orig, W_orig = img_orig.shape[:2]
    print(f"处理原始图像: '{os.path.basename(original_image_path)}' ({W_orig}x{H_orig})")

    # --- 1. 获取导管ROI的边界框 ---
    red_mask = np.all(img_orig == [0, 0, 255], axis=-1)  # BGR
    red_coords = np.argwhere(red_mask)
    bbox_red_pixels = None
    if red_coords.size > 0:
        y1_red, x1_red = red_coords.min(axis=0)
        y2_red, x2_red = red_coords.max(axis=0)
        bbox_red_pixels = [x1_red, y1_red, x2_red + 1, y2_red + 1]

    # --- 2. 解析所有狭窄ROI的边界框 ---
    all_stenosis_bboxes = []
    try:
        with open(yolo_label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:5])
                    x1 = int((cx - w / 2) * W_orig)
                    y1 = int((cy - h / 2) * H_orig)
                    x2 = int((cx + w / 2) * W_orig)
                    y2 = int((cy + h / 2) * H_orig)
                    all_stenosis_bboxes.append({'class_id': class_id, 'bbox': [x1, y1, x2, y2]})
    except FileNotFoundError:
        print(f"警告: YOLO标签文件 '{yolo_label_path}' 未找到。")
    except Exception as e:
        print(f"读取YOLO标签时发生错误: {e}")

    print(f"信息: 找到 {1 if bbox_red_pixels else 0} 个导管ROI, {len(all_stenosis_bboxes)} 个狭窄ROI。")

    # --- 3. 独立处理和保存每个ROI ---
    # 处理导管ROI
    print("\n正在处理导管ROI...")
    catheter_params = params.copy()
    catheter_params['padding_pixels'] = params.get('catheter_padding_pixels', 0)
    _process_and_save_single_roi(
        img_orig, bbox_red_pixels, "catheter_roi", output_folder, original_image_path, catheter_params
    )

    # 循环处理所有狭窄ROI
    if all_stenosis_bboxes:
        print("\n正在处理狭窄ROI...")
        stenosis_params = params.copy()
        stenosis_params['padding_pixels'] = params.get('stenosis_padding_pixels', 0)
        for i, stenosis_info in enumerate(all_stenosis_bboxes):
            stenosis_suffix = f"stenosis_roi_{i + 1}"
            print(f" - 处理第 {i + 1} 个狭窄区域:")
            _process_and_save_single_roi(
                img_orig, stenosis_info['bbox'], stenosis_suffix, output_folder, original_image_path,
                stenosis_params, original_yolo_bbox_with_id=stenosis_info
            )


# ==============================================================================
# 3. 独立运行和测试区
# ==============================================================================
if __name__ == '__main__':
    """
    当这个脚本被直接运行时，会执行这部分代码。
    这使得我们可以独立地测试第一步的功能，而无需运行完整的工作流。
    """
    print("--- 正在独立运行步骤1: ROI预处理 (测试模式) ---")

    # --- 路径配置 ---
    TEST_ORIGINAL_IMAGE_PATH = "path/to/your/test_image.png"
    TEST_YOLO_LABEL_PATH = "path/to/your/test_label.txt"
    TEST_OUTPUT_FOLDER = "test_outputs/step1_rois/"

    # --- 参数配置 (未来这些将由主程序传入) ---
    # 将所有可调参数集中到一个字典中
    processing_params = {
        'magnification_factor': 8,
        'expand_pixels': 10,
        'catheter_padding_pixels': 0,
        'stenosis_padding_pixels': 0
    }

    # (示例文件创建逻辑保持不变)
    if not os.path.exists(TEST_ORIGINAL_IMAGE_PATH):
        print("\n--- 警告: 示例文件不存在。正在创建模拟图像和标签用于测试。---")
        os.makedirs(os.path.dirname(TEST_ORIGINAL_IMAGE_PATH), exist_ok=True)
        os.makedirs(os.path.dirname(TEST_YOLO_LABEL_PATH), exist_ok=True)
        dummy_img = np.zeros((512, 512, 3), dtype=np.uint8)
        cv2.rectangle(dummy_img, (28, 0), (75, 450), (0, 0, 255), -1)
        cv2.rectangle(dummy_img, (75, 40), (450, 80), (255, 255, 255), -1)
        cv2.rectangle(dummy_img, (197, 66), (229, 89), (0, 255, 255), -1)
        cv2.rectangle(dummy_img, (300, 410), (350, 430), (0, 255, 255), -1)
        cv2.imwrite(TEST_ORIGINAL_IMAGE_PATH, dummy_img)
        with open(TEST_YOLO_LABEL_PATH, 'w') as f:
            f.write("0 0.4159 0.1513 0.0625 0.0449\n")
            f.write("0 0.6347 0.8203 0.0976 0.0390\n")
        print("示例文件创建完成。")

    # 调用主功能函数
    run_step1_preprocess_rois(
        original_image_path=TEST_ORIGINAL_IMAGE_PATH,
        yolo_label_path=TEST_YOLO_LABEL_PATH,
        output_folder=TEST_OUTPUT_FOLDER,
        params=processing_params
    )