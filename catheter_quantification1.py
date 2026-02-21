# 文件名: step3a_clean_catheter_roi.py
# 作用: 清理导管ROI图像，将血管部分的黄色中心线转换为白色。

import cv2
import numpy as np
import os
from PIL import Image


# ==============================================================================
# 1. 主调用函数 (高层级) - 这是未来主程序要调用的函数
# ==============================================================================

def run_step3a_clean_catheter_roi(
        catheter_roi_centerline_path,
        output_folder,
        params
):
    """
    对单个导管ROI中心线图像进行清理。
    它会找到所有不与红色导管区域相邻的黄色中心线像素，并将它们转换为白色。

    Args:
        catheter_roi_centerline_path (str): 步骤2生成的导管ROI中心线图像路径。
        output_folder (str): 保存清理后图像的文件夹。
        params (dict): 全局参数字典 (当前未使用，为未来扩展保留)。

    Returns:
        str or None: 成功时返回保存的 cleaned 文件路径，失败时返回 None。
    """
    try:
        img = Image.open(catheter_roi_centerline_path).convert("RGB")
    except FileNotFoundError:
        print(f"  - 错误: 无法找到输入的导管中心线图像 '{os.path.basename(catheter_roi_centerline_path)}'。")
        return None  # 【修改】确保函数有返回值

    img_array = np.array(img)

    # --- 使用NumPy向量化操作进行高效处理 ---
    YELLOW = (255, 255, 0)
    RED = (255, 0, 0)
    WHITE = (255, 255, 255)

    yellow_pixels_mask = np.all(img_array == YELLOW, axis=-1)

    # 【修改】如果图中没有黄色像素，直接复制原图并返回新路径
    if not np.any(yellow_pixels_mask):
        print(f"  - 提示: '{os.path.basename(catheter_roi_centerline_path)}' 中没有黄色中心线，无需清理。")
        os.makedirs(output_folder, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(catheter_roi_centerline_path))[0]
        output_filename = f"{base_name}_cleaned.png"
        save_path = os.path.join(output_folder, output_filename)
        img.save(save_path)  # 直接保存原始PIL Image对象
        print(f"  - 已复制文件为: '{output_filename}'")
        return save_path

    # 创建一个标记红色区域的膨胀掩码
    red_mask = np.all(img_array == RED, axis=-1).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    dilated_red_mask = cv2.dilate(red_mask, kernel, iterations=1)

    # 找到所有需要变白的黄色像素: 那些是黄色，但又不在膨胀红色区域内的点
    pixels_to_change_mask = yellow_pixels_mask & (dilated_red_mask == 0)

    # 一次性将所有需要改变的像素设置为白色
    cleaned_img_array = img_array.copy()
    cleaned_img_array[pixels_to_change_mask] = WHITE

    # 将结果转回 PIL Image 并保存
    cleaned_img = Image.fromarray(cleaned_img_array)

    os.makedirs(output_folder, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(catheter_roi_centerline_path))[0]
    output_filename = f"{base_name}_cleaned.png"
    save_path = os.path.join(output_folder, output_filename)

    cleaned_img.save(save_path)
    print(f"  - 清理后的导管ROI已保存: '{output_filename}'")

    # --- 【核心修改】---
    # 返回新创建的文件的路径
    return save_path


# ==============================================================================
# 2. 独立运行和测试区
# ==============================================================================
if __name__ == '__main__':
    """
    当这个脚本被直接运行时，会执行这部分代码，用于独立测试。
    """
    print("--- 正在独立运行步骤3a: 导管ROI清理 (测试模式) ---")

    # --- 路径配置 ---
    TEST_INPUT_PATH = "path/to/your/test_catheter_centerline.png"
    TEST_OUTPUT_FOLDER = "test_outputs/step3a_cleaned_catheter_rois/"

    # --- 参数配置 ---
    # 当前步骤参数较少，为未来保留
    processing_params = {}

    # (示例文件创建逻辑)
    if not os.path.exists(TEST_INPUT_PATH):
        print("\n--- 警告: 示例文件不存在。正在创建模拟图像用于测试。---")
        os.makedirs(os.path.dirname(TEST_INPUT_PATH), exist_ok=True)
        # 创建一个模拟的导管中心线图
        dummy_img = np.zeros((100, 150, 3), dtype=np.uint8)
        # 血管区域（白色）
        cv2.rectangle(dummy_img, (0, 0), (150, 100), (255, 255, 255), -1)
        # 导管区域（红色）
        cv2.rectangle(dummy_img, (10, 10), (50, 50), (255, 0, 0), -1)
        # 导管中心线（黄色，在红色区域内）
        cv2.line(dummy_img, (20, 20), (40, 40), (255, 255, 0), 1)
        # 血管中心线（黄色，在白色区域内）
        cv2.line(dummy_img, (40, 40), (130, 80), (255, 255, 0), 1)
        # 将PIL格式的图像保存
        Image.fromarray(cv2.cvtColor(dummy_img, cv2.COLOR_BGR2RGB)).save(TEST_INPUT_PATH)
        print("示例导管中心线ROI文件创建完成。")

    # 调用主功能函数
    run_step3a_clean_catheter_roi(
        catheter_roi_centerline_path=TEST_INPUT_PATH,
        output_folder=TEST_OUTPUT_FOLDER,
        params=processing_params
    )