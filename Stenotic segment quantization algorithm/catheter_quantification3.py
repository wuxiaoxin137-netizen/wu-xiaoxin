# 文件名: step3c_quantify_catheter.py
# 作用: 对修剪后的导管中心线进行直径量化分析。

import cv2
import numpy as np
import os
import collections
import math
import matplotlib.pyplot as plt
from collections import Counter

# ==============================================================================
# 1. 核心算法函数 (低层级)
# ==============================================================================

# --- 定义颜色 (BGR格式) ---
YELLOW_BGR = (0, 255, 255)
RED_BGR = (0, 0, 255)
BLACK_BGR = (0, 0, 0)


# --- 辅助函数 (与之前的步骤类似，保持私有) ---
def _get_skeleton_graph_info(skeleton_binary):
    """从二值骨架图中构建图结构，并找出端点。"""
    H, W = skeleton_binary.shape
    graph, endpoints = {}, []
    foreground_pixels = np.argwhere(skeleton_binary > 0)
    for y, x in foreground_pixels:
        neighbors = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0: continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W and skeleton_binary[ny, nx] > 0:
                    neighbors.append((ny, nx))
        graph[(y, x)] = neighbors
        if len(neighbors) == 1:
            endpoints.append((y, x))
    return graph, endpoints


def _find_path_bfs(graph, start_node, end_node):
    """使用BFS在图中寻找路径。"""
    q = collections.deque([(start_node, [start_node])])
    visited = {start_node}
    while q:
        (current_node, path) = q.popleft()
        if current_node == end_node: return path
        for neighbor in graph.get(current_node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                q.append((neighbor, path + [neighbor]))
    return []


def _calculate_diameter_bresenham(center_y, center_x, img_roi, max_search_radius, boundary_threshold_ratio):
    """使用Bresenham圆算法估算直径。"""
    H, W = img_roi.shape[:2]
    is_boundary = lambda y, x: not (0 <= y < H and 0 <= x < W) or np.all(img_roi[int(y), int(x)] == BLACK_BGR)

    for r in range(1, max_search_radius + 1):
        x, y, p = r, 0, 3 - 2 * r
        points = set()
        while x >= y:
            points.update([(center_y + dy, center_x + dx) for dy, dx in
                           {(y, x), (-y, x), (y, -x), (-y, -x), (x, y), (-x, y), (x, -y), (-x, -y)}])
            y += 1
            if p > 0:
                x -= 1; p += 4 * (y - x) + 10
            else:
                p += 4 * y + 6

        if points and sum(1 for py, px in points if is_boundary(py, px)) / len(points) > boundary_threshold_ratio:
            return 2 * (r - 0.5) + 1

    return 2 * float(max_search_radius) + 1


# ==============================================================================
# 2. 主调用函数 (高层级) - 这是未来主程序要调用的函数
# ==============================================================================

def run_step3c_quantify_catheter(
        pruned_catheter_roi_path,
        output_folder,
        params
):
    """
    对修剪后的导管ROI进行直径量化，计算代表性直径，并生成可视化图表。

    Args:
        pruned_catheter_roi_path (str): 步骤3b生成的修剪后的导管中心线图像路径。
        output_folder (str): 保存量化结果（图表和数据）的文件夹。
        params (dict): 全局参数字典。

    Returns:
        float: 校准比例 (mm/px)，如果失败则返回 None
    """
    # --- 1. 从参数字典中解包 ---
    magnification = params.get('magnification_factor', 8)
    sampling_interval = params.get('catheter_sampling_interval', 20)
    boundary_threshold = params.get('diameter_boundary_threshold', 1 / 16)

    # 获取导管校准参数
    catheter_type = params.get('catheter_type', '5F')
    catheter_real_diameter_5F = params.get('catheter_real_diameter_5F', 1.65)
    catheter_real_diameter_7F = params.get('catheter_real_diameter_7F', 2.33)

    # 根据导管类型选择真实直径
    if catheter_type == '7F':
        catheter_real_diameter_mm = catheter_real_diameter_7F
    else:
        catheter_real_diameter_mm = catheter_real_diameter_5F

    # 将放大后图像的像素值转换回原始图像尺度
    scale_to_original = 1 / magnification

    # --- 2. 加载图像并提取中心线 ---
    img_roi = cv2.imread(pruned_catheter_roi_path)
    if img_roi is None:
        print(f"  - 错误: 无法读取输入的导管图像 '{os.path.basename(pruned_catheter_roi_path)}'。")
        return None

    is_yellow_mask = np.all(img_roi == YELLOW_BGR, axis=-1)
    skeleton_binary = is_yellow_mask.astype(np.uint8) * 255
    if np.sum(skeleton_binary) == 0:
        print(f"  - 警告: '{os.path.basename(pruned_catheter_roi_path)}' 中没有中心线，跳过量化。")
        return None

    # --- 3. 获取有序的中心线像素列表 (最长路径) ---
    graph, endpoints = _get_skeleton_graph_info(skeleton_binary)
    ordered_centerline_pixels = []
    if len(endpoints) >= 2:
        max_len = 0
        for i in range(len(endpoints)):
            for j in range(i + 1, len(endpoints)):
                path = _find_path_bfs(graph, endpoints[i], endpoints[j])
                if len(path) > max_len:
                    ordered_centerline_pixels = path
                    max_len = len(path)

    if not ordered_centerline_pixels:
        print(f"  - 警告: 未能找到中心线主干路径，将使用所有骨架点。")
        ordered_centerline_pixels = list(graph.keys())

    # --- 4. 沿中心线采样并计算直径 ---
    diameters_px = []
    positions_sampled = []
    max_radius = min(img_roi.shape[:2]) // 2

    for i in range(0, len(ordered_centerline_pixels), sampling_interval):
        cy, cx = ordered_centerline_pixels[i]
        diameter_pixels = _calculate_diameter_bresenham(cy, cx, img_roi, max_radius, boundary_threshold)
        # 将放大图像上测得的直径转换回原始图像尺度
        diameter_in_original_scale = diameter_pixels * scale_to_original
        diameters_px.append(diameter_in_original_scale)
        positions_sampled.append(i + 1)

    if not diameters_px:
        print("  - 错误: 未能计算出任何直径数据。")
        return None

    # --- 5. 计算代表性直径 (众数平均值) ---
    diameters_array = np.array(diameters_px)
    rounded_diameters = np.round(diameters_array, 3)

    final_diameter = 0.0
    if len(rounded_diameters) > 0:
        counts = Counter(rounded_diameters)
        most_common = counts.most_common(2)
        if len(most_common) == 1:
            final_diameter = most_common[0][0]
        else:
            final_diameter = (most_common[0][0] + most_common[1][0]) / 2.0

    # --- 6. 计算校准系数 (mm/px) ---
    if final_diameter > 0:
        calibration_scale = catheter_real_diameter_mm / final_diameter  # mm/px
        final_diameter_mm = catheter_real_diameter_mm  # 真实值
    else:
        calibration_scale = None
        final_diameter_mm = 0.0

    print(f"  - 计算出的导管代表性直径: {final_diameter:.3f} px (原始图像尺度)")
    if calibration_scale:
        print(f"  - 导管类型: {catheter_type}, 真实直径: {catheter_real_diameter_mm:.2f} mm")
        print(f"  - 校准系数: {calibration_scale:.6f} mm/px")

    # --- 7. 生成并保存可视化图表 ---
    os.makedirs(output_folder, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(pruned_catheter_roi_path))[0]

    plt.figure(figsize=(12, 6))
    plt.plot(positions_sampled, diameters_array, marker='o', linestyle='-', color='b', markersize=4)
    if final_diameter > 0:
        plt.axhline(final_diameter, color='r', linestyle='--', label=f'直径众数平均值: {final_diameter:.3f} px ({final_diameter_mm:.2f} mm)')
    plt.xlabel('中心线点采样序号')
    plt.ylabel('导管直径 (px, 原始图像尺度)')
    if calibration_scale:
        plt.title(f'导管直径变化曲线 - {base_name}\n{catheter_type}导管, 校准系数: {calibration_scale:.6f} mm/px')
    else:
        plt.title(f'导管直径变化曲线 - {base_name}')
    plt.grid(True)
    plt.legend()

    output_plot_path = os.path.join(output_folder, f"{base_name}_diameter_profile.png")
    plt.savefig(output_plot_path)
    # plt.show() # 在工作流中通常不希望自动弹窗
    plt.close()  # 关闭图表以释放内存
    print(f"  - 导管直径图表已保存: '{os.path.basename(output_plot_path)}'")

    # --- 8. (可选) 保存量化数据到文本文件 ---
    output_data_path = os.path.join(output_folder, f"{base_name}_quantification.txt")
    with open(output_data_path, 'w') as f:
        f.write(f"Catheter Type: {catheter_type}\n")
        f.write(f"Representative Catheter Diameter (px, original image scale): {final_diameter:.4f}\n")
        f.write(f"Real Catheter Diameter (mm): {catheter_real_diameter_mm:.2f}\n")
        if calibration_scale:
            f.write(f"Calibration Scale (mm/px): {calibration_scale:.6f}\n")
        f.write(f"Note: Measured on {magnification}x magnified image, then scaled back to original image scale\n")
        f.write("\n--- Sampled Diameters (px, original image scale) ---\n")
        for pos, dia in zip(positions_sampled, diameters_array):
            f.write(f"Position: {pos}, Diameter (px): {dia:.4f}\n")
    print(f"  - 导管量化数据已保存: '{os.path.basename(output_data_path)}'")

    # 返回校准系数供后续使用
    return calibration_scale


# ==============================================================================
# 3. 独立运行和测试区
# ==============================================================================
if __name__ == '__main__':
    """
    当这个脚本被直接运行时，会执行这部分代码，用于独立测试。
    """
    print("--- 正在独立运行步骤3c: 导管量化 (测试模式) ---")

    # --- 路径配置 ---
    TEST_INPUT_PATH = "path/to/your/test_pruned_catheter.png"
    TEST_OUTPUT_FOLDER = "test_outputs/step3c_quantified_catheter/"

    # --- 参数配置 ---
    processing_params = {
        'magnification_factor': 8,
        'catheter_sampling_interval': 10
    }

    # (示例文件创建逻辑)
    if not os.path.exists(TEST_INPUT_PATH):
        print("\n--- 警告: 示例文件不存在。正在创建模拟图像用于测试。---")
        os.makedirs(os.path.dirname(TEST_INPUT_PATH), exist_ok=True)
        dummy_img = np.zeros((200, 100, 3), dtype=np.uint8)
        # 红色导管背景
        cv2.rectangle(dummy_img, (40, 0), (60, 200), (0, 0, 255), -1)  # BGR for Red
        # 黄色中心线
        cv2.line(dummy_img, (50, 10), (50, 190), (0, 255, 255), 1)  # BGR for Yellow
        cv2.imwrite(TEST_INPUT_PATH, dummy_img)
        print("示例文件创建完成。")

    # 调用主功能函数
    run_step3c_quantify_catheter(
        pruned_catheter_roi_path=TEST_INPUT_PATH,
        output_folder=TEST_OUTPUT_FOLDER,
        params=processing_params
    )