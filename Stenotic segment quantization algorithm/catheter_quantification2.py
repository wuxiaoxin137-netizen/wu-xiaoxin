# 文件名: step3b_prune_catheter_centerline.py
# 作用: 修剪导管中心线，只保留最长的主干路径。

import cv2
import numpy as np
import os
import collections


# ==============================================================================
# 1. 核心算法函数 (低层级)
# ==============================================================================

def _get_skeleton_graph_info(skeleton_binary):
    """从二值骨架图中构建图结构，并找出端点。"""
    H, W = skeleton_binary.shape
    graph, endpoints = {}, []
    foreground_pixels = np.argwhere(skeleton_binary > 0)
    for y, x in foreground_pixels:
        neighbors = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if (dy, dx) == (0, 0): continue
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
        if current_node == end_node:
            return path
        for neighbor in graph.get(current_node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                q.append((neighbor, path + [neighbor]))
    return []


# ==============================================================================
# 2. 主调用函数 (高层级) - 这是未来主程序要调用的函数
# ==============================================================================

def run_step3b_prune_catheter_centerline(
        cleaned_catheter_roi_path,
        output_folder,
        params
):
    """
    对清理后的导管ROI中心线图像进行修剪，只保留最长路径。

    Returns:
        str or None: 成功时返回保存的 pruned 文件路径，失败时返回 None。
    """
    # 定义颜色 (BGR格式)
    YELLOW_BGR = (0, 255, 255)
    RED_BGR = (0, 0, 255)

    img_roi = cv2.imread(cleaned_catheter_roi_path)
    if img_roi is None:
        print(f"  - 错误: 无法读取输入的导管中心线图像 '{os.path.basename(cleaned_catheter_roi_path)}'。")
        return None  # 【修改】确保函数有返回值

    # 提取黄色中心线骨架
    is_yellow_mask = np.all(img_roi == YELLOW_BGR, axis=-1)
    skeleton_binary = is_yellow_mask.astype(np.uint8) * 255

    # 如果没有中心线，直接复制原图并返回新路径
    if np.sum(skeleton_binary) == 0:
        print(f"  - 提示: '{os.path.basename(cleaned_catheter_roi_path)}' 中没有黄色中心线，无需修剪。")
        os.makedirs(output_folder, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(cleaned_catheter_roi_path))[0]
        output_filename = f"{base_name}_pruned.png"
        save_path = os.path.join(output_folder, output_filename)
        cv2.imwrite(save_path, img_roi)
        print(f"  - 已复制文件为: '{output_filename}'")
        return save_path

    # 寻找最长路径
    graph, endpoints = _get_skeleton_graph_info(skeleton_binary)

    longest_path = []
    if len(endpoints) >= 2:
        for i in range(len(endpoints)):
            for j in range(i + 1, len(endpoints)):
                path = _find_path_bfs(graph, endpoints[i], endpoints[j])
                if len(path) > len(longest_path):
                    longest_path = path

    if not longest_path:
        print(f"  - 警告: 未能在 '{os.path.basename(cleaned_catheter_roi_path)}' 中找到端到端路径，将保留原始中心线。")
        final_centerline_mask = skeleton_binary
    else:
        # 创建一个只包含最长路径的新掩码
        final_centerline_mask = np.zeros_like(skeleton_binary)
        for y, x in longest_path:
            final_centerline_mask[y, x] = 255

    # 创建并保存结果图像
    output_img = img_roi.copy()
    output_img[is_yellow_mask] = RED_BGR
    output_img[final_centerline_mask > 0] = YELLOW_BGR

    os.makedirs(output_folder, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(cleaned_catheter_roi_path))[0]
    output_filename = f"{base_name}_pruned.png"
    save_path = os.path.join(output_folder, output_filename)
    cv2.imwrite(save_path, output_img)
    print(f"  - 修剪后的导管中心线已保存: '{output_filename}'")

    # --- 【核心修改】---
    # 返回新创建的文件的路径
    return save_path


# ==============================================================================
# 3. 独立运行和测试区
# ==============================================================================
if __name__ == '__main__':
    """
    当这个脚本被直接运行时，会执行这部分代码，用于独立测试。
    """
    print("--- 正在独立运行步骤3b: 导管中心线修剪 (测试模式) ---")

    # --- 路径配置 ---
    TEST_INPUT_PATH = "path/to/your/test_cleaned_catheter_centerline.png"
    TEST_OUTPUT_FOLDER = "test_outputs/step3b_pruned_catheter_centerlines/"

    # --- 参数配置 ---
    processing_params = {}

    # (示例文件创建逻辑)
    if not os.path.exists(TEST_INPUT_PATH):
        print("\n--- 警告: 示例文件不存在。正在创建模拟图像用于测试。---")
        os.makedirs(os.path.dirname(TEST_INPUT_PATH), exist_ok=True)
        # 创建一个带有分支的模拟导管中心线图
        dummy_img = np.zeros((100, 150, 3), dtype=np.uint8)
        # 导管区域（红色）
        cv2.rectangle(dummy_img, (10, 10), (50, 50), (0, 0, 255), -1)  # BGR for Red
        # 导管主中心线（黄色）
        cv2.line(dummy_img, (20, 20), (80, 80), (0, 255, 255), 1)  # BGR for Yellow
        # 一个短的“毛刺”分支（黄色）
        cv2.line(dummy_img, (50, 50), (60, 40), (0, 255, 255), 1)
        cv2.imwrite(TEST_INPUT_PATH, dummy_img)
        print("示例文件创建完成。")

    # 调用主功能函数
    run_step3b_prune_catheter_centerline(
        cleaned_catheter_roi_path=TEST_INPUT_PATH,
        output_folder=TEST_OUTPUT_FOLDER,
        params=processing_params
    )