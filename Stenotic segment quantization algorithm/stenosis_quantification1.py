# 文件名: step4a_prune_stenosis_centerline.py
# 作用: 清理和修剪狭窄段ROI的中心线，只保留连接边界的主路径。

import cv2
import numpy as np
import os
import collections
import math

# ==============================================================================
# 1. 核心算法函数 (低层级)
# ==============================================================================

# --- 定义颜色 (BGR格式) ---
YELLOW_BGR = (0, 255, 255)
WHITE_BGR = (255, 255, 255)
BLACK_BGR = (0, 0, 0)


# --- 辅助函数 (保持私有) ---
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
        (current, path) = q.popleft()
        if current == end_node: return path
        for neighbor in graph.get(current, []):
            if neighbor not in visited:
                visited.add(neighbor)
                q.append((neighbor, path + [neighbor]))
    return []


def _calculate_path_geometric_length(path):
    """计算路径的几何长度。"""
    if not path or len(path) < 2: return 0.0
    return sum(math.hypot(p2[0] - p1[0], p2[1] - p1[1]) for p1, p2 in zip(path, path[1:]))


# ==============================================================================
# 2. 主调用函数 (高层级) - 这是未来主程序要调用的函数
# ==============================================================================

def run_step4a_prune_stenosis_centerline(
        stenosis_roi_centerline_path,
        corresponding_yolo_label_path,
        output_folder,
        params
):
    """
    对单个狭窄段ROI的中心线图像进行清理和修剪。

    Returns:
        str or None: 成功时返回保存的 pruned 文件路径，失败时返回 None。
    """
    boundary_tolerance = params.get('yolo_boundary_tolerance', 2)

    img_roi = cv2.imread(stenosis_roi_centerline_path)
    if img_roi is None:
        print(f"  - 错误: 无法读取狭窄段ROI图像 '{os.path.basename(stenosis_roi_centerline_path)}'。")
        return None  # 【修改】确保函数有返回值

    H_roi, W_roi = img_roi.shape[:2]

    yolo_bbox = None
    try:
        with open(corresponding_yolo_label_path, 'r') as f:
            parts = f.readline().strip().split()
            if len(parts) >= 5:
                cx_n, cy_n, w_n, h_n = map(float, parts[1:5])
                x1 = int((cx_n - w_n / 2) * W_roi);
                y1 = int((cy_n - h_n / 2) * H_roi)
                x2 = int((cx_n + w_n / 2) * W_roi);
                y2 = int((cy_n + h_n / 2) * H_roi)
                yolo_bbox = [max(0, x1), max(0, y1), min(W_roi, x2), min(H_roi, y2)]
    except FileNotFoundError:
        print(f"  - 警告: 未找到对应的YOLO标签。将尝试无框修剪。")
    except Exception as e:
        print(f"  - 错误: 解析YOLO标签时出错: {e}")

    # 提取框内骨架
    skeleton_binary_full = (np.all(img_roi == YELLOW_BGR, axis=-1)).astype(np.uint8) * 255
    skeleton_in_box = np.zeros_like(skeleton_binary_full)
    if yolo_bbox:
        x1, y1, x2, y2 = yolo_bbox
        skeleton_in_box[y1:y2, x1:x2] = skeleton_binary_full[y1:y2, x1:x2]
    else:
        skeleton_in_box = skeleton_binary_full

    if np.sum(skeleton_in_box) == 0:
        print(f"  - 警告: YOLO框内无中心线，无法修剪。")
        final_centerline_mask = skeleton_in_box  # 将使用空掩码
    else:
        graph, endpoints = _get_skeleton_graph_info(skeleton_in_box)
        boundary_endpoints = []
        if yolo_bbox:
            x1, y1, x2, y2 = yolo_bbox
            for p_y, p_x in endpoints:
                if (abs(p_x - x1) <= boundary_tolerance or abs(p_x - x2 - 1) <= boundary_tolerance or
                        abs(p_y - y1) <= boundary_tolerance or abs(p_y - y2 - 1) <= boundary_tolerance):
                    boundary_endpoints.append((p_y, p_x))
        else:
            boundary_endpoints = endpoints

        final_path = []
        if len(boundary_endpoints) >= 2:
            max_len = 0.0
            for i in range(len(boundary_endpoints)):
                for j in range(i + 1, len(boundary_endpoints)):
                    path = _find_path_bfs(graph, boundary_endpoints[i], boundary_endpoints[j])
                    current_len = _calculate_path_geometric_length(path)
                    if current_len > max_len:
                        final_path = path;
                        max_len = current_len

        if not final_path:
            print(f"  - 警告: 未能找到连接边界的路径，将保留框内所有中心线。")
            final_centerline_mask = skeleton_in_box
        else:
            final_centerline_mask = np.zeros_like(skeleton_in_box)
            for y, x in final_path:
                final_centerline_mask[y, x] = 255

    # 创建并保存结果图像
    output_img = img_roi.copy()
    output_img[np.all(output_img == YELLOW_BGR, axis=-1)] = WHITE_BGR
    output_img[final_centerline_mask > 0] = YELLOW_BGR

    os.makedirs(output_folder, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(stenosis_roi_centerline_path))[0]
    output_filename = f"{base_name}_pruned.png"
    save_path = os.path.join(output_folder, output_filename)
    cv2.imwrite(save_path, output_img)
    print(f"  - 修剪后的狭窄段中心线已保存: '{output_filename}'")

    # --- 【核心修改】---
    # 无论成功与否，都返回一个路径，主程序可以根据路径是否存在来判断
    return save_path

# ==============================================================================
# 3. 独立运行和测试区
# ==============================================================================
if __name__ == '__main__':
    print("--- 正在独立运行步骤4a: 狭窄段中心线修剪 (测试模式) ---")

    # --- 路径配置 ---
    TEST_INPUT_PATH = "path/to/your/test_stenosis_centerline.png"
    TEST_LABEL_PATH = "path/to/your/test_stenosis_label.txt"
    TEST_OUTPUT_FOLDER = "test_outputs/step4a_pruned_stenosis_centerlines/"

    # --- 参数配置 ---
    processing_params = {
        'yolo_boundary_tolerance': 2
    }

    # (示例文件创建逻辑)
    if not os.path.exists(TEST_INPUT_PATH):
        print("\n--- 警告: 示例文件不存在。正在创建模拟图像用于测试。---")
        os.makedirs(os.path.dirname(TEST_INPUT_PATH), exist_ok=True)
        os.makedirs(os.path.dirname(TEST_LABEL_PATH), exist_ok=True)

        dummy_img = np.zeros((100, 150, 3), dtype=np.uint8)
        # 白色血管背景
        cv2.rectangle(dummy_img, (0, 0), (150, 100), (255, 255, 255), -1)
        # 黄色中心线主干
        cv2.line(dummy_img, (10, 50), (140, 50), (0, 255, 255), 1)
        # 黄色分支
        cv2.line(dummy_img, (75, 50), (75, 80), (0, 255, 255), 1)
        cv2.imwrite(TEST_INPUT_PATH, dummy_img)

        # 对应的YOLO标签
        # 框体大小: (20, 20) 到 (130, 80)
        with open(TEST_LABEL_PATH, 'w') as f:
            f.write("0 0.500000 0.500000 0.733333 0.600000")  # (150*0.73=110, 100*0.6=60)
        print("示例文件创建完成。")

    run_step4a_prune_stenosis_centerline(
        stenosis_roi_centerline_path=TEST_INPUT_PATH,
        corresponding_yolo_label_path=TEST_LABEL_PATH,
        output_folder=TEST_OUTPUT_FOLDER,
        params=processing_params
    )