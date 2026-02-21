# 文件名: step2_extract_centerlines.py
# 作用: 血管中心线提取与修复模块。

import cv2
import numpy as np
import os
from scipy.spatial import distance

# ==============================================================================
# 1. 核心算法函数 (低层级)
# ==============================================================================

# --- 论文改进的 Zhang-Suen 细化算法 ---
SPECIAL_PATTERNS_TO_DELETE = {
    65, 5, 20, 80, 133, 52, 208, 67, 13, 22, 88, 97, 99, 141, 54, 216
}


def zhang_suen_improved_from_paper(img):
    # 这里是您提供的 zhang_suen_improved_from_paper 函数的完整代码
    H, W = img.shape
    out = np.zeros((H, W), dtype=int)
    out[img > 0] = 1
    out = 1 - out
    while True:
        s1, s2 = [], []
        # Step 1
        for y in range(1, H - 1):
            for x in range(1, W - 1):
                if out[y, x] == 1: continue
                P_inv = [out[y - 1, x], out[y - 1, x + 1], out[y, x + 1], out[y + 1, x + 1], out[y + 1, x],
                         out[y + 1, x - 1], out[y, x - 1], out[y - 1, x - 1]]
                if not (2 <= P_inv.count(0) <= 6): continue
                P_std = [1 - p for p in P_inv]
                if sum(1 for i in range(8) if
                       (P_std + P_std[:1])[i] == 0 and (P_std + P_std[:1])[i + 1] == 1) != 1 and sum(
                    v * (2 ** i) for i, v in enumerate(P_std)) not in SPECIAL_PATTERNS_TO_DELETE: continue
                if P_std[0] * P_std[2] * P_std[4] != 0: continue
                if P_std[2] * P_std[4] * P_std[6] != 0: continue
                s1.append((y, x))
        for y, x in s1: out[y, x] = 1
        # Step 2
        for y in range(1, H - 1):
            for x in range(1, W - 1):
                if out[y, x] == 1: continue
                P_inv = [out[y - 1, x], out[y - 1, x + 1], out[y, x + 1], out[y + 1, x + 1], out[y + 1, x],
                         out[y + 1, x - 1], out[y, x - 1], out[y - 1, x - 1]]
                if not (2 <= P_inv.count(0) <= 6): continue
                P_std = [1 - p for p in P_inv]
                if sum(1 for i in range(8) if
                       (P_std + P_std[:1])[i] == 0 and (P_std + P_std[:1])[i + 1] == 1) != 1 and sum(
                    v * (2 ** i) for i, v in enumerate(P_std)) not in SPECIAL_PATTERNS_TO_DELETE: continue
                if P_std[0] * P_std[2] * P_std[6] != 0: continue
                if P_std[0] * P_std[4] * P_std[6] != 0: continue
                s2.append((y, x))
        for y, x in s2: out[y, x] = 1
        if not s1 and not s2: break
    return (1 - out).astype(np.uint8) * 255


# --- 断点连接后处理函数 (优化版) ---
def _connect_gaps_endpoint_bridging(skeleton_img, max_distance):  # <-- max_distance 是位置参数
    """
    通过查找并连接邻近的端点来修复骨架中的断点。
    优化: 采用贪婪的最近邻匹配策略，确保每个端点最多只参与一个连接。
    """
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    skeleton_norm = skeleton_img // 255
    neighbor_counts = cv2.filter2D(skeleton_norm, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    endpoints_mask = (skeleton_norm == 1) & (neighbor_counts == 1)
    endpoint_coords = np.argwhere(endpoints_mask)

    if len(endpoint_coords) < 2:
        return skeleton_img

    connected_skeleton = skeleton_img.copy()

    dist_matrix = distance.cdist(endpoint_coords, endpoint_coords)

    potential_connections = []
    num_endpoints = len(endpoint_coords)
    for i in range(num_endpoints):
        for j in range(i + 1, num_endpoints):
            dist = dist_matrix[i, j]
            if 0 < dist <= max_distance:
                potential_connections.append((dist, i, j))

    potential_connections.sort(key=lambda x: x[0])

    used_endpoints_indices = set()

    for dist, i, j in potential_connections:
        if i not in used_endpoints_indices and j not in used_endpoints_indices:
            p1_yx, p2_yx = endpoint_coords[i], endpoint_coords[j]
            cv2.line(connected_skeleton, (p1_yx[1], p1_yx[0]), (p2_yx[1], p2_yx[0]), 255, 1)
            used_endpoints_indices.add(i)
            used_endpoints_indices.add(j)

    return connected_skeleton


# ==============================================================================
# 2. 主调用函数 (高层级) - 这是未来主程序要调用的函数
# ==============================================================================

def run_step2_extract_centerline(
        roi_image_path,
        output_folder,
        params
):
    """
    对单个ROI图像进行中心线提取、修复，并将结果叠加在原图上进行保存。
    【修改】现在会保留原始图像的颜色信息。
    """
    max_gap_distance = params.get('max_gap_distance_to_connect', 30.0)

    img_roi = cv2.imread(roi_image_path)
    if img_roi is None:
        print(f"  - 错误: 无法读取ROI图像 '{os.path.basename(roi_image_path)}'。")
        return

    # --- 1. 准备二值化掩码 ---
    # 定义颜色 (BGR)
    WHITE_BGR = (255, 255, 255)
    RED_BGR = (0, 0, 255)
    YELLOW_BGR = (0, 255, 255)

    # 查找所有白色或红色的像素作为细化的前景
    white_mask = np.all(img_roi == WHITE_BGR, axis=-1)
    red_mask = np.all(img_roi == RED_BGR, axis=-1)
    combined_mask_binary = ((white_mask) | (red_mask)).astype(np.uint8) * 255

    if np.sum(combined_mask_binary) == 0:
        print(f"  - 警告: ROI图像 '{os.path.basename(roi_image_path)}' 中没有找到前景像素(白色或红色)。")
        # 仍然复制一份文件以保持工作流文件完整性
        os.makedirs(output_folder, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(roi_image_path))[0]
        output_filename = f"{base_name}_centerline.png"
        save_path = os.path.join(output_folder, output_filename)
        cv2.imwrite(save_path, img_roi)
        return

    # --- 2. 执行细化和后处理 ---
    thinned_mask = zhang_suen_improved_from_paper(combined_mask_binary)
    # 修正这一行：直接传递参数，而不是使用关键字参数
    connected_mask = _connect_gaps_endpoint_bridging(thinned_mask, max_gap_distance)  # <-- 这里是修改的地方

    # --- 3. 【核心修改】创建并保存结果图像 ---
    # a. 创建一个原始输入图像的副本，以保留所有原始颜色
    output_img = img_roi.copy()

    # b. 直接在这个副本上，将提取出的中心线位置涂成黄色
    output_img[connected_mask > 0] = YELLOW_BGR

    # --- 4. 保存文件 ---
    os.makedirs(output_folder, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(roi_image_path))[0]
    output_filename = f"{base_name}_centerline.png"
    save_path = os.path.join(output_folder, output_filename)
    cv2.imwrite(save_path, output_img)
    print(f"  - 中心线图像已保存: '{output_filename}'")


# ==============================================================================
# 3. 独立运行和测试区
# ==============================================================================
if __name__ == '__main__':
    """
    当这个脚本被直接运行时，会执行这部分代码，用于独立测试。
    """
    print("--- 正在独立运行步骤2: 中心线提取 (测试模式) ---")

    # --- 路径配置 ---
    TEST_ROI_IMAGE_PATH = "path/to/your/test_roi_image.png"
    TEST_OUTPUT_FOLDER = "test_outputs/step2_centerlines/"

    # --- 参数配置 ---
    processing_params = {
        'max_gap_distance_to_connect': 5.0
    }

    # (示例文件创建逻辑)
    if not os.path.exists(TEST_ROI_IMAGE_PATH):
        print("\n--- 警告: 示例文件不存在。正在创建模拟图像用于测试。---")
        os.makedirs(os.path.dirname(TEST_ROI_IMAGE_PATH), exist_ok=True)
        # 创建一个包含白色和红色区域的ROI图
        dummy_roi = np.zeros((100, 100, 3), dtype=np.uint8)
        # 模拟一个有断点的骨架，例如一个 'C' 形
        cv2.rectangle(dummy_roi, (10, 10), (90, 90), (255, 255, 255), 1)  # 外框
        cv2.line(dummy_roi, (20, 20), (80, 20), (255, 255, 255), 1)  # 上边
        cv2.line(dummy_roi, (20, 80), (80, 80), (255, 255, 255), 1)  # 下边
        cv2.line(dummy_roi, (20, 20), (20, 80), (255, 255, 255), 1)  # 左边
        # 模拟一个断点 (中间区域故意不连)
        cv2.line(dummy_roi, (40, 40), (45, 45), (255, 255, 255), 1)
        cv2.line(dummy_roi, (55, 55), (60, 60), (255, 255, 255), 1)

        cv2.imwrite(TEST_ROI_IMAGE_PATH, dummy_roi)
        print(f"示例ROI文件创建完成: {TEST_ROI_IMAGE_PATH}")

    # 调用主功能函数
    run_step2_extract_centerline(
        roi_image_path=TEST_ROI_IMAGE_PATH,
        output_folder=TEST_OUTPUT_FOLDER,
        params=processing_params
    )