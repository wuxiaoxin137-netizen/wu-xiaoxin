import cv2
import numpy as np
import os
import collections
import math
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


# ==============================================================================
# 1. 全局配置与初始化
# ==============================================================================

def setup_matplotlib_chinese_font():
    """自动查找并配置Matplotlib以支持中文显示。"""
    font_paths = [
        '/System/Library/Fonts/STHeiti Light.ttc',  # macOS
        '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',  # Linux
        'C:/Windows/Fonts/simhei.ttf',  # Windows 黑体
        'C:/Windows/Fonts/msyh.ttc'  # Windows 微软雅黑
    ]
    chinese_font_path = next((fp for fp in font_paths if os.path.exists(fp)), None)
    if chinese_font_path:
        plt.rcParams['font.sans-serif'] = [os.path.splitext(os.path.basename(chinese_font_path))[0]]
        plt.rcParams['axes.unicode_minus'] = False
        try:
            fm.fontManager.addfont(chinese_font_path)
        except Exception:
            pass
        print(f"信息: 已设置Matplotlib使用字体: {chinese_font_path}")
    else:
        print("警告: 未找到合适的中文系统字体，图表中的中文可能无法正常显示。")


# --- 定义颜色 (BGR格式) ---
YELLOW_BGR = (0, 255, 255)
BLACK_BGR = (0, 0, 0)
RED_BGR = (0, 0, 255)
# 绘图颜色
COLOR_DIAMETER = 'blue'
COLOR_MLD = 'red'
COLOR_PROX_RVD = 'green'
COLOR_DIST_RVD = 'magenta'
COLOR_INTERP_RVD = 'cyan'


# ==============================================================================
# 2. 核心算法辅助函数 (低层级)
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
    """使用广度优先搜索(BFS)在图中寻找两个节点之间的路径。"""
    q = collections.deque([(start_node, [start_node])])
    visited = {start_node}
    while q:
        current_node, path = q.popleft()
        if current_node == end_node:
            return path
        for neighbor in graph.get(current_node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                q.append((neighbor, path + [neighbor]))
    return []


# 【修改】_calculate_diameter_bresenham 函数现在接收 boundary_threshold_ratio 参数
def _calculate_diameter_bresenham(center_y, center_x, img_roi, max_search_radius, boundary_threshold_ratio):
    """使用Bresenham圆算法的变体来估算血管直径。"""
    H, W = img_roi.shape[:2]
    is_boundary_pixel = lambda y, x: not (0 <= y < H and 0 <= x < W) or np.all(img_roi[int(y), int(x)] == BLACK_BGR)

    for r in range(1, max_search_radius + 1):
        x, y, p = r, 0, 3 - 2 * r
        points = set()
        while x >= y:
            points.update([(center_y + dy, center_x + dx) for dy, dx in
                           {(y, x), (-y, x), (y, -x), (-y, -x), (x, y), (-x, y), (x, -y), (-x, -y)}])
            y += 1
            if p > 0:
                x -= 1
                p += 4 * (y - x) + 10
            else:
                p += 4 * y + 6

        # 【修改】使用传入的参数替换硬编码的值
        if points and sum(1 for py, px in points if is_boundary_pixel(py, px)) / len(points) > boundary_threshold_ratio:
            return 2 * (r - 0.5) + 1

    return 2 * float(max_search_radius) + 1


def _calculate_actual_path_distance(path_segment):
    """计算一个像素路径段的实际几何长度（累加距离）。"""
    distance = 0.0
    for i in range(len(path_segment) - 1):
        p1 = path_segment[i]
        p2 = path_segment[i + 1]
        distance += math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    return distance


# ==============================================================================
# 3. 主调用函数 (高层级)
# ==============================================================================
def run_step4b_quantify_stenosis(pruned_stenosis_roi_path, original_segmentation_mask_path, output_folder, params, calibration_scale=None):
    """
    对修剪后的狭窄段ROI进行全面的定量分析。

    Args:
        calibration_scale (float, optional): 校准系数 (mm/px)，如果提供则输出真实mm值
    """
    # --- 1. 参数解包 ---
    magnification = params.get('magnification_factor', 8)
    buffer_pixels = params.get('lesion_buffer_pixels', 15)
    boundary_threshold = params.get('diameter_boundary_threshold', 1 / 16)
    pixel_to_mm_scale = 1 / magnification

    # --- 2. 加载与预处理 ---
    img_roi = cv2.imread(pruned_stenosis_roi_path)
    if img_roi is None:
        print(f"  - 错误: 无法读取 '{os.path.basename(pruned_stenosis_roi_path)}'。")
        return

    seg_mask = cv2.imread(original_segmentation_mask_path)
    catheter_ref_point = None
    if seg_mask is not None:
        red_coords = np.argwhere(np.all(seg_mask == RED_BGR, axis=-1))
        if red_coords.size > 0:
            catheter_ref_point = tuple(red_coords[np.argmin(red_coords[:, 0])])

    skeleton_binary = (np.all(img_roi == YELLOW_BGR, axis=-1)).astype(np.uint8) * 255
    if np.sum(skeleton_binary) == 0:
        print(f"  - 警告: '{os.path.basename(pruned_stenosis_roi_path)}' 中无中心线。")
        return

    graph, endpoints = _get_skeleton_graph_info(skeleton_binary)
    ordered_centerline = []
    if len(endpoints) >= 2:
        max_len = 0
        for i, e1 in enumerate(endpoints):
            for j, e2 in enumerate(endpoints):
                if i >= j: continue
                path = _find_path_bfs(graph, e1, e2)
                if len(path) > max_len:
                    ordered_centerline, max_len = path, len(path)

    if not ordered_centerline:
        print(f"  - 错误: 未能找到中心线路径。")
        return

    if catheter_ref_point:
        ep1, ep2 = ordered_centerline[0], ordered_centerline[-1]
        dist1 = math.hypot(ep1[0] - catheter_ref_point[0], ep1[1] - catheter_ref_point[1])
        dist2 = math.hypot(ep2[0] - catheter_ref_point[0], ep2[1] - catheter_ref_point[1])
        if dist2 < dist1:
            ordered_centerline.reverse()
            print("  - 信息: 中心线方向已校准 (近端->远端)。")

    # --- 3. 计算所有基础指标 ---
    H, W, _ = img_roi.shape
    # 【修改】调用直径计算函数时，传入新的阈值参数
    diameters_pixels = [_calculate_diameter_bresenham(cy, cx, img_roi, min(H, W) // 2, boundary_threshold) for cy, cx in
                        ordered_centerline]
    diameters_mm = np.array(diameters_pixels) * pixel_to_mm_scale

    mld_mm, mld_idx = np.min(diameters_mm), np.argmin(diameters_mm)
    prox_rvd_mm, prox_len_mm, prox_ref_idx = 0.0, 0.0, -1
    dist_rvd_mm, dist_len_mm, dist_ref_idx = 0.0, 0.0, -1

    if (end_idx := mld_idx - buffer_pixels) > 0:
        prox_seg = diameters_mm[:end_idx]
        if prox_seg.size > 0:
            prox_rvd_mm = np.max(prox_seg)
            prox_idxs = np.where(prox_seg == prox_rvd_mm)[0]
            prox_ref_idx = prox_idxs[len(prox_idxs) // 2]
            prox_len_mm = _calculate_actual_path_distance(
                ordered_centerline[prox_ref_idx:mld_idx + 1]) * pixel_to_mm_scale

    if (start_idx := mld_idx + 1 + buffer_pixels) < len(diameters_mm):
        dist_seg = diameters_mm[start_idx:]
        if dist_seg.size > 0:
            dist_rvd_mm = np.max(dist_seg)
            dist_idxs_local = np.where(dist_seg == dist_rvd_mm)[0]
            dist_ref_idx = start_idx + dist_idxs_local[len(dist_idxs_local) // 2]
            dist_len_mm = _calculate_actual_path_distance(
                ordered_centerline[mld_idx:dist_ref_idx + 1]) * pixel_to_mm_scale

    # --- 4. 分叉病变仲裁与最终指标计算 ---
    is_bifurcation, final_rvd_mm, stenosis_rate = False, 0.0, 0.0

    if prox_rvd_mm > 0 and dist_rvd_mm > 0:
        if max(prox_rvd_mm, dist_rvd_mm) > min(prox_rvd_mm, dist_rvd_mm) * 2:
            is_bifurcation = True
            print("  - 信息: 检测到分叉病变，将仅保留主血管段分析。")
            if prox_rvd_mm > dist_rvd_mm:
                dist_rvd_mm, dist_len_mm, dist_ref_idx = 0, 0, -1
            else:
                prox_rvd_mm, prox_len_mm, prox_ref_idx = 0, 0, -1

    if is_bifurcation or ((prox_rvd_mm > 0) != (dist_rvd_mm > 0)):
        final_rvd_mm = max(prox_rvd_mm, dist_rvd_mm)
    elif prox_rvd_mm > 0 and dist_rvd_mm > 0:
        numerator = (prox_rvd_mm * dist_len_mm) + (dist_rvd_mm * prox_len_mm)
        denominator = prox_len_mm + dist_len_mm
        final_rvd_mm = numerator / denominator if denominator > 0 else 0

    if mld_mm > 0 and final_rvd_mm > 0:
        stenosis_rate = (1 - (mld_mm / final_rvd_mm)) * 100

    # --- 5. 打印、绘图和保存 ---
    base_name = os.path.splitext(os.path.basename(pruned_stenosis_roi_path))[0]

    # a. 打印报告
    print(f"\n  --- 量化报告 for {base_name} ---")
    print(f"  最小管腔直径 (MLD): {mld_mm:.3f} px", end="")
    if calibration_scale:
        print(f" ({mld_mm * calibration_scale:.3f} mm)")
    else:
        print()

    if prox_rvd_mm > 0:
        print(f"  近端参考血管直径 (RVD): {prox_rvd_mm:.3f} px", end="")
        if calibration_scale:
            print(f" ({prox_rvd_mm * calibration_scale:.3f} mm)")
        else:
            print()

    if dist_rvd_mm > 0:
        print(f"  远端参考血管直径 (RVD): {dist_rvd_mm:.3f} px", end="")
        if calibration_scale:
            print(f" ({dist_rvd_mm * calibration_scale:.3f} mm)")
        else:
            print()

    if prox_len_mm > 0:
        print(f"  近端病变长度: {prox_len_mm:.3f} px", end="")
        if calibration_scale:
            print(f" ({prox_len_mm * calibration_scale:.3f} mm)")
        else:
            print()

    if dist_len_mm > 0:
        print(f"  远端病变长度: {dist_len_mm:.3f} px", end="")
        if calibration_scale:
            print(f" ({dist_len_mm * calibration_scale:.3f} mm)")
        else:
            print()

    if final_rvd_mm > 0:
        rvd_type = "插值" if not is_bifurcation and (prox_rvd_mm > 0 and dist_rvd_mm > 0) else "最终"
        print("  " + "-" * 20)
        print(f"  {rvd_type}参考血管直径: {final_rvd_mm:.3f} px", end="")
        if calibration_scale:
            print(f" ({final_rvd_mm * calibration_scale:.3f} mm)")
        else:
            print()
        print(f"  直径狭窄程度: {stenosis_rate:.2f}%")

        # 【新增】狭窄分级评估和诊断建议
        stenosis_grade = ""
        diagnosis_suggestion = ""

        if stenosis_rate < 30:
            stenosis_grade = "无显著狭窄"
            diagnosis_suggestion = "血管状况良好，建议定期随访观察"
        elif 30 <= stenosis_rate < 50:
            stenosis_grade = "轻微狭窄"
            diagnosis_suggestion = "建议药物治疗控制，定期复查"
        elif 50 <= stenosis_rate < 70:
            stenosis_grade = "中度狭窄"
            diagnosis_suggestion = "建议进一步评估，考虑介入治疗或药物强化治疗"
        elif 70 <= stenosis_rate < 100:
            stenosis_grade = "重度狭窄或阻塞"
            diagnosis_suggestion = "强烈建议介入治疗（如支架植入术）或冠脉搭桥术"
        else:  # stenosis_rate >= 100
            stenosis_grade = "完全闭塞"
            diagnosis_suggestion = "需要紧急介入治疗或冠脉搭桥术"

        print(f"  狭窄分级: {stenosis_grade}")
        print(f"  诊断建议: {diagnosis_suggestion}")

    # b. 绘图
    fig, ax = plt.subplots(figsize=(14, 7))
    positions = np.arange(1, len(diameters_mm) + 1)

    plot_positions, plot_diameters = positions, diameters_mm
    plot_mld_idx, plot_prox_ref_idx, plot_dist_ref_idx = mld_idx, prox_ref_idx, dist_ref_idx

    if is_bifurcation:
        if prox_rvd_mm > 0:
            plot_slice = slice(0, mld_idx + 1)
        else:
            plot_slice = slice(mld_idx, len(diameters_mm))
        plot_positions, plot_diameters = positions[plot_slice], diameters_mm[plot_slice]
        if prox_rvd_mm > 0:
            plot_mld_idx = len(plot_positions) - 1
            plot_dist_ref_idx = -1
        else:
            plot_prox_ref_idx = -1
            plot_mld_idx = 0
            if dist_ref_idx != -1:
                plot_dist_ref_idx = dist_ref_idx - mld_idx

    ax.plot(plot_positions, plot_diameters, color=COLOR_DIAMETER, marker='o', ls='-', ms=4, label='Diameter')
    ax.axvline(positions[mld_idx], color=COLOR_MLD, ls='--', label=f'MLD: {mld_mm:.3f} px')
    if prox_rvd_mm > 0: ax.axhline(prox_rvd_mm, color=COLOR_PROX_RVD, ls=':', label=f'PRVD: {prox_rvd_mm:.3f} px')
    if dist_rvd_mm > 0: ax.axhline(dist_rvd_mm, color=COLOR_DIST_RVD, ls=':', label=f'DRVD: {dist_rvd_mm:.3f} px')
    if final_rvd_mm > 0 and not is_bifurcation: ax.axhline(final_rvd_mm, color=COLOR_INTERP_RVD, ls='-.', lw=2,
                                                           label=f'RVD: {final_rvd_mm:.3f} px')

    if prox_ref_idx != -1 and plot_prox_ref_idx != -1 and plot_prox_ref_idx < len(plot_positions):
        ax.plot(plot_positions[plot_prox_ref_idx], plot_diameters[plot_prox_ref_idx], 'P', color=COLOR_PROX_RVD, ms=10,
                label='PRP')
    if dist_ref_idx != -1 and plot_dist_ref_idx != -1 and plot_dist_ref_idx < len(plot_positions):
        ax.plot(plot_positions[plot_dist_ref_idx], plot_diameters[plot_dist_ref_idx], 'P', color=COLOR_DIST_RVD, ms=10,
                label='DRP')

    ax.set_xlabel('Centerline Point Index (Proximal -> Distal)')
    ax.set_ylabel('Diameter (px)')
    ax.set_title(f'Stenosis Diameter Variation - {base_name}')
    ax.grid(True)
    ax.legend()

    # c. 保存
    output_plot_path = os.path.join(output_folder, f"{base_name}_quantified.png")
    plt.savefig(output_plot_path)
    plt.close(fig)
    print(f"  - 量化图表已保存: '{os.path.basename(output_plot_path)}'")

    output_data_path = os.path.join(output_folder, f"{base_name}_quantification.txt")
    with open(output_data_path, 'w', encoding='utf-8') as f:
        f.write(f"MLD_px: {mld_mm:.4f}\n")
        f.write(f"Proximal_RVD_px: {prox_rvd_mm:.4f}\n")
        f.write(f"Distal_RVD_px: {dist_rvd_mm:.4f}\n")
        f.write(f"Proximal_Lesion_Length_px: {prox_len_mm:.4f}\n")
        f.write(f"Distal_Lesion_Length_px: {dist_len_mm:.4f}\n")
        f.write(f"Final_RVD_px: {final_rvd_mm:.4f}\n")
        f.write(f"Stenosis_Rate_percent: {stenosis_rate:.2f}\n")

        # 【新增】保存狭窄分级和诊断建议
        if final_rvd_mm > 0:
            f.write(f"\n--- Stenosis Assessment ---\n")
            f.write(f"Stenosis_Grade: {stenosis_grade}\n")
            f.write(f"Diagnosis_Suggestion: {diagnosis_suggestion}\n")

        if calibration_scale:
            f.write(f"\n--- Calibrated Real Values (mm) ---\n")
            f.write(f"Calibration_Scale_mm_per_px: {calibration_scale:.6f}\n")
            f.write(f"MLD_mm: {mld_mm * calibration_scale:.4f}\n")
            f.write(f"Proximal_RVD_mm: {prox_rvd_mm * calibration_scale:.4f}\n")
            f.write(f"Distal_RVD_mm: {dist_rvd_mm * calibration_scale:.4f}\n")
            f.write(f"Proximal_Lesion_Length_mm: {prox_len_mm * calibration_scale:.4f}\n")
            f.write(f"Distal_Lesion_Length_mm: {dist_len_mm * calibration_scale:.4f}\n")
            f.write(f"Final_RVD_mm: {final_rvd_mm * calibration_scale:.4f}\n")
            f.write(f"Stenosis_Rate_percent: {stenosis_rate:.2f}\n")
    print(f"  - 量化数据已保存: '{os.path.basename(output_data_path)}'")


# ==============================================================================
# 4. 独立运行和测试区
# ==============================================================================
if __name__ == '__main__':
    setup_matplotlib_chinese_font()

    TEST_INPUT_PATH = "path/to/pruned_stenosis.png"
    TEST_SEG_MASK_PATH = "path/to/original_mask.png"
    TEST_OUTPUT_FOLDER = "test_outputs/step4b_quantified_stenosis/"

    processing_params = {
        'magnification_factor': 8,
        'lesion_buffer_pixels': 15,
        'diameter_boundary_threshold': 0.0625
    }

    if not os.path.exists(TEST_INPUT_PATH):
        print("\n--- 警告: 示例文件不存在。正在创建模拟图像... ---")
        os.makedirs(os.path.dirname(TEST_INPUT_PATH), exist_ok=True)
        os.makedirs(os.path.dirname(TEST_SEG_MASK_PATH), exist_ok=True)
        dummy_img = np.full((100, 200, 3), BLACK_BGR, dtype=np.uint8)
        cv2.rectangle(dummy_img, (0, 0), (200, 100), (255, 255, 255), -1)
        cv2.line(dummy_img, (10, 50), (190, 50), (0, 255, 255), 1)
        cv2.imwrite(TEST_INPUT_PATH, dummy_img)
        dummy_seg = np.zeros((150, 250, 3), dtype=np.uint8)
        cv2.line(dummy_seg, (10, 10), (30, 10), (0, 0, 255), 5)
        cv2.imwrite(TEST_SEG_MASK_PATH, dummy_seg)
        print("示例文件创建完成。")

    run_step4b_quantify_stenosis(
        pruned_stenosis_roi_path=TEST_INPUT_PATH,
        original_segmentation_mask_path=TEST_SEG_MASK_PATH,
        output_folder=TEST_OUTPUT_FOLDER,
        params=processing_params
    )