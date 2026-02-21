# 文件名: main_workflow.py
# 作用: 自动化QCA分析流程的总指挥。

import os
import glob
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import time # <-- 新增导入 time 模块

# --- 从我们创建的模块脚本中导入主调用函数 ---
try:
    from stenosis_catheter_quantification1 import run_step1_preprocess_rois
    from stenosis_catheter_quantification2 import run_step2_extract_centerline
    from catheter_quantification1 import run_step3a_clean_catheter_roi
    from catheter_quantification2 import run_step3b_prune_catheter_centerline
    from catheter_quantification3 import run_step3c_quantify_catheter
    from stenosis_quantification1 import run_step4a_prune_stenosis_centerline
    from stenosis_quantification2 import run_step4b_quantify_stenosis
except ImportError as e:
    print(f"错误: 无法导入功能模块。请确保所有 stepX_....py 文件与本脚本位于同一目录下。")
    print(f"详细错误: {e}")
    exit()


# ==============================================================================
# --- 1. 全局配置 ---
# ==============================================================================

def setup_matplotlib_for_chinese():
    """自动查找并配置Matplotlib以支持中文显示。"""
    font_paths = [
        'C:/Windows/Fonts/simhei.ttf', 'C:/Windows/Fonts/msyh.ttc',
        '/System/Library/Fonts/STHeiti Light.tc', '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'
    ]
    font_path = next((fp for fp in font_paths if os.path.exists(fp)), None)
    if font_path:
        plt.rcParams['font.sans-serif'] = [os.path.splitext(os.path.basename(font_path))[0]]
        plt.rcParams['axes.unicode_minus'] = False
        print(f"信息: Matplotlib已配置使用中文字体: {font_path}")
    else:
        print("警告: 未找到合适的中文字体，图表中的中文可能无法正常显示。")


# --- 路径配置 ---
ORIGINAL_IMAGES_DIR = "D:/data/Skeletonization_algorithms/quantification_result/original_images/"
YOLO_LABELS_DIR = "D:/data/Skeletonization_algorithms/quantification_result/yolo_labels/"
BASE_OUTPUT_DIR = "D:/data/Skeletonization_algorithms/quantification_result/analysis_results/"

# --- 自动构建输出路径 ---
STEP_DIRS = {"step1": os.path.join(BASE_OUTPUT_DIR, "step1_rois"),
             "step2": os.path.join(BASE_OUTPUT_DIR, "step2_centerlines"),
             "step3a": os.path.join(BASE_OUTPUT_DIR, "step3a_cleaned_catheter_rois"),
             "step3b": os.path.join(BASE_OUTPUT_DIR, "step3b_pruned_catheter_centerlines"),
             "step3c": os.path.join(BASE_OUTPUT_DIR, "step3c_quantified_catheter"),
             "step4a": os.path.join(BASE_OUTPUT_DIR, "step4a_pruned_stenosis_centerlines"),
             "step4b": os.path.join(BASE_OUTPUT_DIR, "step4b_quantified_stenosis"), }

# --- 核心参数配置 ---
WORKFLOW_PARAMS = {
    'magnification_factor': 6, 'expand_pixels': 10,
    'catheter_padding_pixels': 5, 'stenosis_padding_pixels': 0,
    'max_gap_distance_to_connect': 10,
    'catheter_sampling_interval': 10,
    'yolo_boundary_tolerance': 5, 'lesion_buffer_pixels': 5,
    'diameter_boundary_threshold': 1/32,

    # --- 导管校准参数 ---
    'catheter_type': '5F',  # 可选: '5F' 或 '7F'
    'catheter_real_diameter_5F': 1.65,  # 5F导管真实直径 (mm)
    'catheter_real_diameter_7F': 2.33,  # 7F导管真实直径 (mm)
}


# ==============================================================================
# --- 2. 主工作流执行 ---
# ==============================================================================

# 【修改】函数现在可以接收一个可选的 single_image_name 参数
def run_full_workflow(single_image_name=None):
    """
    执行完整的自动化QCA分析流程。
    如果提供了 single_image_name，则只处理该文件；否则处理整个文件夹。
    """
    print(f"{'=' * 20} 开始自动化QCA工作流 {'=' * 20}\n")

    # 预先创建所有输出目录
    for dir_path in STEP_DIRS.values():
        os.makedirs(dir_path, exist_ok=True)

    # 【修改】根据 single_image_name 决定要处理的文件列表
    if single_image_name:
        # 单文件模式
        image_path = os.path.join(ORIGINAL_IMAGES_DIR, single_image_name)
        if not os.path.exists(image_path):
            print(f"错误: 指定的单个图像文件不存在: '{image_path}'")
            return
        image_paths = [image_path]
        print(f"信息: 已指定【单文件模式】，将只处理: {single_image_name}")
    else:
        # 批处理模式 (原有逻辑)
        image_paths = sorted(
            [p for ext in ('*.png', '*.jpg', '*.jpeg') for p in glob.glob(os.path.join(ORIGINAL_IMAGES_DIR, ext))])
        if not image_paths:
            print(f"错误: 在 '{ORIGINAL_IMAGES_DIR}' 中没有找到任何图像文件。")
            return
        print(f"信息: 已启动【批处理模式】，将处理文件夹中所有 {len(image_paths)} 张图像。")

    # --- 遍历处理选定的图像 ---
    for original_image_path in image_paths:
        base_name = os.path.basename(original_image_path)
        file_name_no_ext = os.path.splitext(base_name)[0]
        yolo_label_path = os.path.join(YOLO_LABELS_DIR, f"{file_name_no_ext}.txt")

        print(f"\n{'─' * 15} 正在处理图像: {base_name} {'─' * 15}")

        if not os.path.exists(yolo_label_path):
            print(f"警告: 找不到对应的YOLO标签 '{yolo_label_path}'，跳过。");
            continue

        # --- 步骤1: ROI预处理 ---
        print("\n>>> 执行步骤1: ROI预处理...")
        run_step1_preprocess_rois(original_image_path, yolo_label_path, STEP_DIRS["step1"], WORKFLOW_PARAMS)

        # --- 步骤2: 中心线提取 ---
        print("\n>>> 执行步骤2: 中心线提取...")
        roi_pattern = os.path.join(STEP_DIRS["step1"], f"{file_name_no_ext}_*.png")
        generated_roi_paths = glob.glob(roi_pattern)
        if not generated_roi_paths: print("警告: 步骤1未生成ROI文件，跳过。"); continue
        for roi_path in generated_roi_paths:
            run_step2_extract_centerline(roi_path, STEP_DIRS["step2"], WORKFLOW_PARAMS)

        # --- 导管处理分支 ---
        print("\n>>> 开始导管处理分支...")
        centerline_pattern = os.path.join(STEP_DIRS["step2"], f"{file_name_no_ext}_catheter_roi_*_centerline.png")
        catheter_centerline_paths = glob.glob(centerline_pattern)
        calibration_scale = None  # 初始化校准系数
        if not catheter_centerline_paths:
            print("警告: 未找到导管中心线文件，跳过导管处理。")
        else:
            catheter_path = catheter_centerline_paths[0]
            cleaned_path = run_step3a_clean_catheter_roi(catheter_path, STEP_DIRS["step3a"], WORKFLOW_PARAMS)
            if cleaned_path and os.path.exists(cleaned_path):
                pruned_path = run_step3b_prune_catheter_centerline(cleaned_path, STEP_DIRS["step3b"], WORKFLOW_PARAMS)
                if pruned_path and os.path.exists(pruned_path):
                    calibration_scale = run_step3c_quantify_catheter(pruned_path, STEP_DIRS["step3c"], WORKFLOW_PARAMS)

        # --- 狭窄段处理分支 ---
        print("\n>>> 开始狭窄段处理分支...")
        stenosis_pattern = os.path.join(STEP_DIRS["step2"], f"{file_name_no_ext}_stenosis_roi_*_centerline.png")
        stenosis_centerline_paths = glob.glob(stenosis_pattern)
        if not stenosis_centerline_paths:
            print("警告: 未找到狭窄段中心线文件，跳过。")
        else:
            for stenosis_path in stenosis_centerline_paths:
                centerline_basename = os.path.splitext(os.path.basename(stenosis_path))[0]
                roi_basename = centerline_basename.replace('_centerline', '')
                corresponding_label_path = os.path.join(STEP_DIRS["step1"], f"{roi_basename}.txt")
                if not os.path.exists(corresponding_label_path):
                    print(f"  - 警告: 找不到对应的YOLO标签，跳过修剪。");
                    continue

                pruned_stenosis_path = run_step4a_prune_stenosis_centerline(stenosis_path, corresponding_label_path,
                                                                            STEP_DIRS["step4a"], WORKFLOW_PARAMS)

                if pruned_stenosis_path and os.path.exists(pruned_stenosis_path):
                    run_step4b_quantify_stenosis(pruned_stenosis_path, original_image_path, STEP_DIRS["step4b"],
                                                 WORKFLOW_PARAMS, calibration_scale)
                else:
                    print(f"  - 警告: 步骤4a未能生成修剪后的文件，无法量化。")

        print(f"\n{'─' * 15} 图像 {base_name} 处理完成 {'─' * 15}")

    print(f"\n{'=' * 20} 所有图像处理完毕 {'=' * 20}")


# ==============================================================================
# --- 3. 运行主程序 ---
# ==============================================================================
if __name__ == "__main__":
    setup_matplotlib_for_chinese()

    # --- 在这里选择您的运行模式 ---

    # 【模式1】: 批量处理整个文件夹 (默认)
    # 如果要使用此模式，请取消下面一行的注释，并注释掉模式2的调用。
    # start_time = time.time() # <-- 记录开始时间
    # run_full_workflow()
    # end_time = time.time()   # <-- 记录结束时间
    # print(f"\n{'=' * 20} 工作流总耗时: {end_time - start_time:.2f} 秒 {'=' * 20}")


    # 【模式2】: 只处理单个指定的文件
    # 如果要使用此模式，请取消下面一行的注释，并在这里填入您想处理的图片文件名。
    start_time = time.time() # <-- 记录开始时间
    run_full_workflow(single_image_name="65.png")
    end_time = time.time()   # <-- 记录结束时间
    print(f"\n{'=' * 20} 工作流总耗时: {end_time - start_time:.2f} 秒 {'=' * 20}")