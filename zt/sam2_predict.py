import os
import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2_video_predictor
from glob import glob
from os.path import join
import matplotlib.pyplot as plt

# 使用 SAM 2 进行视频目标跟踪
def predict_sam2(root_dir, sam2_cfg, sam2_weight, result_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    if device.type == "cuda":
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print(
            "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS."
        )
    # 遍历 root_dir 目录中的所有文件夹，每个子文件夹代表一个视频序列 ['/home/zt/Datasets/HOT/challenge2024/datasets/validation/HSI-VIS-FalseColor/cranberries7', ...]
    seq_dirs = [os.path.join(root_dir, folder) for folder in os.listdir(root_dir) if
                os.path.isdir(os.path.join(root_dir, folder))]
    predictor = build_sam2_video_predictor(sam2_cfg, sam2_weight, device=device)

    # 掩码转换为边界框
    def mask_to_bbox(mask):
        if mask.ndim > 2:
            mask = mask[0]
        y_indices, x_indices = np.where(mask)
        if len(x_indices) == 0 or len(y_indices) == 0:
            return [0, 0, 0, 0]
        x_min = np.min(x_indices)
        y_min = np.min(y_indices)
        x_max = np.max(x_indices)
        y_max = np.max(y_indices)
        return [x_min, y_min, x_max, y_max]

    def genConfig(seq_path):
        RGB_img_list = sorted(glob(join(seq_path, '*.jpg')))  # 读取所有 .jpg 图像文件，并排序
        RGB_gt = getGTBbox(join(seq_path, 'groundtruth_rect.txt'))  # groundtruth_rect, init_rect
        return RGB_img_list, RGB_gt
    
    # 提取每一帧的真实边界框
    def getGTBbox(gt_filename):
        with open(gt_filename, 'r') as f:
            arr = f.readlines()
        gt = []
        for dd in arr:
            dd = dd.strip()  # '255\t133\t22\t19'
            if '\t' in dd:
                kk = dd.split('\t')
            else:
                kk = dd.split()  # ['255', '133', '22', '19']
            gt.append(list(map(int, kk)))
        return gt

    # 可视化掩码
    def show_mask(mask, ax, obj_id=None, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            cmap = plt.get_cmap("tab10")
            cmap_idx = 0 if obj_id is None else obj_id
            color = np.array([*cmap(cmap_idx)[:3], 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    for seq in seq_dirs:
        seq_name = os.path.basename(seq)  # 获取当前视频序列的名称

        RGB_img_list, RGB_gt = genConfig(seq)
        # 计算目标的中心点作为初始追踪点
        points_x = RGB_gt[0][0] + RGB_gt[0][2] // 2  # 266
        points_y = RGB_gt[0][1] + RGB_gt[0][3] // 2  # 142

        # 获取所有帧的文件名，并按数字顺序排序
        frame_names = [p for p in os.listdir(seq) if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

        # 初始化 SAM 2 追踪器的状态
        inference_state = predictor.init_state(video_path=seq)
        predictor.reset_state(inference_state)

        ann_frame_idx = 0  # 表示目标是在 第 0 帧（第一帧） 进行初始化的
        ann_obj_id = 1  # 表示这个目标的 对象 ID 是 1，用于在后续帧中追踪该目标

        # 点提示（point prompt）：points 和 labels 组成了 point prompt，用于告诉 SAM 2 目标在哪个位置
        points = np.array([[points_x, points_y]], dtype=np.float32)  # 目标的中心点坐标 array([[266., 142.]], dtype=float32)
        
        # 1 表示该点是 前景（foreground），即该点属于目标。在 SAM 2 里，1 代表目标点，0 代表背景点
        labels = np.array([1], np.int32)

        # 边界框提示（bounding box prompt）：第一帧的真实目标边界框
        box = np.array([RGB_gt[0][0], RGB_gt[0][1], RGB_gt[0][0] + RGB_gt[0][2], RGB_gt[0][1] + RGB_gt[0][3]],
                       dtype=np.float32)  # array([255., 133., 277., 152.], dtype=float32)

        # out_obj_ids：SAM 2 预测的 目标 ID 列表，用于标记不同的目标；out_mask_logits：SAM 2 预测的 掩码（mask），表示目标的分割结果
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,  # 视频推理状态
            frame_idx=ann_frame_idx,  # 目标是在第 0 帧初始化
            obj_id=ann_obj_id,  # 指定 目标 ID = 1
            points=points,  # 点提示
            labels=labels,  
            box=box,  # 边界框提示
        )

        # 进行目标追踪
        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):  # 传播目标信息并获取每一帧的预测掩码
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

        vis_frame_stride = 1  # 表示 每 1 帧 处理一次
        plt.close("all")

        os.makedirs(result_dir, exist_ok=True)

        result_file_path = os.path.join(result_dir, f"{seq_name}.txt")
        with open(result_file_path, "a") as f:
            for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
                for out_obj_id, out_mask in video_segments[out_frame_idx].items():  # video_segments 是 SAM 2 生成的目标分割结果
                    show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
                bbox = mask_to_bbox(out_mask)
                f.write(f"{bbox[0]}\t{bbox[1]}\t{bbox[2] - bbox[0]}\t{bbox[3] - bbox[1]}\n")
            print(f"Saved sequence name: {seq_name}")


if __name__ == '__main__':
    root_dir = "/dataset/ranking/test/"
    sam2_predictions = predict_sam2(root_dir)

    # 输出预测结果以检查
    for category, predictions in sam2_predictions.items():
        print(f"类别: {category}")
        for pred in predictions:
            frame_idx = pred['frame_index']
            bboxes = pred['bboxes']
            print(f"  帧索引: {frame_idx}")
            for bbox in bboxes:
                x, y, w, h = bbox
                print(f"    边界框: x={x}, y={y}, w={w}, h={h}")
