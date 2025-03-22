import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from sam2_predict import *
import argparse

def run(args):
    image_dir = os.path.join(args.root_dir, args.dataset_type + '-FalseColor')
    sam2_cfg = args.sam2_cfg
    sam2_weight = args.sam2_weight

    result_dir = args.result_dir
    dataset_type = args.dataset_type
    
    predict_sam2(image_dir, sam2_cfg=sam2_cfg, sam2_weight=sam2_weight, result_dir=os.path.join(result_dir, dataset_type))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process SAM2.")
    parser.add_argument("--root_dir", default='/home/zt/Datasets/HOT/challenge2024/datasets/validation', type=str, help="Path to the root directory containing dataset folders.")
    parser.add_argument("--dataset_type", default='HSI-VIS', type=str, help="HSI-NIR, HSI-RedNIR, HSI-VIS")
    parser.add_argument("--sam2_cfg", default='sam2.1_hiera_l.yaml', type=str, help="Path to the SAM2 model configuration file.")
    parser.add_argument("--sam2_weight", default='./checkpoints/sam2.1_hiera_large.pt', type=str, help="Path to the SAM2 model weight file.")
    parser.add_argument("--result_dir", default='./result/', type=str, help="Path to the folder for saving SAM2 prediction labels.")
    args = parser.parse_args()
    run(args)

