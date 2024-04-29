import os
import cv2
import glob
import torch
import Imath
import argparse
import numpy as np
import OpenEXR as exr
from PIL import Image
from tqdm import tqdm
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

from unidepth.models import UniDepthV1
from unidepth.utils import colorize, image_grid
from unidepth.utils.visualization import get_pointcloud_from_rgbd, save_file_ply


def read_depth_exr_file(filepath):
    exrfile = exr.InputFile(filepath)
    raw_bytes = exrfile.channel('R', Imath.PixelType(Imath.PixelType.FLOAT))
    depth_vector = np.frombuffer(raw_bytes, dtype=np.float32)
    height = exrfile.header()['displayWindow'].max.y + 1 - exrfile.header()['displayWindow'].min.y
    width = exrfile.header()['displayWindow'].max.x + 1 - exrfile.header()['displayWindow'].min.x
    depth_map = np.reshape(depth_vector, (height, width))
    return depth_map


def demo_single_img(model, img_f, out_f, save_ply=False, cam_intrinsics=None):
    rgb = np.array(Image.open(img_f))
    
    gt_depth_f = img_f.replace(".png", ".exr").replace(".jpg", ".exr").replace("rgb", "depth")
    
    rgb_torch = torch.from_numpy(rgb).permute(2, 0, 1)
    rgb_torch = rgb_torch[:3]
    
    # predict
    predictions = model.infer(rgb_torch, intrinsics=cam_intrinsics)

    # get GT and pred
    depth_pred = predictions["depth"].squeeze().cpu().numpy()
    intrinsics = predictions["intrinsics"].squeeze().cpu().numpy()
    # print("intrinsics:", intrinsics)
        
    pc = get_pointcloud_from_rgbd(rgb, depth_pred, None, intrinsics)
    
    if save_ply:
        save_file_ply(
            pc[..., :3].reshape(-1, 3), pc[..., 3:].reshape(-1, 3), 
            out_f.replace(".png", ".ply").replace(".jpg", ".ply")
        )
    
    if os.path.isfile(gt_depth_f):
        depth_gt = read_depth_exr_file(gt_depth_f)
        depth_gt = cv2.resize(depth_gt, (depth_pred.shape[1], depth_pred.shape[0]), interpolation=cv2.INTER_CUBIC)
        
        # compute error, you have zero divison where depth_gt == 0.0
        depth_arel = np.abs(depth_gt - depth_pred) / depth_gt
        depth_arel[depth_gt == 0.0] = 0.0
        
        depth_pred_col = colorize(depth_pred, vmin=0.01, vmax=10.0, cmap="magma_r")
        depth_gt_col = colorize(depth_gt, vmin=0.01, vmax=10.0, cmap="magma_r")
        depth_error_col = colorize(depth_arel, vmin=0.0, vmax=0.2, cmap="coolwarm")

        # save image with pred and error
        artifact = image_grid([rgb, depth_gt_col, depth_pred_col, depth_error_col], 2, 2)
    else:
        # colorize
        depth_pred_col = colorize(depth_pred, vmin=0.01, vmax=10.0, cmap="magma_r")
        artifact = image_grid([rgb, depth_pred_col], 1, 2)
    
    Image.fromarray(artifact).save(out_f)
    # save depth as npy file
    np.save(out_f.replace(".png", ".npy").replace(".jpg", ".npy"), depth_pred)
    # save pred intrinsics as txt file
    np.savetxt(out_f.replace(".png", "_intrinsics.txt").replace(".jpg", "_intrinsics.txt"), intrinsics)


def demo(model, img_folder, output_folder, intrinsic_file=None, save_ply=False):
    img_files = glob.glob(f"{img_folder}/*.png")
    img_files += glob.glob(f"{img_folder}/*.jpg")
    img_files = sorted(img_files)

    if intrinsic_file is not None:
        cam_intrinsics = np.loadtxt(intrinsic_file, delimiter=" ") if intrinsic_file is not None else None
        cam_int = torch.eye(3)
        cam_int[0, 0] = cam_intrinsics[0]
        cam_int[1, 1] = cam_intrinsics[1]
        cam_int[0, 2] = cam_intrinsics[2]
        cam_int[1, 2] = cam_intrinsics[3]
        cam_int = cam_int[None].float()
    else:
        cam_int = None
    
    for img_f in tqdm(img_files):
        out_f = os.path.join(output_folder, os.path.basename(img_f))
        demo_single_img(model, img_f, out_f, save_ply, cam_int)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UniDepth demo")
    parser.add_argument("--input_folder", type=str, required=True)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--intrinsics_file", type=str, default=None)
    parser.add_argument("--save_ply", action="store_true")
    args = parser.parse_args()
    
    print("Torch version:", torch.__version__)
    model = UniDepthV1.from_pretrained("lpiccinelli/unidepth-v1-vitl14")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    os.makedirs(args.output_folder, exist_ok=True)
    demo(model, args.input_folder, args.output_folder, args.intrinsics_file, save_ply=args.save_ply)
