import os
import time
from argparse import ArgumentParser

import imageio
import numpy as np
import torch
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render
from omg4_editing_utils import add_omg4_runtime_args, create_scene_and_gaussians, extract_source_dataset, merge_config_args
from utils.general_utils import safe_state
from utils.image_utils import psnr


DEFAULT_MODEL_PATH = "./cook_spinach_comp"
DEFAULT_SOURCE_CANDIDATES = (
    "/media/ai2lab/SSD4TB/EV_final/data/N3DV/cook_spinach",
    "/media/ai2lab/SSD4TB/EV_final/data/dynerf/cook_spinach",
    "/media/ai2lab/SSD4TB/EV_final/data/dynerf2/cook_spinach",
)


def background_tensor(dataset):
    return torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")


def image_to_frame(image):
    image = image.detach().cpu().clamp(0, 1)
    return (image.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)


def render_test_video(dataset, pipe, args):
    if args.comp_checkpoint is None:
        args.comp_checkpoint = os.path.join(args.model_path, "comp.xz")
    if not os.path.exists(args.comp_checkpoint):
        raise FileNotFoundError(f"Compressed model not found: {args.comp_checkpoint}")

    with torch.no_grad():
        scene, gaussians, _ = create_scene_and_gaussians(dataset, None, pipe, args, load_checkpoint=True, shuffle=False)
        background = background_tensor(dataset)
        test_cameras = scene.getTestCameras()
        if len(test_cameras) == 0:
            raise RuntimeError("No test cameras found.")

        out_dir = args.output_dir or os.path.join(scene.model_path, "test", "render_comp")
        render_dir = os.path.join(out_dir, "renders")
        os.makedirs(render_dir, exist_ok=True)

        video_path = os.path.join(out_dir, args.video_name)
        frames = []
        psnr_values = []
        render_times = []

        for out_idx, idx in enumerate(tqdm(range(0, len(test_cameras), args.skip_frames), desc="Rendering test")):
            gt_image, cam = test_cameras[idx]
            cam = cam.cuda()

            torch.cuda.synchronize()
            start = time.time()
            image = render(cam, gaussians, pipe, background)["render"].clamp(0, 1)
            torch.cuda.synchronize()
            render_times.append(time.time() - start)

            frame = image_to_frame(image)
            imageio.imwrite(os.path.join(render_dir, f"{out_idx:05d}.png"), frame)
            frames.append(frame)

            if not args.no_psnr and gt_image is not None:
                psnr_values.append(psnr(image, gt_image.cuda()).mean().item())

        imageio.mimwrite(video_path, frames, fps=args.fps, macro_block_size=1)
        print(f"Saved frames: {render_dir}")
        print(f"Saved video: {video_path}")
        if psnr_values:
            print(f"Mean PSNR: {np.mean(psnr_values):.2f} dB")
        if render_times:
            warmed_times = render_times[5:] if len(render_times) > 5 else render_times
            print(f"Avg render time: {np.mean(warmed_times):.4f} sec/frame")
            print(f"Render FPS: {1.0 / np.mean(warmed_times):.2f}")


def repair_source_path(args):
    if os.path.exists(args.source_path):
        return
    for candidate in DEFAULT_SOURCE_CANDIDATES:
        if os.path.exists(candidate):
            print(f"[WARN] source_path not found: {args.source_path}")
            print(f"[WARN] Using fallback source_path: {candidate}")
            args.source_path = candidate
            return
    raise FileNotFoundError(
        f"source_path not found: {args.source_path}. Pass --source_path /path/to/cook_spinach."
    )


if __name__ == "__main__":
    parser = ArgumentParser(description="Render cook_spinach_comp test cameras to PNG frames and an MP4 video.")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    add_omg4_runtime_args(parser)
    parser.set_defaults(
        model_path=DEFAULT_MODEL_PATH,
        gaussian_dim=4,
        time_duration=[0.0, 10.0],
        num_pts=300_000,
        rot_4d=True,
        eval_shfs_4d=True,
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--video_name", type=str, default="renders.mp4")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--skip_frames", type=int, default=1)
    parser.add_argument("--no_psnr", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    args = get_combined_args(parser)
    defaults = parser.parse_args([])
    for key, value in vars(defaults).items():
        if not hasattr(args, key):
            setattr(args, key, value)
    args = merge_config_args(args)
    if args.comp_checkpoint is None:
        args.comp_checkpoint = os.path.join(args.model_path, "comp.xz")

    print(f"Rendering compressed model: {args.comp_checkpoint}")
    safe_state(args.quiet)
    repair_source_path(args)
    dataset = extract_source_dataset(model, args)
    render_test_video(dataset, pipeline.extract(args), args)
