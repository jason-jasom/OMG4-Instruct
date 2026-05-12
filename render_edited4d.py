import os
import time
from argparse import ArgumentParser

import imageio
import numpy as np
import torch
import torchvision
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render
from omg4_editing_utils import add_omg4_runtime_args, create_scene_and_gaussians, extract_source_dataset, merge_config_args
from utils.general_utils import safe_state


def background_tensor(dataset):
    return torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")


def render_view(cam, gaussians, pipe, background):
    cam = cam.cuda()
    image = render(cam, gaussians, pipe, background)["render"]
    return image.clamp(0, 1)


def render_split(model_path, name, iteration, views, gaussians, pipe, background, fps):
    if not views:
        print(f"No {name} cameras to render.")
        return
    out_dir = os.path.join(model_path, name, f"edited_{iteration}")
    render_dir = os.path.join(out_dir, "renders")
    os.makedirs(render_dir, exist_ok=True)

    video_path = os.path.join(out_dir, "renders.mp4")
    frames = []

    times = []
    for idx, (_, cam) in enumerate(tqdm(views, desc=f"Rendering {name}")):
        torch.cuda.synchronize()
        start = time.time()
        image = render_view(cam, gaussians, pipe, background)
        torch.cuda.synchronize()
        times.append(time.time() - start)

        torchvision.utils.save_image(image, os.path.join(render_dir, f"{idx:05d}.png"))
        frame = (image.detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        frames.append(frame)

    imageio.mimwrite(video_path, frames, fps=fps, macro_block_size=1)
    if len(times) > 5:
        print(f"{name} FPS: {1.0 / np.mean(times[5:]):.4f}")
    print(f"Saved {video_path}")


def render_sets(dataset, pipe, args):
    with torch.no_grad():
        scene, gaussians, _ = create_scene_and_gaussians(dataset, None, pipe, args, load_checkpoint=True, shuffle=False)
        background = background_tensor(dataset)
        iteration = getattr(args, "iteration", -1)
        label = iteration if iteration and iteration > 0 else "latest"

        if not args.skip_train:
            render_split(scene.model_path, "train", label, scene.getTrainCameras(), gaussians, pipe, background, args.fps)
        if not args.skip_test:
            render_split(scene.model_path, "test", label, scene.getTestCameras(), gaussians, pipe, background, args.fps)


if __name__ == "__main__":
    parser = ArgumentParser(description="Render an OMG4 4DGS checkpoint saved by train.py or the edit/refine scripts")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    add_omg4_runtime_args(parser)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--fps", default=30, type=int)
    parser.add_argument("--quiet", action="store_true")
    args = merge_config_args(get_combined_args(parser))

    print("Rendering", args.model_path)
    safe_state(args.quiet)
    dataset = extract_source_dataset(model, args)
    render_sets(dataset, pipeline.extract(args), args)
