import os
import random
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams, PipelineParams, get_combined_args
from gaussian_renderer import render
from omg4_editing_utils import (
    add_omg4_runtime_args,
    create_scene_and_gaussians,
    extract_source_dataset,
    merge_config_args,
    save_gaussians,
    safe_prompt_token,
    step_omg4_optimizers,
)
from utils.general_utils import safe_state
from utils.image_utils import psnr
from utils.loader_utils import camera_name_parts
from utils.loss_utils import l1_loss, ssim

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


def background_tensor(dataset):
    return torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")


def render_view(cam, gaussians, pipe, background):
    cam = cam.cuda()
    return render(cam, gaussians, pipe, background)


def update_densification_stats(gaussians, render_pkgs, iteration, opt, scene):
    if iteration >= opt.densify_until_iter:
        return
    if opt.densify_until_num_points >= 0 and gaussians.get_xyz.shape[0] >= opt.densify_until_num_points:
        return

    for pkg in render_pkgs:
        visibility_filter = pkg["visibility_filter"]
        radii = pkg["radii"]
        viewspace_points = pkg["viewspace_points"]
        gaussians.max_radii2D[visibility_filter] = torch.max(
            gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
        )
        t_grad = gaussians._t.grad.detach() if gaussians.gaussian_dim == 4 and gaussians._t.grad is not None else None
        gaussians.add_densification_stats(viewspace_points, visibility_filter, t_grad)

    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
        gaussians.densify_and_prune(
            opt.densify_grad_threshold,
            opt.thresh_opa_prune,
            scene.cameras_extent,
            size_threshold,
            opt.densify_grad_t_threshold,
        )

    if iteration % opt.opacity_reset_interval == 0 or (scene.white_background and iteration == opt.densify_from_iter):
        gaussians.reset_opacity()


def camera_stem(cam):
    return Path(str(getattr(cam, "image_name", getattr(cam, "uid", "camera")))).stem


def find_target_image(edited_dir, cam, index, pattern):
    edited_dir = Path(edited_dir)
    stem, camera_id, frame_id = camera_name_parts(cam)
    if pattern:
        positional_id = camera_id if camera_id is not None else index
        values = {
            "image_name": stem,
            "uid": getattr(cam, "uid", index),
            "index": index,
            "camera_id": camera_id if camera_id is not None else index,
            "frame_id": frame_id if frame_id is not None else index,
            "timestamp": getattr(cam, "timestamp", 0.0),
        }
        try:
            filename = pattern.format(positional_id, **values)
        except (IndexError, KeyError):
            filename = pattern.format(**values)
        path = edited_dir / filename
        if path.exists():
            return path

    for ext in (".png", ".jpg", ".jpeg"):
        names = [stem, str(index), f"{index:05d}", str(getattr(cam, "uid", index))]
        if camera_id is not None:
            names.extend([str(camera_id), f"{camera_id:02d}", f"{camera_id:05d}"])
        for name in names:
            path = edited_dir / f"{name}{ext}"
            if path.exists():
                return path
    return None


def camera_dataset_items(camera_dataset, load_images=False):
    if not load_images and hasattr(camera_dataset, "viewpoint_stack"):
        return [(None, cam) for cam in camera_dataset.viewpoint_stack]
    return [camera_dataset[idx] for idx in tqdm(range(len(camera_dataset)), desc="Loading source images")]


def build_target_pairs(train_views, args):
    pairs = []
    missing = 0
    for idx, (gt, cam) in enumerate(tqdm(train_views, desc="Matching edited targets")):
        target_path = find_target_image(args.edited_images_path, cam, idx, args.edited_pattern)
        if target_path is None:
            missing += 1
            if args.fallback_original:
                pairs.append((idx, gt, cam, None))
        else:
            pairs.append((idx, gt, cam, target_path))

    if not pairs:
        raise RuntimeError(
            f"No edited targets were matched in {args.edited_images_path}. "
            "Use --edited_pattern like 'edited_statue_original_time0_{}.png' "
            "or 'edited_statue_original_time0_{index}.png'."
        )

    if missing:
        print(f"Matched {len(pairs)} training views with targets; skipped {missing} views without edited images.")
    else:
        print(f"Matched all {len(pairs)} training views with edited targets.")
    return pairs


def load_target(path, size):
    image = Image.open(path).convert("RGB")
    target = torchvision.transforms.functional.to_tensor(image).cuda()
    target = F.interpolate(target.unsqueeze(0), size=size, mode="bilinear", align_corners=False).squeeze(0)
    return target.clamp(0, 1)


def finite_image(tensor, label, iteration, cam, target_path=None):
    if torch.isfinite(tensor).all():
        return tensor.clamp(0, 1)
    finite = torch.isfinite(tensor)
    bad = tensor.numel() - finite.sum().item()
    cam_name = getattr(cam, "image_name", getattr(cam, "uid", "camera"))
    print(
        f"[WARN] {label} has {bad}/{tensor.numel()} non-finite values at iter {iteration}, "
        f"camera {cam_name}, target {target_path}. Replacing them for this step."
    )
    return torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1)


def prepare_output(args):
    output_dir = getattr(args, "out_path", None) or args.model_path
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "cfg_args"), "w") as cfg_file:
        cfg_file.write(str(Namespace(**vars(args))))
    if SummaryWriter is None:
        return None
    try:
        return SummaryWriter(output_dir)
    except Exception:
        return None


def report(tb_writer, iteration, scene, gaussians, pipe, background, loss_value, l1_value):
    if tb_writer is None:
        return
    tb_writer.add_scalar("edit/train_loss/l1", l1_value.item(), iteration)
    tb_writer.add_scalar("edit/train_loss/total", loss_value.item(), iteration)
    tb_writer.add_scalar("edit/total_points", gaussians.get_xyz.shape[0], iteration)

    with torch.no_grad():
        views = list(scene.getTestCameras())
        if not views:
            return
        gt, cam = views[min(iteration % len(views), len(views) - 1)]
        image = render_view(cam, gaussians, pipe, background)["render"].clamp(0, 1)
        tb_writer.add_images("edit/test/render", image[None], iteration)
        tb_writer.add_scalar("edit/test/psnr_to_original", psnr(image, gt.cuda()).mean().item(), iteration)


def train_edit(dataset, opt, pipe, args):
    args.densify = False
    tb_writer = prepare_output(args)
    dataset.dataloader = True
    scene, gaussians, _ = create_scene_and_gaussians(dataset, opt, pipe, args, load_checkpoint=True, shuffle=False)
    print("3DGS-only edit: optimizing canonical 3D Gaussian parameters; time/4D and decoded MLP parameters are frozen.")

    background = background_tensor(dataset)
    print("Preparing camera metadata and matching edited targets...")
    train_views = camera_dataset_items(scene.getTrainCameras(), load_images=args.fallback_original)
    if args.include_test_targets:
        train_views.extend(camera_dataset_items(scene.getTestCameras(), load_images=args.fallback_original))
    if not train_views:
        raise RuntimeError("No training cameras were loaded.")

    target_pairs = build_target_pairs(train_views, args)
    progress = tqdm(range(1, opt.iterations + 1), desc="3D edit")
    ema = 0.0
    for iteration in progress:
        gaussians.update_learning_rate(iteration)
        batch = random.sample(target_pairs, min(args.batch_size, len(target_pairs)))

        images = []
        targets = []
        render_pkgs = []
        for idx, gt, cam, target_path in batch:
            pkg = render_view(cam, gaussians, pipe, background)
            image = finite_image(pkg["render"], "render", iteration, cam, target_path)
            if target_path is None:
                target = gt.cuda()
            else:
                target = load_target(target_path, image.shape[-2:])
            target = finite_image(target[:3], "target", iteration, cam, target_path)
            images.append(image.unsqueeze(0))
            targets.append(target.unsqueeze(0))
            if args.densify:
                render_pkgs.append(pkg)

        image_tensor = torch.cat(images, dim=0)
        target_tensor = torch.cat(targets, dim=0)
        l1 = l1_loss(image_tensor, target_tensor)
        if opt.lambda_dssim > 0:
            ssim_value = ssim(image_tensor, target_tensor)
            if not torch.isfinite(ssim_value):
                print(f"[WARN] SSIM became non-finite at iter {iteration}; using L1 for this step.")
                ssim_value = image_tensor.new_tensor(1.0)
        else:
            ssim_value = image_tensor.new_tensor(1.0)
        loss = (1.0 - opt.lambda_dssim) * l1 + opt.lambda_dssim * (1.0 - ssim_value)
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Loss became non-finite at iter {iteration}. "
                f"render range=({image_tensor.min().item():.4g}, {image_tensor.max().item():.4g}), "
                f"target range=({target_tensor.min().item():.4g}, {target_tensor.max().item():.4g}), "
                f"l1={l1.item():.4g}, ssim={ssim_value.item():.4g}"
            )
        loss.backward()

        if args.densify:
            with torch.no_grad():
                update_densification_stats(gaussians, render_pkgs, iteration, opt, scene)

        step_omg4_optimizers(gaussians)

        ema = 0.4 * loss.item() + 0.6 * ema
        if iteration % 10 == 0:
            progress.set_postfix({"loss": f"{ema:.6f}", "points": gaussians.get_xyz.shape[0]})

        if iteration in args.save_iterations:
            print(f"\\n[ITER {iteration}] Saving edited OMG4 4DGS")
            save_gaussians(gaussians, scene.model_path, f"chkpnt_edit_{safe_prompt_token(args.prompt)}_", iteration)
        if iteration in args.test_iterations:
            report(tb_writer, iteration, scene, gaussians, pipe, background, loss, l1)

    if tb_writer is not None:
        tb_writer.close()


if __name__ == "__main__":
    parser = ArgumentParser(description="Supervised edit/refit for an OMG4 4DGS checkpoint saved by train.py")
    lp = ModelParams(parser, sentinel=True)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    add_omg4_runtime_args(parser)
    parser.add_argument("--edited_images_path", default="", type=str, help="Directory containing edited RGB targets.")
    parser.add_argument("--edited_pattern", default="", type=str, help="Optional pattern, e.g. '{image_name}.png' or '{index:05d}.png'.")
    parser.add_argument("--canonical_only", dest="canonical_only", action="store_true", default=False, help="Deprecated: this script now always optimizes canonical 3DGS parameters only.")
    parser.add_argument("--optimize_view_feature", dest="canonical_only", action="store_false", help="Deprecated: this script now always optimizes canonical 3DGS parameters only.")
    parser.add_argument("--comp_net_lr", default=0.0, type=float, help="Deprecated: decoded MLPs are frozen in 3DGS-only editing.")
    parser.add_argument("--appearance_mlp_lr", default=0.0, type=float, help="Deprecated: decoded MLPs are frozen in 3DGS-only editing.")
    parser.add_argument("--cont_mlp_lr", default=0.0, type=float, help="Deprecated: decoded MLPs are frozen in 3DGS-only editing.")
    parser.add_argument("--densify", dest="densify", action="store_true", default=False, help="Deprecated: densification/pruning is disabled in 3DGS-only editing.")
    parser.add_argument("--disable_densify", dest="densify", action="store_false")
    parser.add_argument("--include_test_targets", action="store_true", default=False, help="Also use test camera metadata when matching edited targets.")
    parser.add_argument("--train_only_targets", dest="include_test_targets", action="store_false", help="Use train camera metadata only.")
    parser.add_argument("--fallback_original", action="store_true", help="Use original training images when an edited target is missing.")
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[500, 1000, 3000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[500, 1000, 3000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--dataset", default="", type=str, help="Kept for compatibility with old launch scripts.")
    parser.add_argument("--scene", default="", type=str, help="Kept for compatibility with old launch scripts.")
    parser.add_argument("--prompt", default="", type=str, help="Kept for compatibility with old launch scripts.")
    args = merge_config_args(get_combined_args(parser))
    args.optimize_decoded_appearance = False
    args.optimize_canonical_3dgs = True
    args.optimize_all_except_mlp = False
    args.canonical_only = False
    args.comp_net_lr = 0.0
    args.appearance_mlp_lr = 0.0
    args.cont_mlp_lr = 0.0
    args.densify = False

    if not args.edited_images_path:
        if args.dataset and args.scene and args.prompt:
            args.edited_images_path = f"./data/{args.dataset}/{args.scene}/{args.prompt.split(' ')[-1].replace('?', '')}"
        else:
            print("Please pass --edited_images_path.", file=sys.stderr)
            sys.exit(2)
    args.save_iterations = sorted(set(args.save_iterations + [args.iterations]))

    print("Editing", args.model_path)
    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    train_edit(extract_source_dataset(lp, args), op.extract(args), pp.extract(args), args)
    print("\\nEditing complete.")
