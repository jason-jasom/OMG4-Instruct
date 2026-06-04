import math
import os
import random
from argparse import ArgumentParser, Namespace

import torch
import torch.nn.functional as F
from einops import rearrange
from torch.utils.data import DataLoader
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
from utils.loader_utils import FineSampler

from pytorch_lightning import seed_everything   

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
    if not render_pkgs:
        return

    visibility_stack = torch.stack([pkg["visibility_filter"] for pkg in render_pkgs], dim=1)
    visibility_count = visibility_stack.sum(dim=1)
    visibility_filter = visibility_count > 0
    radii = torch.stack([pkg["radii"] for pkg in render_pkgs], dim=1).max(dim=1).values
    gaussians.max_radii2D[visibility_filter] = torch.max(
        gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
    )

    point_grads = []
    for pkg in render_pkgs:
        viewspace_grad = pkg["viewspace_points"].grad
        if viewspace_grad is None:
            viewspace_grad = torch.zeros_like(pkg["viewspace_points"])
        point_grads.append(torch.norm(viewspace_grad[:, :2], dim=-1))
    viewspace_point_grad = torch.stack(point_grads, dim=1).sum(dim=1)
    viewspace_point_grad[visibility_filter] *= len(render_pkgs) / visibility_count[visibility_filter].clamp_min(1)
    viewspace_point_grad = viewspace_point_grad.unsqueeze(1)

    t_grad = None
    if gaussians.gaussian_dim == 4:
        if gaussians._t.grad is None:
            t_grad = torch.zeros_like(gaussians._t)
        else:
            t_grad = gaussians._t.grad.detach().clone()
            t_grad[visibility_filter] *= len(render_pkgs) / visibility_count[visibility_filter].clamp_min(1).unsqueeze(1)
    gaussians.add_densification_stats_grad(viewspace_point_grad, visibility_filter, t_grad)

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


def encode_sample(ip2p, images):
    return ip2p.vae.encode(2 * images - 1).latent_dist.sample() * 0.18215


def encode_mode(ip2p, images):
    return ip2p.vae.encode(2 * images - 1).latent_dist.mode()


def build_ip2p(device, dtype):
    from diffusers import AutoencoderKL, DDIMScheduler
    from transformers import CLIPTextModel, CLIPTokenizer

    from ip2p_models.models.ip2p_pipeline import InstructPix2PixPipeline
    from ip2p_models.models.ip2p_unet import UNet3DConditionModel

    ddim_source = "CompVis/stable-diffusion-v1-4"
    ip2p_source = "timbrooks/instruct-pix2pix"
    seed_everything(20211202)
    tokenizer = CLIPTokenizer.from_pretrained(ip2p_source, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(ip2p_source, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(ip2p_source, subfolder="vae")
    unet = UNet3DConditionModel.from_pretrained_2d(ip2p_source, subfolder="unet")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    return InstructPix2PixPipeline(
        vae=vae.to(device, dtype=dtype),
        text_encoder=text_encoder.to(device, dtype=dtype),
        tokenizer=tokenizer,
        unet=unet.to(device, dtype=dtype),
        scheduler=DDIMScheduler.from_pretrained(ddim_source, subfolder="scheduler"),
    )


def resize_for_diffusion(images, max_side):
    _, _, height, width = images.shape
    factor = max_side / max(width, height)
    short = max(64, math.ceil(min(width, height) * factor / 64) * 64)
    factor = short / min(width, height)
    new_width = max(64, int(width * factor) // 64 * 64)
    new_height = max(64, int(height * factor) // 64 * 64)
    return F.interpolate(images, size=(new_height, new_width), mode="bilinear", align_corners=False)


def sds_loss(ip2p, rendered, cond_images, prompt, args, device, dtype):
    sequence_length = rendered.shape[0]
    rendered = resize_for_diffusion(rendered, args.resize).to(device=device, dtype=dtype)
    cond_images = resize_for_diffusion(cond_images, args.resize).to(device=device, dtype=dtype)

    latents = encode_sample(ip2p, rendered)
    image_latents = encode_mode(ip2p, cond_images)
    latents = rearrange(latents, "(b f) c h w -> b c f h w", b=1, f=sequence_length)
    image_latents = rearrange(image_latents, "(b f) c h w -> b c f h w", b=1, f=sequence_length)
    uncond_image_latents = torch.zeros_like(image_latents)

    prompt_embeds = ip2p._encode_prompt(
        prompt,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
    )

    ip2p.scheduler.config.num_train_timesteps = args.num_train_timesteps
    ip2p.scheduler.set_timesteps(args.diffusion_steps)

    noise = torch.randn_like(latents)
    t = torch.randint(
        int(args.num_train_timesteps * args.t_min),
        int(args.num_train_timesteps * args.t_max),
        [1],
        dtype=torch.long,
        device=device,
    )
    noisy_latents = ip2p.scheduler.add_noise(latents, noise, t)

    image_latents = torch.cat([image_latents, image_latents, uncond_image_latents], dim=0)
    latent_model_input = torch.cat([noisy_latents] * 3, dim=0)
    latent_model_input = torch.cat([latent_model_input, image_latents], dim=1)

    with torch.no_grad():
        noise_pred = ip2p.unet(latent_model_input, t, prompt_embeds, None, None, False)[0]
        noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
        noise_pred = (
            noise_pred_uncond
            + args.guidance_scale * (noise_pred_text - noise_pred_image)
            + args.image_guidance_scale * (noise_pred_image - noise_pred_uncond)
        )

    alphas = ip2p.scheduler.alphas_cumprod.to(device)
    weight = (1 - alphas[t]).view(-1, 1, 1, 1, 1)
    grad = torch.nan_to_num(weight * (noise_pred - noise))
    target = (noisy_latents - grad).detach()
    return 0.5 * F.mse_loss(noisy_latents.float(), target.float(), reduction="sum") / sequence_length


def build_view_loader(train_views, args):
    if args.custom_sampler:
        sampler = FineSampler(
            train_views,
            repeats_per_frame=args.sampler_repeats_per_frame,
            history_mix=args.sampler_history_mix,
        )
        return DataLoader(
            train_views,
            batch_size=args.sequence_length,
            sampler=sampler,
            num_workers=args.sampler_workers,
            collate_fn=list,
            drop_last=True,
        )

    return DataLoader(
        train_views,
        batch_size=args.sequence_length,
        shuffle=True,
        num_workers=args.sampler_workers,
        collate_fn=list,
        drop_last=True,
    )


def train_sds(dataset, opt, pipe, args):
    tb_writer = prepare_output(args)
    dataset.dataloader = True
    scene, gaussians, _ = create_scene_and_gaussians(dataset, opt, pipe, args, load_checkpoint=True, shuffle=False)
    print("3DGS-only SDS refine: optimizing canonical 3D Gaussian parameters; time/4D and decoded MLP parameters are frozen.")
    print(f"Densification/pruning is {'enabled' if args.densify else 'disabled'}.")

    device = torch.device("cuda:0")
    dtype = torch.float16
    ip2p = build_ip2p(device, dtype)
    background = background_tensor(dataset)
    train_views = scene.getTrainCameras()
    if len(train_views) < args.sequence_length:
        raise RuntimeError(f"Need at least {args.sequence_length} training views for SDS refinement.")
    view_loader = build_view_loader(train_views, args)
    view_iter = iter(view_loader)

    progress = tqdm(range(1, opt.iterations + 1), desc="SDS refine")
    ema = 0.0
    for iteration in progress:
        gaussians.update_learning_rate(iteration)
        try:
            batch = next(view_iter)
        except StopIteration:
            view_iter = iter(view_loader)
            batch = next(view_iter)

        renders = []
        conds = []
        render_pkgs = []
        for gt, cam in batch:
            pkg = render_view(cam, gaussians, pipe, background)
            renders.append(pkg["render"].unsqueeze(0))
            conds.append(gt[:3].cuda().unsqueeze(0))
            if args.densify:
                render_pkgs.append(pkg)
        render_tensor = torch.cat(renders, dim=0).clamp(0, 1)
        cond_tensor = torch.cat(conds, dim=0).clamp(0, 1)

        loss = sds_loss(ip2p, render_tensor, cond_tensor, args.prompt, args, device, dtype)
        loss.backward()
        if torch.isnan(loss):
            raise RuntimeError("SDS loss became NaN during refinement.")

        if args.densify:
            with torch.no_grad():
                update_densification_stats(gaussians, render_pkgs, iteration, opt, scene)

        step_omg4_optimizers(gaussians)

        ema = 0.4 * loss.item() + 0.6 * ema
        if iteration % 10 == 0:
            progress.set_postfix({"loss": f"{ema:.6f}", "points": gaussians.get_xyz.shape[0]})
        if tb_writer is not None and iteration % args.log_interval == 0:
            tb_writer.add_scalar("sds/train_loss", loss.item(), iteration)
            tb_writer.add_scalar("sds/total_points", gaussians.get_xyz.shape[0], iteration)
            tb_writer.add_images("sds/train/render", render_tensor[:1], iteration)
        if iteration in args.save_iterations:
            print(f"\\n[ITER {iteration}] Saving SDS-refined OMG4 4DGS")
            save_gaussians(gaussians, scene.model_path, f"chkpnt_sds_{safe_prompt_token(args.prompt)}_", iteration)

    if tb_writer is not None:
        tb_writer.close()


if __name__ == "__main__":
    parser = ArgumentParser(description="InstructPix2Pix SDS refinement for an OMG4 4DGS checkpoint saved by train.py")
    lp = ModelParams(parser, sentinel=True)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    add_omg4_runtime_args(parser)
    parser.add_argument("--prompt", default="", type=str)
    parser.add_argument("--canonical_only", dest="canonical_only", action="store_true", default=False, help="Deprecated: this script now always optimizes canonical 3DGS parameters only.")
    parser.add_argument("--optimize_view_feature", dest="canonical_only", action="store_false", help="Deprecated: this script now always optimizes canonical 3DGS parameters only.")
    parser.add_argument("--comp_net_lr", default=0.0, type=float, help="Deprecated: decoded MLPs are frozen in 3DGS-only SDS refinement.")
    parser.add_argument("--appearance_mlp_lr", default=0.0, type=float, help="Deprecated: decoded MLPs are frozen in 3DGS-only SDS refinement.")
    parser.add_argument("--cont_mlp_lr", default=0.0, type=float, help="Deprecated: decoded MLPs are frozen in 3DGS-only SDS refinement.")
    parser.add_argument("--sequence_length", default=4, type=int)
    parser.add_argument("--custom_sampler", dest="custom_sampler", action="store_true", help="Use utils.loader_utils.FineSampler for frame-aware view sampling.")
    parser.add_argument("--sampler_workers", default=0, type=int)
    parser.add_argument("--sampler_repeats_per_frame", default=4, type=int)
    parser.add_argument("--sampler_history_mix", default=2, type=int)
    parser.add_argument("--resize", default=512, type=int)
    parser.add_argument("--diffusion_steps", default=20, type=int)
    parser.add_argument("--num_train_timesteps", default=1000, type=int)
    parser.add_argument("--t_min", default=0.02, type=float)
    parser.add_argument("--t_max", default=0.98, type=float)
    parser.add_argument("--guidance_scale", default=10.5, type=float)
    parser.add_argument("--image_guidance_scale", default=1.2, type=float)
    parser.add_argument("--densify", dest="densify", action="store_true", default=False, help="Enable Gaussian densification/pruning during SDS refinement.")
    parser.add_argument("--disable_densify", dest="densify", action="store_false")
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[100, 300, 500, 800])
    parser.add_argument("--log_interval", default=25, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    args = merge_config_args(get_combined_args(parser))
    args.optimize_decoded_appearance = False
    args.optimize_canonical_3dgs = True
    args.optimize_all_except_mlp = False
    args.canonical_only = False
    args.comp_net_lr = 0.0
    args.appearance_mlp_lr = 0.0
    args.cont_mlp_lr = 0.0

    if not args.prompt:
        raise ValueError("Please provide --prompt for SDS refinement.")
    args.save_iterations = sorted(set(args.save_iterations + [args.iterations]))

    print("SDS refining", args.model_path)
    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    train_sds(extract_source_dataset(lp, args), op.extract(args), pp.extract(args), args)
    print("\\nSDS refinement complete.")
