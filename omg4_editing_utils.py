import glob
import os
import re
import sys
from argparse import Namespace

import torch
from torch import nn
from omegaconf import DictConfig, OmegaConf
from PIL import Image
import torchvision.transforms as transforms

from scene import Scene, GaussianModel
from utils.compress_utils import load_comp
from utils.general_utils import get_expon_lr_func


def add_omg4_runtime_args(parser):
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--configs", type=str, default=None, help="Alias for --config.")
    parser.add_argument("--checkpoint", type=str, default=None, help="OMG4 train.py chkpnt*.pth to load.")
    parser.add_argument("--comp_checkpoint", type=str, default=None, help="OMG4 encoded comp.xz to decode as the source model.")
    parser.add_argument("--iteration", type=int, default=-1, help="Checkpoint iteration when --checkpoint is omitted.")
    parser.add_argument("--gaussian_dim", type=int, default=4)
    parser.add_argument("--time_duration", nargs=2, type=float, default=[-0.5, 0.5])
    parser.add_argument("--num_pts", type=int, default=100_000)
    parser.add_argument("--num_pts_ratio", type=float, default=1.0)
    parser.add_argument("--rot_4d", action="store_true")
    parser.add_argument("--force_sh_3d", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=6666)
    parser.add_argument("--out_path", type=str, default=None)


def merge_config_args(args):
    config_path = args.config or args.configs
    if not config_path:
        return args
    cfg = OmegaConf.load(config_path)
    explicit_cli_args = {
        token[2:].split("=", 1)[0].replace("-", "_")
        for token in sys.argv[1:]
        if token.startswith("--")
    }

    def recursive_merge(key, host):
        if isinstance(host[key], DictConfig):
            for child in host[key].keys():
                recursive_merge(child, host[key])
        else:
            if key not in explicit_cli_args:
                setattr(args, key, host[key])

    for key in cfg.keys():
        recursive_merge(key, cfg)
    return args


def extract_source_dataset(model_params, args):
    output_path = getattr(args, "out_path", None)
    if hasattr(args, "out_path"):
        args.out_path = None
    dataset = model_params.extract(args)
    if hasattr(args, "out_path"):
        args.out_path = output_path
    return dataset


def normalize_time_duration(dataset, time_duration):
    if getattr(dataset, "frame_ratio", 1) > 1:
        return [time_duration[0] / dataset.frame_ratio, time_duration[1] / dataset.frame_ratio]
    return list(time_duration)


def resolve_checkpoint(model_path, checkpoint=None, iteration=-1):
    if checkpoint:
        return checkpoint
    if iteration is not None and iteration > 0:
        path = os.path.join(model_path, f"chkpnt{iteration}.pth")
        if os.path.exists(path):
            return path
    candidates = glob.glob(os.path.join(model_path, "chkpnt*.pth"))
    if not candidates:
        raise FileNotFoundError(
            f"No OMG4 checkpoint found under {model_path}. Pass --checkpoint /path/to/chkpnt*.pth."
        )

    def checkpoint_iter(path):
        match = re.search(r"chkpnt(?:_.*?_)?(\d+)\.pth$", os.path.basename(path))
        return int(match.group(1)) if match else -1

    return max(candidates, key=checkpoint_iter)


def _as_parameter(value):
    if isinstance(value, nn.Parameter):
        return value
    return nn.Parameter(value.detach().clone().float().cuda().requires_grad_(True))


def setup_decoded_comp_training(gaussians, opt, optimize_geometry=False, net_lr=0.0):
    gaussians._xyz = _as_parameter(gaussians._xyz)
    gaussians._scaling = _as_parameter(gaussians._scaling)
    gaussians._rotation = _as_parameter(gaussians._rotation)
    gaussians._t = _as_parameter(gaussians._t)
    gaussians._scaling_t = _as_parameter(gaussians._scaling_t)
    if gaussians.rot_4d:
        gaussians._rotation_r = _as_parameter(gaussians._rotation_r)
    gaussians._features_static = _as_parameter(gaussians._features_static)
    gaussians._features_view = _as_parameter(gaussians._features_view)

    gaussians.percent_dense = opt.percent_dense
    gaussians.xyz_gradient_accum = torch.zeros((gaussians.get_xyz.shape[0], 1), device="cuda")
    gaussians.t_gradient_accum = torch.zeros((gaussians.get_xyz.shape[0], 1), device="cuda")
    gaussians.denom = torch.zeros((gaussians.get_xyz.shape[0], 1), device="cuda")
    gaussians.max_radii2D = torch.zeros((gaussians.get_xyz.shape[0]), device="cuda")

    groups = [
        {"params": [gaussians._features_static], "lr": opt.feature_lr, "name": "f_static"},
        {"params": [gaussians._features_view], "lr": opt.feature_lr, "name": "f_view"},
    ]
    if optimize_geometry:
        groups.extend([
            {"params": [gaussians._xyz], "lr": opt.position_lr_init * gaussians.spatial_lr_scale, "name": "xyz"},
            {"params": [gaussians._scaling], "lr": opt.scaling_lr, "name": "scaling"},
            {"params": [gaussians._rotation], "lr": opt.rotation_lr, "name": "rotation"},
            {"params": [gaussians._t], "lr": (opt.position_t_lr_init if opt.position_t_lr_init > 0 else opt.position_lr_init) * gaussians.spatial_lr_scale, "name": "t"},
            {"params": [gaussians._scaling_t], "lr": opt.scaling_lr, "name": "scaling_t"},
        ])
        if gaussians.rot_4d:
            groups.append({"params": [gaussians._rotation_r], "lr": opt.rotation_lr, "name": "rotation_r"})
    gaussians.optimizer = torch.optim.Adam(groups, lr=0.0, eps=1e-8)
    gaussians.xyz_scheduler_args = get_expon_lr_func(
        lr_init=opt.position_lr_init * gaussians.spatial_lr_scale,
        lr_final=opt.position_lr_final * gaussians.spatial_lr_scale,
        lr_delay_mult=opt.position_lr_delay_mult,
        max_steps=opt.position_lr_max_steps,
    )

    gaussians.optimizer_net = None
    if net_lr > 0:
        mlp_params = []
        for module in (gaussians.mlp_cont, gaussians.mlp_view, gaussians.mlp_dc, gaussians.mlp_opacity):
            mlp_params.extend(list(module.parameters()))
        # tcnn stores these params in half precision; a tiny Adam eps can underflow and
        # turn the first update into Inf/NaN. Keep this opt-in and use a fp16-safe eps.
        gaussians.optimizer_net = torch.optim.Adam(mlp_params, lr=net_lr, eps=1e-4)


def load_decoded_comp(gaussians, comp_path, opt=None):
    gaussians.construct_net(train=False)
    gaussians.decode(load_comp(comp_path), decompress=True)
    gaussians.active_sh_degree = gaussians.max_sh_degree
    gaussians.active_sh_degree_t = gaussians.max_sh_degree_t
    if hasattr(gaussians, "env_map") and gaussians.env_map.numel() and gaussians.env_map.device.type != "cuda":
        gaussians.env_map = gaussians.env_map.cuda()
    if opt is not None:
        setup_decoded_comp_training(gaussians, opt)


def capture_decoded_comp(gaussians):
    return {
        "format": "omg4_decoded_comp_v1",
        "active_sh_degree": gaussians.active_sh_degree,
        "active_sh_degree_t": gaussians.active_sh_degree_t,
        "xyz": gaussians._xyz.detach().cpu(),
        "scaling": gaussians._scaling.detach().cpu(),
        "rotation": gaussians._rotation.detach().cpu(),
        "t": gaussians._t.detach().cpu(),
        "scaling_t": gaussians._scaling_t.detach().cpu(),
        "rotation_r": gaussians._rotation_r.detach().cpu() if gaussians.rot_4d else None,
        "features_static": gaussians._features_static.detach().cpu(),
        "features_view": gaussians._features_view.detach().cpu(),
        "MLP_cont": gaussians.mlp_cont.params.detach().cpu(),
        "MLP_dc": gaussians.mlp_dc.params.detach().cpu(),
        "MLP_sh": gaussians.mlp_view.params.detach().cpu(),
        "MLP_opacity": gaussians.mlp_opacity.params.detach().cpu(),
    }


def restore_decoded_comp(gaussians, state, opt=None):
    gaussians.construct_net(train=False)
    gaussians.net_enabled = True
    gaussians.vq_enabled = False
    gaussians.active_sh_degree = state["active_sh_degree"]
    gaussians.active_sh_degree_t = state["active_sh_degree_t"]
    gaussians._xyz = state["xyz"].cuda().float()
    gaussians._scaling = state["scaling"].cuda().float()
    gaussians._rotation = state["rotation"].cuda().float()
    gaussians._t = state["t"].cuda().float()
    gaussians._scaling_t = state["scaling_t"].cuda().float()
    if gaussians.rot_4d and state.get("rotation_r") is not None:
        gaussians._rotation_r = state["rotation_r"].cuda().float()
    gaussians._features_static = nn.Parameter(state["features_static"].cuda().float().requires_grad_(True))
    gaussians._features_view = nn.Parameter(state["features_view"].cuda().float().requires_grad_(True))
    gaussians.mlp_cont.params = nn.Parameter(state["MLP_cont"].cuda().half().requires_grad_(True))
    gaussians.mlp_dc.params = nn.Parameter(state["MLP_dc"].cuda().half().requires_grad_(True))
    gaussians.mlp_view.params = nn.Parameter(state["MLP_sh"].cuda().half().requires_grad_(True))
    gaussians.mlp_opacity.params = nn.Parameter(state["MLP_opacity"].cuda().half().requires_grad_(True))
    if opt is not None:
        setup_decoded_comp_training(gaussians, opt)


def create_scene_and_gaussians(dataset, opt, pipe, args, load_checkpoint=True, shuffle=True):
    checkpoint = getattr(args, "checkpoint", None)
    comp_checkpoint = getattr(args, "comp_checkpoint", None)
    iteration = getattr(args, "iteration", -1)
    output_path = getattr(args, "out_path", None)
    if comp_checkpoint and not checkpoint:
        source_model_path = os.path.dirname(comp_checkpoint) or args.model_path
    else:
        source_model_path = os.path.dirname(checkpoint) if checkpoint else args.model_path
    output_model_path = output_path or source_model_path
    dataset.model_path = source_model_path
    time_duration = normalize_time_duration(dataset, getattr(args, "time_duration", [-0.5, 0.5]))
    gaussians = GaussianModel(
        dataset.sh_degree,
        gaussian_dim=getattr(args, "gaussian_dim", 4),
        time_duration=time_duration,
        rot_4d=getattr(args, "rot_4d", False),
        force_sh_3d=getattr(args, "force_sh_3d", False),
        sh_degree_t=2 if pipe.eval_shfs_4d else 0,
    )
    scene = Scene(
        dataset,
        gaussians,
        shuffle=shuffle,
        num_pts=getattr(args, "num_pts", 100_000),
        num_pts_ratio=getattr(args, "num_pts_ratio", 1.0),
        time_duration=time_duration,
    )
    if opt is not None and not comp_checkpoint:
        gaussians.training_setup(opt)
    loaded_from = None
    loaded_iter = None
    if load_checkpoint and comp_checkpoint:
        loaded_from = comp_checkpoint
        load_decoded_comp(gaussians, loaded_from, opt)
        if opt is not None and getattr(args, "comp_net_lr", 0.0) > 0:
            setup_decoded_comp_training(gaussians, opt, net_lr=args.comp_net_lr)
        loaded_iter = "comp"
        print(f"Loaded OMG4 compressed model: {loaded_from}")
    elif load_checkpoint:
        loaded_from = resolve_checkpoint(source_model_path, checkpoint, iteration)
        loaded_obj = torch.load(loaded_from, weights_only=False)
        if isinstance(loaded_obj, dict) and loaded_obj.get("format") == "omg4_decoded_comp_v1":
            restore_decoded_comp(gaussians, loaded_obj, opt)
            loaded_iter = "decoded_comp"
        else:
            model_params, loaded_iter = loaded_obj
            gaussians.restore(model_params, opt)
        print(f"Loaded OMG4 checkpoint: {loaded_from} (iteration {loaded_iter})")
    scene.model_path = output_model_path
    dataset.model_path = output_model_path
    return scene, gaussians, loaded_from


def get_camera_sample(sample):
    if isinstance(sample, tuple) and len(sample) == 2:
        gt_image, camera = sample
        return gt_image, camera
    if isinstance(sample, list) and len(sample) == 2:
        gt_image, camera = sample
        return gt_image, camera
    return getattr(sample, "image", None), sample


def camera_to_cuda(sample):
    gt_image, camera = get_camera_sample(sample)
    if gt_image is not None:
        gt_image = gt_image.cuda()
    if hasattr(camera, "cuda"):
        camera = camera.cuda()
    return gt_image, camera


def camera_name(camera):
    return str(getattr(camera, "image_name", getattr(camera, "uid", "camera")))


def save_gaussians(gaussians, output_dir, prefix, iteration):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{prefix}{iteration}.pth")
    if getattr(gaussians, "net_enabled", False) and hasattr(gaussians, "_features_static"):
        torch.save(capture_decoded_comp(gaussians), path)
    else:
        torch.save((gaussians.capture(), iteration), path)
    print(f"Saved checkpoint: {path}")
    return path


def step_omg4_optimizers(gaussians):
    for optimizer in (gaussians.optimizer, getattr(gaussians, "optimizer_net", None)):
        if optimizer is None:
            continue
        for group in optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None and not torch.isfinite(param.grad).all():
                    param.grad = torch.nan_to_num(param.grad, nan=0.0, posinf=0.0, neginf=0.0)
    gaussians.optimizer.step()
    gaussians.optimizer.zero_grad(set_to_none=True)
    if hasattr(gaussians, "optimizer_net") and gaussians.optimizer_net is not None:
        gaussians.optimizer_net.step()
        gaussians.optimizer_net.zero_grad(set_to_none=True)


def safe_prompt_token(prompt):
    token = re.sub(r"[^0-9A-Za-z._-]+", "_", prompt.strip()).strip("_")
    return token or "edit"


def load_edited_target(edited_images_path, camera, prompt="", scene_name="", mapping=None):
    name = camera_name(camera)
    prompt_tail = prompt.split(" ")[-1].replace("?", "") if prompt else "*"
    candidates = [
        os.path.join(edited_images_path, f"{name}.png"),
        os.path.join(edited_images_path, f"{name}.jpg"),
        os.path.join(edited_images_path, f"{name}.jpeg"),
        os.path.join(edited_images_path, f"edited_{name}.png"),
        os.path.join(edited_images_path, f"edited_{prompt_tail}_original_time0_{name}.png"),
    ]
    if mapping is not None:
        try:
            mapped = mapping.get(int(name))
            if mapped is not None:
                candidates.insert(0, os.path.join(edited_images_path, f"edited_{prompt_tail}_original_time0_{mapped}.png"))
        except ValueError:
            pass
    candidates.extend(sorted(glob.glob(os.path.join(edited_images_path, f"*{name}*.png"))))
    for path in candidates:
        if path and os.path.exists(path):
            return transforms.ToTensor()(Image.open(path).convert("RGB"))
    raise FileNotFoundError(f"Could not find an edited image for camera {name} in {edited_images_path}")


class SimpleTimer:
    def __init__(self):
        self.elapsed = 0.0

    def start(self):
        return None

    def pause(self):
        return None

    def get_elapsed_time(self):
        return self.elapsed
