import random
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data.sampler import Sampler


def unwrap_camera(item):
    """Return the camera object from either a Camera or a dataset item."""
    if isinstance(item, (tuple, list)) and len(item) >= 2:
        return item[1]
    return item


def camera_stem(camera):
    return Path(str(getattr(camera, "image_name", getattr(camera, "uid", "camera")))).stem


def camera_name_parts(camera):
    stem = camera_stem(camera)
    camera_id = None
    frame_id = None
    parts = stem.split("_")

    if parts and parts[0].startswith("cam"):
        try:
            camera_id = int(parts[0][3:])
        except ValueError:
            camera_id = None

    if len(parts) > 1:
        try:
            frame_id = int(parts[-1])
        except ValueError:
            frame_id = None

    return stem, camera_id, frame_id


def get_camera_id(camera, fallback=0):
    _, camera_id, _ = camera_name_parts(camera)
    if camera_id is not None:
        return camera_id
    for attr in ("colmap_id", "uid"):
        value = getattr(camera, attr, None)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
    return fallback


def get_frame_id(camera, fallback=0):
    _, _, frame_id = camera_name_parts(camera)
    if frame_id is not None:
        return frame_id
    timestamp = getattr(camera, "timestamp", None)
    if timestamp is not None:
        try:
            return int(round(float(timestamp)))
        except (TypeError, ValueError):
            pass
    return fallback


def _iter_indexed_cameras(dataset):
    if hasattr(dataset, "viewpoint_stack"):
        for idx, camera in enumerate(dataset.viewpoint_stack):
            yield idx, camera
        return

    for idx in range(len(dataset)):
        yield idx, unwrap_camera(dataset[idx])


def build_frame_index(dataset):
    frame_to_indices = defaultdict(list)
    camera_to_indices = defaultdict(list)

    for idx, camera in _iter_indexed_cameras(dataset):
        frame_to_indices[get_frame_id(camera, idx)].append(idx)
        camera_to_indices[get_camera_id(camera, idx)].append(idx)

    frame_to_indices = dict(sorted(frame_to_indices.items()))
    camera_to_indices = dict(sorted(camera_to_indices.items()))
    return frame_to_indices, camera_to_indices


def get_stamp_list(dataset, timestamp):
    """Return all dataset items whose camera image belongs to the requested frame.

    Original dynamic-3DGS code assumed indices were laid out as
    pose_id * frame_length + timestamp. 4D-Scaffold-GS datasets expose
    camera names like cam01_0000, so this version groups by the parsed
    frame_id while still returning dataset items just like the old helper.
    """
    frame_to_indices, _ = build_frame_index(dataset)
    if timestamp not in frame_to_indices:
        available = list(frame_to_indices.keys())
        raise IndexError(f"timestamp/frame {timestamp} not found. Available range: {available[:3]} ... {available[-3:]}")

    indices = frame_to_indices[timestamp]
    print("select index:", indices)
    return [dataset[idx] for idx in indices]


class FineSampler(Sampler):
    """Frame-aware sampler compatible with old datasets and CameraDataset.

    It iterates cameras grouped by frame_id. For each frame, the camera order
    is randomized several times, with a small amount of history mixing to keep
    the original sampler's cross-frame behavior.
    """

    def __init__(self, dataset, repeats_per_frame=4, history_mix=2, seed=None):
        self.dataset = dataset
        self.repeats_per_frame = repeats_per_frame
        self.history_mix = history_mix
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

        self.frame_to_indices, self.camera_to_indices = build_frame_index(dataset)
        self.frame_ids = list(self.frame_to_indices.keys())
        self.len_dataset = len(dataset)
        self.len_pose = len(self.camera_to_indices)
        self.frame_length = len(self.frame_ids)
        self.sample_list = self._build_sample_list()
        print("one epoch containing:", len(self.sample_list))

    def _build_sample_list(self):
        sample_list = []
        for frame_id in self.frame_ids:
            frame_indices = self.frame_to_indices[frame_id]
            for _ in range(self.repeats_per_frame):
                perm = torch.randperm(len(frame_indices), generator=self.generator).tolist()
                now_list = []
                for count, perm_idx in enumerate(perm, start=1):
                    now_list.append(frame_indices[perm_idx])
                    if count % 2 == 0 and len(sample_list) > self.history_mix:
                        now_list.extend(random.sample(sample_list, self.history_mix))
                sample_list.extend(now_list)
        return sample_list

    def __iter__(self):
        return iter(self.sample_list)

    def __len__(self):
        return len(self.sample_list)
