# Optimized Minimal 4D Gaussian Splatting

Minseo Lee*, Byeonghyeon Lee*, Lucas Yunkyu Lee, Eunsoo Lee, Sangmin Kim, Seunghyeon Song, Joo Chan Lee, Jong Hwan Ko, Jaesik Park, and Eunbyung Park†

[Project Page](https://minshirley.github.io/OMG4/) &nbsp; [Paper](https://arxiv.org/abs/2510.03857)

![Teaser](https://github.com/MinShirley/OMG4/blob/main/assets/teaser.jpg?raw=true)

Our code is built based on [4D-GS](https://github.com/fudan-zvg/4d-gaussian-splatting)


## Setup
We ran the experiments in the following environment:
```
- ubuntu: 20.04
- python: 3.11
- cuda: 12.1
- pytorch: 2.5.1  ( > 2.5.0 is required for svq)
- GPU: RTX 3090
```

###  1. Installation
```
conda create -n OMG4 python=3.11
conda activate OMG4
pip install -r requirement.txt
```

Then, please download the pretrained 4D-GS weight and gradients.  
You can download the weights from [Google Drive](https://drive.google.com/drive/folders/1WB7WYOUlvemfYZE35lkl_WV4fiF3p68v?usp=sharing).


### 2. Data preparation
Data preprocessing follows the method used in [4D-GS](https://github.com/fudan-zvg/4d-gaussian-splatting).
Run the following command to prepare the data:
```
python scripts/n3v2blender.py data/N3V/$scene_name
```

The directory data/N3V/$scene_name should contain the following files before preprocessing:
```
data/N3V/$scene_name
├── cam00.mp4
├── cam01.mp4
├── ...
└── poses_bounds.npy
```

After running the script, the directory structure will look like this:
```
data/N3V/$scene_name
├── cam00.mp4
├── cam01.mp4
├── ...
├── poses_bounds.npy
├── transforms_train.json
├── transforms_test.json
└── images
    ├── cam00_0000.png
    ├── cam00_0001.png
    ├── ...
```

### 3. Training
Gradient (2D mean, t) should be calculated in advance to sample important Gaussians.
If you want to compute gradients, run the following command
```
python compute_gradient.py \
  --config ./configs/dynerf/cook_spinach.yaml \
  --start_checkpoint PATH_TO_4DGS_PRETRAINED \
  --out_path PATH_TO_GRADIENT
```

Once you compute gradients (or download provided gradients), please set --grad to your gradient path, not to compute them repeatedly.
```
python train.py \
  --config ./configs/dynerf/cook_spinach.yaml \
  --start_checkpoint PATH_TO_4DGS_PRETRAINED \
  --grad PATH_TO_GRADIENT \
  --out_path ./cook_spinach_comp
```
You can check the result (w/ various metrics, encoded model size, etc.) at **./res.txt**

### 4. Evaluation
At the end of training, the evaluation process is implemented. Or you can evaluate the trained model with the encoded "comp.xz" file with the following command
```
python test.py \
--config ./configs/dynerf/cook_spinach.yaml \
--comp_checkpoint ./cook_spinach_comp/comp.xz
```

The weights reported in our paper are available for download on [Google Drive](https://drive.google.com/drive/folders/1WB7WYOUlvemfYZE35lkl_WV4fiF3p68v?usp=sharing).

To evaluate OMG4-FTGS using a trained model, you can use the provided checkpoints.
The checkpoints are available on [Google Drive](https://drive.google.com/drive/folders/1WB7WYOUlvemfYZE35lkl_WV4fiF3p68v?usp=sharing).
```
python -m OMG4_FTGS.test \
    --comp_checkpoint ./OMG4-FTGS_weights/cook_spinach.xz \
    --data_path data/N3V/cook_spinach
```

## Instruction Edit

### 1. Install Instruct-4DGS packages

Install the packages required by [Instruct-4DGS](https://github.com/CHINHUICHU/Instruct-4DGS/tree/test).

### 2. Generate edited images with Instruct-4DGS

Use the following command from Instruct-4DGS to generate edited multi-view images:

```
python ./ip2p_models/multiview_edit.py \
    --dataset "${DATASET}" \
    --scene "${SCENE_NAME}" \
    --prompt "${PROMPT}" \
    ${RESIZE_FLAG} \
    --steps 20 \
    --guidance_scale ${GUIDANCE_SCALE} \
    --image_guidance_scale ${IMAGE_GUIDANCE_SCALE}
```

Arguments:
- `<DATASET>`: dataset name or dataset path used by Instruct-4DGS.
- `<SCENE_NAME>`: scene name to edit.
- `<PROMPT>`: text instruction for image editing.
- `<RESIZE_FLAG>`: optional resize flag used by Instruct-4DGS. Leave empty if resizing is not needed.
- `<GUIDANCE_SCALE>`: text guidance scale.
- `<IMAGE_GUIDANCE_SCALE>`: image guidance scale.

### 3. Generate the edited video

After obtaining the edited images, use the same pipeline as `script.sh` to optimize and render the edited 4D result:

```
conda activate OMG4

python edit_3d.py \
  --config <CONFIG_PATH> \
  --comp_checkpoint <COMP_CHECKPOINT_PATH> \
  --edited_images_path <EDITED_IMAGES_DIR> \
  --edited_pattern <EDITED_IMAGE_PATTERN> \
  --prompt <PROMPT> \
  --out_path <EDIT_3D_OUTPUT_DIR> \
  --iterations <EDIT_3D_ITERATIONS>

python refine_sds.py \
  --config <CONFIG_PATH> \
  --checkpoint <EDIT_3D_CHECKPOINT_PATH> \
  --prompt <PROMPT> \
  --out_path <SDS_OUTPUT_DIR> \
  --iterations <SDS_ITERATIONS> \
  --save_iterations <SAVE_ITERATIONS>

python render_edited4d.py \
  --config <CONFIG_PATH> \
  --checkpoint <SDS_CHECKPOINT_PATH> \
  --out_path <RENDER_OUTPUT_DIR> \
  --skip_train \
  --fps <FPS>

conda deactivate
```

Arguments:
- `<CONFIG_PATH>`: scene config file, for example `configs/dynerf/cook_spinach.yaml`.
- `<COMP_CHECKPOINT_PATH>`: compressed OMG4 checkpoint, for example `./cook_spinach_comp/comp.xz`.
- `<EDITED_IMAGES_DIR>`: directory containing the edited images generated by Instruct-4DGS.
- `<EDITED_IMAGE_PATTERN>`: edited image filename pattern, for example `edited_painting_original_time{frame_id}_{camera_id}.png`.
- `<PROMPT>`: the same edit instruction used to generate the edited images.
- `<EDIT_3D_OUTPUT_DIR>`: output directory for `edit_3d.py`.
- `<EDIT_3D_ITERATIONS>`: number of iterations for `edit_3d.py`.
- `<EDIT_3D_CHECKPOINT_PATH>`: checkpoint produced by `edit_3d.py`.
- `<SDS_OUTPUT_DIR>`: output directory for `refine_sds.py`.
- `<SDS_ITERATIONS>`: number of iterations for SDS refinement.
- `<SAVE_ITERATIONS>`: checkpoint save iterations, for example `100 300 500 800`.
- `<SDS_CHECKPOINT_PATH>`: checkpoint produced by `refine_sds.py`.
- `<RENDER_OUTPUT_DIR>`: output directory for rendered edited videos.
- `<FPS>`: rendered video FPS.
