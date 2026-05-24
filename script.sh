echo "Editing with prompt: Make it look like a Fauvism painting"
python edit_3d.py \
  --config configs/dynerf/cook_spinach.yaml \
  --comp_checkpoint ./cook_spinach_comp/comp.xz \
  --edited_images_path /media/ai2lab/SSD4TB/EV_final/data/N3DV/cook_spinach/painting \
  --edited_pattern "edited_painting_original_time{frame_id}_{camera_id}.png" \
  --prompt "Make it look like a fauvism painting" \
  --out_path ./cook_spinach_edit_painting \
  --iterations 1000

python refine_sds.py \
  --config configs/dynerf/cook_spinach.yaml \
  --checkpoint ./cook_spinach_edit_painting/chkpnt_edit_Make_it_look_like_a_fauvism_painting_1000.pth \
  --prompt "Make it look like a fauvism painting" \
  --out_path ./cook_spinach_edit_sds_painting \
  --iterations 800 \
  --save_iterations 100 300 500 800

python render_edited4d.py \
  --config configs/dynerf/cook_spinach.yaml \
  --checkpoint ./cook_spinach_edit_sds_painting/chkpnt_sds_Make_it_look_like_a_fauvism_painting_800.pth \
  --out_path ./cook_spinach_edit_sds_render_painting \
  --skip_train \
  --fps 30



echo "Editing with prompt: Make it look like a sculpture"
python edit_3d.py \
  --config configs/dynerf/cook_spinach.yaml \
  --comp_checkpoint ./cook_spinach_comp/comp.xz \
  --edited_images_path /media/ai2lab/SSD4TB/EV_final/data/N3DV/cook_spinach/sculpture \
  --edited_pattern "edited_sculpture_original_time{frame_id}_{camera_id}.png" \
  --prompt "Make it look like a sculpture" \
  --out_path ./cook_spinach_edit_sculpture \
  --iterations 1000

python refine_sds.py \
  --config configs/dynerf/cook_spinach.yaml \
  --checkpoint ./cook_spinach_edit_sculpture/chkpnt_edit_Make_it_look_like_a_sculpture_1000.pth \
  --prompt "Make it look like a sculpture" \
  --out_path ./cook_spinach_edit_sds_sculpture \
  --iterations 800 \
  --save_iterations 100 300 500 800

python render_edited4d.py \
  --config configs/dynerf/cook_spinach.yaml \
  --checkpoint ./cook_spinach_edit_sds_sculpture/chkpnt_sds_Make_it_look_like_a_sculpture_800.pth \
  --out_path ./cook_spinach_edit_sds_render_sculpture \
  --skip_train \
  --fps 30

echo "Editing with prompt: Turn the man into a woman"
python edit_3d.py \
  --config configs/dynerf/cook_spinach.yaml \
  --comp_checkpoint ./cook_spinach_comp/comp.xz \
  --edited_images_path /media/ai2lab/SSD4TB/EV_final/data/N3DV/cook_spinach/woman \
  --edited_pattern "edited_woman_original_time{frame_id}_{camera_id}.png" \
  --prompt "Turn the man into a woman" \
  --out_path ./cook_spinach_edit_woman \
  --iterations 1000

python refine_sds.py \
  --config configs/dynerf/cook_spinach.yaml \
  --checkpoint ./cook_spinach_edit_woman/chkpnt_edit_Turn_the_man_into_a_woman_1000.pth \
  --prompt "Turn the man into a woman" \
  --out_path ./cook_spinach_edit_sds_woman \
  --iterations 800 \
  --save_iterations 100 300 500 800

python render_edited4d.py \
  --config configs/dynerf/cook_spinach.yaml \
  --checkpoint ./cook_spinach_edit_sds_woman/chkpnt_sds_Turn_the_man_into_a_woman_800.pth \
  --out_path ./cook_spinach_edit_sds_render_woman \
  --skip_train \
  --fps 30