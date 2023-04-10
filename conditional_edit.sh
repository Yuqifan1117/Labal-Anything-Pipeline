export CUDA_VISIBLE_DEVICES=0
python grounded_edit.py \
  --config GroundingDINO/groundingdino/config/GroundingDINO_SwinB.cfg.py \
  --grounded_checkpoint checkpoints/groundingdino_swinb_cogcoor.pth \
  --sam_checkpoint checkpoints/sam_vit_h_4b8939.pth \
  --input_image example.jpg \
  --output_dir "outputs" \
  --box_threshold 0.3 \
  --text_threshold 0.25 \
  --text_prompt "a surfboard man" \
  --device "cuda"
