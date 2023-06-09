export CUDA_VISIBLE_DEVICES=1
python grounded_sam.py \
  --config GroundingDINO/groundingdino/config/GroundingDINO_SwinB.cfg.py \
  --grounded_checkpoint checkpoints/groundingdino_swinb_cogcoor.pth \
  --sam_checkpoint checkpoints/sam_vit_h_4b8939.pth \
  --input_image example.jpg \
  --specific_label 'human face' \
  --output_dir "outputs" \
  --box_threshold 0.3 \
  --text_threshold 0.25 \
  --device "cuda"
